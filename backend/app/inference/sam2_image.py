"""SAM 2.1 단일 프레임 분할 서비스 (공식 SAM2ImagePredictor 사용).

- 같은 (세션, 프레임)에 대한 연속 클릭은 이미지 임베딩을 재사용합니다 (set_image 1회 → predict 여러 번).
- 추론은 float32 로 실행합니다. 장치별 처리는 runtime 계층이 담당합니다.
- 실패 시 오류를 그대로 전달합니다. Mock 결과로 대체하지 않습니다.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from ..errors import AppError
from .engine import Sam2Engine

log = logging.getLogger(__name__)


@dataclass
class SegmentOutput:
    mask: np.ndarray  # (H, W) uint8, 원본 프레임 해상도
    score: float
    metrics: dict
    #: 한 번 클릭(multimask)일 때 SAM 이 낸 후보 3개 [(마스크, 점수)], 그 외에는 None
    candidates: list[tuple[np.ndarray, float]] | None = None
    #: candidates 중 기본 선택(점수 최고) 번호
    best: int = 0


class Sam2ImageService:
    #: 추적 작업이 모델을 쓰고 있을 때 분할 요청이 기다리는 최대 시간(초)
    BUSY_TIMEOUT_S = 2.0

    def __init__(self, engine: Sam2Engine):
        self.engine = engine
        self._image_key: tuple[str, int] | None = None

    def status(self) -> dict:
        return self.engine.status()

    def warmup(self) -> dict:
        """첫 요청의 지연 초기화 비용을 미리 치르고, 그 시간을 측정해 둡니다."""
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        img[100:260, 250:390] = (220, 80, 40)
        with self.engine.exclusive("warmup"):
            out = self._segment_locked(("__warmup__", 0), lambda: img, np.array([[320.0, 180.0]]), np.array([1]))
            self._image_key = None
            self.engine.image_predictor.reset_predictor()
        return {k: out.metrics[k] for k in ("setImageMs", "predictMs")}

    def segment(
        self,
        image_key: tuple[str, int],
        load_image: Callable[[], np.ndarray],
        points_px: np.ndarray,
        labels: np.ndarray,
        features=None,
    ) -> SegmentOutput:
        """features: 새 프레임일 때 set_image 대신 쓸 공유 특징 공급자 (원격 GPU 워커, features.py 참고)."""
        self.engine.require_ready()
        runtime = self.engine.runtime
        with self.engine.exclusive("segmentation", timeout=self.BUSY_TIMEOUT_S):
            try:
                return self._segment_locked(image_key, load_image, points_px, labels, features)
            except AppError:
                raise
            except Exception as e:
                self._recover_after_failure()
                if runtime.is_oom(e):
                    raise AppError(
                        "gpu_oom",
                        "GPU out of memory during segmentation. Restart the backend with OMS_DEVICE=cpu if this persists.",
                        503,
                    ) from e
                log.exception("Segmentation failed")
                raise AppError("inference_failed", f"SAM 2.1 inference failed: {type(e).__name__}: {e}", 500) from e

    def _recover_after_failure(self) -> None:
        self._image_key = None
        try:
            self.engine.image_predictor.reset_predictor()
        except Exception:  # noqa: BLE001
            pass
        self.engine.runtime.empty_cache()

    def _segment_locked(self, image_key, load_image, points_px, labels, features=None) -> SegmentOutput:
        runtime = self.engine.runtime
        predictor = self.engine.image_predictor
        runtime.reset_peak_memory()
        t_total = time.perf_counter()
        cached = self._image_key == image_key
        set_image_ms = 0.0
        with torch.inference_mode(), runtime.autocast():
            shared = False
            if not cached:
                t = time.perf_counter()
                if features is not None:
                    # 원격 GPU 워커: 추적과 공유하는 백본 특징 (캐시에 있으면 인코딩 생략)
                    from .features import set_image_from_features

                    feats, shared = features()
                    set_image_from_features(predictor, feats)
                else:
                    image = load_image()
                    predictor.set_image(image)  # 내부적으로 1024x1024 로 리사이즈 후 인코딩
                runtime.synchronize()
                set_image_ms = (time.perf_counter() - t) * 1000
                self._image_key = image_key

            # 점이 하나면 모호성이 크므로 3개 후보 중 최고 점수 선택 (SAM 권장 방식)
            multimask = len(points_px) == 1
            t = time.perf_counter()
            masks, scores, _ = predictor.predict(
                point_coords=points_px.astype(np.float32),
                point_labels=labels.astype(np.int32),
                multimask_output=multimask,
                normalize_coords=True,
            )
            runtime.synchronize()
            predict_ms = (time.perf_counter() - t) * 1000

        best = int(np.argmax(scores))
        mask = (masks[best] > 0).astype(np.uint8)
        info = self.engine.device_info
        metrics = {
            "device": runtime.name,
            "deviceName": (info.gpu.name if runtime.name == "cuda" else "CPU"),
            "dtype": runtime.precision,
            "embeddingCached": cached,
            "embeddingShared": shared,
            "setImageMs": round(set_image_ms, 1),
            "predictMs": round(predict_ms, 1),
            "inferenceMs": round((time.perf_counter() - t_total) * 1000, 1),
            "multimask": multimask,
            "score": round(float(scores[best]), 4),
            **runtime.memory_snapshot(),
        }
        candidates = [((m > 0).astype(np.uint8), float(sc)) for m, sc in zip(masks, scores)] if multimask else None
        return SegmentOutput(mask=mask, score=float(scores[best]), metrics=metrics, candidates=candidates, best=best)
