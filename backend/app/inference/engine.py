"""SAM 2.1 모델 엔진: 장치 선택, 모델 1회 로드, 상태 보고, 연산 직렬화.

- 공식 build_sam2_video_predictor 로 SAM2VideoPredictor 를 만들고,
  같은 모델 인스턴스를 SAM2ImagePredictor 에도 넘겨 가중치를 한 번만 메모리에 올립니다
  (SAM2VideoPredictor 는 SAM2Base 의 하위 클래스이므로 이미지 예측기가 그대로 사용 가능).
- 단일 모델이므로 분할/추적 연산은 하나의 락으로 직렬화합니다 (CPU 과부하 방지).
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time

from ..config import Settings
from ..errors import AppError
from .device import DeviceInfo, select_device
from .runtime import InferenceRuntime, create_runtime

log = logging.getLogger(__name__)


class Sam2Engine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.state = "not_loaded"  # not_loaded | loading | ready | error
        self.error: str | None = None
        self.device_info: DeviceInfo | None = None
        self.runtime: InferenceRuntime | None = None
        self.load_ms: float | None = None
        self.warmup: dict | None = None
        self.video_predictor = None
        self.image_predictor = None
        self._lock = threading.Lock()
        self.busy_with: str | None = None

    # ------------------------------------------------------------------ 로딩
    def load(self, warmup=None) -> None:
        self.state = "loading"
        t0 = time.perf_counter()
        try:
            if not self.settings.sam2_checkpoint.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {self.settings.sam2_checkpoint}. See README for the download command."
                )
            from sam2.build_sam import build_sam2_video_predictor
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            self.device_info = select_device(self.settings.device_preference)
            self.runtime = create_runtime(self.device_info, self.settings.precision, self.settings.offload_state)
            log.info("Loading SAM 2.1 Hiera Tiny on %s (%s)", self.runtime.name, self.device_info.reason)
            self.video_predictor = build_sam2_video_predictor(
                self.settings.sam2_config, str(self.settings.sam2_checkpoint), device=self.runtime.name
            )
            self.image_predictor = SAM2ImagePredictor(self.video_predictor)
            self.load_ms = round((time.perf_counter() - t0) * 1000, 1)
            # 워밍업이 끝난 뒤에 ready 로 전환 (워밍업 중 요청이 busy 오류를 받지 않도록)
            if warmup is not None:
                self.warmup = warmup()
            self.state = "ready"
            log.info("SAM 2.1 ready (load %.0f ms, warmup %s)", self.load_ms, self.warmup)
        except Exception as e:  # noqa: BLE001 — 상태로 노출
            self.state = "error"
            self.error = f"{type(e).__name__}: {e}"
            log.exception("SAM 2.1 failed to load")

    def require_ready(self) -> None:
        if self.state == "ready":
            return
        msg = {
            "loading": "The SAM 2.1 model is still loading. Try again in a moment.",
            "error": f"The SAM 2.1 model failed to load: {self.error}",
        }.get(self.state, "The SAM 2.1 model is not loaded.")
        raise AppError("model_unavailable", msg, 503)

    @contextlib.contextmanager
    def exclusive(self, what: str, timeout: float | None = None):
        """모델 연산 구간. timeout 안에 획득하지 못하면 busy 오류."""
        acquired = self._lock.acquire(timeout=-1 if timeout is None else timeout)
        if not acquired:
            raise AppError("busy", f"The model is busy ({self.busy_with or 'another task'}). Try again when it finishes.", 409)
        self.busy_with = what
        try:
            yield
        finally:
            self.busy_with = None
            self._lock.release()

    def status(self) -> dict:
        return {
            "state": self.state,
            "error": self.error,
            "model": "SAM 2.1 Hiera Tiny",
            "checkpoint": self.settings.sam2_checkpoint.name,
            "dtype": self.runtime.precision if self.runtime else self.settings.precision,
            "loadMs": self.load_ms,
            "warmup": self.warmup,
            "busyWith": self.busy_with,
            "device": self.device_info.to_json() if self.device_info else None,
            "runtime": self.runtime.describe() if self.runtime else None,
        }
