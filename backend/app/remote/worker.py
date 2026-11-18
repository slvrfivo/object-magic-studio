"""원격 GPU 워커: GPU 컨테이너(Modal) 안에서 SAM 2.1 분할과 추적을 실행합니다.

- 모델은 컨테이너 시작 시 1회 로드합니다 (Sam2Engine: 영상 예측기와 이미지 예측기가 가중치 공유).
- 영상은 SHA-256 으로 식별합니다. 업로드된 MP4 를 서버와 같은 코드(app.media.extract_frames)로 프레임 추출합니다.
- 분할과 추적은 같은 프레임 특징 캐시(FeatureCache)를 공유합니다:
  첫 프레임 클릭 → 인코딩 1회 → 추가 클릭은 디코더만 → 추적 시작 시 그 특징을 다시 사용.
- 임시 파일은 컨테이너 로컬 디스크에만 둡니다 (영구 저장소 없음). 영상 수가 한도를 넘으면 오래된 것부터 삭제하고,
  drop_video 호출이나 컨테이너 종료 시 모두 삭제됩니다.
- 모든 메서드는 예외 대신 {"ok": False, "error": {...}} 를 반환합니다 (전송 계층과 무관하게 오류 코드 유지).
"""

from __future__ import annotations

import hashlib
import logging
import queue
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np
import torch

from ..config import Settings
from ..errors import AppError
from ..inference.engine import Sam2Engine
from ..inference.features import FeatureCache, encode_frame
from ..inference.sam2_image import Sam2ImageService
from ..inference.sam2_video import TrackRequest, VideoTracker
from ..jobs import Job
from ..masks import rle_decode, rle_encode
from ..media import VideoMeta, extract_frames, load_frame_rgb

log = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
_SHA_LEN = 64


def ok(**data) -> dict:
    return {"ok": True, **data}


def err(code: str, message: str, status: int = 400) -> dict:
    return {"ok": False, "error": {"code": code, "message": message, "status": status}}


def _is_sha(s: str) -> bool:
    return isinstance(s, str) and len(s) == _SHA_LEN and all(c in "0123456789abcdef" for c in s)


class _Video:
    def __init__(self, sha: str, root: Path, meta: VideoMeta):
        self.sha = sha
        self.root = root
        self.meta = meta

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"


class _JobAdapter(Job):
    """VideoTracker 가 쓰는 Job 인터페이스 → 진행 콜백 / 취소 확인 함수."""

    def __init__(self, on_update: Callable[[float, str, dict], None], should_cancel: Callable[[], bool], check_every_s: float):
        super().__init__(id=f"remote-{uuid.uuid4().hex[:12]}", kind="propagate", session_id="remote")
        self._on_update = on_update
        self._should_cancel = should_cancel
        self._check_every = check_every_s
        self._last_check = 0.0
        self._cancelled = False

    @property
    def cancelled(self) -> bool:  # type: ignore[override]
        # 취소 확인(원격 저장소 조회)은 비용이 있으므로 간격을 둡니다
        now = time.monotonic()
        if not self._cancelled and now - self._last_check >= self._check_every:
            self._last_check = now
            try:
                self._cancelled = bool(self._should_cancel())
            except Exception:  # noqa: BLE001 — 확인 실패는 취소로 보지 않음
                log.warning("cancel check failed", exc_info=True)
        return self._cancelled

    def update(self, progress=None, message=None, stats=None) -> None:
        super().update(progress, message, stats)
        self._on_update(self.progress, self.message, self.stats)


class GpuWorker:
    def __init__(self, settings: Settings, work_root: Path, *, max_videos: int = 3, feature_cache: int = 2, engine: Sam2Engine | None = None):
        self.settings = settings
        self.work_root = Path(work_root)
        self.max_videos = max_videos
        self.engine = engine or Sam2Engine(settings)
        self._owns_engine = engine is None
        self.images = Sam2ImageService(self.engine)
        self.tracker = VideoTracker(self.engine, settings)
        self.features = FeatureCache(feature_cache)
        self._videos: OrderedDict[str, _Video] = OrderedDict()
        self._videos_lock = threading.Lock()
        self.started_at = time.time()
        self.counters = {"segments": 0, "tracks": 0, "uploads": 0, "encodes": 0}

    # ------------------------------------------------------------------ 수명
    def load(self) -> None:
        if self._owns_engine and self.engine.state == "not_loaded":
            self.engine.load(warmup=self.images.warmup)
        if self.engine.state != "ready":
            raise RuntimeError(f"SAM 2.1 failed to load: {self.engine.error}")
        self.work_root.mkdir(parents=True, exist_ok=True)

    def shutdown(self) -> None:
        with self._videos_lock:
            self._videos.clear()
        self.features.clear()
        shutil.rmtree(self.work_root, ignore_errors=True)

    def info(self) -> dict:
        st = self.engine.status()
        return ok(
            protocol=PROTOCOL_VERSION,
            model=st,
            maxTrackFrames=self.tracker.max_frames(),
            videos=len(self._videos),
            featureCache={"size": len(self.features), "hits": self.features.hits, "misses": self.features.misses},
            counters=dict(self.counters),
            uptimeS=round(time.time() - self.started_at, 1),
        )

    # ------------------------------------------------------------------ 영상
    def has_video(self, video_sha: str) -> dict:
        with self._videos_lock:
            v = self._videos.get(video_sha)
        return ok(present=v is not None, meta=v.meta.to_json() if v else None)

    def put_video(self, video_sha: str, data: bytes) -> dict:
        if not _is_sha(video_sha):
            return err("validation", "Invalid video id.", 422)
        if not isinstance(data, (bytes, bytearray)) or not data:
            return err("validation", "Empty video upload.", 422)
        if len(data) > self.settings.max_file_size_bytes:
            return err("validation", f"Video is larger than {self.settings.max_file_size_bytes // 2**20} MB.", 413)
        if hashlib.sha256(data).hexdigest() != video_sha:
            return err("validation", "Uploaded video does not match its SHA-256 (corrupted upload).", 422)
        with self._videos_lock:
            if video_sha in self._videos:
                self._videos.move_to_end(video_sha)
                return ok(meta=self._videos[video_sha].meta.to_json(), cached=True)
        root = self.work_root / "videos" / video_sha
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        try:
            (root / "input.mp4").write_bytes(data)
            t = time.perf_counter()
            meta = extract_frames(root / "input.mp4", root / "frames")  # 서버와 같은 검증·추출 코드
            extract_ms = round((time.perf_counter() - t) * 1000, 1)
        except AppError as e:
            shutil.rmtree(root, ignore_errors=True)
            return err(e.code, e.message, e.status)
        except Exception as e:  # noqa: BLE001
            shutil.rmtree(root, ignore_errors=True)
            return err("validation", f"Could not decode the video on the GPU worker: {type(e).__name__}", 422)
        (root / "input.mp4").unlink(missing_ok=True)  # 프레임만 있으면 됨
        with self._videos_lock:
            self._videos[video_sha] = _Video(video_sha, root, meta)
            evicted = []
            while len(self._videos) > self.max_videos:
                old_sha, old = self._videos.popitem(last=False)
                evicted.append((old_sha, old))
        for old_sha, old in evicted:
            self.features.drop_video(old_sha)
            shutil.rmtree(old.root, ignore_errors=True)
        self.counters["uploads"] += 1
        return ok(meta=meta.to_json(), extractMs=extract_ms, cached=False)

    def drop_video(self, video_sha: str) -> dict:
        with self._videos_lock:
            v = self._videos.pop(video_sha, None)
        self.features.drop_video(video_sha)
        if v:
            shutil.rmtree(v.root, ignore_errors=True)
        return ok(dropped=v is not None)

    def _video(self, video_sha: str) -> _Video | None:
        with self._videos_lock:
            v = self._videos.get(video_sha)
            if v:
                self._videos.move_to_end(video_sha)
            return v

    # ------------------------------------------------------------------ 공유 특징
    def _features_for(self, video: _Video, frame: int):
        """(특징, 캐시 적중 여부). 호출하는 쪽이 inference_mode + autocast 안에서 부릅니다."""
        key = (video.sha, frame)
        f = self.features.get(key)
        if f is not None:
            return f, True
        f = encode_frame(self.engine.video_predictor, video.frames_dir / f"{frame:05d}.jpg", self.engine.runtime.device)
        self.features.put(key, f)
        self.counters["encodes"] += 1
        return f, False

    @staticmethod
    def _prompt(video: _Video, frame: int, points_px, labels) -> tuple[np.ndarray, np.ndarray] | dict:
        m = video.meta
        if not isinstance(frame, int) or not 0 <= frame < m.frame_count:
            return err("validation", f"frameIndex {frame} is out of range (0..{m.frame_count - 1}).", 422)
        pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 2)
        lab = np.asarray(labels, dtype=np.int32).reshape(-1)
        if len(pts) == 0 or len(pts) != len(lab) or len(pts) > 64:
            return err("validation", "Invalid prompt points.", 422)
        if not np.isfinite(pts).all() or (pts[:, 0] < 0).any() or (pts[:, 1] < 0).any() or (pts[:, 0] > m.width).any() or (pts[:, 1] > m.height).any():
            return err("validation", "Prompt points are outside the frame.", 422)
        if not set(lab.tolist()) <= {0, 1} or 1 not in lab:
            return err("validation", "At least one positive (include) point is required.", 422)
        return pts, lab

    # ------------------------------------------------------------------ 분할
    def segment(self, video_sha: str, frame_index: int, points_px, labels) -> dict:
        video = self._video(video_sha)
        if video is None:
            return err("video_missing", "The video is not on the GPU worker (container restarted). Upload it again.", 409)
        p = self._prompt(video, frame_index, points_px, labels)
        if isinstance(p, dict):
            return p
        pts, lab = p
        runtime = self.engine.runtime

        def features():
            with torch.inference_mode(), runtime.autocast():
                return self._features_for(video, frame_index)

        try:
            out = self.images.segment(
                (video.sha, frame_index),
                lambda: load_frame_rgb(video.frames_dir, frame_index),
                pts,
                lab,
                features=features,
            )
        except AppError as e:
            return err(e.code, e.message, e.status)
        self.counters["segments"] += 1
        h, w = out.mask.shape
        cands = None
        if out.candidates:
            cands = [{"mask": {"width": w, "height": h, "counts": rle_encode(m)}, "score": sc} for m, sc in out.candidates]
        return ok(mask={"width": w, "height": h, "counts": rle_encode(out.mask)}, score=out.score, metrics=out.metrics, candidates=cands, best=out.best)

    # ------------------------------------------------------------------ 추적
    def track(
        self,
        video_sha: str,
        prompt_frame: int,
        points_px,
        labels,
        max_frames: int,
        on_progress: Callable[[dict], None] = lambda _e: None,
        should_cancel: Callable[[], bool] = lambda: False,
        cancel_check_s: float = 0.5,
        mask_rle: dict | None = None,
    ) -> dict:
        """추적 실행 → {"ok", "stats", "masks": {프레임: PNG bytes}} (취소되면 완료된 프레임까지)."""
        video = self._video(video_sha)
        if video is None:
            return err("video_missing", "The video is not on the GPU worker (container restarted). Upload it again.", 409)
        p = self._prompt(video, prompt_frame, points_px, labels)
        if isinstance(p, dict):
            return p
        pts, lab = p
        mask = None
        if mask_rle is not None:
            # 사용자가 고른 마스크 후보 (원본 해상도 RLE)
            try:
                if (mask_rle["width"], mask_rle["height"]) != (video.meta.width, video.meta.height):
                    return err("validation", "Prompt mask size does not match the video.", 422)
                mask = rle_decode(mask_rle["counts"], mask_rle["width"], mask_rle["height"])
            except (KeyError, TypeError, ValueError):
                return err("validation", "Invalid prompt mask.", 422)
            if not mask.any():
                return err("validation", "The selected mask is empty.", 422)
        req = TrackRequest(prompt_frame=prompt_frame, points_px=pts, labels=lab, max_frames=int(max_frames), mask=mask)
        try:
            self.tracker.validate(req, video.meta.frame_count)
        except AppError as e:
            return err(e.code, e.message, e.status)

        # 작업마다 별도 폴더 (VideoTracker 가 tracks/<job>/ 아래에 마스크를 씀)
        run_root = self.work_root / "runs" / uuid.uuid4().hex
        run_root.mkdir(parents=True)
        session = _RunSession(run_root, video.frames_dir, video.meta)  # 프레임은 VideoTracker 가 하드링크

        last = {"t": 0.0}

        def on_update(progress: float, message: str, stats: dict) -> None:
            now = time.monotonic()
            if now - last["t"] < 0.25 and progress < 1.0:
                return  # 진행 이벤트는 초당 4회까지
            last["t"] = now
            keep = {k: stats.get(k) for k in ("framesDone", "framesRequested", "avgFrameMs", "fps", "rangeStart", "rangeEnd", "lostFrames", "unstableFrames") if k in stats}
            on_progress({"progress": round(progress, 4), "message": message, "stats": keep})

        job = _JobAdapter(on_update, should_cancel, cancel_check_s)
        runtime = self.engine.runtime
        try:
            # 분할에서 이미 인코딩한 프롬프트 프레임이면 그 특징을 추적에 재사용
            prompt_features = self.features.get((video.sha, prompt_frame))
            stats = self.tracker.track(session, req, job, prompt_features=prompt_features)
            masks_dir = run_root / "tracks" / job.id / "masks"
            masks = {int(f): (masks_dir / f"{f:05d}.png").read_bytes() for f in stats["trackedFrames"]}
        except AppError as e:
            return err(e.code, e.message, e.status)
        except Exception as e:  # noqa: BLE001
            log.exception("remote tracking failed")
            if self.engine.runtime and self.engine.runtime.is_oom(e):
                return err("gpu_oom", "GPU out of memory during tracking. Try fewer frames.", 503)
            return err("inference_failed", f"{type(e).__name__}: {e}", 500)
        finally:
            shutil.rmtree(run_root, ignore_errors=True)
        self.counters["tracks"] += 1
        stats["cancelled"] = job.cancelled
        info = self.engine.device_info
        env = {
            "device": runtime.name,
            "precision": runtime.precision,
            "gpuName": info.gpu.name if info and runtime.name == "cuda" and info.gpu else None,
            "torch": torch.__version__,
        }
        return ok(stats=stats, masks=masks, environment=env)

    def track_stream(self, *args, **kwargs) -> Iterator[dict]:
        """track() 을 백그라운드 스레드에서 실행하며 진행 이벤트를 차례로 내보내는 제너레이터.

        이벤트: {"type": "progress", ...} 여러 번 → 마지막에 {"type": "result", ...track() 결과}
        """
        q: queue.Queue = queue.Queue()

        def run():
            try:
                res = self.track(*args, on_progress=lambda e: q.put({"type": "progress", **e}), **kwargs)
            except BaseException as e:  # noqa: BLE001
                res = err("inference_failed", f"{type(e).__name__}: {e}", 500)
            q.put({"type": "result", **res})

        th = threading.Thread(target=run, name="remote-track", daemon=True)
        th.start()
        while True:
            ev = q.get()
            yield ev
            if ev["type"] == "result":
                break
        th.join(timeout=5)


class _RunSession:
    """VideoTracker 가 쓰는 세션 인터페이스 (root: 작업 폴더, frames_dir: 영상 프레임)."""

    def __init__(self, root: Path, frames_dir: Path, meta: VideoMeta):
        self.id = "remote"
        self.root = root
        self.frames_dir = frames_dir
        self.meta = meta


def png_to_mask(data: bytes) -> np.ndarray:
    m = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError("invalid mask PNG")
    return (m > 127).astype(np.uint8)
