"""로컬 FastAPI 쪽 원격 GPU 클라이언트.

역할 분리:
  - 로컬 서버: 세션·프레임·효과·MP4 내보내기, 작업 상태/취소 API, 사용량 예산 (기존 그대로)
  - 원격 워커: SAM 2.1 분할과 추적만 (app.remote.worker.GpuWorker)

흐름:
  1) 영상이 원격에 없으면 원본 MP4 를 SHA-256 과 함께 업로드 (세션당 1회, 컨테이너가 재시작되면 자동 재업로드)
     → 원격에서 같은 코드로 프레임을 추출하고, 크기·프레임 수가 로컬과 같은지 확인
  2) 분할: 원격 결과(원본 해상도 RLE)를 로컬 Sam2ImageService 와 같은 SegmentOutput 으로 변환
  3) 추적: 진행 이벤트를 받아 로컬 Job 에 반영하고, 끝나면 마스크 PNG 를 세션의 tracks/<job>/masks 에 저장
     → 기존 미리보기(/tracks/{id}/masks), 렌더링, MP4 내보내기를 그대로 사용
  4) 취소: 로컬 Job 이 취소되면 원격 취소 플래그를 켜고, 원격이 완료한 프레임까지 받아 유지 (로컬 CPU 와 같은 동작)
"""

from __future__ import annotations

import logging
import threading
import time
import uuid

import numpy as np

from ..config import Settings
from ..errors import AppError
from ..inference.sam2_image import SegmentOutput
from ..inference.sam2_video import TrackRequest, _snapshot
from ..jobs import Job
from ..masks import rle_decode, rle_encode
from .transport import RemoteTransportError, Transport
from .usage import BudgetExceeded, UsageLedger
from .worker import png_to_mask

log = logging.getLogger(__name__)

#: 원격 오류 코드 → 로컬 API 오류 코드 (frontend ErrorCode 와 일치)
_PASS_CODES = {"validation", "busy", "gpu_oom", "inference_failed", "model_unavailable", "not_found"}


def _app_error(e: dict) -> AppError:
    code = e.get("code", "remote_failed")
    status = int(e.get("status", 502))
    if code not in _PASS_CODES:
        code, status = "remote_failed", 502
    return AppError(code, f"Remote GPU: {e.get('message', 'unknown error')}", status)


class RemoteGpuService:
    def __init__(self, settings: Settings, transport: Transport, ledger: UsageLedger):
        self.settings = settings
        self.transport = transport
        self.ledger = ledger
        #: session id → 업로드한 영상 sha (원격에 있다고 알고 있는 것)
        self._uploaded: dict[str, str] = {}
        self._lock = threading.Lock()
        self.state = "idle"  # idle | ready | error
        self.last_error: str | None = None
        self.last_info: dict | None = None

    # ------------------------------------------------------------------ 상태
    @property
    def max_frames(self) -> int:
        return self.settings.remote_max_frames

    def status(self) -> dict:
        return {
            "configured": True,
            "provider": self.transport.provider,
            "gpu": self.settings.remote_gpu,
            # Modal 워커는 fp16 고정, inprocess(개발용)는 로컬 엔진 정밀도
            "precision": ((self.last_info or {}).get("model") or {}).get("dtype") or ("fp16" if self.transport.provider == "modal" else self.settings.precision),
            "state": self.state,
            "lastError": self.last_error,
            "maxFrames": self.max_frames,
            "worker": self.last_info,
            "usage": self.ledger.totals(),
        }

    def _ok(self) -> None:
        self.state, self.last_error = "ready", None

    def _fail(self, message: str) -> None:
        self.state, self.last_error = "error", message

    def _call(self, method: str, **kw) -> dict:
        try:
            r = self.transport.call(method, **kw)
        except RemoteTransportError as e:
            self._fail(e.message)
            raise AppError("remote_unavailable", e.message, 503) from e
        self._ok()
        return r

    def _budget(self, expected_active_s: float) -> None:
        try:
            self.ledger.check(expected_active_s)
        except BudgetExceeded as e:
            raise AppError("budget_exceeded", str(e), 429) from e

    def warmup(self) -> dict:
        """컨테이너를 미리 켜고 모델 상태를 확인 (요금 발생: 콜드 스타트 + 대기 시간)."""
        self._budget(1.0)
        t0 = time.time()
        r = self._call("info")
        self.ledger.record("warmup", t0, time.time())
        if not r.get("ok"):
            raise _app_error(r["error"])
        self.last_info = {k: r.get(k) for k in ("maxTrackFrames", "featureCache", "counters", "uptimeS")}
        self.last_info["model"] = {k: (r.get("model") or {}).get(k) for k in ("state", "dtype", "loadMs", "warmup")}
        dev = (r.get("model") or {}).get("device") or {}
        self.last_info["gpuName"] = (dev.get("gpu") or {}).get("name") if isinstance(dev, dict) else None
        return self.status()

    # ------------------------------------------------------------------ 영상
    def _ensure_video(self, session, force: bool = False) -> float:
        """원격에 영상이 있게 함. 반환: 업로드에 걸린 시간(ms, 없으면 0)."""
        sha = session.video_sha256
        with self._lock:
            known = self._uploaded.get(session.id) == sha
        if known and not force:
            return 0.0
        if not force:
            r = self._call("has_video", video_sha=sha)
            if r.get("ok") and r.get("present"):
                self._check_meta(session, r.get("meta"))
                with self._lock:
                    self._uploaded[session.id] = sha
                return 0.0
        data = (session.root / "input.mp4").read_bytes()
        if len(data) > self.settings.remote_max_upload_bytes:
            raise AppError("validation", f"Video is too large to send to the remote GPU (limit {self.settings.remote_max_upload_bytes // 2**20} MB).", 413)
        t = time.perf_counter()
        r = self._call("put_video", video_sha=sha, data=data)
        ms = (time.perf_counter() - t) * 1000
        if not r.get("ok"):
            raise _app_error(r["error"])
        self._check_meta(session, r.get("meta"))
        with self._lock:
            self._uploaded[session.id] = sha
        return ms

    @staticmethod
    def _check_meta(session, meta: dict | None) -> None:
        m = session.meta
        if not meta or (meta.get("width"), meta.get("height"), meta.get("frameCount")) != (m.width, m.height, m.frame_count):
            raise AppError(
                "remote_failed",
                f"The GPU worker decoded the video differently ({meta and (meta.get('width'), meta.get('height'), meta.get('frameCount'))} vs {(m.width, m.height, m.frame_count)}).",
                502,
            )

    def forget_session(self, session) -> None:
        """세션 삭제 시 원격 임시 파일도 삭제 (실패해도 무시: 컨테이너가 꺼지면 어차피 사라짐)."""
        with self._lock:
            sha = self._uploaded.pop(session.id, None)
            still_used = sha in self._uploaded.values()
        if sha and not still_used and self.state == "ready":
            try:
                self.transport.call("drop_video", video_sha=sha)
            except Exception:  # noqa: BLE001
                log.info("remote drop_video failed (ignored)", exc_info=True)

    def _with_video(self, session, fn):
        """영상이 없다는 오류(컨테이너 재시작)면 한 번 다시 올리고 재시도."""
        upload_ms = self._ensure_video(session)
        r = fn()
        if not r.get("ok") and r["error"].get("code") == "video_missing":
            upload_ms += self._ensure_video(session, force=True)
            r = fn()
        return r, upload_ms

    # ------------------------------------------------------------------ 분할
    def segment(self, session, frame_index: int, points_px: np.ndarray, labels: np.ndarray) -> SegmentOutput:
        self._budget(2.0)
        t0 = time.time()
        try:
            r, upload_ms = self._with_video(
                session,
                lambda: self._call("segment", video_sha=session.video_sha256, frame_index=int(frame_index), points_px=points_px.tolist(), labels=[int(x) for x in labels]),
            )
        finally:
            usage = self.ledger.record("segment", t0, time.time())
        if not r.get("ok"):
            raise _app_error(r["error"])
        m = r["mask"]
        if (m["width"], m["height"]) != (session.meta.width, session.meta.height):
            raise AppError("remote_failed", "Remote GPU returned a mask with unexpected dimensions.", 502)
        mask = rle_decode(m["counts"], m["width"], m["height"])
        candidates = None
        if r.get("candidates"):
            candidates = []
            for c in r["candidates"]:
                cm = c["mask"]
                if (cm["width"], cm["height"]) != (session.meta.width, session.meta.height):
                    raise AppError("remote_failed", "Remote GPU returned a candidate mask with unexpected dimensions.", 502)
                candidates.append((rle_decode(cm["counts"], cm["width"], cm["height"]), float(c["score"])))
        metrics = {
            **r["metrics"],
            "backend": "remote",
            "provider": self.transport.provider,
            "uploadMs": round(upload_ms, 1),
            "remoteRoundTripMs": round((time.time() - t0) * 1000, 1),
            "estimatedCostUSD": usage["estimatedUSD"],
            "coldStart": usage["coldStart"],
        }
        return SegmentOutput(mask=mask, score=float(r["score"]), metrics=metrics, candidates=candidates, best=int(r.get("best", 0)))

    # ------------------------------------------------------------------ 추적
    def validate(self, req: TrackRequest, frame_count: int) -> int:
        if req.max_frames < 2:
            raise AppError("validation", "Track at least 2 frames.", 422)
        if req.max_frames > self.max_frames:
            raise AppError("validation", f"At most {self.max_frames} frames can be tracked per remote GPU job.", 422)
        n = min(req.max_frames, frame_count - req.prompt_frame)
        self._budget(n * self.settings.remote_expected_frame_s)
        return n

    def track(self, session, req: TrackRequest, job: Job) -> dict:
        n = self.validate(req, session.meta.frame_count)
        stats: dict = {
            "source": "remote-gpu",
            "provider": self.transport.provider,
            "rangeStart": req.prompt_frame,
            "rangeEnd": req.prompt_frame + n - 1,
            "framesRequested": n,
            "framesDone": 0,
            "trackedFrames": [],
        }
        job.update(0.0, "Sending video to the remote GPU…", _snapshot(stats))
        key = f"oms-{job.id}-{uuid.uuid4().hex[:8]}"
        t0 = time.time()
        result = None
        upload_ms = 0.0
        # 취소 감시: 사용자가 취소하면 진행 이벤트를 기다리지 않고 즉시 원격 플래그를 켬
        # (콜드 스타트 중이거나 프레임이 느려도 바로 전달됨)
        done = threading.Event()

        def watch_cancel():
            while not done.is_set():
                if job.cancelled:
                    try:
                        self.transport.request_cancel(key)
                    except Exception:  # noqa: BLE001 — 실패하면 아래 이벤트 루프에서 다시 시도
                        log.warning("remote cancel request failed", exc_info=True)
                    return
                done.wait(0.2)

        threading.Thread(target=watch_cancel, name=f"remote-cancel-{job.id[:8]}", daemon=True).start()
        try:
            for attempt in range(2):
                upload_ms += self._ensure_video(session, force=attempt > 0)
                job.update(0.0, f"Waiting for the remote GPU ({self.settings.remote_gpu})… a cold start can take up to a minute", _snapshot(stats))
                cancel_sent = False
                try:
                    extra = {}
                    if req.mask is not None:
                        # 선택한 마스크 후보로 추적 시작 (없으면 이전 배포 워커와도 호환되도록 인자를 보내지 않음)
                        extra["mask_rle"] = {"width": session.meta.width, "height": session.meta.height, "counts": rle_encode(req.mask)}
                    events = self.transport.stream(
                        "track_stream",
                        video_sha=session.video_sha256,
                        prompt_frame=int(req.prompt_frame),
                        points_px=req.points_px.tolist(),
                        labels=[int(x) for x in req.labels],
                        max_frames=n,
                        cancel_key=key,
                        **extra,
                    )
                    for ev in events:
                        if job.cancelled and not cancel_sent:
                            self.transport.request_cancel(key)
                            cancel_sent = True
                            job.update(message="Cancelling on the remote GPU…")
                        if ev.get("type") == "progress":
                            s = ev.get("stats") or {}
                            stats.update({k: v for k, v in s.items() if k in ("framesDone", "avgFrameMs", "fps")})
                            job.update(min(ev.get("progress", 0.0), 0.97), f"Remote GPU · {ev.get('message', '')}", _snapshot(stats))
                        elif ev.get("type") == "result":
                            result = ev
                except RemoteTransportError as e:
                    self._fail(e.message)
                    raise AppError("remote_unavailable", e.message, 503) from e
                self._ok()
                if result and not result.get("ok") and result["error"].get("code") == "video_missing" and attempt == 0:
                    result = None
                    continue
                break
        finally:
            done.set()
            self.transport.clear_cancel(key)
            usage = self.ledger.record("track", t0, time.time(), tracked_frames=len((result or {}).get("masks") or {}))

        if result is None:
            raise AppError("remote_failed", "The remote GPU ended the stream without a result.", 502)
        if not result.get("ok"):
            raise _app_error(result["error"])

        # 원격 결과 → 로컬 세션의 추적 결과 (기존 미리보기/내보내기 경로 그대로)
        rstats = result["stats"]
        frames = [int(f) for f in rstats["trackedFrames"]]
        masks = result["masks"]
        expected = list(range(req.prompt_frame, req.prompt_frame + len(frames)))
        if frames != expected or set(int(k) for k in masks) != set(frames):
            raise AppError("remote_failed", "Remote GPU returned an inconsistent set of frames.", 502)
        if not rstats.get("cancelled") and len(frames) != n:
            raise AppError("remote_failed", f"Remote GPU tracked {len(frames)} of {n} frames.", 502)
        masks_dir = session.root / "tracks" / job.id / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        coverage = []
        for f in frames:
            data = masks[f] if f in masks else masks[str(f)]
            m = png_to_mask(data)
            if m.shape != (session.meta.height, session.meta.width):
                raise AppError("remote_failed", f"Remote mask for frame {f} has the wrong size.", 502)
            (masks_dir / f"{f:05d}.png").write_bytes(data)
            coverage.append(round(float(m.mean()), 5))

        keep = ("frameMs", "avgFrameMs", "medianFrameMs", "maxFrameMs", "fps", "initMs", "promptMs", "peakAllocatedMB", "peakReservedMB", "lostFrames", "unstableFrames", "promptFeaturesReused", "promptType")
        stats.update({k: rstats.get(k) for k in keep if k in rstats})
        env = result.get("environment") or {}
        stats.update(
            trackedFrames=frames,
            framesDone=len(frames),
            coverage=coverage,
            remoteTotalMs=rstats.get("totalMs"),
            totalMs=round((time.time() - t0) * 1000, 1),
            uploadMs=round(upload_ms, 1),
            device=env.get("device"),
            gpuName=env.get("gpuName"),
            precision=env.get("precision"),
            estimatedCostUSD=usage["estimatedUSD"],
            billableS=usage["billableS"],
            coldStart=usage["coldStart"],
            cancelled=bool(rstats.get("cancelled")),
        )
        if job.cancelled or stats["cancelled"]:
            job.update(message=f"Cancelled after {len(frames)}/{n} frames (remote GPU)", stats=_snapshot(stats))
        else:
            lost = len(stats.get("lostFrames") or [])
            note = f", object not visible in {lost} frame(s)" if lost else ""
            job.update(1.0, f"Tracked {len(frames)} frames on {env.get('gpuName') or 'remote GPU'} in {stats['totalMs'] / 1000:.1f}s{note}", _snapshot(stats))
        return stats
