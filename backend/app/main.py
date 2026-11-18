"""Object Magic Studio API.

Phase 2: 세션 업로드 + SAM 2.1 단일 프레임 분할
Phase 3: SAM 2.1 영상 객체 추적 (백그라운드 작업)
Phase 4: 추적 마스크 + 효과 → MP4 내보내기 (FFmpeg, SAM 2 재추론 없음)
Phase 5B-1: 오프라인 GPU 작업 (작업 파일 내보내기 → Colab 추적 → 결과 가져오기)
Phase 5B-2: 원격 GPU 자동 연결 (분할·추적을 서버리스 GPU 워커에서 실행, OMS_REMOTE 설정 시)
"""

import io
import json
import logging
import re
import shutil
import subprocess
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from fractions import Fraction
from functools import lru_cache
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .config import settings
from .effects import any_effect
from .errors import AppError, app_error_handler, not_found, validation_error
from .exporter import Exporter, tracked_range
from .ffmpeg import find_ffmpeg, video_frame_rate
from .gpu_jobs import MAX_RESULT_BYTES, GpuJobError, build_job, sha256_file, validate_result_zip, write_job_zip
from .inference.sam2_video import classify_frames
from .inference.engine import Sam2Engine
from .inference.sam2_image import Sam2ImageService
from .inference.sam2_video import TrackRequest, VideoTracker
from .jobs import JobManager
from .masks import normalized_to_pixel, resize_mask, rle_encode
from .media import load_frame_rgb
from .remote.service import RemoteGpuService
from .remote.transport import InProcessTransport, ModalTransport
from .remote.usage import UsageLedger
from .remote.worker import GpuWorker
from .sessions import SessionStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = SessionStore(settings.sessions_dir)
    store.cleanup_stale(settings.session_ttl_hours)
    _cleanup_gpu_jobs()
    engine = Sam2Engine(settings)
    image_service = Sam2ImageService(engine)
    app.state.sessions = store
    app.state.engine = engine
    app.state.model = image_service
    app.state.tracker = VideoTracker(engine, settings)
    app.state.jobs = JobManager()
    app.state.exporter = Exporter(settings)
    app.state.remote = create_remote_service(settings, engine)
    if settings.load_model:
        # 서버는 바로 응답하고, 모델은 백그라운드에서 로드 (/api/health 로 상태 확인)
        threading.Thread(target=engine.load, kwargs={"warmup": image_service.warmup}, name="sam2-loader", daemon=True).start()
    yield
    app.state.jobs.shutdown()


def create_remote_service(s, engine: Sam2Engine) -> RemoteGpuService | None:
    """OMS_REMOTE 설정에 따라 원격 GPU 서비스 생성 (없으면 None → 로컬 CPU 만)."""
    if not s.remote_provider:
        return None
    ledger = UsageLedger(
        # 제공자별 파일 (개발용 inprocess 기록이 실제 Modal 사용량에 섞이지 않도록)
        s.remote_usage_file.with_name(f"{s.remote_usage_file.stem}_{s.remote_provider}.json"),
        cost_per_s=s.remote_cost_per_s,
        idle_s=s.remote_idle_s,
        cold_start_s=s.remote_cold_start_s,
        daily_limit_s=s.remote_daily_gpu_seconds,
        monthly_limit_usd=s.remote_monthly_usd,
    )
    if s.remote_provider == "modal":
        transport = ModalTransport(s.modal_app_name, s.modal_class_name, s.modal_cancel_dict)
    elif s.remote_provider == "inprocess":
        # 개발·데모용: 원격 경로(업로드 → 워커 → 결과 수신)를 로컬 모델로 실행. 모델은 로컬과 공유 (메모리 추가 없음)
        transport = InProcessTransport(GpuWorker(s, s.sessions_dir.parent / "remote_worker", engine=engine))
    else:
        raise RuntimeError(f"Unknown OMS_REMOTE provider: {s.remote_provider!r} (use 'modal' or 'inprocess')")
    return RemoteGpuService(s, transport, ledger)


app = FastAPI(title="Object Magic Studio API", version="0.5.0", lifespan=lifespan)
app.add_exception_handler(AppError, app_error_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PointIn(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    label: Literal[0, 1]


Backend = Literal["auto", "local", "remote"]


class SegmentIn(BaseModel):
    frameIndex: int = Field(ge=0)
    #: auto = 원격 GPU 가 설정되어 있으면 원격, 아니면 로컬 CPU
    backend: Backend = "auto"
    points: list[PointIn] = Field(min_length=1, max_length=64)
    maskWidth: int = Field(ge=1, le=settings.max_mask_side)
    maskHeight: int = Field(ge=1, le=settings.max_mask_side)


class NeonIn(BaseModel):
    enabled: bool
    intensity: float = Field(ge=0.0, le=1.0)
    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    thickness: float = Field(ge=1, le=10)


class AmountIn(BaseModel):
    enabled: bool
    intensity: float = Field(ge=0.0, le=1.0)


class EffectsIn(BaseModel):
    neon: NeonIn
    colorPop: AmountIn
    blur: AmountIn


class ExportIn(BaseModel):
    trackJobId: str
    effects: EffectsIn


class RenderFrameIn(BaseModel):
    frame: int = Field(ge=0)
    effects: EffectsIn
    width: int | None = Field(default=None, ge=16, le=settings.max_mask_side)
    height: int | None = Field(default=None, ge=16, le=settings.max_mask_side)


class GpuJobIn(BaseModel):
    frameIndex: int = Field(ge=0)
    maskCandidate: int | None = Field(default=None, ge=0, le=2)
    points: list[PointIn] = Field(min_length=1, max_length=64)
    maxFrames: int = Field(ge=2, le=10_000)
    precision: Literal["fp16", "fp32"] = "fp16"


class PropagateIn(BaseModel):
    frameIndex: int = Field(ge=0)
    backend: Backend = "auto"
    #: 한 번 클릭의 마스크 후보 중 사용자가 고른 번호 (마지막 분할 응답의 candidates 기준)
    maskCandidate: int | None = Field(default=None, ge=0, le=2)
    points: list[PointIn] = Field(min_length=1, max_length=64)
    maxFrames: int = Field(ge=2, le=10_000)


def _prompt_points(session, frame_index: int, points: list[PointIn]) -> tuple[np.ndarray, np.ndarray]:
    meta = session.meta
    if frame_index >= meta.frame_count:
        raise validation_error(f"frameIndex {frame_index} is out of range (0..{meta.frame_count - 1}).")
    if not any(p.label == 1 for p in points):
        raise validation_error("At least one positive (include) point is required.")
    # 정규화 좌표 → 원본 해상도 픽셀 좌표 (분할과 추적이 같은 규칙 사용)
    points_px = normalized_to_pixel([(p.x, p.y) for p in points], meta.width, meta.height)
    return points_px, np.asarray([p.label for p in points])


def _remote(request: Request, backend: str) -> RemoteGpuService | None:
    """요청한 연산 위치 결정. 원격이 필요한데 설정이 없으면 오류 (로컬로 몰래 바꾸지 않음)."""
    remote: RemoteGpuService | None = request.app.state.remote
    if backend == "local":
        return None
    if backend == "remote" and remote is None:
        raise AppError("remote_unavailable", "Remote GPU is not configured on this server (set OMS_REMOTE=modal).", 409)
    return remote


@app.get("/api/health")
def health(request: Request) -> dict:
    tracker = request.app.state.tracker
    ff = find_ffmpeg()
    return {
        "status": "ok",
        "phase": 4,
        "capabilities": {"segmentation": True, "tracking": True, "export": ff is not None},
        "model": request.app.state.engine.status(),
        "tracking": {"maxFrames": tracker.max_frames(), "lastAvgFrameMs": getattr(tracker, "last_avg_frame_ms", None)},
        "ffmpeg": {"version": ff.version, "source": ff.source} if ff else None,
        # Colab 등 외부 GPU 에서 추적한 결과 가져오기 (CPU 추적 한도와 별도)
        "gpuImport": {"maxFrames": settings.max_gpu_import_frames, "precisions": ["fp16", "fp32"], "defaultPrecision": "fp16"},
        # 원격 GPU (Phase 5B-2). 상태 조회만으로는 원격을 호출하지 않음 (요금 방지)
        "remote": request.app.state.remote.status() if request.app.state.remote else {"configured": False},
    }


@app.post("/api/remote/warmup")
async def remote_warmup(request: Request) -> dict:
    """원격 GPU 컨테이너를 미리 켜고 모델 상태 확인 (요금 발생)."""
    remote = _remote(request, "remote")
    return await run_in_threadpool(remote.warmup)


@app.post("/api/sessions")
async def create_session(request: Request, file: UploadFile = File(...)) -> dict:
    session = await request.app.state.sessions.create(file)
    return {"sessionId": session.id, "meta": session.meta.to_json()}


@app.delete("/api/sessions/{session_id}", status_code=204)
def delete_session(request: Request, session_id: str) -> None:
    request.app.state.jobs.cancel_session(session_id)
    remote = request.app.state.remote
    if remote is not None:
        try:
            remote.forget_session(request.app.state.sessions.get(session_id))
        except AppError:
            pass
    request.app.state.sessions.delete(session_id)


@app.post("/api/sessions/{session_id}/segment")
async def segment(request: Request, session_id: str, body: SegmentIn) -> dict:
    session = request.app.state.sessions.get(session_id)
    points_px, labels = _prompt_points(session, body.frameIndex, body.points)
    remote = _remote(request, body.backend)
    if remote is not None:
        out = await run_in_threadpool(remote.segment, session, body.frameIndex, points_px, labels)
    else:
        out = await run_in_threadpool(
            request.app.state.model.segment,
            (session.id, body.frameIndex),
            lambda: load_frame_rgb(session.frames_dir, body.frameIndex),
            points_px,
            labels,
        )
        out.metrics["backend"] = "local"
    preview = resize_mask(out.mask, body.maskWidth, body.maskHeight)
    meta = session.meta
    resp = {
        "mask": {"width": body.maskWidth, "height": body.maskHeight, "counts": rle_encode(preview)},
        "metrics": {**out.metrics, "frameSize": [meta.width, meta.height], "coverage": round(float(out.mask.mean()), 5)},
    }
    if out.candidates:
        # 한 번 클릭: SAM 후보 3개를 모두 돌려주고 원본 해상도 후보는 세션에 보관 (추적 때 선택 반영)
        session.last_segment = {
            "frame": body.frameIndex,
            "points_px": points_px.copy(),
            "labels": labels.copy(),
            "masks": [m for m, _ in out.candidates],
            "scores": [s for _, s in out.candidates],
            "best": out.best,
        }
        resp["candidates"] = [
            {"mask": {"width": body.maskWidth, "height": body.maskHeight, "counts": rle_encode(resize_mask(m, body.maskWidth, body.maskHeight))},
             "score": round(s, 4), "coverage": round(float(m.mean()), 5)}
            for m, s in out.candidates
        ]
        resp["selected"] = out.best
    else:
        session.last_segment = None
    return resp


def _selected_mask(session, frame_index: int, points_px, labels, candidate: int | None):
    """추적에 쓸 마스크 후보. 기본(점수 최고) 후보면 None → 기존처럼 클릭 점으로 추적 (동작 동일)."""
    if candidate is None:
        return None
    seg = session.last_segment
    if (
        seg is None
        or seg["frame"] != frame_index
        or seg["points_px"].shape != points_px.shape
        or not np.allclose(seg["points_px"], points_px, atol=1e-3)
        or not np.array_equal(seg["labels"], labels)
    ):
        raise AppError("validation", "The mask options are out of date. Click the object again, then choose a mask.", 409)
    if candidate >= len(seg["masks"]):
        raise validation_error("Unknown mask option.")
    return None if candidate == seg["best"] else seg["masks"][candidate]


@app.post("/api/sessions/{session_id}/propagate")
def propagate(request: Request, session_id: str, body: PropagateIn) -> dict:
    session = request.app.state.sessions.get(session_id)
    points_px, labels = _prompt_points(session, body.frameIndex, body.points)
    mask = _selected_mask(session, body.frameIndex, points_px, labels, body.maskCandidate)
    req = TrackRequest(prompt_frame=body.frameIndex, points_px=points_px, labels=labels, max_frames=body.maxFrames, mask=mask)
    remote = _remote(request, body.backend)
    if remote is not None:
        remote.validate(req, session.meta.frame_count)  # 프레임 한도와 예산을 큐에 넣기 전에 확인
        # 원격 작업은 별도 큐: 로컬 CPU 작업과 서로 기다리지 않음
        job = request.app.state.jobs.submit("propagate", session.id, lambda job: remote.track(session, req, job), queue="remote")
        return job.to_json()

    request.app.state.engine.require_ready()
    tracker: VideoTracker = request.app.state.tracker
    tracker.validate(req, session.meta.frame_count)  # 큐에 넣기 전에 즉시 검증

    def run(job):
        stats = tracker.track(session, req, job)
        if stats.get("avgFrameMs"):
            tracker.last_avg_frame_ms = stats["avgFrameMs"]
        return stats

    engine = request.app.state.engine
    job = request.app.state.jobs.submit("propagate", session.id, run, is_oom=lambda e: engine.runtime.is_oom(e))
    return job.to_json()


@app.get("/api/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> dict:
    return request.app.state.jobs.get(job_id).to_json()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(request: Request, job_id: str) -> dict:
    return request.app.state.jobs.cancel(job_id).to_json()


def _finished_track(request: Request, session, job_id: str):
    """내보내기/렌더링에 쓸 수 있는 추적 결과 (성공했거나, 취소됐지만 일부 프레임이 있는 작업)."""
    job = request.app.state.jobs.get(job_id)
    if job.session_id != session.id or job.kind != "propagate":
        raise not_found("Tracking result not found for this session.")
    if job.status not in ("succeeded", "cancelled") or not job.stats.get("trackedFrames"):
        raise AppError("validation", f"Tracking job is {job.status} and has no usable frames.", 409)
    return job, session.root / "tracks" / job.id


@app.get("/api/sessions/{session_id}/tracks/{job_id}/masks")
def track_masks(
    request: Request,
    session_id: str,
    job_id: str,
    width: int = Query(ge=1, le=settings.max_mask_side),
    height: int = Query(ge=1, le=settings.max_mask_side),
) -> dict:
    """추적 결과 마스크를 미리보기 크기로 리샘플해 RLE로 반환 (취소된 작업은 완료된 프레임만)."""
    session = request.app.state.sessions.get(session_id)
    job = request.app.state.jobs.get(job_id)
    if job.session_id != session.id or job.kind != "propagate":
        raise not_found("Tracking result not found for this session.")
    if job.status not in ("succeeded", "cancelled"):
        raise AppError("validation", f"Tracking job is {job.status}.", 409)
    masks_dir = session.root / "tracks" / job.id / "masks"
    frames = []
    for frame in job.stats.get("trackedFrames", []):
        m = cv2.imread(str(masks_dir / f"{frame:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise AppError("internal", f"Mask for frame {frame} is missing.", 500)
        preview = resize_mask((m > 127).astype(np.uint8), width, height)
        frames.append({"index": frame, "counts": rle_encode(preview)})
    return {"width": width, "height": height, "frames": frames}


@app.post("/api/sessions/{session_id}/tracks/{job_id}/render-frame")
def render_frame_png(request: Request, session_id: str, job_id: str, body: RenderFrameIn) -> Response:
    """내보내기와 같은 코드로 한 프레임을 렌더링한 PNG (미리보기와 결과물 일치 검증용, 무손실)."""
    session = request.app.state.sessions.get(session_id)
    job, track_dir = _finished_track(request, session, job_id)
    if body.frame not in job.stats["trackedFrames"]:
        raise validation_error(f"Frame {body.frame} was not tracked.")
    size = (body.width, body.height) if body.width and body.height else None
    img = request.app.state.exporter.render_single(session, track_dir, body.frame, body.effects.model_dump(), size)
    ok, png = cv2.imencode(".png", img)
    return Response(content=png.tobytes(), media_type="image/png")


# ---------------------------------------------------------------- 오프라인 GPU 작업 (Phase 5B-1)
_JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_BACKEND_ROOT = settings.sam2_checkpoint.parent.parent


def _cleanup_gpu_jobs() -> None:
    root = settings.gpu_jobs_dir
    if not root.exists():
        return
    cutoff = time.time() - settings.gpu_job_ttl_days * 86400
    for d in root.iterdir():
        if d.is_dir() and _JOB_ID_RE.match(d.name) and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)


@lru_cache(maxsize=4)
def _checkpoint_sha256(path: str, mtime: float) -> str:
    return sha256_file(settings.sam2_checkpoint)


def _current_checkpoint_sha256() -> str:
    ckpt = settings.sam2_checkpoint
    if not ckpt.exists():
        raise AppError("model_unavailable", "Checkpoint not found on the server.", 503)
    return _checkpoint_sha256(str(ckpt), ckpt.stat().st_mtime)


def _sam2_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(_BACKEND_ROOT / "third_party" / "sam2"), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _session_fps(session) -> str:
    """원본의 정확한 프레임레이트 (ffprobe, 없으면 OpenCV 값을 유리수로 근사)."""
    if session.fps_rational is None:
        tools = find_ffmpeg()
        if tools:
            fr = video_frame_rate(tools.ffprobe, session.root / "input.mp4", session.meta.fps)
        else:
            fr = Fraction(session.meta.fps).limit_denominator(1001)
        session.fps_rational = f"{fr.numerator}/{fr.denominator}"
    return session.fps_rational


def _package_code_files() -> list[tuple]:
    files = [(p, p.relative_to(_BACKEND_ROOT).as_posix()) for p in sorted((_BACKEND_ROOT / "app").rglob("*.py"))]
    files.append((_BACKEND_ROOT / "scripts" / "colab_track.py", "scripts/colab_track.py"))
    return files


@app.post("/api/sessions/{session_id}/gpu-jobs")
def create_gpu_job(request: Request, session_id: str, body: GpuJobIn) -> dict:
    """영상 구간 + 클릭 프롬프트를 Colab 에서 실행할 GPU 작업 파일로 만듦 (모델 불필요)."""
    session = request.app.state.sessions.get(session_id)
    points_px, labels = _prompt_points(session, body.frameIndex, body.points)
    if _selected_mask(session, body.frameIndex, points_px, labels, body.maskCandidate) is not None:
        # Colab 작업 파일은 클릭 점만 담음 → 다른 후보를 고른 경우 결과가 화면과 달라지므로 거부
        raise validation_error("Colab GPU jobs use the click points only. Keep the default mask option, or track with Local CPU / Remote GPU.")
    if body.maxFrames > settings.max_gpu_import_frames:
        raise validation_error(f"At most {settings.max_gpu_import_frames} frames per GPU job.")
    n = min(body.maxFrames, session.meta.frame_count - body.frameIndex)
    if n < 2:
        raise validation_error("Track at least 2 frames.")
    try:
        job, frames = build_job(
            video_sha256=session.video_sha256,
            width=session.meta.width,
            height=session.meta.height,
            fps=_session_fps(session),
            frame_count=session.meta.frame_count,
            prompt_frame=body.frameIndex,
            points_norm=[p.model_dump() for p in body.points],
            points_px=points_px.tolist(),
            num_frames=n,
            precision=body.precision,
            checkpoint_sha256=_current_checkpoint_sha256(),
            sam2_commit=_sam2_commit(),
            frames_dir=session.frames_dir,
        )
    except GpuJobError as e:
        raise validation_error(str(e)) from e
    job_dir = settings.gpu_jobs_dir / job["jobId"]
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(json.dumps(job, indent=1))
    write_job_zip(job, frames, _package_code_files(), job_dir / "package.zip")
    return {
        "jobId": job["jobId"],
        "startFrame": body.frameIndex,
        "numFrames": n,
        "precision": body.precision,
        "packageBytes": (job_dir / "package.zip").stat().st_size,
        "downloadUrl": f"/api/gpu-jobs/{job['jobId']}/package",
    }


@app.get("/api/gpu-jobs/{job_id}/package")
def download_gpu_job(job_id: str) -> FileResponse:
    if not _JOB_ID_RE.match(job_id) or not (settings.gpu_jobs_dir / job_id / "package.zip").exists():
        raise not_found("GPU job package not found.")
    job = json.loads((settings.gpu_jobs_dir / job_id / "job.json").read_text())
    name = f"oms_gpu_job_{job_id[:8]}_{job['tracking']['numFrames']}f_{job['precision']}.zip"
    return FileResponse(settings.gpu_jobs_dir / job_id / "package.zip", media_type="application/zip", filename=name)


@app.post("/api/sessions/{session_id}/gpu-results")
async def import_gpu_result(request: Request, session_id: str, file: UploadFile = File(...)) -> dict:
    """Colab 결과 zip 을 검증하고 일반 추적 결과로 등록 (재추론 없음)."""
    session = request.app.state.sessions.get(session_id)
    data = await file.read(MAX_RESULT_BYTES + 1)
    if len(data) > MAX_RESULT_BYTES:
        raise AppError("invalid_result", "Result file is too large.", 422)
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            gpu_job_id = json.loads(z.read("result.json")).get("jobId", "")
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        raise AppError("invalid_result", "Not a valid GPU result file (expected a zip with result.json).", 422) from None
    job_json = settings.gpu_jobs_dir / str(gpu_job_id) / "job.json"
    if not isinstance(gpu_job_id, str) or not _JOB_ID_RE.match(gpu_job_id) or not job_json.exists():
        raise AppError("invalid_result", "Unknown GPU job. Export the GPU job from this app before importing its result.", 422)
    expected = json.loads(job_json.read_text())
    ckpt_sha = _current_checkpoint_sha256()
    fps = _session_fps(session)

    def validate():
        return validate_result_zip(
            data,
            expected_job=expected,
            video_sha256=session.video_sha256,
            width=session.meta.width,
            height=session.meta.height,
            fps=fps,
            frame_count=session.meta.frame_count,
            checkpoint_sha256=ckpt_sha,
            cpu_frame_limit=settings.max_track_frames_cpu,
            gpu_frame_limit=settings.max_gpu_import_frames,
        )

    try:
        v = await run_in_threadpool(validate)
    except GpuJobError as e:
        raise AppError("invalid_result", str(e), 422) from e

    r = v.result
    frames = r["tracking"]["trackedFrames"]
    coverage = []
    for f in frames:
        m = cv2.imdecode(np.frombuffer(v.masks[f], np.uint8), cv2.IMREAD_GRAYSCALE)
        coverage.append(round(float((m > 127).mean()), 5))
    lost, unstable = classify_frames(coverage, settings.lost_coverage, settings.unstable_jump_ratio)
    env, rs = r.get("environment", {}), r.get("stats", {})
    stats = {
        "source": "gpu-import",
        "gpuJobId": gpu_job_id,
        "prompt": {"frameIndex": r["prompt"]["frameIndex"], "points": r["prompt"]["points"]},
        "precision": r["precision"],
        "device": env.get("device"),
        "gpuName": env.get("gpuName"),
        "rangeStart": frames[0],
        "rangeEnd": frames[-1],
        "framesRequested": len(frames),
        "framesDone": len(frames),
        "trackedFrames": frames,
        "coverage": coverage,
        "lostFrames": [frames[0] + i for i in lost],
        "unstableFrames": [frames[0] + i for i in unstable],
    }
    for k in ("avgFrameMs", "medianFrameMs", "maxFrameMs", "fps", "initMs", "totalMs", "peakAllocatedMB", "peakReservedMB", "frameMs"):
        stats[k] = rs.get(k)
    label = f"{env.get('gpuName') or env.get('device')} {r['precision']}"
    job = request.app.state.jobs.register("propagate", session.id, stats, f"Imported {len(frames)} frames tracked on {label}", result={"environment": env})
    masks_dir = session.root / "tracks" / job.id / "masks"
    masks_dir.mkdir(parents=True)
    for f, b in v.masks.items():
        (masks_dir / f"{f:05d}.png").write_bytes(b)
    return job.to_json()


@app.get("/api/sessions/{session_id}/tracks/compare")
def compare_tracks(request: Request, session_id: str, a: str = Query(), b: str = Query()) -> dict:
    """두 추적 결과(예: GPU fp16 vs fp32)의 프레임별 IoU."""
    session = request.app.state.sessions.get(session_id)
    ja, da = _finished_track(request, session, a)
    jb, db = _finished_track(request, session, b)
    common = sorted(set(ja.stats["trackedFrames"]) & set(jb.stats["trackedFrames"]))
    if not common:
        raise validation_error("The two tracking results have no frames in common.")
    ious, worst = [], (2.0, None, 0)
    for f in common:
        ma = cv2.imread(str(da / "masks" / f"{f:05d}.png"), cv2.IMREAD_GRAYSCALE) > 127
        mb = cv2.imread(str(db / "masks" / f"{f:05d}.png"), cv2.IMREAD_GRAYSCALE) > 127
        union = (ma | mb).sum()
        iou = float((ma & mb).sum() / union) if union else 1.0
        ious.append(iou)
        if iou < worst[0]:
            worst = (iou, f, int((ma ^ mb).sum()))
    return {"frames": len(common), "minIoU": round(min(ious), 4), "meanIoU": round(float(np.mean(ious)), 4), "worstFrame": worst[1], "worstDiffPx": worst[2]}


@app.post("/api/sessions/{session_id}/export")
def export(request: Request, session_id: str, body: ExportIn) -> dict:
    session = request.app.state.sessions.get(session_id)
    track_job, track_dir = _finished_track(request, session, body.trackJobId)
    effects = body.effects.model_dump()
    if not any_effect(effects):
        raise validation_error("Enable at least one effect before exporting.")
    exporter: Exporter = request.app.state.exporter
    exporter.require_tools()
    tracked_range(track_job.stats)  # 큐에 넣기 전에 즉시 검증
    track_stats = dict(track_job.stats)

    def run(job):
        job.result_url = f"/api/exports/{job.id}/download"
        return exporter.export(session, track_stats, track_dir, effects, job)

    # 모델을 쓰지 않으므로 별도 큐에서 실행 (추적 작업 뒤에서 기다리지 않음)
    job = request.app.state.jobs.submit("export", session.id, run, queue="export")
    return job.to_json()


@app.get("/api/exports/{job_id}/download")
def download_export(request: Request, job_id: str) -> FileResponse:
    job = request.app.state.jobs.get(job_id)
    if job.kind != "export" or job.status != "succeeded":
        raise not_found("Export not found or not finished.")
    session = request.app.state.sessions.get(job.session_id)
    path = session.root / "exports" / job.id / "output.mp4"
    if not path.exists():
        raise not_found("Export file is missing.")
    st = job.stats
    names = [n for k, n in (("neon", "neon"), ("colorPop", "colorpop"), ("blur", "blur")) if job.result and job.result.get("effects", {}).get(k)]
    fname = f"object-magic_{'-'.join(names) or 'fx'}_f{st['rangeStart'] + 1}-{st['rangeEnd'] + 1}.mp4"
    return FileResponse(path, media_type="video/mp4", filename=fname)
