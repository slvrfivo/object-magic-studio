"""오프라인 GPU 작업: 웹앱 ↔ Colab 사이에서 주고받는 파일 형식과 검증 (서버와 Colab 스크립트가 공용).

작업 파일 (웹앱 → Colab), oms-gpu-job v1 zip:
  job.json          작업 정보: 영상 해시/해상도/FPS/프레임 수, 프롬프트, 추적 구간, 모델, 정밀도, 프레임 해시
  frames/00000.jpg  추적 구간의 프레임 (업로드 시 추출한 JPEG 그대로, 로컬 0번 = 프롬프트 프레임)
  app/..., scripts/colab_track.py   Colab 에서 실행할 추론 코드 (서버와 같은 코드)

결과 파일 (Colab → 웹앱), oms-gpu-result v1 zip:
  result.json       작업 정보 사본 + 실행 환경(장치/정밀도/버전) + 측정값 + 마스크 해시 목록
  masks/NNNNN.png   전역 프레임 번호별 원본 해상도 이진 마스크 (8-bit 단일 채널, 0/255)

가져오기 검증: 형식/버전, 서버에 기록된 작업과의 일치(jobId, 영상 해시, 프롬프트, 구간),
현재 세션 영상과의 일치(해시, 해상도, FPS, 프레임 수), 체크포인트 해시, 정밀도, 장치별 프레임 한도,
마스크 파일 목록/해시/디코딩/크기/이진 여부, zip 경로 안전성.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
import uuid
import zipfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

JOB_FORMAT, RESULT_FORMAT, VERSION = "oms-gpu-job", "oms-gpu-result", 1
PRECISIONS = ("fp16", "fp32")
MASK_NAME = re.compile(r"^masks/(\d{5})\.png$")
MAX_RESULT_BYTES = 300 * 1024 * 1024
MAX_RESULT_ENTRIES = 1000


class GpuJobError(ValueError):
    """잘못된 작업/결과 파일. message 는 사용자에게 그대로 보여 줌."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fps_string(fps: float | Fraction) -> str:
    fr = fps if isinstance(fps, Fraction) else Fraction(fps).limit_denominator(1001)
    return f"{fr.numerator}/{fr.denominator}"


# ---------------------------------------------------------------- 작업 파일 생성 (서버)
def build_job(
    *,
    video_sha256: str,
    width: int,
    height: int,
    fps: str,
    frame_count: int,
    prompt_frame: int,
    points_norm: list[dict],
    points_px: list[list[float]],
    num_frames: int,
    precision: str,
    checkpoint_sha256: str,
    sam2_commit: str | None,
    frames_dir: Path,
) -> tuple[dict, list[Path]]:
    if precision not in PRECISIONS:
        raise GpuJobError(f"precision must be one of {PRECISIONS}")
    frames = [frames_dir / f"{prompt_frame + i:05d}.jpg" for i in range(num_frames)]
    missing = [p.name for p in frames if not p.exists()]
    if missing:
        raise GpuJobError(f"Frames missing on server: {missing[:3]}")
    job = {
        "format": JOB_FORMAT,
        "version": VERSION,
        "jobId": uuid.uuid4().hex,
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "video": {"sha256": video_sha256, "width": width, "height": height, "fps": fps, "frameCount": frame_count},
        "prompt": {
            "frameIndex": prompt_frame,
            "points": [{"x": p["x"], "y": p["y"], "label": int(p["label"])} for p in points_norm],
            "pointsPx": [[float(x), float(y)] for x, y in points_px],
        },
        "tracking": {"startFrame": prompt_frame, "numFrames": num_frames, "direction": "forward"},
        "model": {"name": "SAM 2.1 Hiera Tiny", "checkpointSha256": checkpoint_sha256, "sam2Commit": sam2_commit},
        "precision": precision,
        "frames": {"format": "jpg", "sha256": [sha256_file(p) for p in frames]},
    }
    return job, frames


def write_job_zip(job: dict, frames: list[Path], code_files: list[tuple[Path, str]], out: Path) -> None:
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:  # JPEG/PNG 는 이미 압축됨
        z.writestr("job.json", json.dumps(job, indent=1))
        for i, p in enumerate(frames):
            z.write(p, f"frames/{i:05d}.jpg")
        for src, arc in code_files:
            z.write(src, arc)


# ---------------------------------------------------------------- 작업 파일 검증 (Colab)
def verify_job_dir(job_dir: Path) -> dict:
    job = json.loads((job_dir / "job.json").read_text())
    if job.get("format") != JOB_FORMAT or job.get("version") != VERSION:
        raise GpuJobError("Not an Object Magic Studio GPU job (format/version mismatch).")
    frames = sorted((job_dir / "frames").glob("*.jpg"))
    n = job["tracking"]["numFrames"]
    if len(frames) != n or [p.name for p in frames] != [f"{i:05d}.jpg" for i in range(n)]:
        raise GpuJobError(f"Expected {n} frames in the job package, found {len(frames)}.")
    for i, (p, h) in enumerate(zip(frames, job["frames"]["sha256"])):
        if sha256_file(p) != h:
            raise GpuJobError(f"Frame {i} is corrupted (sha256 mismatch).")
    return job


# ---------------------------------------------------------------- 결과 파일 생성 (Colab)
def build_result(job: dict, masks_local: list[np.ndarray], environment: dict, stats: dict, out: Path) -> dict:
    """masks_local: 로컬 0..n-1 순서의 (H, W) 0/1 마스크. 전역 프레임 번호로 저장."""
    start = job["tracking"]["startFrame"]
    h, w = job["video"]["height"], job["video"]["width"]
    entries, hashes = [], {}
    for i, m in enumerate(masks_local):
        if m.shape != (h, w):
            raise GpuJobError(f"mask {i} has shape {m.shape}, expected {(h, w)}")
        ok, png = cv2.imencode(".png", (m > 0).astype(np.uint8) * 255)
        name = f"masks/{start + i:05d}.png"
        entries.append((name, png.tobytes()))
        hashes[f"{start + i:05d}"] = sha256_bytes(png.tobytes())
    result = {
        "format": RESULT_FORMAT,
        "version": VERSION,
        "jobId": job["jobId"],
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "video": job["video"],
        "prompt": job["prompt"],
        "tracking": {**job["tracking"], "trackedFrames": [start + i for i in range(len(masks_local))]},
        "model": job["model"],
        "precision": environment["precision"],
        "environment": environment,
        "stats": stats,
        "masks": {"format": "png", "encoding": "8-bit grayscale, 0=background 255=object", "sha256": hashes},
    }
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        z.writestr("result.json", json.dumps(result, indent=1))
        for name, data in entries:
            z.writestr(name, data)
    return result


# ---------------------------------------------------------------- 결과 파일 검증 (서버)
@dataclass
class ValidatedResult:
    result: dict
    masks: dict[int, bytes]  # 전역 프레임 번호 → PNG 바이트 (검증 완료)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise GpuJobError(msg)


def validate_result_zip(
    data: bytes,
    *,
    expected_job: dict,
    video_sha256: str,
    width: int,
    height: int,
    fps: str,
    frame_count: int,
    checkpoint_sha256: str,
    cpu_frame_limit: int,
    gpu_frame_limit: int,
) -> ValidatedResult:
    _require(len(data) <= MAX_RESULT_BYTES, "Result file is too large.")
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise GpuJobError("Not a zip file.") from None
    names = z.namelist()
    _require(len(names) <= MAX_RESULT_ENTRIES, "Result zip has too many entries.")
    for n in names:
        _require(not n.startswith(("/", "\\")) and ".." not in n.split("/") and ":" not in n, f"Unsafe path in zip: {n!r}")
        _require(n == "result.json" or MASK_NAME.match(n) is not None or n.endswith("/"), f"Unexpected file in result zip: {n!r}")
    _require("result.json" in names, "result.json is missing.")
    try:
        r = json.loads(z.read("result.json"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise GpuJobError("result.json is not valid JSON.") from None

    # 형식
    _require(r.get("format") == RESULT_FORMAT, "Not an Object Magic Studio GPU result (format mismatch).")
    _require(r.get("version") == VERSION, f"Unsupported result version {r.get('version')!r}.")
    # 서버가 만든 작업과 일치
    _require(r.get("jobId") == expected_job["jobId"], "Result belongs to a different GPU job.")
    _require(r.get("prompt") == expected_job["prompt"], "Prompt in the result does not match the exported job.")
    _require(r.get("model", {}).get("checkpointSha256") == checkpoint_sha256, "Result was produced with a different checkpoint.")
    # 영상 일치 (현재 세션)
    v = r.get("video", {})
    _require(v.get("sha256") == expected_job["video"]["sha256"] == video_sha256, "Result was produced for a different video (sha256 mismatch).")
    _require((v.get("width"), v.get("height")) == (width, height), f"Resolution mismatch: result {v.get('width')}x{v.get('height')}, video {width}x{height}.")
    _require(v.get("fps") == fps, f"FPS mismatch: result {v.get('fps')}, video {fps}.")
    _require(v.get("frameCount") == frame_count, f"Frame count mismatch: result {v.get('frameCount')}, video {frame_count}.")
    # 구간
    t = r.get("tracking", {})
    jt = expected_job["tracking"]
    _require((t.get("startFrame"), t.get("numFrames"), t.get("direction")) == (jt["startFrame"], jt["numFrames"], jt["direction"]), "Tracking range does not match the exported job.")
    start, n = jt["startFrame"], jt["numFrames"]
    expected_frames = list(range(start, start + n))
    _require(t.get("trackedFrames") == expected_frames, f"Result must contain all {n} tracked frames {start}–{start + n - 1} in order.")
    _require(start + n <= frame_count, "Tracking range exceeds the video.")
    # 정밀도 / 장치 한도
    precision = r.get("precision")
    _require(precision in PRECISIONS, f"Unsupported precision {precision!r}.")
    device = r.get("environment", {}).get("device")
    _require(device in ("cuda", "cpu"), f"Unknown device {device!r}.")
    limit = gpu_frame_limit if device == "cuda" else cpu_frame_limit
    _require(n <= limit, f"{n} frames exceeds the limit for {device} results ({limit}).")
    # 마스크
    hashes = r.get("masks", {}).get("sha256", {})
    mask_names = sorted(nm for nm in names if MASK_NAME.match(nm))
    _require(mask_names == [f"masks/{f:05d}.png" for f in expected_frames], f"Expected {n} mask files for frames {start}–{start + n - 1}.")
    _require(sorted(hashes) == [f"{f:05d}" for f in expected_frames], "Mask hash list does not match the frames.")
    masks: dict[int, bytes] = {}
    for f in expected_frames:
        b = z.read(f"masks/{f:05d}.png")
        _require(sha256_bytes(b) == hashes[f"{f:05d}"], f"Mask for frame {f} is corrupted (sha256 mismatch).")
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
        _require(img is not None, f"Mask for frame {f} is not a valid PNG.")
        _require(img.ndim == 2 and img.dtype == np.uint8, f"Mask for frame {f} must be 8-bit single-channel.")
        _require(img.shape == (height, width), f"Mask for frame {f} is {img.shape[1]}x{img.shape[0]}, expected {width}x{height}.")
        _require(set(np.unique(img).tolist()) <= {0, 255}, f"Mask for frame {f} is not binary (values other than 0/255).")
        masks[f] = b
    return ValidatedResult(result=r, masks=masks)
