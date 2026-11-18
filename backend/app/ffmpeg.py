"""FFmpeg / ffprobe 탐색과 호출 도우미.

탐색 순서 (시스템 설정을 바꾸지 않음):
  1) 환경 변수 OMS_FFMPEG (ffmpeg 실행 파일 경로)
  2) 프로젝트 로컬: backend/third_party/ffmpeg/*/bin/ffmpeg(.exe)  ← scripts/setup_ffmpeg.py 로 설치
  3) PATH 의 ffmpeg
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from .config import BACKEND_ROOT

_EXE = ".exe" if os.name == "nt" else ""


@dataclass(frozen=True)
class FFmpegTools:
    ffmpeg: Path
    ffprobe: Path
    version: str
    source: str  # env | project | path


def _probe_version(ffmpeg: Path) -> str:
    out = subprocess.run([str(ffmpeg), "-hide_banner", "-version"], capture_output=True, text=True, timeout=15)
    return out.stdout.splitlines()[0] if out.returncode == 0 and out.stdout else ""


def _candidates() -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    env = os.getenv("OMS_FFMPEG")
    if env:
        found.append((Path(env), "env"))
    local_root = BACKEND_ROOT / "third_party" / "ffmpeg"
    if local_root.exists():
        for p in sorted(local_root.glob(f"*/bin/ffmpeg{_EXE}"), reverse=True):
            found.append((p, "project"))
    which = shutil.which("ffmpeg")
    if which:
        found.append((Path(which), "path"))
    return found


@lru_cache(maxsize=1)
def find_ffmpeg() -> FFmpegTools | None:
    for ffmpeg, source in _candidates():
        ffprobe = ffmpeg.with_name(f"ffprobe{_EXE}")
        if ffmpeg.is_file() and ffprobe.is_file():
            try:
                version = _probe_version(ffmpeg)
            except (OSError, subprocess.SubprocessError):
                continue
            if version:
                return FFmpegTools(ffmpeg, ffprobe, version, source)
    return None


def ffprobe_json(ffprobe: Path, path: Path, count_frames: bool = False) -> dict:
    args = [str(ffprobe), "-v", "error", "-print_format", "json", "-show_streams", "-show_format"]
    if count_frames:
        args.append("-count_frames")
    out = subprocess.run(args + [str(path)], capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr.strip()[-500:]}")
    return json.loads(out.stdout)


def video_frame_rate(ffprobe: Path, path: Path, fallback: float) -> Fraction:
    """원본 영상의 정확한 프레임레이트(예: 30000/1001). 실패 시 OpenCV 값을 유리수로 근사."""
    try:
        info = ffprobe_json(ffprobe, path)
        v = next(s for s in info["streams"] if s.get("codec_type") == "video")
        num, den = (int(x) for x in v["r_frame_rate"].split("/"))
        if num > 0 and den > 0:
            return Fraction(num, den)
    except (RuntimeError, StopIteration, KeyError, ValueError, subprocess.SubprocessError):
        pass
    return Fraction(fallback).limit_denominator(1001)


def has_audio(ffprobe: Path, path: Path) -> bool:
    try:
        info = ffprobe_json(ffprobe, path)
    except (RuntimeError, subprocess.SubprocessError):
        return False
    return any(s.get("codec_type") == "audio" for s in info["streams"])
