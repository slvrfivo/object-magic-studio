import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))


def synthetic_frame(t: float, w: int = 640, h: int = 360) -> np.ndarray:
    """scripts/make_test_clip.py 와 같은 장면 (BGR): 그라데이션 배경 + 주황 원 + 초록 사각형."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = np.linspace(40, 120, w, dtype=np.uint8)[None, :, None]
    img[:, :, 0] = 90
    cx = int(120 + 400 * (0.5 + 0.5 * np.sin(t * 1.2)))
    cy = 180 + int(40 * np.cos(t * 2))
    cv2.circle(img, (cx, cy), 55, (40, 90, 240), -1)
    cv2.rectangle(img, (40, 260), (200, 330), (60, 200, 60), -1)
    return img


def write_clip(path: Path, seconds: float, fps: int = 30, size=(640, 360)) -> Path:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(int(seconds * fps)):
        vw.write(synthetic_frame(i / fps, *size))
    vw.release()
    return path


@pytest.fixture
def clip_2s(tmp_path) -> Path:
    return write_clip(tmp_path / "clip.mp4", 2)


@pytest.fixture
def isolated_sessions(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "sessions_dir", tmp_path / "sessions")
    return settings


def pytest_configure(config):
    config.addinivalue_line("markers", "real_model: runs actual SAM 2.1 inference (slow, needs checkpoint)")
    os.environ.setdefault("OMS_LOAD_MODEL", "0")
