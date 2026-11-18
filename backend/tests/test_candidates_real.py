"""실제 SAM 2.1 (CPU): 한 번 클릭의 마스크 후보 중 기본이 아닌 것을 골라 추적하면, 그 마스크로 추적이 시작되는지.

실행: pytest -m real_model tests/test_candidates_real.py -s   (CPU 약 1~2분)
장면: 파란 사각형 안의 빨간 원 (원 클릭 → 원 / 사각형 등 모호한 후보), 3px/프레임 이동
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.inference.sam2_video import TrackRequest
from app.jobs import Job

pytestmark = pytest.mark.real_model
pytest.importorskip("sam2")


def frame(i: int) -> np.ndarray:
    img = np.full((360, 640, 3), 200, np.uint8)
    x = 180 + 3 * i
    cv2.rectangle(img, (x, 90), (x + 200, 290), (200, 80, 30), -1)  # 파란 사각형 (BGR)
    cv2.circle(img, (x + 100, 190), 32, (40, 40, 230), -1)  # 빨간 원
    return img


def iou(a, b) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


@pytest.fixture(scope="module")
def engine():
    from app.inference.engine import Sam2Engine

    s = Settings(device_preference="cpu", precision="fp32")
    if not s.sam2_checkpoint.exists():
        pytest.skip("checkpoint missing")
    e = Sam2Engine(s)
    e.load()
    assert e.state == "ready", e.error
    return e


def test_chosen_candidate_drives_tracking(engine, tmp_path):
    from app.inference.sam2_image import Sam2ImageService
    from app.inference.sam2_video import VideoTracker
    from app.media import VideoMeta
    from app.sessions import Session

    root = tmp_path / "s"
    (root / "frames").mkdir(parents=True)
    for i in range(6):
        cv2.imwrite(str(root / "frames" / f"{i:05d}.jpg"), frame(i), [cv2.IMWRITE_JPEG_QUALITY, 95])
    session = Session(id="c" * 32, root=root, meta=VideoMeta(width=640, height=360, fps=30.0, frame_count=6))

    pts, labels = np.array([[280.0, 190.0]], np.float32), np.array([1])
    out = Sam2ImageService(engine).segment(("cand", 0), lambda: cv2.cvtColor(frame(0), cv2.COLOR_BGR2RGB), pts, labels)
    assert out.candidates and len(out.candidates) == 3
    areas = [float(m.mean()) for m, _ in out.candidates]
    print("\ncandidate areas", [round(a, 4) for a in areas], "scores", [round(s, 3) for _, s in out.candidates], "best", out.best)

    # 기본 후보와 가장 다른 후보를 고름
    best_mask = out.candidates[out.best][0]
    k = min((i for i in range(3) if i != out.best), key=lambda i: iou(out.candidates[i][0], best_mask))
    chosen = out.candidates[k][0]
    if iou(chosen, best_mask) > 0.9:
        pytest.skip("SAM returned near-identical candidates for this scene")

    tracker = VideoTracker(engine, engine.settings)
    st_mask = tracker.track(session, TrackRequest(0, pts, labels, 4, mask=chosen), Job(id="m", kind="propagate", session_id=session.id))
    st_pts = tracker.track(session, TrackRequest(0, pts, labels, 4), Job(id="p", kind="propagate", session_id=session.id))
    assert st_mask["promptType"] == "mask" and st_pts["promptType"] == "points"

    def mask_at(job_id, f):
        return cv2.imread(str(root / "tracks" / job_id / "masks" / f"{f:05d}.png"), cv2.IMREAD_GRAYSCALE) > 127

    first_mask, first_pts = mask_at("m", 0), mask_at("p", 0)
    print(f"frame0 IoU: mask-prompt vs chosen {iou(first_mask, chosen):.3f}, point-prompt vs chosen {iou(first_pts, chosen):.3f}, "
          f"point-prompt vs default {iou(first_pts, best_mask):.3f}")
    assert iou(first_mask, chosen) > 0.9  # 고른 후보로 시작
    assert iou(first_pts, best_mask) > 0.8  # 클릭 점 추적은 기본 후보와 비슷 (기존 동작)
    assert iou(first_mask, chosen) > iou(first_pts, chosen) + 0.1
    # 이후 프레임도 고른 후보의 크기를 유지하며 이동
    for f in range(1, 4):
        m = mask_at("m", f)
        assert abs(m.mean() - chosen.mean()) / chosen.mean() < 0.35, f
        ys, xs = np.nonzero(m)
        cy, cx = np.nonzero(chosen)
        assert xs.mean() - cx.mean() == pytest.approx(3 * f, abs=4)
