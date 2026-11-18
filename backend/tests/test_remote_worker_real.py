"""원격 GPU 워커를 실제 SAM 2.1 로 실행 (CPU, fp32). Modal 없이 InProcessTransport 로 전체 원격 경로를 검증합니다.

실행: pytest -m real_model tests/test_remote_worker_real.py -s   (CPU 에서 약 2~4분)

확인 항목
- 분할: 공유 특징 경로(영상 예측기 전처리)의 마스크가 기존 set_image 경로와 거의 같음
- 추가 클릭은 인코딩 없이 디코더만 실행 (embeddingCached)
- 다른 프레임을 보고 돌아와도 특징 캐시에서 재사용 (embeddingShared)
- 추적 시작 시 프롬프트 프레임 특징을 재사용 → 인코더 호출이 1회 줄고, 결과는 재사용하지 않은 경우와 같음
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from app.config import Settings
from app.inference.sam2_video import TrackRequest
from app.jobs import Job

pytestmark = pytest.mark.real_model
pytest.importorskip("sam2")

from tests.conftest import write_clip  # noqa: E402


def gt_center(frame: int, fps: int = 30) -> tuple[float, float]:
    t = frame / fps
    return 120 + 400 * (0.5 + 0.5 * np.sin(t * 1.2)), 180 + int(40 * np.cos(t * 2))


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


@pytest.fixture
def setup(engine, tmp_path):
    from app.remote.worker import GpuWorker

    clip = write_clip(tmp_path / "clip.mp4", 1.0)
    data = clip.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    w = GpuWorker(engine.settings, tmp_path / "worker", engine=engine)
    assert w.put_video(sha, data)["ok"]
    calls = {"n": 0}
    pred = engine.video_predictor
    orig = pred.forward_image

    def counting(img):
        calls["n"] += 1
        return orig(img)

    pred.forward_image = counting
    yield w, sha, calls, clip
    del pred.forward_image


def test_segment_shares_features_and_matches_set_image_path(setup, engine):
    from app.inference.sam2_image import Sam2ImageService
    from app.masks import rle_decode
    from app.media import load_frame_rgb

    w, sha, calls, _ = setup
    cx, cy = gt_center(3)
    r1 = w.segment(sha, 3, [[cx, cy]], [1])
    assert r1["ok"] and r1["metrics"]["embeddingShared"] is False and calls["n"] == 1
    m1 = rle_decode(r1["mask"]["counts"], 640, 360)
    yy, xx = np.mgrid[:360, :640]
    gt = (xx - cx) ** 2 + (yy - cy) ** 2 <= 55**2
    assert iou(m1, gt) > 0.9

    r2 = w.segment(sha, 3, [[cx, cy], [600, 40]], [1, 0])  # 추가 클릭: 디코더만
    assert r2["metrics"]["embeddingCached"] is True and calls["n"] == 1

    w.segment(sha, 10, [[*gt_center(10)]], [1])  # 다른 프레임
    assert calls["n"] == 2
    r3 = w.segment(sha, 3, [[cx, cy]], [1])  # 돌아오면 특징 캐시에서 재사용
    assert r3["metrics"]["embeddingShared"] is True and calls["n"] == 2

    # 기존 로컬 경로(set_image)와 비교: 전처리 리사이즈 방식만 달라 거의 같아야 함
    local = Sam2ImageService(engine).segment(("local", 3), lambda: load_frame_rgb(w._video(sha).frames_dir, 3), np.array([[cx, cy]], np.float32), np.array([1]))
    assert iou(m1, local.mask) > 0.97, iou(m1, local.mask)
    print(f"\nshared-feature vs set_image IoU {iou(m1, local.mask):.4f}, GT IoU {iou(m1, gt):.4f}")


def test_track_reuses_prompt_frame_features_with_identical_result(setup):
    from app.remote.worker import png_to_mask

    w, sha, calls, _ = setup
    cx, cy = gt_center(3)
    w.segment(sha, 3, [[cx, cy]], [1])
    assert calls["n"] == 1
    n = 8
    before = calls["n"]
    reused = w.track(sha, 3, [[cx, cy]], [1], n)
    assert reused["ok"] and reused["stats"]["promptFeaturesReused"] is True
    enc_reused = calls["n"] - before

    w.features.clear()
    before = calls["n"]
    fresh = w.track(sha, 3, [[cx, cy]], [1], n)
    assert fresh["stats"]["promptFeaturesReused"] is False
    enc_fresh = calls["n"] - before
    assert enc_reused == enc_fresh - 1, (enc_reused, enc_fresh)  # 프롬프트 프레임 인코딩 1회 생략

    assert reused["stats"]["trackedFrames"] == fresh["stats"]["trackedFrames"] == list(range(3, 3 + n))
    ious = [iou(png_to_mask(reused["masks"][f]), png_to_mask(fresh["masks"][f])) for f in range(3, 3 + n)]
    assert min(ious) >= 0.999, ious
    for f in range(3, 3 + n):
        m = png_to_mask(reused["masks"][f])
        ys, xs = np.nonzero(m)
        gx, gy = gt_center(f)
        assert abs(xs.mean() - gx) < 6 and abs(ys.mean() - gy) < 6
    print(f"\nencoder calls: reused {enc_reused} vs fresh {enc_fresh}; min IoU {min(ious):.4f}")


def test_full_remote_path_through_service_and_transport(setup, tmp_path):
    """로컬 서비스 → InProcessTransport(pickle 왕복) → 실제 워커 → 마스크 저장."""
    from app.remote.service import RemoteGpuService
    from app.remote.transport import InProcessTransport
    from app.remote.usage import UsageLedger
    from app.sessions import Session
    from app.media import extract_frames

    w, sha, _, clip = setup
    root = tmp_path / "sess"
    root.mkdir()
    (root / "input.mp4").write_bytes(clip.read_bytes())
    meta = extract_frames(root / "input.mp4", root / "frames")
    session = Session(id="b" * 32, root=root, meta=meta, video_sha256=sha)
    s = Settings(remote_provider="inprocess")
    ledger = UsageLedger(tmp_path / "u.json", cost_per_s=0.0002, idle_s=120, cold_start_s=45, daily_limit_s=0, monthly_limit_usd=0)
    svc = RemoteGpuService(s, InProcessTransport(w), ledger)
    cx, cy = gt_center(0)
    out = svc.segment(session, 0, np.array([[cx, cy]], np.float32), np.array([1]))
    assert out.mask.sum() > 5000 and out.metrics["backend"] == "remote"
    job = Job(id="rt", kind="propagate", session_id=session.id)
    stats = svc.track(session, TrackRequest(0, np.array([[cx, cy]], np.float32), np.array([1]), 6), job)
    assert stats["framesDone"] == 6 and stats["promptFeaturesReused"] is True
    assert len(list((root / "tracks" / "rt" / "masks").glob("*.png"))) == 6
