"""실제 SAM 2.1 영상 추적 테스트 (CPU에서 장면당 수 분 소요).

실행: pytest -m real_model tests/test_tracking_real.py -s
교차(distractor) 장면처럼 결과가 모델 판단에 달린 경우는 scripts/run_tracking_scenarios.py 보고서로 기록합니다.
"""

import numpy as np
import pytest

from app.config import Settings

pytestmark = pytest.mark.real_model
pytest.importorskip("sam2")

from scripts.run_tracking_scenarios import run_scenario  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from tests.scenarios import SCENARIOS, contact_sheet, evaluate  # noqa: E402

#: 결과 이미지/JSON 저장 위치 (보고서용)
REPORT_DIR = Path(__file__).resolve().parents[2] / "docs" / "phase3"


@pytest.fixture(scope="module")
def tracker():
    from app.inference.engine import Sam2Engine
    from app.inference.sam2_video import VideoTracker

    s = Settings()
    if not s.sam2_checkpoint.exists():
        pytest.skip("checkpoint missing")
    engine = Sam2Engine(s)
    engine.load()
    assert engine.state == "ready", engine.error
    return VideoTracker(engine, s)


def run(tracker, name, tmp_path):
    scn = SCENARIOS[name]
    stats, masks = run_scenario(tracker, scn, tmp_path)
    rows = evaluate(scn, masks)
    contact_sheet(scn, masks, REPORT_DIR / f"{name}.png")
    keep = ("framesDone", "avgFrameMs", "medianFrameMs", "maxFrameMs", "peakRssMB", "totalMs", "lostFrames", "unstableFrames", "frameMs")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{name}.json").write_text(
        json.dumps({"description": scn.description, "stats": {k: stats.get(k) for k in keep}, "frames": rows}, indent=1)
    )
    print(f"\n[{name}] avg {stats.get('avgFrameMs')} ms/frame, lost={stats['lostFrames']}")
    for r in rows:
        print("  ", r)
    return stats, masks, rows


def test_mask_follows_moving_object(tracker, tmp_path):
    stats, masks, rows = run(tracker, "moving", tmp_path)
    assert stats["framesDone"] == SCENARIOS["moving"].frames
    assert min(r["iou"] for r in rows) >= 0.9
    assert max(r["centerErrPx"] for r in rows) <= 3.0
    # 복사된 마스크가 아님: 첫 프레임과 마지막 프레임 마스크가 정답처럼 이동해 있어야 함
    first, last = masks[0], masks[max(masks)]
    assert (first & last).sum() < first.sum()


def test_object_leaving_frame_is_reported_lost(tracker, tmp_path):
    stats, masks, rows = run(tracker, "exit", tmp_path)
    for r in rows:
        if r["gtVisiblePx"] > 2000:  # 충분히 보일 때는 잘 따라가야 함
            assert r["iou"] >= 0.85, r
        if r["gtVisiblePx"] == 0:  # 완전히 나간 뒤에는 마스크가 비어야 함 (다른 곳을 잡지 않음)
            assert r["predPx"] < 50, r
            assert r["frame"] in stats["lostFrames"]


def test_fully_occluded_object_is_not_hallucinated(tracker, tmp_path):
    stats, masks, rows = run(tracker, "occlusion", tmp_path)
    circle_area = np.pi * 50**2
    for r in rows:
        if r["frame"] < 5:
            assert r["iou"] >= 0.85, r
        if r["gtVisiblePx"] == 0:
            # 가려진 동안 회색 판이나 배경을 객체로 잡지 않아야 함
            assert r["predPx"] < 0.05 * circle_area, r
