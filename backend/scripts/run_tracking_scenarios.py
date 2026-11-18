"""합성 장면으로 실제 SAM 2.1 추적을 실행하고 결과(JSON + 오버레이 이미지)를 저장합니다.

사용: python scripts/run_tracking_scenarios.py [moving exit occlusion distractor] --out ../docs/phase3
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.inference.engine import Sam2Engine  # noqa: E402
from app.inference.sam2_video import TrackRequest, VideoTracker  # noqa: E402
from app.jobs import Job  # noqa: E402
from app.media import VideoMeta  # noqa: E402
from app.sessions import Session  # noqa: E402
from tests.scenarios import FPS, H, SCENARIOS, W, contact_sheet, evaluate, write_frames  # noqa: E402


def run_scenario(tracker: VideoTracker, scn, tmp: Path):
    root = tmp / scn.name
    write_frames(scn, root / "frames")
    session = Session(id=scn.name, root=root, meta=VideoMeta(W, H, FPS, scn.frames))
    job = Job(id=f"{scn.name}-{int(time.time())}", kind="propagate", session_id=scn.name)
    req = TrackRequest(0, np.array([scn.click]), np.array([1]), scn.frames)
    stats = tracker.track(session, req, job)
    masks_dir = root / "tracks" / job.id / "masks"
    masks = {f: (cv2.imread(str(masks_dir / f"{f:05d}.png"), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) for f in stats["trackedFrames"]}
    return stats, masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--out", default=str(ROOT.parent / "docs" / "phase3"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    settings = Settings()
    engine = Sam2Engine(settings)
    engine.load()
    assert engine.state == "ready", engine.error
    tracker = VideoTracker(engine, settings)
    report = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in args.names:
            scn = SCENARIOS[name]
            stats, masks = run_scenario(tracker, scn, Path(tmp))
            rows = evaluate(scn, masks)
            contact_sheet(scn, masks, out / f"{name}.png")
            summary = {k: stats.get(k) for k in ("framesDone", "avgFrameMs", "medianFrameMs", "maxFrameMs", "peakRssMB", "totalMs", "lostFrames", "unstableFrames")}
            report[name] = {"description": scn.description, "summary": summary, "frames": rows}
            print(f"\n== {name}: {scn.description}")
            print(json.dumps(summary))
            for r in rows:
                print("  ", r)
            (out / "scenarios.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
