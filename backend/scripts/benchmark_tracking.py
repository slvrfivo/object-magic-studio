"""SAM 2.1 영상 추적 실측 벤치마크 (서비스와 동일한 코드 경로 사용).

사용법:
  python scripts/benchmark_tracking.py <video.mp4> --frames 10 30 --start 0 --point 320,220 [--gt circle]

--gt circle: scripts/make_test_clip.py 로 만든 영상이면 프레임별 정답 원 위치와 비교해
             IoU / 중심 오차를 계산합니다 ("마스크가 실제로 객체를 따라가는지" 검증).
"""

import argparse
import json
import math
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
from app.media import extract_frames  # noqa: E402
from app.sessions import Session  # noqa: E402


def gt_circle(frame: int, fps: float, w: int, h: int) -> tuple[np.ndarray, tuple[int, int]]:
    """make_test_clip.py 의 원 궤적 (반지름 55)."""
    t = frame / fps
    cx = int(120 + 400 * (0.5 + 0.5 * np.sin(t * 1.2)))
    cy = 180 + int(40 * np.cos(t * 2))
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (cx, cy), 55, 1, -1)
    return m, (cx, cy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, nargs="+", default=[10])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--point", default=None, help="x,y in pixels (default: GT circle center at start)")
    ap.add_argument("--gt", choices=["circle", "none"], default="none")
    ap.add_argument("--out", default=None, help="write JSON results here")
    args = ap.parse_args()

    settings = Settings()
    settings.max_track_frames_cpu = 10_000  # 벤치마크에서는 한도 해제
    engine = Sam2Engine(settings)
    t = time.perf_counter()
    engine.load()
    assert engine.state == "ready", engine.error
    print(f"model load {time.perf_counter() - t:.1f}s on {engine.runtime.name}")
    tracker = VideoTracker(engine, settings)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "session"
        meta = extract_frames(Path(args.video), root / "frames")
        session = Session(id="bench", root=root, meta=meta)
        print(f"video {meta.width}x{meta.height} {meta.fps}fps {meta.frame_count} frames")

        if args.point:
            px = [float(v) for v in args.point.split(",")]
        else:
            _, c = gt_circle(args.start, meta.fps, meta.width, meta.height)
            px = list(map(float, c))

        results = []
        for n in args.frames:
            job = Job(id=f"bench-{n}-{int(time.time())}", kind="propagate", session_id="bench")
            req = TrackRequest(args.start, np.array([px]), np.array([1]), n)
            stats = tracker.track(session, req, job)
            row = {k: stats.get(k) for k in ("framesDone", "initMs", "promptMs", "avgFrameMs", "medianFrameMs", "maxFrameMs", "fps", "peakRssMB", "totalMs", "lostFrames", "unstableFrames", "frameMs")}
            if args.gt == "circle":
                masks_dir = root / "tracks" / job.id / "masks"
                ious, errs = [], []
                for f in stats["trackedFrames"]:
                    pred = cv2.imread(str(masks_dir / f"{f:05d}.png"), cv2.IMREAD_GRAYSCALE) > 127
                    gt, (cx, cy) = gt_circle(f, meta.fps, meta.width, meta.height)
                    inter = (pred & (gt > 0)).sum()
                    union = (pred | (gt > 0)).sum()
                    ious.append(inter / max(union, 1))
                    ys, xs = np.nonzero(pred)
                    errs.append(math.hypot(xs.mean() - cx, ys.mean() - cy) if len(xs) else float("inf"))
                first, last = stats["trackedFrames"][0], stats["trackedFrames"][-1]
                _, c0 = gt_circle(first, meta.fps, meta.width, meta.height)
                _, c1 = gt_circle(last, meta.fps, meta.width, meta.height)
                row.update(
                    minIoU=round(min(ious), 3),
                    meanIoU=round(float(np.mean(ious)), 3),
                    maxCenterErrPx=round(max(errs), 1),
                    objectMovedPx=round(math.hypot(c1[0] - c0[0], c1[1] - c0[1]), 1),
                )
            results.append(row)
            print(json.dumps(row))
        if args.out:
            Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
