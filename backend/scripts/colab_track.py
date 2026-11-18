"""GPU 작업 파일(oms-gpu-job)을 실행해 결과 파일(oms-gpu-result zip)을 만듭니다. Colab 노트북에서 호출합니다.

서버와 같은 app.inference 코드(Sam2Engine + VideoTracker, 공식 SAM2VideoPredictor)를 사용합니다.

사용:
  python scripts/colab_track.py --job-dir /content/oms_job --precision fp16 --out /content/result.zip
  (로컬 CPU 검증용: --device cpu --precision fp32)
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from app.config import Settings  # noqa: E402
from app.gpu_jobs import build_result, sha256_file, verify_job_dir  # noqa: E402
from app.inference.engine import Sam2Engine  # noqa: E402
from app.inference.sam2_video import TrackRequest, VideoTracker  # noqa: E402
from app.jobs import Job  # noqa: E402
from app.media import VideoMeta  # noqa: E402
from app.sessions import Session  # noqa: E402


def sam2_commit() -> str | None:
    try:
        direct = md.distribution("SAM-2").read_text("direct_url.json")
        return json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--precision", default=None, choices=["fp16", "fp32"], help="default: the precision stored in job.json")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    ap.add_argument("--checkpoint", default=None)
    a = ap.parse_args()

    job_dir = Path(a.job_dir).resolve()
    t0 = time.perf_counter()
    job = verify_job_dir(job_dir)
    precision = a.precision or job["precision"]
    n = job["tracking"]["numFrames"]
    v = job["video"]
    print(f"job {job['jobId'][:8]}: {n} frames from frame {job['tracking']['startFrame']} of {v['frameCount']} "
          f"({v['width']}x{v['height']} @ {v['fps']}), precision {precision}", flush=True)

    settings = Settings()
    settings.device_preference = a.device
    settings.precision = precision
    settings.offload_state = "auto"
    settings.max_track_frames_cpu = settings.max_track_frames_cuda = max(n, 2)
    if a.checkpoint:
        settings.sam2_checkpoint = Path(a.checkpoint).resolve()
    ck = sha256_file(settings.sam2_checkpoint)
    if ck != job["model"]["checkpointSha256"]:
        print(f"checkpoint sha256 {ck} does not match the job ({job['model']['checkpointSha256']})")
        return 2

    engine = Sam2Engine(settings)
    t = time.perf_counter()
    engine.load()
    load_s = time.perf_counter() - t
    if engine.state != "ready":
        print("MODEL LOAD FAILED:", engine.error)
        return 1
    print(f"model loaded in {load_s:.1f}s on {engine.runtime.name} ({engine.runtime.precision})", flush=True)

    num, den = (int(x) for x in v["fps"].split("/"))
    fps = num / den
    session = Session(id=job["jobId"], root=job_dir, meta=VideoMeta(width=v["width"], height=v["height"], fps=fps, frame_count=n))
    tracker = VideoTracker(engine, settings)
    pts = np.array(job["prompt"]["pointsPx"], np.float32)
    labels = np.array([p["label"] for p in job["prompt"]["points"]])
    tjob = Job(id=f"colab-{job['jobId'][:8]}", kind="propagate", session_id=job["jobId"])

    last = [0.0]

    def progress_printer():
        # VideoTracker 가 job.update 로 갱신하는 메시지를 10% 단위로 출력
        orig = tjob.update

        def update(progress=None, message=None, stats=None):
            orig(progress, message, stats)
            if progress is not None and (progress - last[0] >= 0.1 or progress >= 1.0):
                last[0] = progress
                print(f"  {progress * 100:5.1f}%  {tjob.message}", flush=True)

        tjob.update = update

    progress_printer()
    stats = tracker.track(session, TrackRequest(0, pts, labels, n), tjob)
    if stats["framesDone"] != n:
        print(f"TRACKING INCOMPLETE: {stats['framesDone']}/{n}")
        return 1

    masks_dir = job_dir / "tracks" / tjob.id / "masks"
    masks = []
    for i in range(n):
        m = cv2.imread(str(masks_dir / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
        masks.append((m > 127).astype(np.uint8))
    shutil.rmtree(job_dir / "tracks", ignore_errors=True)

    env = {
        "device": engine.runtime.name,
        "precision": engine.runtime.precision,
        "gpuName": torch.cuda.get_device_name(0) if engine.runtime.name == "cuda" else None,
        "torch": torch.__version__,
        "torchCuda": torch.version.cuda,
        "python": platform.python_version(),
        "sam2Commit": sam2_commit(),
        "checkpointSha256": ck,
        "stateOffload": engine.runtime.video_state_options()["offload_state_to_cpu"],
        "modelLoadS": round(load_s, 2),
    }
    keep = ("avgFrameMs", "medianFrameMs", "maxFrameMs", "fps", "initMs", "promptMs", "totalMs",
            "peakAllocatedMB", "peakReservedMB", "peakRssMB", "lostFrames", "unstableFrames", "frameMs")
    # lostFrames 등은 로컬(0부터) 번호 → 전역 번호로 변환
    out_stats = {k: stats.get(k) for k in keep}
    for k in ("lostFrames", "unstableFrames"):
        out_stats[k] = [job["tracking"]["startFrame"] + (f - stats["rangeStart"]) for f in stats.get(k, [])]
    out_stats["wallS"] = round(time.perf_counter() - t0, 2)

    out = Path(a.out)
    build_result(job, masks, env, out_stats, out)
    print(f"tracked {n} frames: avg {stats.get('avgFrameMs')} ms/frame ({stats.get('fps')} fps), "
          f"total {stats.get('totalMs', 0) / 1000:.1f}s, GPU peak {stats.get('peakAllocatedMB')} MB")
    print(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
