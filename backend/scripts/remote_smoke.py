"""원격 GPU 연결 점검 (Phase 5B-2) — 배포 직후 한 번 실행해 연결·정확도·속도·비용을 확인합니다.

사용:
  python scripts/remote_smoke.py --provider modal [--frames 300]      # 실제 Modal GPU (요금 발생, 약 $0.05 이하)
  python scripts/remote_smoke.py --provider inprocess --frames 8      # GPU 없이 로컬 모델로 같은 경로 점검

순서: 워커 상태(info) → 합성 영상 업로드 → 클릭 분할 2회(두 번째는 임베딩 재사용) → 추적 → 정답 원과 비교
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.masks import rle_decode  # noqa: E402
from app.remote.transport import InProcessTransport, ModalTransport  # noqa: E402
from app.remote.worker import png_to_mask  # noqa: E402
from tests.conftest import write_clip  # noqa: E402


def gt_center(frame: int, fps: int = 30) -> tuple[float, float]:
    t = frame / fps
    return 120 + 400 * (0.5 + 0.5 * np.sin(t * 1.2)), 180 + int(40 * np.cos(t * 2))


def gt_mask(frame: int) -> np.ndarray:
    cx, cy = gt_center(frame)
    yy, xx = np.mgrid[:360, :640]
    m = (xx - cx) ** 2 + (yy - cy) ** 2 <= 55**2
    m[260:331, 40:201] = False  # 원 위에 그려지는 초록 사각형
    return m


def iou(a, b) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["modal", "inprocess"], required=True)
    ap.add_argument("--frames", type=int, default=300)
    a = ap.parse_args()
    s = Settings()
    if a.provider == "modal":
        tr = ModalTransport(s.modal_app_name, s.modal_class_name, s.modal_cancel_dict)
    else:
        from app.remote.worker import GpuWorker

        s = Settings(device_preference="cpu", precision="fp32")
        w = GpuWorker(s, Path(tempfile.mkdtemp()) / "w")
        w.load()
        tr = InProcessTransport(w)

    t0 = time.time()
    info = tr.call("info")
    t_info = time.time() - t0
    dev = (info.get("model") or {}).get("device") or {}
    print(f"info: {t_info:.1f}s (includes cold start if the container was off) · model {info['model']['state']} "
          f"{info['model'].get('dtype')} on {dev.get('device')}"
          f"{' · ' + str((dev.get('gpu') or {}).get('name')) if dev.get('device') == 'cuda' else ''}")

    seconds = min(10.0, max(a.frames, 2) / 30 + 0.1)
    clip = write_clip(Path(tempfile.mkdtemp()) / "smoke.mp4", seconds)
    data = clip.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    t = time.time()
    r = tr.call("put_video", video_sha=sha, data=data)
    assert r["ok"], r
    print(f"upload {len(data) / 2**20:.1f} MB + frame extraction: {time.time() - t:.1f}s · {r['meta']}")

    cx, cy = gt_center(0)
    t = time.time()
    r1 = tr.call("segment", video_sha=sha, frame_index=0, points_px=[[cx, cy]], labels=[1])
    t1 = time.time() - t
    t = time.time()
    r2 = tr.call("segment", video_sha=sha, frame_index=0, points_px=[[cx, cy], [600.0, 40.0]], labels=[1, 0])
    t2 = time.time() - t
    m1 = rle_decode(r1["mask"]["counts"], 640, 360).astype(bool)
    print(f"segment #1 {t1:.2f}s (encoder {r1['metrics']['setImageMs']} ms) IoU {iou(m1, gt_mask(0)):.3f} · "
          f"segment #2 {t2:.2f}s (embedding cached: {r2['metrics']['embeddingCached']})")

    n = min(a.frames, r["meta"]["frameCount"] if "meta" in r else a.frames)
    t = time.time()
    result, events = None, 0
    for ev in tr.stream("track_stream", video_sha=sha, prompt_frame=0, points_px=[[cx, cy]], labels=[1], max_frames=n, cancel_key=f"smoke-{sha[:8]}"):
        if ev["type"] == "progress":
            events += 1
        else:
            result = ev
    wall = time.time() - t
    assert result and result["ok"], result
    st = result["stats"]
    ious = [iou(png_to_mask(result["masks"][f]).astype(bool), gt_mask(f)) for f in st["trackedFrames"]]
    print(f"track {len(ious)} frames: wall {wall:.1f}s, GPU {st.get('avgFrameMs')} ms/frame, peak {st.get('peakAllocatedMB')} MB, "
          f"prompt features reused: {st.get('promptFeaturesReused')}, progress events {events}")
    print(f"  IoU vs ground truth: mean {np.mean(ious):.3f}, min {min(ious):.3f}")
    tr.call("drop_video", video_sha=sha)
    total = time.time() - t0
    rate = s.remote_cost_per_s
    print(f"total {total:.0f}s → estimated cost incl. {s.remote_idle_s:.0f}s idle: ${(total + s.remote_idle_s) * rate:.4f} (at ${rate}/s)")
    ok = np.mean(ious) > 0.9 and st["trackedFrames"] == list(range(len(ious)))
    print("SMOKE TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
