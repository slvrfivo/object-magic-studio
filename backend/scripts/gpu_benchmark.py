"""SAM 2.1 Hiera Tiny 벤치마크 — 로컬 CPU 와 Colab GPU 에서 같은 코드로 실행합니다.

추론은 백엔드와 동일한 app.inference (Sam2Engine / Sam2ImageService / VideoTracker) 를 사용합니다.
입력은 미리 추출한 JPEG 프레임 폴더(번들에 포함)라서 CPU/GPU 가 바이트 단위로 같은 입력을 봅니다.

사용 예:
  # 실험 실행
  python scripts/gpu_benchmark.py run --frames-dir data/frames --out results/gpu_fp32 --label gpu_fp32 \\
      --segment --track 10 30 60 300 --precision fp32 --offload off --reference reference/cpu
  # 결과 요약 (여러 실험 + CPU 기준)
  python scripts/gpu_benchmark.py summarize results/* --reference reference/cpu

정답(GT): 테스트 영상은 scripts/make_test_clip.py 로 만든 합성 영상이라 프레임별 원(반지름 55px) 위치를 정확히 압니다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from app.config import Settings  # noqa: E402
from app.inference.engine import Sam2Engine  # noqa: E402
from app.inference.sam2_image import Sam2ImageService  # noqa: E402
from app.inference.sam2_video import TrackRequest, VideoTracker  # noqa: E402
from app.jobs import Job  # noqa: E402
from app.media import VideoMeta, load_frame_rgb  # noqa: E402
from app.sessions import Session  # noqa: E402

FPS = 30.0


# ---------------------------------------------------------------- 정답 / 비교
def gt_circle(frame: int) -> tuple[int, int, int]:
    t = frame / FPS
    cx = int(120 + 400 * (0.5 + 0.5 * np.sin(t * 1.2)))
    cy = 180 + int(40 * np.cos(t * 2))
    return cx, cy, 55


def gt_mask(frame: int, shape: tuple[int, int]) -> np.ndarray:
    """보이는 원 영역. make_test_clip.py 는 원을 그린 뒤 초록 사각형을 위에 그리므로 겹친 부분은 가려짐."""
    m = np.zeros(shape, np.uint8)
    cx, cy, r = gt_circle(frame)
    cv2.circle(m, (cx, cy), r, 1, -1)
    cv2.rectangle(m, (40, 260), (200, 330), 0, -1)
    return m


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def read_mask(path: Path) -> np.ndarray | None:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if m is None else (m > 127).astype(np.uint8)


def summarize_ious(values: list[float]) -> dict:
    if not values:
        return {}
    return {"min": round(min(values), 4), "mean": round(statistics.fmean(values), 4), "n": len(values)}


# ---------------------------------------------------------------- 환경 정보
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment(settings: Settings) -> dict:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchCuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cudaAvailable": torch.cuda.is_available(),
        "tf32Matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32Cudnn": torch.backends.cudnn.allow_tf32,
        "checkpoint": settings.sam2_checkpoint.name,
        "checkpointSha256": sha256(settings.sam2_checkpoint) if settings.sam2_checkpoint.exists() else None,
    }
    try:
        import importlib.metadata as md

        env["sam2Version"] = md.version("SAM-2")
        direct = md.distribution("SAM-2").read_text("direct_url.json")
        if direct:
            env["sam2Source"] = json.loads(direct)
    except Exception:  # noqa: BLE001
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update(
            gpuName=p.name,
            gpuMemoryMB=p.total_memory // 2**20,
            computeCapability=f"{p.major}.{p.minor}",
            bf16Hardware=bool(torch.cuda.is_bf16_supported(including_emulation=False)),
            archList=torch.cuda.get_arch_list(),
        )
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
            env["driver"] = out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return env


# ---------------------------------------------------------------- 실험
SEG_PROMPTS = {
    # 이름: (프레임, [(x, y, label)...]) — 640x360 테스트 영상 기준 픽셀 좌표
    "circle": (0, [(320, 220, 1)]),
    "rect": (0, [(120, 295, 1)]),
    "both": (0, [(320, 220, 1), (120, 295, 1)]),
    "both_minus_rect": (0, [(320, 220, 1), (120, 295, 1), (120, 295, 0)]),
    "circle_f100": (100, None),  # None → 해당 프레임의 정답 원 중심
    "circle_f200": (200, None),
}


def run_segmentation(svc: Sam2ImageService, session: Session, out: Path, ref: Path | None) -> dict:
    (out / "seg").mkdir(parents=True, exist_ok=True)
    res = {}
    for name, (frame, pts) in SEG_PROMPTS.items():
        if frame >= session.meta.frame_count:
            continue
        if pts is None:
            cx, cy, _ = gt_circle(frame)
            pts = [(cx, cy, 1)]
        pxy = np.array([[x, y] for x, y, _ in pts], np.float32)
        labels = np.array([lab for *_, lab in pts])
        o = svc.segment((session.id, frame), lambda f=frame: load_frame_rgb(session.frames_dir, f), pxy, labels)
        cv2.imwrite(str(out / "seg" / f"{name}.png"), o.mask * 255)
        row = dict(o.metrics)
        if name.startswith("circle"):
            row["iouVsGT"] = round(iou(o.mask, gt_mask(frame, o.mask.shape)), 4)
        if ref is not None and (r := read_mask(ref / "seg" / f"{name}.png")) is not None:
            row["iouVsCPU"] = round(iou(o.mask, r), 4)
        res[name] = row
        print(f"  seg {name:16s} encoder {row['setImageMs']:8.1f} ms  decoder {row['predictMs']:7.1f} ms  "
              f"IoU GT {row.get('iouVsGT', '-')}  IoU CPU {row.get('iouVsCPU', '-')}", flush=True)
    enc = [r["setImageMs"] for r in res.values() if not r["embeddingCached"]]
    dec = [r["predictMs"] for r in res.values()]
    return {"prompts": res, "encoderMsMedian": round(statistics.median(enc), 1), "decoderMsMedian": round(statistics.median(dec), 1)}


def run_tracking(tracker: VideoTracker, session: Session, n: int, out: Path, ref: Path | None) -> dict:
    n = min(n, session.meta.frame_count)
    cx, cy, _ = gt_circle(0)
    job = Job(id=f"bench-{n}-{int(time.time() * 1000)}", kind="propagate", session_id=session.id)
    t = time.perf_counter()
    stats = tracker.track(session, TrackRequest(0, np.array([[cx, cy]], np.float32), np.array([1]), n), job)
    wall = time.perf_counter() - t

    src = session.root / "tracks" / job.id / "masks"
    dst = out / f"track_{n}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    shutil.rmtree(session.root / "tracks" / job.id, ignore_errors=True)

    iou_gt, iou_cpu = [], []
    ref_dir = None
    if ref is not None:
        # 정방향 추적은 인과적(이전 프레임만 사용)이라 60프레임 기준 결과의 앞부분 = 10/30프레임 결과
        candidates = sorted((p for p in ref.glob("track_*") if p.is_dir()), key=lambda p: int(p.name.split("_")[1]))
        ref_dir = next((p for p in candidates if int(p.name.split("_")[1]) >= n), candidates[-1] if candidates else None)
    for f in stats["trackedFrames"]:
        m = read_mask(dst / f"{f:05d}.png")
        iou_gt.append(iou(m, gt_mask(f, m.shape)))
        if ref_dir is not None and (r := read_mask(ref_dir / f"{f:05d}.png")) is not None:
            iou_cpu.append(iou(m, r))

    keep = ("framesDone", "initMs", "promptMs", "avgFrameMs", "medianFrameMs", "maxFrameMs", "fps", "totalMs",
            "peakRssMB", "peakAllocatedMB", "peakReservedMB", "lostFrames", "unstableFrames", "frameMs")
    row = {k: stats.get(k) for k in keep}
    row["wallS"] = round(wall, 2)
    row["endToEndFps"] = round(stats["framesDone"] / wall, 3)  # 초기화·프롬프트 포함
    row["iouVsGT"] = summarize_ious(iou_gt)
    row["iouVsCPU"] = summarize_ious(iou_cpu)
    if ref_dir is not None:
        row["cpuReference"] = ref_dir.name
    print(f"  track {n:4d}: {stats['framesDone']} frames, avg {row['avgFrameMs']} ms/frame ({row['fps']} fps), "
          f"init {row['initMs']} ms, total {wall:.1f}s, GPU peak {row.get('peakAllocatedMB')} MB, "
          f"IoU GT {row['iouVsGT'].get('min')}, IoU CPU {row['iouVsCPU'].get('min', '-')}", flush=True)
    return row


def cmd_run(a) -> int:
    frames_dir = Path(a.frames_dir).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ref = Path(a.reference).resolve() if a.reference else None

    settings = Settings()
    settings.device_preference = a.device
    settings.precision = a.precision
    settings.offload_state = a.offload
    settings.max_track_frames_cpu = settings.max_track_frames_cuda = 10_000
    if a.checkpoint:
        settings.sam2_checkpoint = Path(a.checkpoint).resolve()

    result: dict = {"label": a.label, "config": {"device": a.device, "precision": a.precision, "offload": a.offload}}
    result["environment"] = environment(settings)
    save = lambda: (out / "results.json").write_text(json.dumps(result, indent=1))  # noqa: E731
    save()
    print(json.dumps(result["environment"], indent=1), flush=True)

    engine = Sam2Engine(settings)
    svc = Sam2ImageService(engine)
    t = time.perf_counter()
    engine.load(warmup=svc.warmup)
    result["load"] = {"wallS": round(time.perf_counter() - t, 2), "loadMs": engine.load_ms, "warmup": engine.warmup,
                      "state": engine.state, "error": engine.error, "runtime": engine.runtime.describe() if engine.runtime else None}
    save()
    if engine.state != "ready":
        print("MODEL LOAD FAILED:", engine.error)
        return 1
    print(f"model loaded in {result['load']['wallS']}s on {engine.runtime.name} ({engine.runtime.precision}); warmup {engine.warmup}", flush=True)

    n_frames = len(list(frames_dir.glob("*.jpg")))
    sample = cv2.imread(str(frames_dir / "00000.jpg"))
    meta = VideoMeta(width=sample.shape[1], height=sample.shape[0], fps=FPS, frame_count=n_frames)
    # Session.frames_dir == root/"frames" 이므로 프레임 폴더의 부모를 세션 루트로 사용
    assert frames_dir.name == "frames", "--frames-dir must point to a folder named 'frames'"
    session = Session(id=f"bench-{a.label}", root=frames_dir.parent, meta=meta)

    if a.segment:
        try:
            result["segmentation"] = run_segmentation(svc, session, out, ref)
        except Exception as e:  # noqa: BLE001
            result["segmentation"] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
            print("SEGMENTATION FAILED:", e)
        save()

    tracker = VideoTracker(engine, settings)
    result["tracking"] = {}
    for n in a.track:
        try:
            result["tracking"][str(n)] = run_tracking(tracker, session, n, out, ref)
        except Exception as e:  # noqa: BLE001 — 예: 300프레임 OOM 도 기록하고 다음 실험 진행
            result["tracking"][str(n)] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
            print(f"TRACKING {n} FAILED:", e)
            engine.runtime.empty_cache()
        save()
    print(f"saved {out / 'results.json'}")
    return 0


# ---------------------------------------------------------------- 요약
def _fmt(v, unit="", nd=1):
    if v is None:
        return "—"
    return f"{v:,.{nd}f}{unit}" if isinstance(v, (int, float)) else str(v)


def _cpu_prefix_ms(ref: dict, n: int) -> float | None:
    """CPU 기준 60프레임 실행의 앞 n프레임 평균 (프롬프트 프레임 제외). 추적이 인과적이라 n프레임 단독 실행과 같은 계산."""
    tr = ref.get("tracking", {})
    for key in sorted(tr, key=int):
        ms = tr[key].get("frameMs")
        if ms and int(key) >= n:
            sub = ms[1:n]
            return statistics.fmean(sub) if sub else None
    return None


def cmd_summarize(a) -> int:
    runs = []
    for d in a.dirs:
        p = Path(d) / "results.json"
        if p.exists():
            runs.append(json.loads(p.read_text()))
    ref = json.loads((Path(a.reference) / "results.json").read_text()) if a.reference else None
    lines = []
    if ref:
        runs = [ref] + [r for r in runs if r.get("label") != ref.get("label")]
    lines.append("### Environment")
    for r in runs:
        e = r["environment"]
        lines.append(f"- **{r['label']}**: torch {e['torch']} (CUDA {e.get('torchCuda')}), "
                     f"{e.get('gpuName', 'CPU')} {e.get('computeCapability', '')} {e.get('gpuMemoryMB', '')}MB, "
                     f"driver {e.get('driver', '-')}, bf16 hw {e.get('bf16Hardware', '-')}, ckpt {str(e.get('checkpointSha256'))[:12]}…")
    lines.append("\n### Model load & single-frame segmentation")
    lines.append("| run | load (s) | encoder median (ms) | decoder median (ms) | circle IoU vs GT | IoU vs CPU (min over prompts) |")
    lines.append("|---|---|---|---|---|---|")
    for r in runs:
        seg = r.get("segmentation", {})
        prompts = seg.get("prompts", {})
        gt = [v["iouVsGT"] for v in prompts.values() if "iouVsGT" in v]
        cpu = [v["iouVsCPU"] for v in prompts.values() if "iouVsCPU" in v]
        lines.append(f"| {r['label']} | {_fmt(r.get('load', {}).get('wallS'))} | {_fmt(seg.get('encoderMsMedian'))} | "
                     f"{_fmt(seg.get('decoderMsMedian'))} | {_fmt(min(gt) if gt else None, nd=4)} | {_fmt(min(cpu) if cpu else None, nd=4)} |")
    lines.append("\n### Tracking")
    lines.append("| run | frames | avg ms/frame | tracking fps | speed-up vs CPU | init (ms) | total (s) | GPU peak alloc / reserved (MB) | IoU vs GT (min / mean) | IoU vs CPU (min / mean) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        for n, t in sorted(r.get("tracking", {}).items(), key=lambda kv: int(kv[0])):
            if "error" in t:
                lines.append(f"| {r['label']} | {n} | **failed**: {t['error'][:80]} | | | | | | | |")
                continue
            cpu_ms = _cpu_prefix_ms(ref, int(n)) if ref and r is not ref else None
            avg = t.get("avgFrameMs")
            speed = f"×{cpu_ms / avg:.1f}" if cpu_ms and avg else ("(CPU)" if r is ref else "—")
            g, c = t.get("iouVsGT", {}), t.get("iouVsCPU", {})
            lines.append(
                f"| {r['label']} | {t.get('framesDone')} | {_fmt(avg)} | {_fmt(t.get('fps'), nd=2)} | {speed} | {_fmt(t.get('initMs'))} | "
                f"{_fmt(t.get('wallS'))} | {_fmt(t.get('peakAllocatedMB'))} / {_fmt(t.get('peakReservedMB'))} | "
                f"{_fmt(g.get('min'), nd=4)} / {_fmt(g.get('mean'), nd=4)} | {_fmt(c.get('min'), nd=4)} / {_fmt(c.get('mean'), nd=4)} |"
            )
    if ref:
        lines.append("\nSpeed-up and IoU vs CPU use the first N frames of the 60-frame CPU reference run. "
                     "Note: SAM 2 uses min(num_frames, 16) object pointers (sam2_base.py), so runs shorter than 16 frames "
                     "compute slightly differently from longer runs (10-frame run vs first 10 frames of a 60-frame run: IoU ~0.997 even on the same device). "
                     "Runs of equal length are identical on CPU and GPU fp32 (IoU 1.0).")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    return 0


def main() -> int:
    try:  # Windows 콘솔(cp949)에서도 표/기호 출력이 깨지지 않도록
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--frames-dir", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    r.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"])
    r.add_argument("--offload", default="auto", choices=["auto", "on", "off"])
    r.add_argument("--segment", action="store_true")
    r.add_argument("--track", type=int, nargs="*", default=[])
    r.add_argument("--reference", default=None, help="CPU reference results dir (results.json, seg/, track_N/)")
    r.add_argument("--checkpoint", default=None)
    s = sub.add_parser("summarize")
    s.add_argument("dirs", nargs="+")
    s.add_argument("--reference", default=None)
    s.add_argument("--out", default=None)
    a = ap.parse_args()
    return cmd_run(a) if a.cmd == "run" else cmd_summarize(a)


if __name__ == "__main__":
    raise SystemExit(main())
