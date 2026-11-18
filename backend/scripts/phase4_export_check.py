"""Phase 4 통합 검증: 실제 SAM 2.1 추적 → 효과별 MP4 내보내기 → ffprobe/프레임 단위 검증.

사용: python scripts/phase4_export_check.py [--frames 10] [--out ../docs/phase4]

1) 원이 움직이는 합성 영상(640x360, 1920x1080)을 만들고 실제 SAM 2.1 로 N 프레임 추적
2) 모델을 메모리에서 제거한 뒤(내보내기가 SAM 2 를 쓰지 않음을 보장) Neon / Color Pop / Blur 를 각각 내보냄
3) ffprobe: 프레임 수, FPS, 길이, 해상도, 코덱
4) 디코딩한 각 프레임을: (a) 같은 마스크로 만든 무손실 기준 렌더와 비교(PSNR = 코덱 손실만 남아야 함)
   (b) 효과 적용 영역이 추적 마스크/정답 원 위치와 일치하는지 검사
"""

import argparse
import gc
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.effects import effect_params, render_frame  # noqa: E402
from app.exporter import Exporter, load_mask, read_frames  # noqa: E402
from app.ffmpeg import find_ffmpeg  # noqa: E402
from app.inference.engine import Sam2Engine  # noqa: E402
from app.inference.sam2_video import TrackRequest, VideoTracker  # noqa: E402
from app.jobs import Job  # noqa: E402
from app.media import extract_frames  # noqa: E402
from app.sessions import Session  # noqa: E402
from tests.conftest import synthetic_frame  # noqa: E402

DEFAULT_FX = {  # frontend DEFAULT_EFFECTS 와 같은 강도
    "neon": {"enabled": False, "intensity": 0.75, "color": "#22d3ee", "thickness": 3},
    "colorPop": {"enabled": False, "intensity": 1.0},
    "blur": {"enabled": False, "intensity": 0.5},
}


def only(name: str) -> dict:
    fx = json.loads(json.dumps(DEFAULT_FX))
    fx[name]["enabled"] = True
    return fx


def gt_circle(frame: int, scale: int) -> tuple[float, float, float]:
    t = frame / 30
    cx = int(120 + 400 * (0.5 + 0.5 * np.sin(t * 1.2)))
    cy = 180 + int(40 * np.cos(t * 2))
    return cx * scale, cy * scale, 55 * scale


def make_clip(path: Path, scale: int, frames: int, ffmpeg: Path) -> None:
    tmp = path.with_suffix(".src")
    tmp.mkdir()
    for i in range(frames):
        img = synthetic_frame(i / 30)
        if scale != 1:
            # 원을 고해상도에서 다시 그려 경계가 선명하도록 함
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            cx, cy, r = gt_circle(i, scale)
            bg = cv2.resize(synthetic_frame(i / 30), None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            img = bg
            cv2.circle(img, (int(cx), int(cy)), int(r), (40, 90, 240), -1)
        cv2.imwrite(str(tmp / f"{i:05d}.png"), img)
    subprocess.run(
        [str(ffmpeg), "-v", "error", "-y", "-framerate", "30", "-i", str(tmp / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(path)],
        check=True,
    )


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255**2 / mse)


def check_region(name: str, out: np.ndarray, orig: np.ndarray, mask: np.ndarray, p) -> dict:
    """효과가 마스크 기준의 올바른 영역에만 적용됐는지 (코덱 오차 허용)."""
    m = mask > 0
    inner = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0
    far = cv2.dilate(mask, np.ones((61, 61), np.uint8)) == 0
    diff = np.abs(out.astype(int) - orig.astype(int)).mean(axis=2)
    r = {}
    if name == "colorPop":
        chroma = np.abs(out[..., 2].astype(int) - out[..., 0].astype(int))
        r["bgChromaMean"] = round(float(chroma[far].mean()), 2)  # 배경 무채색 → 0 근처
        r["objChromaMean"] = round(float(chroma[inner].mean()), 1)  # 객체는 색 유지 (채도 강화)
        colored = chroma > 60
        ys, xs = np.nonzero(colored)
        my, mx = np.nonzero(m)
        r["coloredCentroidErrPx"] = round(math.hypot(xs.mean() - mx.mean(), ys.mean() - my.mean()), 2)
    elif name == "blur":
        r["objDiffMean"] = round(float(diff[inner].mean()), 2)  # 객체 내부는 그대로
        rect = np.zeros_like(mask)
        h, w = mask.shape
        s = w / 640
        cv2.rectangle(rect, (int(40 * s) - 3, int(260 * s) - 3), (int(200 * s) + 3, int(330 * s) + 3), 1, int(8 * s))
        r["bgEdgeDiffMean"] = round(float(diff[(rect > 0) & far].mean()), 1)  # 배경 사각형 경계는 흐려짐
    elif name == "neon":
        t = max(1, int(round(p.neon.thickness_px)))
        ring = (cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * t + 1, 2 * t + 1))) - mask) > 0
        neon = np.array(p.neon.color_bgr)
        r["ringColorErr"] = round(float(np.abs(out[ring].astype(float) - neon).mean()), 1)  # 링 = 네온 색
        r["innerDiffMean"] = round(float(diff[inner & (cv2.erode(mask, np.ones((41, 41), np.uint8)) > 0)].mean()), 2)
        r["farDiffMean"] = round(float(diff[far].mean()), 2)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT.parent / "docs" / "phase4"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tools = find_ffmpeg()
    assert tools, "FFmpeg not found"
    settings = Settings()
    report = {"ffmpeg": tools.version, "clips": {}}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        clips = {"640x360": 1, "1920x1080": 3}
        sessions = {}
        # ---------------- 1) 실제 SAM 2.1 추적 (모델은 이 단계에서만 사용)
        engine = Sam2Engine(settings)
        engine.load()
        assert engine.state == "ready", engine.error
        tracker = VideoTracker(engine, settings)
        for label, scale in clips.items():
            root = tmp / label
            root.mkdir()
            make_clip(root / "input.mp4", scale, 60, tools.ffmpeg)
            meta = extract_frames(root / "input.mp4", root / "frames")
            session = Session(id=label, root=root, meta=meta)
            cx, cy, _ = gt_circle(0, scale)
            job = Job(id=f"track-{label}", kind="propagate", session_id=label)
            t = time.perf_counter()
            stats = tracker.track(session, TrackRequest(0, np.array([[cx, cy]]), np.array([1]), args.frames), job)
            print(f"[{label}] tracked {stats['framesDone']} frames in {time.perf_counter() - t:.0f}s (avg {stats['avgFrameMs']} ms/frame)")
            sessions[label] = (session, stats, root / "tracks" / job.id)
            report["clips"][label] = {"tracking": {k: stats.get(k) for k in ("framesDone", "avgFrameMs", "totalMs", "peakRssMB", "lostFrames")}}

        # 모델 제거: 이후 내보내기는 SAM 2 없이 저장된 마스크만 사용해야 함
        engine.video_predictor = engine.image_predictor = None
        engine.state = "not_loaded"
        del engine, tracker
        gc.collect()

        # ---------------- 2) 효과별 내보내기 + 검증
        exporter = Exporter(settings)
        for label, (session, tstats, track_dir) in sessions.items():
            meta = session.meta
            originals = {i: f for i, f in read_frames(session.root / "input.mp4", 0, args.frames - 1)}
            masks = {i: load_mask(track_dir / "masks", i, (meta.height, meta.width)) for i in range(args.frames)}
            # 추적 마스크가 실제 원을 따라갔는지 (정답 대비)
            track_iou = []
            for i, m in masks.items():
                cx, cy, r = gt_circle(i, clips[label])
                g = np.zeros_like(m)
                cv2.circle(g, (int(cx), int(cy)), int(r), 1, -1)
                track_iou.append((m & g).sum() / max(1, (m | g).sum()))
            report["clips"][label]["tracking"]["minIoUvsGT"] = round(float(min(track_iou)), 3)

            for fx_name in ("neon", "colorPop", "blur"):
                fx = only(fx_name)
                job = Job(id=f"export-{label}-{fx_name}", kind="export", session_id=label)
                t = time.perf_counter()
                st = exporter.export(session, tstats, track_dir, fx, job)
                wall = time.perf_counter() - t
                mp4 = session.root / "exports" / job.id / "output.mp4"
                dst = out_dir / f"{label}_{fx_name}.mp4"
                dst.write_bytes(mp4.read_bytes())

                cap = cv2.VideoCapture(str(mp4))
                decoded = []
                while (f := cap.read())[0]:
                    decoded.append(f[1])
                cap.release()  # Windows: 파일 잠금 해제
                p = effect_params(fx, meta.width, meta.height)
                psnrs, regions = [], []
                for i, frame in enumerate(decoded):
                    ref = render_frame(originals[i], masks[i], p)  # 무손실 기준
                    psnrs.append(psnr(frame, ref))
                    regions.append(check_region(fx_name, frame, originals[i], masks[i], p))
                summary = {
                    "probe": {k: st[k] for k in ("frameCount", "outputFps", "fpsRational", "duration", "outputWidth", "outputHeight", "codec", "pixFmt", "hasAudio", "fileSizeBytes")},
                    "expected": {"frameCount": args.frames, "duration": round(args.frames / 30, 4), "fps": 30, "size": [meta.width, meta.height]},
                    "decodedFrames": len(decoded),
                    "psnrVsLosslessRender": {"min": round(min(psnrs), 2), "mean": round(float(np.mean(psnrs)), 2)},
                    "regionChecks": {k: [r[k] for r in regions] for k in regions[0]},
                    "timing": {"wallS": round(wall, 2), "avgRenderMs": st["avgRenderMs"], "totalMs": st["totalMs"]},
                }
                report["clips"][label][fx_name] = summary
                print(f"[{label}/{fx_name}] {json.dumps(summary)}", flush=True)
                (out_dir / "report.json").write_text(json.dumps(report, indent=1))
                # 대표 프레임 비교 이미지 (원본 | 출력)
                k = args.frames - 1
                sheet = np.hstack([originals[k], decoded[k]])
                if sheet.shape[1] > 1600:
                    sheet = cv2.resize(sheet, None, fx=1600 / sheet.shape[1], fy=1600 / sheet.shape[1], interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out_dir / f"{label}_{fx_name}_frame{k}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])

    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    print("REPORT", out_dir / "report.json")


if __name__ == "__main__":
    main()
