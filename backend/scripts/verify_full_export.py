"""Phase 5B-1 검증: GPU 결과로 내보낸 10초 MP4 를 프레임 단위로 검사합니다.

사용: python scripts/verify_full_export.py <output.mp4> <result.zip> <source.mp4> [--report out.json]

- ffprobe: 코덱, 해상도, 프레임 수, FPS, 길이, 프레임별 PTS (= i / fps, 단조 증가)
- 프레임 순서: 출력 k번째 프레임의 '원본 부분'(효과가 없는 영역 제외 대신 마스크 내부)을 원본 디코딩 프레임과 비교해
  가장 잘 맞는 원본 프레임 번호가 k 인지 (모든 프레임)
- 효과 위치: Color Pop 의 채색 영역 중심 vs 가져온 마스크 중심 / 정답 원 중심, 네온 링이 마스크 경계를 둘러싸는지
"""

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.effects import effect_params, render_frame  # noqa: E402
from app.ffmpeg import find_ffmpeg  # noqa: E402

DEFAULT_FX = {
    "neon": {"enabled": True, "intensity": 0.75, "color": "#22d3ee", "thickness": 3},
    "colorPop": {"enabled": True, "intensity": 1.0},
    "blur": {"enabled": False, "intensity": 0.5},
}


def gt_visible(f: int, shape=(360, 640)) -> np.ndarray:
    t = f / 30
    cx = int(120 + 400 * (0.5 + 0.5 * np.sin(t * 1.2)))
    cy = 180 + int(40 * np.cos(t * 2))
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, (cx, cy), 55, 1, -1)
    m[260:331, 40:201] = 0  # 원 위에 그려지는 초록 사각형
    return m


def decode(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    out = []
    while (r := cap.read())[0]:
        out.append(r[1])
    cap.release()
    return out


def centroid(m: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(m)
    return float(xs.mean()), float(ys.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("result_zip")
    ap.add_argument("source")
    ap.add_argument("--report")
    ap.add_argument("--effects", default=None, help="effect settings JSON used for the export (default: neon 0.75 + color pop 1.0)")
    a = ap.parse_args()
    tools = find_ffmpeg()

    probe = json.loads(subprocess.run(
        [str(tools.ffprobe), "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:frame=pts_time",
         "-of", "json", a.output], capture_output=True, text=True, check=True).stdout)  # fmt: skip
    st = probe["streams"][0]
    pts = [float(f["pts_time"]) for f in probe["frames"]]
    num, den = (int(x) for x in st["r_frame_rate"].split("/"))
    fps = num / den
    expected_pts = [i / fps for i in range(len(pts))]
    pts_err = max(abs(p - e) for p, e in zip(pts, expected_pts))

    z = zipfile.ZipFile(a.result_zip)
    r = json.loads(z.read("result.json"))
    frames = r["tracking"]["trackedFrames"]
    masks = {f: cv2.imdecode(np.frombuffer(z.read(f"masks/{f:05d}.png"), np.uint8), 0) > 127 for f in frames}

    out = decode(Path(a.output))
    src = decode(Path(a.source))
    assert len(out) == len(frames), (len(out), len(frames))

    fx = json.loads(a.effects) if a.effects else DEFAULT_FX
    params = effect_params(fx, src[0].shape[1], src[0].shape[0])
    gt_centers = np.array([centroid(gt_visible(i) > 0) for i in range(len(src))])
    order_exact, order_ambiguous, order_wrong = 0, 0, []
    color_err, gt_err, ring_ok, psnrs = [], [], 0, []
    neon_bgr = np.array(params.neon.color_bgr, np.float32) if params.neon else None
    for k, frame in enumerate(out):
        f = frames[k]
        m = masks[f]
        # Color Pop: 채색(빨강) 영역 = 객체
        chroma = frame[..., 2].astype(int) - frame[..., 0].astype(int)
        colored = chroma > 60
        vis = m & ~(np.zeros_like(m))
        vis[260:331, 40:201] = False
        cx, cy = centroid(colored)
        mx, my = centroid(vis)
        gx, gy = centroid(gt_visible(f) > 0)
        color_err.append(float(np.hypot(cx - mx, cy - my)))
        gt_err.append(float(np.hypot(cx - gx, cy - gy)))
        # 프레임 순서: 출력 객체 중심과 가장 가까운 정답 프레임(영상 전체에서 탐색)이 k 인가
        d = np.hypot(gt_centers[:, 0] - cx, gt_centers[:, 1] - cy)
        best = int(np.argmin(d))
        if best == f:
            order_exact += 1
        elif np.hypot(*(gt_centers[best] - gt_centers[f])) < 1.0:
            order_ambiguous += 1  # 원이 거의 멈춘 순간: 정답 위치가 1px 미만 차이라 구분 불가
        else:
            order_wrong.append((k, best))
        # 무손실 기준 렌더(같은 원본 프레임 + 같은 마스크 + 같은 효과) 대비 PSNR → 코덱 손실만 남아야 함
        ref = render_frame(src[f], m.astype(np.uint8), params)
        mse = np.mean((frame.astype(np.float64) - ref.astype(np.float64)) ** 2)
        psnrs.append(99.0 if mse == 0 else 10 * np.log10(255**2 / mse))
        # 네온: 마스크 바깥 3px 링의 평균 색이 네온 색에 가까움
        ring = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) - m.astype(np.uint8)
        if neon_bgr is not None:
            ring_ok += int(np.abs(frame[ring > 0].astype(np.float32) - neon_bgr).mean() < 40)

    report = {
        "probe": {k: st.get(k) for k in ("codec_name", "pix_fmt", "width", "height", "r_frame_rate", "avg_frame_rate", "nb_frames", "duration")},
        "decodedFrames": len(out),
        "ptsFirstLast": [pts[0], pts[-1]],
        "ptsMaxErrorS": round(pts_err, 6),
        "ptsMonotonic": all(b > a_ for a_, b in zip(pts, pts[1:])),
        "frameOrder": {"exact": order_exact, "indistinguishable(<1px motion)": order_ambiguous, "wrong": order_wrong, "total": len(out)},
        "psnrVsLosslessReference": {"min": round(min(psnrs), 2), "mean": round(float(np.mean(psnrs)), 2)},
        "coloredCenterVsMaskPx": {"max": round(max(color_err), 2), "mean": round(float(np.mean(color_err)), 2)},
        "coloredCenterVsGroundTruthPx": {"max": round(max(gt_err), 2), "mean": round(float(np.mean(gt_err)), 2)},
        "neonRingPresent": f"{ring_ok}/{len(out)}",
        "effects": fx,
        "resultPrecision": r["precision"],
        "resultDevice": r["environment"].get("gpuName"),
    }
    print(json.dumps(report, indent=1))
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
