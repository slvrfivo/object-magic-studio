"""README 용 데모 GIF: 원본 | 효과 (SA-V 영상 + 실제 추적 마스크 + 내보내기와 같은 효과 렌더러).

사용: python scripts/make_demo_gif.py <video.mp4> <result.zip> <out.gif> [--start 0 --count 72 --step 2 --height 400]
- result.zip: GPU 추적 결과 (oms-gpu-result, masks/NNNNN.png)
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.effects import effect_params, render_frame  # noqa: E402

FX = {
    "neon": {"enabled": True, "intensity": 0.8, "color": "#22d3ee", "thickness": 3},
    "colorPop": {"enabled": True, "intensity": 1.0},
    "blur": {"enabled": False, "intensity": 0.5},
}


def label(img: np.ndarray, text: str) -> None:
    cv2.rectangle(img, (0, 0), (12 + 11 * len(text), 30), (24, 20, 20), -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("result_zip")
    ap.add_argument("out")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=72)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--fps", type=float, default=12)
    a = ap.parse_args()

    z = zipfile.ZipFile(a.result_zip)
    cap = cv2.VideoCapture(a.video)
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    params = effect_params(FX, w0, h0)
    want = {a.start + i * a.step for i in range(a.count)}
    frames, k = [], 0
    while len(frames) < a.count:
        ok, fr = cap.read()
        if not ok:
            break
        if k in want:
            m = cv2.imdecode(np.frombuffer(z.read(f"masks/{k:05d}.png"), np.uint8), cv2.IMREAD_GRAYSCALE) > 127
            out = render_frame(fr, m.astype(np.uint8), params)
            s = a.height / h0
            l, r = (cv2.resize(x, (round(w0 * s), a.height), interpolation=cv2.INTER_AREA) for x in (fr, out))
            label(l, "Original")
            label(r, "Color Pop + Neon")
            pair = np.hstack([l, np.full((a.height, 4, 3), 255, np.uint8), r])
            frames.append(Image.fromarray(cv2.cvtColor(pair, cv2.COLOR_BGR2RGB)))
        k += 1
    cap.release()
    # 전체 프레임에서 공통 팔레트 → 프레임 간 색 깜빡임 방지
    sample = Image.fromarray(np.vstack([np.asarray(f) for f in frames[:: max(1, len(frames) // 8)]]))
    pal = sample.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    buf = io.BytesIO()
    q[0].save(buf, format="GIF", save_all=True, append_images=q[1:], duration=round(1000 / a.fps), loop=0, optimize=True)
    Path(a.out).write_bytes(buf.getvalue())
    print(f"{a.out}: {len(q)} frames, {q[0].size}, {len(buf.getvalue()) / 2**20:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
