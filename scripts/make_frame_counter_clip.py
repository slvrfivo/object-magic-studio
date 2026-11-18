"""재생 동기화 검증용 영상: 각 프레임에 자기 번호를 9비트 흰 블록(+숫자)으로 새깁니다.

사용: python scripts/make_frame_counter_clip.py out.mp4 [--frames 300] [--rate 30000/1001]
브라우저에서 캔버스의 블록을 읽으면 '실제로 그려진 프레임 번호'를 알 수 있어,
마스크에 쓰는 프레임 번호(store.currentFrame)와 정확히 비교할 수 있습니다 (docs/DEVELOPMENT_LOG.md 재생 동기화 참고).
블록 위치: y=100..260, x=20+68*b .. 80+68*b (b = 0..8, 하위 비트부터), 640x360 기준.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.ffmpeg import find_ffmpeg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--rate", default="30000/1001")
    a = ap.parse_args()
    tools = find_ffmpeg()
    if tools is None:
        print("FFmpeg not found (python backend/scripts/setup_ffmpeg.py)")
        return 1
    tmp = Path(tempfile.mkdtemp())
    try:
        for i in range(a.frames):
            img = np.full((360, 640, 3), 40, np.uint8)
            for b in range(9):
                if (i >> b) & 1:
                    img[100:260, 20 + b * 68 : 80 + b * 68] = 255
            cv2.putText(img, str(i), (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 2)
            cv2.imwrite(str(tmp / f"{i:05d}.png"), img)
        # B-프레임(-bf 2)을 넣어 실제 영상처럼 표시 순서 ≠ 디코딩 순서가 되도록 함
        subprocess.run([str(tools.ffmpeg), "-v", "error", "-y", "-framerate", a.rate, "-i", str(tmp / "%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-g", "30", "-bf", "2", a.out], check=True)  # fmt: skip
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
