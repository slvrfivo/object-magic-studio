"""실제 영상 MP4 내보내기 검증 (realworld_eval.py evaluate 이후 실행).

사용: python scripts/realworld_verify_exports.py
- ffprobe: 코덱/해상도/FPS(유리수)/프레임 수/길이, 오디오 스트림 유무와 길이
- 프레임별 PSNR: 출력 프레임 vs 무손실 기준 렌더(같은 원본 프레임 + 가져온 GPU 마스크 + 같은 효과 설정)
  → 코덱 손실만 남아야 하고, 프레임 순서·마스크 위치가 어긋나면 PSNR 이 크게 떨어집니다.
"""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
from app.effects import effect_params, render_frame  # noqa: E402
from app.ffmpeg import find_ffmpeg  # noqa: E402

DOCS = PROJECT / "docs" / "realworld"
VIDEOS = PROJECT / "realworld" / "videos"
RESULTS = PROJECT / "colab" / "results" / "realworld"
VIDEO_OF = {"01_dog": "01_dog", "bedroom": "bedroom", "04_coffee": "04_coffee"}
FX = {
    "neon": {"neon": {"enabled": True, "intensity": 0.75, "color": "#22d3ee", "thickness": 3}, "colorPop": {"enabled": False, "intensity": 1.0}, "blur": {"enabled": False, "intensity": 0.5}},
    "colorPop": {"neon": {"enabled": False, "intensity": 0.75, "color": "#22d3ee", "thickness": 3}, "colorPop": {"enabled": True, "intensity": 1.0}, "blur": {"enabled": False, "intensity": 0.5}},
    "blur": {"neon": {"enabled": False, "intensity": 0.75, "color": "#22d3ee", "thickness": 3}, "colorPop": {"enabled": False, "intensity": 1.0}, "blur": {"enabled": True, "intensity": 0.5}},
}


def probe(ffprobe: Path, path: Path) -> dict:
    out = subprocess.run([str(ffprobe), "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames,duration,pix_fmt",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout  # fmt: skip
    return json.loads(out)


def decode(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while (r := cap.read())[0]:
        frames.append(r[1])
    cap.release()
    return frames


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    tools = find_ffmpeg()
    report = []
    src_cache: dict[str, list] = {}
    for mp4 in sorted((DOCS / "exports").glob("*.mp4")):
        stem = mp4.stem  # 예: 01_dog__dog_neon
        target, fx_name = stem.rsplit("_", 1)
        video = target.split("__")[0]
        z = zipfile.ZipFile(RESULTS / f"{target}__fp16.zip")
        r = json.loads(z.read("result.json"))
        frames = r["tracking"]["trackedFrames"]
        if video not in src_cache:
            src_cache[video] = decode(VIDEOS / f"{VIDEO_OF.get(video, video)}.mp4")
        src = src_cache[video]
        out = decode(mp4)
        p = probe(tools.ffprobe, mp4)
        vs = next(s for s in p["streams"] if s["codec_type"] == "video")
        aus = [s for s in p["streams"] if s["codec_type"] == "audio"]
        params = effect_params(FX[fx_name], src[0].shape[1], src[0].shape[0])
        psnrs = []
        for k, frame in enumerate(out[: len(frames)]):
            f = frames[k]
            m = cv2.imdecode(np.frombuffer(z.read(f"masks/{f:05d}.png"), np.uint8), 0) > 127
            ref = render_frame(src[f], m.astype(np.uint8), params)
            mse = float(np.mean((frame.astype(np.float64) - ref.astype(np.float64)) ** 2))
            psnrs.append(99.0 if mse == 0 else 10 * np.log10(255**2 / mse))
        row = {
            "file": mp4.name, "sizeMB": round(mp4.stat().st_size / 2**20, 2),
            "codec": vs["codec_name"], "pixFmt": vs.get("pix_fmt"), "size": f"{vs['width']}x{vs['height']}", "rFrameRate": vs["r_frame_rate"],
            "decodedFrames": len(out), "trackedFrames": len(frames), "sourceFrames": len(src),
            "videoDuration": vs.get("duration"), "audio": [{"codec": a["codec_name"], "duration": a.get("duration")} for a in aus],
            "psnrMin": round(min(psnrs), 2), "psnrMean": round(float(np.mean(psnrs)), 2), "worstFrame": int(np.argmin(psnrs)),
        }  # fmt: skip
        report.append(row)
        print(json.dumps(row, ensure_ascii=False))
    (DOCS / "exports_verify.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
