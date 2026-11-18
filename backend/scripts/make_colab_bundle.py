"""Colab GPU 벤치마크용 번들(zip) 생성.

포함: 백엔드 추론 코드(app/), scripts/gpu_benchmark.py, 테스트 프레임(JPEG 300장), CPU 기준 결과, manifest.json
제외: 체크포인트(Colab 에서 공식 URL 로 받고 SHA-256 검증), 가상환경, 세션 데이터

사용: python scripts/make_colab_bundle.py   →  ../colab/build/oms_phase5a_bundle.zip
"""

import hashlib
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
PROJECT = BACKEND.parent
BUILD = PROJECT / "colab" / "build"
OUT = BUILD / "oms_phase5a_bundle.zip"
CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    frames = sorted((BUILD / "data" / "frames").glob("*.jpg"))
    ref = BUILD / "reference" / "cpu"
    if len(frames) != 300 or not (ref / "results.json").exists():
        print("Missing frames or CPU reference. See README (Phase 5A) for the preparation steps.")
        return 1
    commit = subprocess.run(["git", "-C", str(BACKEND / "third_party" / "sam2"), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    manifest = {
        "createdAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sam2Repo": "https://github.com/facebookresearch/sam2.git",
        "sam2Commit": commit,
        "checkpointUrl": CKPT_URL,
        "checkpointSha256": sha256(BACKEND / "checkpoints" / "sam2.1_hiera_tiny.pt"),
        "testClip": {"width": 640, "height": 360, "fps": 30, "frames": len(frames), "generator": "scripts/make_test_clip.py out.mp4 10"},
    }
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=1))
        for p in sorted((BACKEND / "app").rglob("*.py")):
            z.write(p, p.relative_to(BACKEND).as_posix())
        z.write(BACKEND / "scripts" / "gpu_benchmark.py", "scripts/gpu_benchmark.py")
        for p in frames:
            z.write(p, f"data/frames/{p.name}")
        for p in sorted(ref.rglob("*")):
            if p.is_file():
                z.write(p, (Path("reference/cpu") / p.relative_to(ref)).as_posix())
    print(f"wrote {OUT} ({OUT.stat().st_size / 2**20:.1f} MB)")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
