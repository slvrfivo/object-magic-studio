"""FFmpeg 공식 안내 Windows 배포본(gyan.dev release essentials)을 프로젝트 폴더에 설치합니다.

ffmpeg.org/download.html 의 Windows 항목이 안내하는 빌드입니다. 시스템 PATH 는 변경하지 않습니다.
설치 위치: backend/third_party/ffmpeg/ffmpeg-<버전>-essentials_build/bin/ffmpeg.exe
백엔드는 이 위치를 자동으로 찾습니다 (app/ffmpeg.py).

사용: python scripts/setup_ffmpeg.py
"""

import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
SHA_URL = URL + ".sha256"
DEST = Path(__file__).resolve().parent.parent / "third_party" / "ffmpeg"


def fetch(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "object-magic-studio-setup"})
    with urllib.request.urlopen(req, timeout=60) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)


def main() -> int:
    if sys.platform != "win32":
        print("This helper is for Windows. On other systems install ffmpeg with your package manager.")
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "ffmpeg.zip"
        print(f"Downloading {URL} …")
        fetch(URL, zpath)
        expected = urllib.request.urlopen(urllib.request.Request(SHA_URL, headers={"User-Agent": "object-magic-studio-setup"}), timeout=30).read().decode().split()[0].lower()
        actual = hashlib.sha256(zpath.read_bytes()).hexdigest()
        if actual != expected:
            print(f"SHA-256 mismatch!\n expected {expected}\n actual   {actual}")
            return 1
        print(f"SHA-256 OK ({actual})")
        if DEST.exists():
            shutil.rmtree(DEST)
        DEST.mkdir(parents=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(DEST)
    exe = next(DEST.glob("*/bin/ffmpeg.exe"), None)
    print(f"Installed: {exe}")
    return 0 if exe else 1


if __name__ == "__main__":
    raise SystemExit(main())
