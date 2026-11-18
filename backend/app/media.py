"""업로드 영상 검증 및 프레임 추출 (OpenCV)."""

from dataclasses import dataclass
from pathlib import Path

import cv2

from .config import settings
from .errors import validation_error


@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps

    def to_json(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "frameCount": self.frame_count,
            "duration": round(self.duration, 3),
            "fpsSource": "probed",
        }


def check_resolution(width: int, height: int) -> None:
    long_side, short_side = max(width, height), min(width, height)
    if long_side > settings.max_long_side or short_side > settings.max_short_side:
        raise validation_error(f"Resolution {width}x{height} is too high. Maximum is 1920x1080 (or 1080x1920).")
    if short_side < settings.min_side:
        raise validation_error(f"Resolution {width}x{height} is too small. Minimum side is {settings.min_side}px.")


def extract_frames(video_path: Path, frames_dir: Path) -> VideoMeta:
    """모든 프레임을 디코딩해 JPEG(00000.jpg ...)로 저장하고 실제 프레임 수를 셉니다.

    CAP_PROP_FRAME_COUNT는 컨테이너에 따라 부정확하므로 직접 디코딩한 수를 신뢰합니다.
    이 JPEG 폴더는 Phase 3에서 SAM 2 비디오 예측기의 입력 형식으로도 사용됩니다.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise validation_error("Could not decode this video. Please upload an H.264 MP4.")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not (0 < fps <= settings.max_fps):
            raise validation_error(f"Unsupported frame rate ({fps:.2f} fps).")
        check_resolution(width, height)

        max_frames = int((settings.max_duration_sec + settings.duration_tolerance_sec) * fps) + 1
        frames_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if count >= max_frames:
                raise validation_error(f"Video is longer than {settings.max_duration_sec:.0f}s.")
            cv2.imwrite(str(frames_dir / f"{count:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            count += 1
    finally:
        cap.release()

    if count == 0:
        raise validation_error("The video contains no decodable frames.")
    meta = VideoMeta(width=width, height=height, fps=fps, frame_count=count)
    if meta.duration > settings.max_duration_sec + settings.duration_tolerance_sec:
        raise validation_error(f"Video is {meta.duration:.1f}s long. Maximum length is {settings.max_duration_sec:.0f}s.")
    return meta


def load_frame_rgb(frames_dir: Path, index: int):
    path = frames_dir / f"{index:05d}.jpg"
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise validation_error(f"Frame {index} is not available.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
