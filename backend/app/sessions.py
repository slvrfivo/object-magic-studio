"""세션 관리: 세션마다 독립된 디렉터리에 업로드 영상과 프레임을 보관합니다.

data/sessions/<session_id>/
    input.mp4
    frames/00000.jpg ...
"""

import hashlib
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import UploadFile

from .config import settings
from .errors import not_found, validation_error
from .media import VideoMeta, extract_frames

_SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_CHUNK = 1024 * 1024


@dataclass
class Session:
    id: str
    root: Path
    meta: VideoMeta
    created_at: float = field(default_factory=time.time)
    #: 업로드한 원본 파일의 SHA-256 (GPU 결과 검증에 사용)
    video_sha256: str = ""
    #: ffprobe 로 읽은 정확한 프레임레이트 문자열 (예: "30000/1001"), 처음 필요할 때 채움
    fps_rational: str | None = None
    #: 마지막 한 번 클릭 분할의 후보 (원본 해상도). 추적 때 선택한 후보를 찾는 데 사용
    #: {"frame", "points_px", "labels", "masks": [np.ndarray], "scores": [float], "best": int}
    last_segment: dict | None = None

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"


class SessionStore:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def cleanup_stale(self, max_age_hours: float) -> int:
        """서버 재시작 시 남은 오래된 세션 디렉터리를 정리합니다."""
        cutoff = time.time() - max_age_hours * 3600
        removed = 0
        for d in self.base_dir.iterdir():
            if d.is_dir() and _SESSION_ID_RE.match(d.name) and d.stat().st_mtime < cutoff and d.name not in self._sessions:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        return removed

    async def create(self, upload: UploadFile) -> Session:
        name = (upload.filename or "").lower()
        if not (name.endswith(".mp4") or upload.content_type == "video/mp4"):
            raise validation_error("Unsupported file type. Please upload an MP4 video.")

        session_id = uuid.uuid4().hex
        root = self.base_dir / session_id
        root.mkdir(parents=True)
        video_path = root / "input.mp4"
        try:
            size = 0
            digest = hashlib.sha256()
            with video_path.open("wb") as f:
                while chunk := await upload.read(_CHUNK):
                    size += len(chunk)
                    if size > settings.max_file_size_bytes:
                        raise validation_error(
                            f"File is too large. Maximum is {settings.max_file_size_bytes // (1024 * 1024)} MB."
                        )
                    f.write(chunk)
                    digest.update(chunk)
            if size == 0:
                raise validation_error("The file is empty.")
            # 디코딩은 CPU 작업이므로 스레드풀에서 실행
            from starlette.concurrency import run_in_threadpool

            meta = await run_in_threadpool(extract_frames, video_path, root / "frames")
        except BaseException:
            shutil.rmtree(root, ignore_errors=True)
            raise

        session = Session(id=session_id, root=root, meta=meta, video_sha256=digest.hexdigest())
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session:
        if not _SESSION_ID_RE.match(session_id):
            raise not_found("Session not found.")
        with self._lock:
            s = self._sessions.get(session_id)
        if s is None:
            raise not_found("Session not found. Please upload the video again.")
        return s

    def delete(self, session_id: str) -> None:
        if not _SESSION_ID_RE.match(session_id):
            return
        with self._lock:
            s = self._sessions.pop(session_id, None)
        if s:
            shutil.rmtree(s.root, ignore_errors=True)
