"""MP4 내보내기 테스트 (실제 FFmpeg 사용, SAM 2 모델은 불필요).

정답을 아는 합성 영상 + 정답 마스크를 '추적 결과'로 저장해 두고 내보낸 뒤,
ffprobe 로 길이/FPS/프레임 수/코덱을 확인하고, 디코딩한 프레임에서 효과가 마스크 위치에 정확히 적용됐는지 검사합니다.
"""

import subprocess
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.errors import AppError
from app.exporter import Exporter, probe_output
from app.ffmpeg import find_ffmpeg
from app.jobs import Job
from app.main import app
from app.media import extract_frames
from app.sessions import Session

TOOLS = find_ffmpeg()
pytestmark = pytest.mark.skipif(TOOLS is None, reason="FFmpeg not installed (python scripts/setup_ffmpeg.py)")

W, H = 320, 180


def circle_center(i: int) -> tuple[int, int]:
    # 25프레임마다 왼쪽으로 돌아옴 → 긴 테스트 영상에서도 원이 화면(320px) 안에 있음 (0~24프레임은 이전과 동일)
    return 60 + (8 * i) % 200, 90


def make_clip(path: Path, frames: int, rate: str = "30", size=(W, H), audio=False) -> Path:
    """FFmpeg 로 정확한 프레임레이트의 H.264 영상 생성: 회색 배경 + 오른쪽으로 이동하는 빨간 원."""
    w, h = size
    tmp = path.with_suffix(".frames")
    tmp.mkdir()
    for i in range(frames):
        img = np.full((h, w, 3), 128, np.uint8)
        img[::8] = 140  # 약간의 질감
        cv2.circle(img, circle_center(i), 30, (40, 60, 230), -1)
        cv2.imwrite(str(tmp / f"{i:05d}.png"), img)
    cmd = [str(TOOLS.ffmpeg), "-v", "error", "-y", "-framerate", rate, "-i", str(tmp / "%05d.png")]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-shortest", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", str(path)]
    subprocess.run(cmd, check=True)
    return path


def make_session(tmp_path: Path, clip: Path, track_frames: range) -> tuple[Session, dict, Path]:
    root = tmp_path / "session"
    root.mkdir()
    (root / "input.mp4").write_bytes(clip.read_bytes())
    meta = extract_frames(root / "input.mp4", root / "frames")
    track_dir = root / "tracks" / "trk"
    (track_dir / "masks").mkdir(parents=True)
    for i in track_frames:
        m = np.zeros((meta.height, meta.width), np.uint8)
        cv2.circle(m, circle_center(i), 30, 255, -1)
        cv2.imwrite(str(track_dir / "masks" / f"{i:05d}.png"), m)
    stats = {"trackedFrames": list(track_frames)}
    return Session(id="s", root=root, meta=meta), stats, track_dir


def fx(neon=False, pop=False, blur=False):
    return {
        "neon": {"enabled": neon, "intensity": 0.8, "color": "#22d3ee", "thickness": 3},
        "colorPop": {"enabled": pop, "intensity": 1.0},
        "blur": {"enabled": blur, "intensity": 0.6},
    }


def decode(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            return frames
        frames.append(f)


def test_export_color_pop_matches_tracked_range_and_mask(tmp_path):
    clip = make_clip(tmp_path / "in.mp4", 40)
    session, stats, track_dir = make_session(tmp_path, clip, range(12, 22))
    job = Job(id="exp1", kind="export", session_id="s")
    out = Exporter(Settings()).export(session, stats, track_dir, fx(pop=True), job)

    final = session.root / "exports" / "exp1" / "output.mp4"
    assert final.exists() and not (session.root / "exports" / "exp1" / "output.part.mp4").exists()
    p = probe_output(TOOLS, final)
    assert p["codec"] == "h264" and p["pixFmt"] == "yuv420p"
    assert (p["outputWidth"], p["outputHeight"]) == (W, H)
    assert p["frameCount"] == 10  # 추적된 10프레임만
    assert p["outputFps"] == 30
    assert p["duration"] == pytest.approx(10 / 30, abs=0.01)
    assert out["rangeStart"] == 12 and out["rangeEnd"] == 21

    # 효과 적용 영역 = 마스크 위치 (프레임마다 이동)
    for k, frame in enumerate(decode(final)):
        i = 12 + k
        chroma = frame[..., 2].astype(int) - frame[..., 0].astype(int)  # R - B
        colored = chroma > 60
        ys, xs = np.nonzero(colored)
        cx, cy = circle_center(i)
        assert abs(xs.mean() - cx) < 1.5 and abs(ys.mean() - cy) < 1.5, (i, xs.mean(), ys.mean())
        gt = np.zeros((H, W), np.uint8)
        cv2.circle(gt, (cx, cy), 30, 1, -1)
        bg = cv2.dilate(gt, np.ones((5, 5), np.uint8)) == 0
        assert np.abs(chroma[bg]).mean() < 3  # 배경은 무채색


def test_export_preserves_exact_ntsc_rate_and_audio(tmp_path):
    clip = make_clip(tmp_path / "in.mp4", 30, rate="30000/1001", audio=True)
    session, stats, track_dir = make_session(tmp_path, clip, range(5, 15))
    job = Job(id="exp2", kind="export", session_id="s")
    out = Exporter(Settings()).export(session, stats, track_dir, fx(neon=True), job)
    assert out["fpsRational"] == "30000/1001"
    p = probe_output(TOOLS, session.root / "exports" / "exp2" / "output.mp4")
    assert p["frameCount"] == 10 and p["hasAudio"]
    assert p["outputFps"] == pytest.approx(float(Fraction(30000, 1001)), abs=1e-3)
    assert p["duration"] == pytest.approx(10 * 1001 / 30000, abs=0.03)


@pytest.mark.parametrize(
    "rate, frames, track",
    [
        ("30000/1001", 60, range(0, 60)),  # 영상 전체 (실제 영상 검증에서 288/289 로 재현된 경우)
        ("30000/1001", 60, range(45, 60)),  # 영상 끝까지 가는 뒷부분 구간
        ("30", 45, range(0, 45)),
        ("24", 40, range(10, 40)),
    ],
)
def test_export_with_audio_keeps_every_tracked_frame(tmp_path, rate, frames, track):
    """회귀: 오디오가 있으면 `-shortest` 때문에 마지막 영상 프레임이 잘리던 버그 (P0)."""
    clip = make_clip(tmp_path / "in.mp4", frames, rate=rate, audio=True)
    session, stats, track_dir = make_session(tmp_path, clip, track)
    job = Job(id="exp_a", kind="export", session_id="s")
    Exporter(Settings()).export(session, stats, track_dir, fx(pop=True), job)
    final = session.root / "exports" / "exp_a" / "output.mp4"
    p = probe_output(TOOLS, final)
    n = len(track)
    assert p["hasAudio"]
    assert p["frameCount"] == n, f"expected {n} frames, got {p['frameCount']}"
    out = decode(final)
    assert len(out) == n
    # 마지막 출력 프레임 = 추적 구간의 마지막 원본 프레임 (채색된 원의 위치로 확인)
    chroma = out[-1][..., 2].astype(int) - out[-1][..., 0].astype(int)
    ys, xs = np.nonzero(chroma > 60)
    assert abs(xs.mean() - circle_center(track[-1])[0]) < 1.5
    # 오디오는 추적 구간 길이를 넘지 않음 (-t 로 제한)
    fr = Fraction(rate)
    expected = n / float(fr)
    audio_dur = float(subprocess.run(
        [str(TOOLS.ffprobe), "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration", "-of", "csv=p=0", str(final)],
        capture_output=True, text=True, check=True).stdout.strip())  # fmt: skip
    assert audio_dur <= expected + 0.03
    assert p["duration"] == pytest.approx(expected, abs=0.03)


def test_odd_resolution_is_padded_not_cropped(tmp_path):
    clip = tmp_path / "odd.mp4"
    # 홀수 크기 영상은 libx264/yuv420p 로 만들 수 없으므로 무손실 코덱으로 생성
    frames_dir = tmp_path / "odd_frames"
    frames_dir.mkdir()
    for i in range(6):
        img = np.full((181, 321, 3), 128, np.uint8)
        cv2.circle(img, circle_center(i), 30, (40, 60, 230), -1)
        cv2.imwrite(str(frames_dir / f"{i:05d}.png"), img)
    subprocess.run([str(TOOLS.ffmpeg), "-v", "error", "-y", "-framerate", "30", "-i", str(frames_dir / "%05d.png"), "-c:v", "ffv1", str(clip.with_suffix(".mkv"))], check=True)
    clip.with_suffix(".mkv").rename(clip)
    session, stats, track_dir = make_session(tmp_path, clip, range(0, 6))
    Exporter(Settings()).export(session, stats, track_dir, fx(blur=True), Job(id="exp3", kind="export", session_id="s"))
    p = probe_output(TOOLS, session.root / "exports" / "exp3" / "output.mp4")
    assert (p["outputWidth"], p["outputHeight"]) == (322, 182) and p["frameCount"] == 6


def test_cancel_removes_partial_output(tmp_path):
    clip = make_clip(tmp_path / "in.mp4", 20)
    session, stats, track_dir = make_session(tmp_path, clip, range(0, 20))
    job = Job(id="exp4", kind="export", session_id="s")
    job._cancel.set()  # 첫 프레임 직후 취소
    out = Exporter(Settings()).export(session, stats, track_dir, fx(pop=True), job)
    assert out["framesDone"] == 1
    assert not (session.root / "exports" / "exp4").exists()
    # 원본 영상 핸들이 해제되어 세션 폴더를 바로 지울 수 있어야 함 (Windows 파일 잠금)
    import shutil

    shutil.rmtree(session.root)
    assert not session.root.exists()


def test_ffmpeg_failure_is_reported_and_cleaned(tmp_path):
    clip = make_clip(tmp_path / "in.mp4", 10)
    session, stats, track_dir = make_session(tmp_path, clip, range(0, 10))
    bad = Settings(export_preset="not-a-preset")
    with pytest.raises(AppError) as e:
        Exporter(bad).export(session, stats, track_dir, fx(pop=True), Job(id="exp5", kind="export", session_id="s"))
    assert e.value.code == "export_failed"
    assert not (session.root / "exports" / "exp5").exists()


def test_missing_mask_fails_cleanly(tmp_path):
    clip = make_clip(tmp_path / "in.mp4", 10)
    session, stats, track_dir = make_session(tmp_path, clip, range(0, 5))
    (track_dir / "masks" / "00003.png").unlink()
    with pytest.raises(AppError) as e:
        Exporter(Settings()).export(session, stats, track_dir, fx(pop=True), Job(id="exp6", kind="export", session_id="s"))
    assert "frame 3" in e.value.message
    assert not (session.root / "exports" / "exp6").exists()


# ---------------------------------------------------------------- API


@pytest.fixture
def client(isolated_sessions):
    with TestClient(app) as c:
        c.app.state.engine.state = "ready"
        yield c


def _wait(client, job_id):
    t0 = time.time()
    while time.time() - t0 < 60:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("succeeded", "failed", "cancelled"):
            return j
        time.sleep(0.05)
    raise TimeoutError


def test_export_api_download_and_render_frame(client, tmp_path, monkeypatch):
    from tests.test_tracking_api import fake_track

    clip = make_clip(tmp_path / "in.mp4", 30)
    with open(clip, "rb") as f:
        sid = client.post("/api/sessions", files={"file": ("in.mp4", f, "video/mp4")}).json()["sessionId"]
    monkeypatch.setattr(client.app.state.tracker, "track", fake_track([]))
    track = client.post(f"/api/sessions/{sid}/propagate", json={"frameIndex": 3, "points": [{"x": 0.5, "y": 0.5, "label": 1}], "maxFrames": 6}).json()
    assert _wait(client, track["id"])["status"] == "succeeded"

    # 효과 없이 내보내기 → 422
    r = client.post(f"/api/sessions/{sid}/export", json={"trackJobId": track["id"], "effects": fx()})
    assert r.status_code == 422

    job = client.post(f"/api/sessions/{sid}/export", json={"trackJobId": track["id"], "effects": fx(pop=True, neon=True)}).json()
    done = _wait(client, job["id"])
    assert done["status"] == "succeeded", done
    assert done["resultUrl"] == f"/api/exports/{job['id']}/download"
    assert done["stats"]["frameCount"] == 6 and done["stats"]["rangeStart"] == 3

    dl = client.get(done["resultUrl"])
    assert dl.status_code == 200 and dl.headers["content-type"] == "video/mp4"
    assert 'filename="object-magic_neon-colorpop_f4-9.mp4"' in dl.headers["content-disposition"]
    assert dl.content[4:8] == b"ftyp"

    # 단일 프레임 렌더 = 내보내기와 같은 코드 경로 (무손실 PNG)
    png = client.post(f"/api/sessions/{sid}/tracks/{track['id']}/render-frame", json={"frame": 4, "effects": fx(pop=True)})
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    small = client.post(f"/api/sessions/{sid}/tracks/{track['id']}/render-frame", json={"frame": 4, "effects": fx(pop=True), "width": 160, "height": 90})
    assert cv2.imdecode(np.frombuffer(small.content, np.uint8), cv2.IMREAD_COLOR).shape == (90, 160, 3)
    # 추적하지 않은 프레임은 렌더링 거부
    assert client.post(f"/api/sessions/{sid}/tracks/{track['id']}/render-frame", json={"frame": 20, "effects": fx(pop=True)}).status_code == 422


def test_export_rejects_unfinished_or_foreign_track(client, tmp_path, monkeypatch):
    from tests.test_tracking_api import fake_track

    clip = make_clip(tmp_path / "in.mp4", 30)
    with open(clip, "rb") as f:
        sid = client.post("/api/sessions", files={"file": ("in.mp4", f, "video/mp4")}).json()["sessionId"]
    monkeypatch.setattr(client.app.state.tracker, "track", fake_track([], slow=0.3))
    track = client.post(f"/api/sessions/{sid}/propagate", json={"frameIndex": 0, "points": [{"x": 0.5, "y": 0.5, "label": 1}], "maxFrames": 10}).json()
    r = client.post(f"/api/sessions/{sid}/export", json={"trackJobId": track["id"], "effects": fx(pop=True)})
    assert r.status_code == 409  # 아직 추적 중
    assert client.post(f"/api/sessions/{sid}/export", json={"trackJobId": "nope", "effects": fx(pop=True)}).status_code == 404
    client.post(f"/api/jobs/{track['id']}/cancel")
    _wait(client, track["id"])
