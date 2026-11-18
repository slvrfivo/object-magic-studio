"""오프라인 GPU 작업 테스트: 작업 파일 생성 → (Colab 결과를 흉내 낸) 결과 파일 가져오기 → 검증/거부 → 내보내기.

결과 파일은 Colab 스크립트와 같은 app.gpu_jobs.build_result 로 만듭니다 (모델 불필요).
"""

import io
import json
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.gpu_jobs import build_result
from app.main import app
from app.masks import rle_decode
from tests.conftest import write_clip


@pytest.fixture
def client(isolated_sessions, tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "gpu_jobs_dir", tmp_path / "gpu_jobs")
    with TestClient(app) as c:
        yield c


def upload(client, path: Path) -> dict:
    with open(path, "rb") as f:
        r = client.post("/api/sessions", files={"file": (path.name, f, "video/mp4")})
    assert r.status_code == 200, r.text
    return r.json()


PROMPT = [{"x": 0.5, "y": 0.6, "label": 1}, {"x": 0.1, "y": 0.9, "label": 0}]


def create_job(client, sid, frame=0, n=20, precision="fp16"):
    r = client.post(f"/api/sessions/{sid}/gpu-jobs", json={"frameIndex": frame, "points": PROMPT, "maxFrames": n, "precision": precision})
    assert r.status_code == 200, r.text
    return r.json()


def download_job(client, info) -> tuple[dict, zipfile.ZipFile]:
    r = client.get(info["downloadUrl"])
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    return json.loads(z.read("job.json")), z


def moving_masks(job, shift=0):
    """프레임마다 오른쪽으로 이동하는 사각형 마스크 (복사가 아님을 확인하기 위함)."""
    h, w = job["video"]["height"], job["video"]["width"]
    out = []
    for i in range(job["tracking"]["numFrames"]):
        m = np.zeros((h, w), np.uint8)
        x = 10 + 5 * i + shift
        m[100:160, x : x + 60] = 1
        out.append(m)
    return out


def make_result(job, tmp_path, *, device="cuda", precision="fp16", masks=None, mutate=None, files=None, name="r.zip") -> bytes:
    path = tmp_path / name
    env = {"device": device, "precision": precision, "gpuName": "Tesla T4" if device == "cuda" else None}
    build_result(job, masks if masks is not None else moving_masks(job), env, {"avgFrameMs": 180.0, "totalMs": 3600.0}, path)
    if mutate is None and files is None:
        return path.read_bytes()
    # result.json 수정 또는 파일 교체/추가
    src = zipfile.ZipFile(path)
    r = json.loads(src.read("result.json"))
    if mutate:
        mutate(r)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("result.json", json.dumps(r))
        for n in src.namelist():
            if n != "result.json" and not (files and n in files):
                z.writestr(n, src.read(n))
        for n, data in (files or {}).items():
            if data is not None:
                z.writestr(n, data)
    return buf.getvalue()


def import_result(client, sid, data: bytes):
    return client.post(f"/api/sessions/{sid}/gpu-results", files={"file": ("result.zip", data, "application/zip")})


# ---------------------------------------------------------------- 정상 흐름


def test_job_package_contents(client, clip_2s):
    s = upload(client, clip_2s)
    info = create_job(client, s["sessionId"], frame=5, n=20)
    assert info["numFrames"] == 20 and info["precision"] == "fp16"
    job, z = download_job(client, info)
    names = z.namelist()
    assert job["format"] == "oms-gpu-job" and job["version"] == 1
    assert job["video"]["width"] == 640 and job["video"]["frameCount"] == 60 and job["video"]["fps"] == "30/1"
    assert len(job["video"]["sha256"]) == 64
    assert job["tracking"] == {"startFrame": 5, "numFrames": 20, "direction": "forward"}
    assert np.allclose(job["prompt"]["pointsPx"], [[320.0, 216.0], [64.0, 324.0]], atol=1e-3)  # 정규화 → 원본 픽셀
    assert len(job["model"]["checkpointSha256"]) == 64
    assert [n for n in names if n.startswith("frames/")] == [f"frames/{i:05d}.jpg" for i in range(20)]
    assert "scripts/colab_track.py" in names and "app/inference/sam2_video.py" in names and "app/gpu_jobs.py" in names
    # 로컬 0번 프레임 = 영상의 5번 프레임 (서버 JPEG 그대로)
    assert job["frames"]["sha256"][0] != job["frames"]["sha256"][1]


def test_import_registers_tracking_and_export_uses_it(client, clip_2s, tmp_path):
    s = upload(client, clip_2s)
    sid = s["sessionId"]
    job, _ = download_job(client, create_job(client, sid, frame=5, n=20))
    r = import_result(client, sid, make_result(job, tmp_path))
    assert r.status_code == 200, r.text
    t = r.json()
    assert t["kind"] == "propagate" and t["status"] == "succeeded"
    st = t["stats"]
    assert st["source"] == "gpu-import" and st["precision"] == "fp16" and st["device"] == "cuda"
    assert st["trackedFrames"] == list(range(5, 25)) and st["rangeStart"] == 5 and st["rangeEnd"] == 24

    # 미리보기용 마스크: 프레임마다 위치가 다름 (가져온 마스크 그대로)
    m = client.get(f"/api/sessions/{sid}/tracks/{t['id']}/masks", params={"width": 640, "height": 360}).json()
    first = rle_decode(m["frames"][0]["counts"], 640, 360)
    last = rle_decode(m["frames"][-1]["counts"], 640, 360)
    assert np.argwhere(first)[:, 1].min() == 10 and np.argwhere(last)[:, 1].min() == 10 + 5 * 19

    # 기존 내보내기 경로와 연결
    fx = {
        "neon": {"enabled": False, "intensity": 0.5, "color": "#22d3ee", "thickness": 3},
        "colorPop": {"enabled": True, "intensity": 1.0},
        "blur": {"enabled": False, "intensity": 0.5},
    }
    e = client.post(f"/api/sessions/{sid}/export", json={"trackJobId": t["id"], "effects": fx}).json()
    t0 = time.time()
    while (j := client.get(f"/api/jobs/{e['id']}").json())["status"] not in ("succeeded", "failed") and time.time() - t0 < 60:
        time.sleep(0.05)
    assert j["status"] == "succeeded", j
    assert j["stats"]["frameCount"] == 20 and j["stats"]["rangeStart"] == 5


def test_import_into_new_session_with_same_video(client, clip_2s, tmp_path):
    """Colab 왕복 중 브라우저를 새로 고쳐도: 같은 영상을 다시 올리면 가져올 수 있음 (영상 해시로 검증)."""
    s1 = upload(client, clip_2s)
    job, _ = download_job(client, create_job(client, s1["sessionId"]))
    s2 = upload(client, clip_2s)
    assert import_result(client, s2["sessionId"], make_result(job, tmp_path)).status_code == 200


def test_compare_fp16_and_fp32_results(client, clip_2s, tmp_path):
    s = upload(client, clip_2s)
    sid = s["sessionId"]
    job16, _ = download_job(client, create_job(client, sid, precision="fp16"))
    job32, _ = download_job(client, create_job(client, sid, precision="fp32"))
    a = import_result(client, sid, make_result(job16, tmp_path, name="a.zip")).json()
    b = import_result(client, sid, make_result(job32, tmp_path, precision="fp32", masks=moving_masks(job32, shift=6), name="b.zip")).json()
    assert b["stats"]["precision"] == "fp32"
    c = client.get(f"/api/sessions/{sid}/tracks/compare", params={"a": a["id"], "b": b["id"]}).json()
    assert c["frames"] == 20 and c["minIoU"] == pytest.approx(54 / 66, abs=1e-3)  # 60px 폭 사각형이 6px 어긋남


def test_gpu_results_allow_more_frames_than_cpu_limit(client, tmp_path):
    clip = write_clip(tmp_path / "long.mp4", 3)  # 90 프레임
    s = upload(client, clip)
    sid = s["sessionId"]
    job, _ = download_job(client, create_job(client, sid, n=80))
    # CPU 로 만든 80프레임 결과는 거부 (CPU 한도 60)
    r = import_result(client, sid, make_result(job, tmp_path, device="cpu", precision="fp32", name="cpu.zip"))
    assert r.status_code == 422 and "limit for cpu" in r.json()["error"]["message"]
    # GPU 결과는 허용
    assert import_result(client, sid, make_result(job, tmp_path, name="gpu.zip")).status_code == 200


def test_job_frame_limit(client, clip_2s):
    s = upload(client, clip_2s)
    r = client.post(f"/api/sessions/{s['sessionId']}/gpu-jobs", json={"frameIndex": 0, "points": PROMPT, "maxFrames": 301})
    assert r.status_code == 422
    r = client.post(f"/api/sessions/{s['sessionId']}/gpu-jobs", json={"frameIndex": 0, "points": [{"x": 0.5, "y": 0.5, "label": 0}], "maxFrames": 10})
    assert r.status_code == 422  # positive 점 필요
    r = client.post(f"/api/sessions/{s['sessionId']}/gpu-jobs", json={"frameIndex": 0, "points": PROMPT, "maxFrames": 10, "precision": "bf16"})
    assert r.status_code == 422


# ---------------------------------------------------------------- 잘못된 결과 파일 거부


def _png(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


@pytest.mark.parametrize(
    "case,expect",
    [
        ("video_sha", "different video"),
        ("frame_count", "Frame count mismatch"),
        ("fps", "FPS mismatch"),
        ("resolution", "Resolution mismatch"),
        ("prompt", "Prompt"),
        ("checkpoint", "different checkpoint"),
        ("precision", "Unsupported precision"),
        ("format", "format mismatch"),
        ("version", "Unsupported result version"),
        ("range", "Tracking range"),
        ("partial", "must contain all"),
        ("missing_mask", "Expected 20 mask files"),
        ("extra_file", "Unexpected file"),
        ("traversal", "Unsafe path"),
        ("tampered_mask", "corrupted"),
        ("non_binary", "not binary"),
        ("rgb_mask", "single-channel"),
        ("wrong_shape", "expected 640x360"),
    ],
)
def test_invalid_results_are_rejected(client, clip_2s, tmp_path, case, expect):
    s = upload(client, clip_2s)
    sid = s["sessionId"]
    job, _ = download_job(client, create_job(client, sid, frame=5, n=20))
    mutate, files = None, None
    bad_gray = np.zeros((360, 640), np.uint8)
    if case == "video_sha":
        mutate = lambda r: r["video"].update(sha256="0" * 64)  # noqa: E731
    elif case == "frame_count":
        mutate = lambda r: r["video"].update(frameCount=61)  # noqa: E731
    elif case == "fps":
        mutate = lambda r: r["video"].update(fps="30000/1001")  # noqa: E731
    elif case == "resolution":
        mutate = lambda r: r["video"].update(width=1280)  # noqa: E731
    elif case == "prompt":
        mutate = lambda r: r["prompt"]["points"][0].update(x=0.3)  # noqa: E731
    elif case == "checkpoint":
        mutate = lambda r: r["model"].update(checkpointSha256="f" * 64)  # noqa: E731
    elif case == "precision":
        mutate = lambda r: r.update(precision="int8")  # noqa: E731
    elif case == "format":
        mutate = lambda r: r.update(format="something-else")  # noqa: E731
    elif case == "version":
        mutate = lambda r: r.update(version=2)  # noqa: E731
    elif case == "range":
        mutate = lambda r: r["tracking"].update(startFrame=6)  # noqa: E731
    elif case == "partial":
        mutate = lambda r: r["tracking"].update(trackedFrames=list(range(5, 24)))  # noqa: E731
    elif case == "missing_mask":
        files = {"masks/00010.png": None}
    elif case == "extra_file":
        files = {"evil.py": b"print('x')"}
    elif case == "traversal":
        files = {"../outside.txt": b"x"}
    elif case == "tampered_mask":
        files = {"masks/00010.png": _png(bad_gray)}
    elif case == "non_binary":
        g = bad_gray.copy()
        g[0, 0] = 128

        def mutate(r, g=g):  # 해시는 맞추고 값만 잘못된 마스크
            from app.gpu_jobs import sha256_bytes

            r["masks"]["sha256"]["00010"] = sha256_bytes(_png(g))

        files = {"masks/00010.png": _png(g)}
    elif case == "rgb_mask":
        rgb = np.zeros((360, 640, 3), np.uint8)

        def mutate(r, rgb=rgb):
            from app.gpu_jobs import sha256_bytes

            r["masks"]["sha256"]["00010"] = sha256_bytes(_png(rgb))

        files = {"masks/00010.png": _png(rgb)}
    elif case == "wrong_shape":
        small = np.zeros((180, 320), np.uint8)

        def mutate(r, small=small):
            from app.gpu_jobs import sha256_bytes

            r["masks"]["sha256"]["00010"] = sha256_bytes(_png(small))

        files = {"masks/00010.png": _png(small)}
    r = import_result(client, sid, make_result(job, tmp_path, mutate=mutate, files=files))
    assert r.status_code == 422, (case, r.text)
    assert r.json()["error"]["code"] == "invalid_result"
    assert expect in r.json()["error"]["message"], (case, r.json())


def test_result_for_other_video_rejected(client, clip_2s, tmp_path):
    s1 = upload(client, clip_2s)
    job, _ = download_job(client, create_job(client, s1["sessionId"]))
    other = upload(client, write_clip(tmp_path / "other.mp4", 2.5))
    r = import_result(client, other["sessionId"], make_result(job, tmp_path))
    assert r.status_code == 422 and "different video" in r.json()["error"]["message"]


def test_unknown_job_and_garbage(client, clip_2s, tmp_path):
    s = upload(client, clip_2s)
    sid = s["sessionId"]
    job, _ = download_job(client, create_job(client, sid))
    fake = dict(job, jobId="a" * 32)
    r = import_result(client, sid, make_result(fake, tmp_path))
    assert r.status_code == 422 and "Unknown GPU job" in r.json()["error"]["message"]
    r = import_result(client, sid, b"not a zip at all")
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_result"


def test_full_10s_result_imports_and_exports_300_frames(client, tmp_path):
    """10초(300프레임) GPU 결과 → 가져오기 → 전체 MP4 내보내기: 길이/FPS/프레임 수/마스크 위치 확인."""
    from app.ffmpeg import find_ffmpeg
    from app.exporter import probe_output

    tools = find_ffmpeg()
    if tools is None:
        pytest.skip("FFmpeg not installed")
    clip = write_clip(tmp_path / "ten.mp4", 10)  # 300 프레임, 30 fps
    s = upload(client, clip)
    sid = s["sessionId"]
    info = create_job(client, sid, frame=0, n=300)
    assert info["numFrames"] == 300
    job, _ = download_job(client, info)
    # 정답 원 위치를 그대로 마스크로 사용 (conftest.synthetic_frame 과 같은 궤적, 반지름 55)
    masks = []
    for i in range(300):
        t = i / 30
        cx = int(120 + 400 * (0.5 + 0.5 * np.sin(t * 1.2)))
        cy = 180 + int(40 * np.cos(t * 2))
        m = np.zeros((360, 640), np.uint8)
        cv2.circle(m, (cx, cy), 55, 1, -1)
        masks.append(m)
    r = import_result(client, sid, make_result(job, tmp_path, masks=masks))
    assert r.status_code == 200, r.text
    t = r.json()
    assert t["stats"]["framesDone"] == 300 and t["stats"]["rangeEnd"] == 299

    fx = {
        "neon": {"enabled": False, "intensity": 0.5, "color": "#22d3ee", "thickness": 3},
        "colorPop": {"enabled": True, "intensity": 1.0},
        "blur": {"enabled": False, "intensity": 0.5},
    }
    e = client.post(f"/api/sessions/{sid}/export", json={"trackJobId": t["id"], "effects": fx}).json()
    t0 = time.time()
    while (j := client.get(f"/api/jobs/{e['id']}").json())["status"] not in ("succeeded", "failed") and time.time() - t0 < 300:
        time.sleep(0.2)
    assert j["status"] == "succeeded", j
    st = j["stats"]
    assert st["frameCount"] == 300 and st["outputFps"] == 30 and st["duration"] == pytest.approx(10.0, abs=0.01)

    # 디코딩한 출력에서 채색 영역(객체)의 중심이 프레임마다 마스크 위치와 일치 → 프레임 순서가 맞음
    dl = client.get(j["resultUrl"])
    out = tmp_path / "out.mp4"
    out.write_bytes(dl.content)
    cap = cv2.VideoCapture(str(out))
    k = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if k % 25 == 0 or k == 299:
            chroma = frame[..., 2].astype(int) - frame[..., 0].astype(int)
            ys, xs = np.nonzero(chroma > 60)
            visible = masks[k].copy()
            visible[260:331, 40:201] = 0  # 영상에서 초록 사각형이 원 위에 그려져 가려지는 부분 제외
            my, mx = np.nonzero(visible)
            assert abs(xs.mean() - mx.mean()) < 1.5 and abs(ys.mean() - my.mean()) < 1.5, k
        k += 1
    cap.release()
    assert k == 300
    assert probe_output(tools, out)["frameCount"] == 300
