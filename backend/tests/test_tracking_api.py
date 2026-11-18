"""추적 API 배선 테스트. 실제 모델 대신 '움직이는 사각형' 마스크를 쓰는 가짜 추적 함수를 사용합니다."""

import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.masks import rle_decode


@pytest.fixture
def client(isolated_sessions):
    with TestClient(app) as c:
        c.app.state.engine.state = "ready"  # 모델 로드 없이 API 경로만 검증
        yield c


@pytest.fixture
def sid(client, clip_2s):
    with open(clip_2s, "rb") as f:
        return client.post("/api/sessions", files={"file": ("clip.mp4", f, "video/mp4")}).json()["sessionId"]


def fake_track(record, slow=0.0):
    def track(session, req, job):
        record.append({"prompt": req.prompt_frame, "points": req.points_px.tolist(), "labels": req.labels.tolist(), "max": req.max_frames})
        masks = session.root / "tracks" / job.id / "masks"
        masks.mkdir(parents=True)
        stats = {"rangeStart": req.prompt_frame, "trackedFrames": [], "framesDone": 0, "lostFrames": []}
        for i in range(req.max_frames):
            f = req.prompt_frame + i
            m = np.zeros((session.meta.height, session.meta.width), np.uint8)
            m[100:150, 10 * i : 10 * i + 50] = 255  # 프레임마다 오른쪽으로 이동
            cv2.imwrite(str(masks / f"{f:05d}.png"), m)
            stats["trackedFrames"].append(f)
            stats["framesDone"] = i + 1
            job.update((i + 1) / req.max_frames, f"frame {i + 1}", dict(stats))
            if job.cancelled:
                break
            time.sleep(slow)
        return stats

    return track


def wait_job(client, job_id, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("succeeded", "failed", "cancelled"):
            return j
        time.sleep(0.02)
    raise TimeoutError


def body(**kw):
    return {"frameIndex": 4, "points": [{"x": 0.5, "y": 0.5, "label": 1}, {"x": 0.25, "y": 0.75, "label": 0}], "maxFrames": 5, **kw}


def test_health_reports_tracking(client):
    h = client.get("/api/health").json()
    assert h["phase"] == 4
    assert h["capabilities"]["tracking"] is True
    assert h["tracking"]["maxFrames"] >= 2


def test_propagate_runs_job_and_returns_per_frame_masks(client, sid, monkeypatch):
    calls = []
    monkeypatch.setattr(client.app.state.tracker, "track", fake_track(calls))
    job = client.post(f"/api/sessions/{sid}/propagate", json=body()).json()
    assert job["kind"] == "propagate"
    done = wait_job(client, job["id"])
    assert done["status"] == "succeeded" and done["stats"]["framesDone"] == 5
    # 분할과 같은 좌표 변환 규칙
    assert calls[0] == {"prompt": 4, "points": [[320.0, 180.0], [160.0, 270.0]], "labels": [1, 0], "max": 5}

    r = client.get(f"/api/sessions/{sid}/tracks/{job['id']}/masks", params={"width": 320, "height": 180}).json()
    assert [f["index"] for f in r["frames"]] == [4, 5, 6, 7, 8]
    first = rle_decode(r["frames"][0]["counts"], 320, 180)
    last = rle_decode(r["frames"][-1]["counts"], 320, 180)
    # 프레임마다 다른 마스크 (복사가 아님): 마지막 프레임 마스크가 오른쪽으로 이동
    assert np.argwhere(last).mean(axis=0)[1] > np.argwhere(first).mean(axis=0)[1] + 10


def test_cancel_keeps_partial_masks(client, sid, monkeypatch):
    monkeypatch.setattr(client.app.state.tracker, "track", fake_track([], slow=0.05))
    job = client.post(f"/api/sessions/{sid}/propagate", json=body(maxFrames=40)).json()
    time.sleep(0.2)
    assert client.post(f"/api/jobs/{job['id']}/cancel").status_code == 200
    done = wait_job(client, job["id"])
    assert done["status"] == "cancelled"
    n = done["stats"]["framesDone"]
    assert 0 < n < 40
    r = client.get(f"/api/sessions/{sid}/tracks/{job['id']}/masks", params={"width": 64, "height": 36}).json()
    assert len(r["frames"]) == n


@pytest.mark.parametrize(
    "override,status",
    [
        ({"maxFrames": 1}, 422),
        ({"maxFrames": 5000}, 422),  # 장치 한도 초과
        ({"frameIndex": 999}, 422),
        ({"points": [{"x": 0.5, "y": 0.5, "label": 0}]}, 422),
    ],
)
def test_propagate_validation(client, sid, override, status):
    assert client.post(f"/api/sessions/{sid}/propagate", json=body(**override)).status_code == status


def test_propagate_requires_loaded_model(client, sid):
    client.app.state.engine.state = "loading"
    r = client.post(f"/api/sessions/{sid}/propagate", json=body())
    assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"


def test_masks_of_other_session_or_running_job_rejected(client, sid, clip_2s, monkeypatch):
    monkeypatch.setattr(client.app.state.tracker, "track", fake_track([], slow=0.2))
    job = client.post(f"/api/sessions/{sid}/propagate", json=body(maxFrames=10)).json()
    r = client.get(f"/api/sessions/{sid}/tracks/{job['id']}/masks", params={"width": 8, "height": 8})
    assert r.status_code == 409  # 아직 실행 중
    with open(clip_2s, "rb") as f:
        other = client.post("/api/sessions", files={"file": ("c.mp4", f, "video/mp4")}).json()["sessionId"]
    assert client.get(f"/api/sessions/{other}/tracks/{job['id']}/masks", params={"width": 8, "height": 8}).status_code == 404
    client.post(f"/api/jobs/{job['id']}/cancel")
    wait_job(client, job["id"])


def test_unknown_job(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.post("/api/jobs/nope/cancel").status_code == 404
