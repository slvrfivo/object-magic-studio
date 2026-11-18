"""API 배선 테스트. 모델은 FakeModel로 대체해 빠르게 실행합니다 (실제 추론 테스트는 test_sam2_real.py)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.errors import AppError
from app.inference.sam2_image import SegmentOutput
from app.main import app
from app.masks import rle_decode
from tests.conftest import write_clip


class FakeModel:
    """받은 픽셀 좌표를 기록하고, 첫 positive 점 주변 사각형을 마스크로 반환."""

    def __init__(self, fail_with: AppError | None = None):
        self.calls = []
        self.fail_with = fail_with

    def status(self):
        return {"state": "ready", "fake": True}

    def segment(self, key, load_image, points_px, labels):
        if self.fail_with:
            raise self.fail_with
        img = load_image()
        self.calls.append({"key": key, "points": points_px.tolist(), "labels": labels.tolist(), "shape": img.shape})
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        x, y = points_px[labels.tolist().index(1)].astype(int)
        mask[max(0, y - 20) : y + 20, max(0, x - 20) : x + 20] = 1
        return SegmentOutput(mask=mask, score=0.9, metrics={"device": "fake"})


@pytest.fixture
def client(isolated_sessions):
    with TestClient(app) as c:
        yield c


def upload(client, path, name="clip.mp4"):
    with open(path, "rb") as f:
        return client.post("/api/sessions", files={"file": (name, f, "video/mp4")})


def test_health_reports_capabilities(client):
    r = client.get("/api/health").json()
    assert r["phase"] == 4
    assert r["capabilities"]["segmentation"] and r["capabilities"]["tracking"]
    # export 는 FFmpeg 가 있을 때만 True
    assert r["capabilities"]["export"] == (r["ffmpeg"] is not None)


def test_upload_probes_meta_and_isolates_frames(client, clip_2s, isolated_sessions):
    r = upload(client, clip_2s)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["meta"]["frameCount"] == 60
    assert body["meta"]["fps"] == 30
    assert body["meta"]["fpsSource"] == "probed"
    session_dir = isolated_sessions.sessions_dir / body["sessionId"]
    assert len(list((session_dir / "frames").glob("*.jpg"))) == 60

    assert client.delete(f"/api/sessions/{body['sessionId']}").status_code == 204
    assert not session_dir.exists()


def test_upload_rejects_long_video_and_cleans_up(client, tmp_path, isolated_sessions):
    r = upload(client, write_clip(tmp_path / "long.mp4", 11))
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation"
    assert list(isolated_sessions.sessions_dir.iterdir()) == []


def test_upload_rejects_non_mp4_and_garbage(client, tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello")
    assert upload(client, p, "a.txt").status_code == 422
    r = upload(client, p, "fake.mp4")
    assert r.status_code == 422 and "decode" in r.json()["error"]["message"]


def test_segment_converts_normalized_points_to_pixels(client, clip_2s):
    sid = upload(client, clip_2s).json()["sessionId"]
    fake = FakeModel()
    client.app.state.model = fake
    r = client.post(
        f"/api/sessions/{sid}/segment",
        json={
            "frameIndex": 5,
            "points": [{"x": 0.5, "y": 0.25, "label": 1}, {"x": 0.1, "y": 0.9, "label": 0}],
            "maskWidth": 320,
            "maskHeight": 180,
        },
    )
    assert r.status_code == 200, r.text
    call = fake.calls[0]
    assert call["key"] == (sid, 5)
    assert call["points"] == [[320.0, 90.0], [64.0, 324.0]]
    assert call["labels"] == [1, 0]
    assert call["shape"] == (360, 640, 3)  # 원본 해상도 RGB 프레임
    m = r.json()["mask"]
    mask = rle_decode(m["counts"], m["width"], m["height"])
    assert mask.shape == (180, 320)
    assert mask[45, 160] == 1 and mask[0, 0] == 0  # 클릭 위치에 마스크


@pytest.mark.parametrize(
    "payload,status",
    [
        ({"frameIndex": 999, "points": [{"x": 0.5, "y": 0.5, "label": 1}], "maskWidth": 10, "maskHeight": 10}, 422),
        ({"frameIndex": 0, "points": [{"x": 0.5, "y": 0.5, "label": 0}], "maskWidth": 10, "maskHeight": 10}, 422),
        ({"frameIndex": 0, "points": [{"x": 1.5, "y": 0.5, "label": 1}], "maskWidth": 10, "maskHeight": 10}, 422),
        ({"frameIndex": 0, "points": [], "maskWidth": 10, "maskHeight": 10}, 422),
    ],
)
def test_segment_validation(client, clip_2s, payload, status):
    sid = upload(client, clip_2s).json()["sessionId"]
    client.app.state.model = FakeModel()
    assert client.post(f"/api/sessions/{sid}/segment", json=payload).status_code == status


def test_segment_unknown_session(client):
    r = client.post(
        "/api/sessions/" + "0" * 32 + "/segment",
        json={"frameIndex": 0, "points": [{"x": 0.5, "y": 0.5, "label": 1}], "maskWidth": 10, "maskHeight": 10},
    )
    assert r.status_code == 404
    assert client.post("/api/sessions/../../etc/segment", json={}).status_code in (404, 422)


def test_model_not_loaded_returns_error_not_mock(client, clip_2s):
    """모델이 없으면 503 model_unavailable — 가짜 마스크를 반환하지 않음."""
    sid = upload(client, clip_2s).json()["sessionId"]
    r = client.post(
        f"/api/sessions/{sid}/segment",
        json={"frameIndex": 0, "points": [{"x": 0.5, "y": 0.5, "label": 1}], "maskWidth": 10, "maskHeight": 10},
    )
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "model_unavailable"
    assert "mask" not in r.json()


def test_gpu_oom_is_reported(client, clip_2s):
    sid = upload(client, clip_2s).json()["sessionId"]
    client.app.state.model = FakeModel(fail_with=AppError("gpu_oom", "GPU out of memory", 503))
    r = client.post(
        f"/api/sessions/{sid}/segment",
        json={"frameIndex": 0, "points": [{"x": 0.5, "y": 0.5, "label": 1}], "maskWidth": 10, "maskHeight": 10},
    )
    assert r.status_code == 503 and r.json()["error"]["code"] == "gpu_oom"


def test_unknown_session_export_and_tracking(client):
    sid = "0" * 32
    fx = {k: {"enabled": True, "intensity": 0.5} for k in ("colorPop", "blur")}
    fx["neon"] = {"enabled": False, "intensity": 0.5, "color": "#22d3ee", "thickness": 3}
    assert client.post(f"/api/sessions/{sid}/export", json={"trackJobId": "x", "effects": fx}).status_code == 404
    r = client.post(f"/api/sessions/{sid}/propagate", json={"frameIndex": 0, "points": [{"x": 0.5, "y": 0.5, "label": 1}], "maxFrames": 5})
    assert r.status_code == 404
