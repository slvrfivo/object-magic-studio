"""한 번 클릭의 마스크 후보 3개 선택 → 추적 반영 테스트 (모델 없이, 로컬·원격 경로 모두).

실제 SAM 2.1 로 "선택한 후보가 추적 첫 프레임에 그대로 반영되는지"는 tests/test_candidates_real.py (real_model).
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.inference.sam2_image import SegmentOutput
from app.main import app
from app.masks import rle_decode
from app.remote.service import RemoteGpuService
from app.remote.transport import InProcessTransport
from tests.test_remote import FakeWorker, make_ledger, make_session, wait_job
from tests.test_tracking_api import fake_track

H, W = 360, 640


def rect(y0, y1, x0, x1) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


# 부분(작음) / 전체(중간) / 배경 포함(큼) 후보 — 점수는 1번이 최고
CANDS = [(rect(150, 210, 290, 350), 0.62), (rect(100, 260, 240, 400), 0.91), (rect(50, 330, 150, 520), 0.40)]


def fake_segment(calls):
    def segment(image_key, load_image, points_px, labels, **kw):
        calls.append(len(points_px))
        if len(points_px) == 1:
            return SegmentOutput(CANDS[1][0], CANDS[1][1], {"device": "cpu"}, candidates=list(CANDS), best=1)
        return SegmentOutput(rect(120, 240, 260, 380), 0.95, {"device": "cpu"})

    return segment


@pytest.fixture
def client(isolated_sessions, monkeypatch):
    with TestClient(app) as c:
        c.app.state.engine.state = "ready"
        calls, tracks = [], []
        monkeypatch.setattr(c.app.state.model, "segment", fake_segment(calls))
        orig = fake_track([])

        def track(session, req, job):
            tracks.append({"mask": None if req.mask is None else req.mask.copy()})
            return orig(session, req, job)

        monkeypatch.setattr(c.app.state.tracker, "track", track)
        c.track_calls = tracks
        yield c


@pytest.fixture
def sid(client, clip_2s):
    with open(clip_2s, "rb") as f:
        return client.post("/api/sessions", files={"file": ("clip.mp4", f, "video/mp4")}).json()["sessionId"]


ONE = [{"x": 0.5, "y": 0.5, "label": 1}]
TWO = [{"x": 0.5, "y": 0.5, "label": 1}, {"x": 0.1, "y": 0.1, "label": 0}]


def seg(client, sid, points, frame=3):
    return client.post(f"/api/sessions/{sid}/segment", json={"frameIndex": frame, "points": points, "maskWidth": 320, "maskHeight": 180, "backend": "local"}).json()


def track(client, sid, points, cand, frame=3):
    body = {"frameIndex": frame, "points": points, "maxFrames": 5, "backend": "local"}
    if cand is not None:
        body["maskCandidate"] = cand
    return client.post(f"/api/sessions/{sid}/propagate", json=body)


def test_single_click_returns_three_candidates_at_preview_size(client, sid):
    r = seg(client, sid, ONE)
    assert r["selected"] == 1 and len(r["candidates"]) == 3
    assert [c["score"] for c in r["candidates"]] == [0.62, 0.91, 0.40]
    small = rle_decode(r["candidates"][0]["mask"]["counts"], 320, 180)
    assert small.sum() == pytest.approx(CANDS[0][0].sum() / 4, rel=0.05)  # 원본의 1/2 크기로 축소
    assert r["candidates"][2]["coverage"] > r["candidates"][1]["coverage"] > r["candidates"][0]["coverage"]
    assert "candidates" not in seg(client, sid, TWO)  # 여러 번 클릭은 후보 없음 (기존과 같음)


def test_selected_candidate_is_used_for_tracking(client, sid):
    seg(client, sid, ONE)
    job = track(client, sid, ONE, 2).json()
    assert wait_job(client, job["id"])["status"] == "succeeded"
    assert np.array_equal(client.track_calls[-1]["mask"], CANDS[2][0])  # 원본 해상도 후보 그대로


def test_default_candidate_keeps_point_prompt(client, sid):
    seg(client, sid, ONE)
    for cand in (None, 1):  # 선택하지 않음 / 기본 후보 선택 → 기존과 같은 클릭 점 추적
        job = track(client, sid, ONE, cand).json()
        wait_job(client, job["id"])
        assert client.track_calls[-1]["mask"] is None


def test_stale_or_missing_candidates_are_rejected(client, sid):
    r = track(client, sid, ONE, 0)  # 분할 전
    assert r.status_code == 409 and "out of date" in r.json()["error"]["message"]
    seg(client, sid, ONE)
    assert track(client, sid, ONE, 0, frame=4).status_code == 409  # 다른 프레임
    seg(client, sid, TWO)  # 점을 추가하면 후보가 사라짐
    assert track(client, sid, ONE, 0).status_code == 409
    assert track(client, sid, TWO, None).status_code == 200


def test_colab_job_rejects_non_default_candidate(client, sid):
    seg(client, sid, ONE)
    body = {"frameIndex": 3, "points": ONE, "maxFrames": 10, "precision": "fp16"}
    r = client.post(f"/api/sessions/{sid}/gpu-jobs", json={**body, "maskCandidate": 0})
    assert r.status_code == 422 and "click points only" in r.json()["error"]["message"]
    assert client.post(f"/api/sessions/{sid}/gpu-jobs", json={**body, "maskCandidate": 1}).status_code == 200


# ---------------------------------------------------------------- 원격 경로


class CandidateWorker(FakeWorker):
    def segment(self, video_sha, frame_index, points_px, labels):
        from app.masks import rle_encode
        from app.remote.worker import ok

        r = super().segment(video_sha, frame_index, points_px, labels)
        if len(points_px) == 1 and r["ok"]:
            r["candidates"] = [{"mask": {"width": W, "height": H, "counts": rle_encode(m)}, "score": s} for m, s in CANDS]
            r["best"] = 1
        return r

    def track_stream(self, *a, mask_rle=None, **k):
        self.mask_rle = mask_rle
        yield from super().track_stream(*a, **k)


def test_remote_candidates_and_mask_prompt(tmp_path, clip_2s):
    from app.config import Settings
    from app.inference.sam2_video import TrackRequest
    from app.jobs import Job

    worker = CandidateWorker()
    svc = RemoteGpuService(Settings(remote_provider="inprocess"), InProcessTransport(worker), make_ledger(tmp_path))
    s = make_session(tmp_path, clip_2s)
    out = svc.segment(s, 0, np.array([[320.0, 180.0]], np.float32), np.array([1]))
    assert out.best == 1 and len(out.candidates) == 3 and np.array_equal(out.candidates[0][0], CANDS[0][0])
    job = Job(id="jc", kind="propagate", session_id=s.id)
    svc.track(s, TrackRequest(0, np.array([[320.0, 180.0]], np.float32), np.array([1]), 5, mask=CANDS[0][0]), job)
    m = worker.mask_rle
    assert (m["width"], m["height"]) == (W, H) and np.array_equal(rle_decode(m["counts"], W, H), CANDS[0][0])
    svc.track(s, TrackRequest(0, np.array([[320.0, 180.0]], np.float32), np.array([1]), 5), Job(id="jd", kind="propagate", session_id=s.id))
    assert worker.mask_rle is None  # 기본 후보 → 마스크를 보내지 않음 (이전 배포 워커와 호환)


def test_worker_rejects_bad_prompt_mask(tmp_path, clip_2s):
    import hashlib

    from app.masks import rle_encode
    from tests.test_remote import bare_worker

    w = bare_worker(tmp_path)
    data = clip_2s.read_bytes()
    h = hashlib.sha256(data).hexdigest()
    w.put_video(h, data)
    wrong = {"width": 320, "height": 180, "counts": rle_encode(np.ones((180, 320), np.uint8))}
    assert "size" in w.track(h, 0, [[1, 1]], [1], 5, mask_rle=wrong)["error"]["message"]
    empty = {"width": W, "height": H, "counts": rle_encode(np.zeros((H, W), np.uint8))}
    assert "empty" in w.track(h, 0, [[1, 1]], [1], 5, mask_rle=empty)["error"]["message"]
