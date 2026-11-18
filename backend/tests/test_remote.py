"""원격 GPU 연결 (Phase 5B-2) 테스트 — GPU·모델·Modal 계정 없이 실행.

- FakeWorker: GpuWorker 와 같은 메서드/반환 형식을 가진 가짜 워커 (원 모양 마스크, 프레임마다 이동)
- InProcessTransport: 인자/결과를 pickle 로 왕복시켜 원격 호출처럼 직렬화 가능한 값만 오가는지 확인
- GpuWorker 의 업로드 검증/영상 관리 부분은 실제 코드를 모델 없이 검증
실제 모델로 워커를 돌리는 테스트는 tests/test_remote_worker_real.py (real_model).
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.errors import AppError
from app.inference.sam2_video import TrackRequest
from app.jobs import Job
from app.main import app
from app.masks import rle_decode, rle_encode
from app.media import extract_frames
from app.remote.service import RemoteGpuService
from app.remote.transport import InProcessTransport, ModalTransport, RemoteTransportError, Transport
from app.remote.usage import BudgetExceeded, UsageLedger
from app.remote.worker import GpuWorker, err, ok


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def circle(h, w, cx, cy, r=25) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (int(cx), int(cy)), r, 1, -1)
    return m


class FakeWorker:
    """GpuWorker 와 같은 인터페이스. 영상 디코딩은 실제 extract_frames 로 해서 메타데이터를 비교 가능하게 함."""

    def __init__(self, frame_delay: float = 0.0):
        self.videos: dict[str, dict] = {}
        self.calls: list[tuple] = []
        self.frame_delay = frame_delay
        self.started = threading.Event()

    def info(self):
        self.calls.append(("info",))
        return ok(protocol=1, model={"state": "ready", "dtype": "fp16", "loadMs": 1.0, "warmup": None, "device": {"gpu": {"name": "Fake T4"}}}, maxTrackFrames=300, featureCache={}, counters={}, uptimeS=1.0)

    def has_video(self, video_sha):
        self.calls.append(("has_video", video_sha))
        v = self.videos.get(video_sha)
        return ok(present=v is not None, meta=v)

    def put_video(self, video_sha, data):
        self.calls.append(("put_video", video_sha, len(data)))
        if sha(data) != video_sha:
            return err("validation", "sha mismatch", 422)
        d = Path(tempfile.mkdtemp())
        try:
            (d / "v.mp4").write_bytes(data)
            meta = extract_frames(d / "v.mp4", d / "frames").to_json()
        finally:
            shutil.rmtree(d, ignore_errors=True)
        self.videos[video_sha] = meta
        return ok(meta=meta, extractMs=1.0, cached=False)

    def drop_video(self, video_sha):
        self.calls.append(("drop_video", video_sha))
        return ok(dropped=self.videos.pop(video_sha, None) is not None)

    def _missing(self, video_sha):
        return None if video_sha in self.videos else err("video_missing", "restarted", 409)

    def segment(self, video_sha, frame_index, points_px, labels):
        self.calls.append(("segment", video_sha, frame_index, points_px, labels))
        if e := self._missing(video_sha):
            return e
        m = self.videos[video_sha]
        x, y = points_px[labels.index(1)]
        mask = circle(m["height"], m["width"], x, y)
        return ok(mask={"width": m["width"], "height": m["height"], "counts": rle_encode(mask)}, score=0.97, metrics={"device": "cuda", "deviceName": "Fake T4", "dtype": "fp16", "embeddingShared": False})

    def track_stream(self, video_sha, prompt_frame, points_px, labels, max_frames, should_cancel):
        self.calls.append(("track", video_sha, prompt_frame, points_px, labels, max_frames))
        if e := self._missing(video_sha):
            yield {"type": "result", **e}
            return
        self.started.set()
        m = self.videos[video_sha]
        n = min(max_frames, m["frameCount"] - prompt_frame)
        x, y = points_px[labels.index(1)]
        masks, frames, cancelled = {}, [], False
        for i in range(n):
            f = prompt_frame + i
            ok_, png = cv2.imencode(".png", circle(m["height"], m["width"], x + 5 * i, y) * 255)
            masks[f] = png.tobytes()
            frames.append(f)
            yield {"type": "progress", "progress": (i + 1) / n, "message": f"Frame {f + 1} ({i + 1}/{n})", "stats": {"framesDone": i + 1, "avgFrameMs": 200.0}}
            time.sleep(self.frame_delay)
            if should_cancel():
                cancelled = True
                break
        stats = {"trackedFrames": frames, "framesDone": len(frames), "rangeStart": prompt_frame, "avgFrameMs": 200.0, "totalMs": 1000.0, "lostFrames": [], "unstableFrames": [], "promptFeaturesReused": True, "cancelled": cancelled}
        yield {"type": "result", **ok(stats=stats, masks=masks, environment={"device": "cuda", "precision": "fp16", "gpuName": "Fake T4"})}


class FailingTransport(Transport):
    provider = "modal"

    def call(self, method, **kw):
        raise RemoteTransportError("remote_unavailable", "Modal authentication failed")

    def stream(self, method, **kw):
        raise RemoteTransportError("remote_unavailable", "Modal authentication failed")
        yield  # pragma: no cover

    def request_cancel(self, key):
        pass

    def clear_cancel(self, key):
        pass


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def make_ledger(tmp_path, **kw) -> UsageLedger:
    args = dict(cost_per_s=0.0002, idle_s=120, cold_start_s=45, daily_limit_s=1800, monthly_limit_usd=5)
    args.update(kw)
    return UsageLedger(tmp_path / "usage.json", **args)


def make_session(tmp_path, clip: Path):
    from app.sessions import Session

    root = tmp_path / "sess"
    root.mkdir()
    shutil.copy(clip, root / "input.mp4")
    meta = extract_frames(root / "input.mp4", root / "frames")
    return Session(id="a" * 32, root=root, meta=meta, video_sha256=sha((root / "input.mp4").read_bytes()))


@pytest.fixture
def remote_settings(tmp_path):
    return Settings(remote_provider="inprocess", remote_usage_file=tmp_path / "usage.json")


# ---------------------------------------------------------------- 서비스 (로컬 클라이언트)


def test_segment_uploads_video_once_and_returns_full_res_mask(tmp_path, clip_2s, remote_settings):
    worker = FakeWorker()
    svc = RemoteGpuService(remote_settings, InProcessTransport(worker), make_ledger(tmp_path))
    s = make_session(tmp_path, clip_2s)
    pts, labels = np.array([[320.0, 180.0], [100.0, 100.0]], np.float32), np.array([1, 0])
    out1 = svc.segment(s, 3, pts, labels)
    out2 = svc.segment(s, 3, pts, labels)
    assert [c[0] for c in worker.calls].count("put_video") == 1  # 같은 세션은 한 번만 업로드
    assert out1.mask.shape == (360, 640) and out1.mask[180, 320] == 1
    assert out1.metrics["backend"] == "remote" and out1.metrics["uploadMs"] > 0 and out2.metrics["uploadMs"] == 0
    assert out1.metrics["coldStart"] is True and out2.metrics["coldStart"] is False
    seg = [c for c in worker.calls if c[0] == "segment"][0]
    assert seg[2] == 3 and seg[3] == [[320.0, 180.0], [100.0, 100.0]] and seg[4] == [1, 0]  # 원본 픽셀 좌표 그대로


def test_reuploads_when_worker_restarted(tmp_path, clip_2s, remote_settings):
    worker = FakeWorker()
    svc = RemoteGpuService(remote_settings, InProcessTransport(worker), make_ledger(tmp_path))
    s = make_session(tmp_path, clip_2s)
    svc.segment(s, 0, np.array([[300.0, 200.0]]), np.array([1]))
    worker.videos.clear()  # 컨테이너 재시작으로 임시 파일이 사라진 상황
    out = svc.segment(s, 0, np.array([[300.0, 200.0]]), np.array([1]))
    assert out.mask.sum() > 0
    assert [c[0] for c in worker.calls].count("put_video") == 2


def test_existing_remote_video_is_not_uploaded_again(tmp_path, clip_2s, remote_settings):
    worker = FakeWorker()
    s = make_session(tmp_path, clip_2s)
    worker.put_video(s.video_sha256, (s.root / "input.mp4").read_bytes())  # 이전 세션이 이미 올림
    worker.calls.clear()
    svc = RemoteGpuService(remote_settings, InProcessTransport(worker), make_ledger(tmp_path))
    svc.segment(s, 0, np.array([[300.0, 200.0]]), np.array([1]))
    assert [c[0] for c in worker.calls] == ["has_video", "segment"]


def test_decode_mismatch_on_worker_is_reported(tmp_path, clip_2s, remote_settings):
    class Lying(FakeWorker):
        def put_video(self, video_sha, data):
            r = super().put_video(video_sha, data)
            r["meta"]["frameCount"] -= 1
            return r

    svc = RemoteGpuService(remote_settings, InProcessTransport(Lying()), make_ledger(tmp_path))
    with pytest.raises(AppError) as e:
        svc.segment(make_session(tmp_path, clip_2s), 0, np.array([[300.0, 200.0]]), np.array([1]))
    assert e.value.code == "remote_failed" and "decoded the video differently" in e.value.message


def test_track_writes_masks_and_stats_like_local_tracking(tmp_path, clip_2s, remote_settings):
    worker = FakeWorker()
    svc = RemoteGpuService(remote_settings, InProcessTransport(worker), make_ledger(tmp_path))
    s = make_session(tmp_path, clip_2s)
    job = Job(id="j1", kind="propagate", session_id=s.id)
    progress = []
    orig = job.update
    job.update = lambda p=None, m=None, st=None: (progress.append(p), orig(p, m, st))[1]
    stats = svc.track(s, TrackRequest(5, np.array([[200.0, 180.0]], np.float32), np.array([1]), 20), job)
    assert stats["trackedFrames"] == list(range(5, 25)) and stats["framesDone"] == 20
    assert stats["source"] == "remote-gpu" and stats["gpuName"] == "Fake T4" and stats["promptFeaturesReused"] is True
    assert len(stats["coverage"]) == 20 and stats["estimatedCostUSD"] > 0
    for i, f in enumerate(stats["trackedFrames"]):
        m = cv2.imread(str(s.root / "tracks" / "j1" / "masks" / f"{f:05d}.png"), cv2.IMREAD_GRAYSCALE) > 127
        ys, xs = np.nonzero(m)
        assert abs(xs.mean() - (200 + 5 * i)) < 1.0
    ps = [p for p in progress if p is not None]
    assert ps == sorted(ps) and ps[-1] == 1.0 and len(ps) > 5


def test_track_cancel_keeps_frames_done_on_gpu(tmp_path, clip_2s, remote_settings):
    worker = FakeWorker(frame_delay=0.03)
    transport = InProcessTransport(worker)
    svc = RemoteGpuService(remote_settings, transport, make_ledger(tmp_path))
    s = make_session(tmp_path, clip_2s)
    job = Job(id="j2", kind="propagate", session_id=s.id)
    threading.Timer(0.2, job._cancel.set).start()
    stats = svc.track(s, TrackRequest(0, np.array([[200.0, 180.0]], np.float32), np.array([1]), 60), job)
    assert 0 < stats["framesDone"] < 60 and stats["cancelled"] is True
    assert stats["trackedFrames"] == list(range(stats["framesDone"]))
    assert len(list((s.root / "tracks" / "j2" / "masks").glob("*.png"))) == stats["framesDone"]
    assert not transport._cancel  # 취소 플래그 정리


def test_cancel_reaches_worker_during_cold_start(tmp_path, clip_2s, remote_settings):
    """진행 이벤트가 오기 전(콜드 스타트 중)에도 취소가 바로 원격에 전달되어야 함."""

    class SlowStart(FakeWorker):
        def track_stream(self, *a, should_cancel, **k):
            t0 = time.time()
            while time.time() - t0 < 3:  # 컨테이너 시작·모델 로드 중
                if should_cancel():
                    self.cancel_seen_after = time.time() - t0
                    yield {"type": "result", **ok(stats={"trackedFrames": [], "framesDone": 0, "cancelled": True}, masks={}, environment={})}
                    return
                time.sleep(0.02)
            yield from super().track_stream(*a, should_cancel=should_cancel, **k)

    worker = SlowStart()
    svc = RemoteGpuService(remote_settings, InProcessTransport(worker), make_ledger(tmp_path))
    s = make_session(tmp_path, clip_2s)
    job = Job(id="j3", kind="propagate", session_id=s.id)
    threading.Timer(0.3, job._cancel.set).start()
    stats = svc.track(s, TrackRequest(0, np.array([[200.0, 180.0]], np.float32), np.array([1]), 30), job)
    assert stats["framesDone"] == 0 and stats["cancelled"] is True
    assert worker.cancel_seen_after < 1.5  # 3초 시작 시간을 기다리지 않음


def test_track_limits_and_budget_are_checked_before_queueing(tmp_path, clip_2s, remote_settings):
    s = make_session(tmp_path, clip_2s)
    svc = RemoteGpuService(remote_settings, InProcessTransport(FakeWorker()), make_ledger(tmp_path, daily_limit_s=100))
    with pytest.raises(AppError) as e:
        svc.validate(TrackRequest(0, np.zeros((1, 2)), np.array([1]), 301), s.meta.frame_count)
    assert e.value.code == "validation" and "300" in e.value.message
    # 예상 청구: 콜드 45 + 60프레임×0.25 + 대기 120 = 180초 > 하루 한도 100초
    with pytest.raises(AppError) as e:
        svc.validate(TrackRequest(0, np.zeros((1, 2)), np.array([1]), 60), s.meta.frame_count)
    assert e.value.code == "budget_exceeded" and e.value.status == 429


def test_transport_failure_is_reported_not_silently_replaced(tmp_path, clip_2s, remote_settings):
    svc = RemoteGpuService(remote_settings, FailingTransport(), make_ledger(tmp_path))
    with pytest.raises(AppError) as e:
        svc.segment(make_session(tmp_path, clip_2s), 0, np.array([[1.0, 1.0]]), np.array([1]))
    assert e.value.code == "remote_unavailable" and e.value.status == 503
    assert svc.status()["state"] == "error" and "authentication" in svc.status()["lastError"]


def test_worker_errors_map_to_app_errors(tmp_path, clip_2s, remote_settings):
    class Oom(FakeWorker):
        def segment(self, *a, **k):
            return err("gpu_oom", "GPU out of memory", 503)

    svc = RemoteGpuService(remote_settings, InProcessTransport(Oom()), make_ledger(tmp_path))
    with pytest.raises(AppError) as e:
        svc.segment(make_session(tmp_path, clip_2s), 0, np.array([[1.0, 1.0]]), np.array([1]))
    assert e.value.code == "gpu_oom" and e.value.message.startswith("Remote GPU:")


def test_forget_session_drops_remote_video(tmp_path, clip_2s, remote_settings):
    worker = FakeWorker()
    svc = RemoteGpuService(remote_settings, InProcessTransport(worker), make_ledger(tmp_path))
    s = make_session(tmp_path, clip_2s)
    svc.segment(s, 0, np.array([[300.0, 200.0]]), np.array([1]))
    svc.forget_session(s)
    assert ("drop_video", s.video_sha256) in worker.calls and not worker.videos


def test_modal_transport_without_package_or_token_gives_clear_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_modal(name, *a, **k):
        if name == "modal":
            raise ImportError("no modal")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_modal)
    with pytest.raises(RemoteTransportError) as e:
        ModalTransport("app", "Cls", "dict").call("info")
    assert e.value.code == "remote_unavailable" and "pip install modal" in e.value.message


# ---------------------------------------------------------------- 사용량·예산


def test_usage_ledger_counts_cold_start_active_and_idle_time(tmp_path):
    clock = Clock()
    led = make_ledger(tmp_path, clock=clock)
    t = clock.t
    r1 = led.record("segment", t, t + 2)  # 콜드: 45 + 2 + 대기 120
    assert r1["coldStart"] and r1["billableS"] == pytest.approx(167)
    r2 = led.record("track", t + 32, t + 92, tracked_frames=300)  # 30초 뒤: 미리 더한 대기 120 중 90 환불 + 60 + 120
    assert not r2["coldStart"] and r2["billableS"] == pytest.approx(-90 + 60 + 120)
    clock.t = t + 100
    tot = led.totals()
    assert tot["todayBillableS"] == pytest.approx(167 + 90) and tot["todayTrackedFrames"] == 300 and tot["likelyWarm"]
    # 실제 켜져 있던 시간: 0~92 사용 + 120 대기 = 212 + 콜드 45 = 257
    assert tot["todayBillableS"] == pytest.approx(257)
    # 다시 열면 같은 값 (파일에 저장)
    assert make_ledger(tmp_path, clock=clock).totals()["todayBillableS"] == pytest.approx(257)


def test_usage_ledger_blocks_daily_and_monthly_budget(tmp_path):
    clock = Clock()
    led = make_ledger(tmp_path, clock=clock, daily_limit_s=300, monthly_limit_usd=0.05)
    led.check(10)  # 45 + 10 + 120 = 175 ≤ 300
    led.record("track", clock.t, clock.t + 100)
    clock.t += 1000
    with pytest.raises(BudgetExceeded, match="Daily"):
        led.check(10)
    led2 = make_ledger(tmp_path / "m", clock=clock, daily_limit_s=0, monthly_limit_usd=0.05)
    with pytest.raises(BudgetExceeded, match="Monthly"):
        led2.check(100)  # (45 + 100 + 120) × 0.0002 = $0.053 > $0.05


# ---------------------------------------------------------------- GpuWorker (모델 없이 검증 가능한 부분)


def bare_worker(tmp_path, **kw) -> GpuWorker:
    engine = SimpleNamespace(runtime=None, state="ready", status=lambda: {"state": "ready"})
    return GpuWorker(Settings(**kw), tmp_path / "w", engine=engine)


def test_worker_put_video_validates_and_evicts(tmp_path, clip_2s):
    from tests.conftest import write_clip

    w = bare_worker(tmp_path)
    w.max_videos = 2
    data = clip_2s.read_bytes()
    assert w.put_video("0" * 64, data)["error"]["code"] == "validation"  # sha 불일치
    assert w.put_video("bad", data)["error"]["code"] == "validation"
    junk = b"not a video" * 100
    assert w.put_video(sha(junk), junk)["error"]["code"] == "validation"
    r = w.put_video(sha(data), data)
    assert r["ok"] and r["meta"]["frameCount"] == 60 and r["meta"]["width"] == 640
    assert w.put_video(sha(data), data)["cached"] is True
    clips = [write_clip(tmp_path / f"c{i}.mp4", 0.5 + 0.1 * i) for i in range(2)]
    for c in clips:
        b = c.read_bytes()
        assert w.put_video(sha(b), b)["ok"]
    assert not w.has_video(sha(data))["present"]  # 가장 오래된 영상 삭제
    assert not (tmp_path / "w" / "videos" / sha(data)).exists()
    w.drop_video(sha(clips[0].read_bytes()))
    assert len(list((tmp_path / "w" / "videos").iterdir())) == 1


def test_worker_rejects_oversized_upload(tmp_path, clip_2s):
    w = bare_worker(tmp_path, max_file_size_bytes=1000)
    data = clip_2s.read_bytes()
    assert w.put_video(sha(data), data)["error"]["status"] == 413


def test_worker_validates_prompts_and_missing_video(tmp_path, clip_2s):
    w = bare_worker(tmp_path)
    data = clip_2s.read_bytes()
    assert w.segment(sha(data), 0, [[1, 1]], [1])["error"]["code"] == "video_missing"
    w.put_video(sha(data), data)
    h = sha(data)
    assert "out of range" in w.segment(h, 60, [[1, 1]], [1])["error"]["message"]
    assert "outside" in w.segment(h, 0, [[700, 1]], [1])["error"]["message"]
    assert "positive" in w.segment(h, 0, [[1, 1]], [0])["error"]["message"]
    assert w.track(h, 0, [[1, 1]], [1], 61)["error"]["code"] == "validation"  # CPU 워커 한도 60


def test_pickle_roundtrip_rejects_unserializable_results(tmp_path):
    class Bad:
        def info(self):
            return {"ok": True, "lock": threading.Lock()}

    with pytest.raises(TypeError):
        InProcessTransport(Bad()).call("info")


# ---------------------------------------------------------------- API


@pytest.fixture
def client(isolated_sessions, tmp_path):
    with TestClient(app) as c:
        c.app.state.engine.state = "ready"
        yield c


@pytest.fixture
def remote_client(client, tmp_path, remote_settings):
    worker = FakeWorker()
    client.app.state.remote = RemoteGpuService(remote_settings, InProcessTransport(worker), make_ledger(tmp_path))
    client.worker = worker
    yield client
    client.app.state.remote = None


def upload(client, clip) -> str:
    with open(clip, "rb") as f:
        return client.post("/api/sessions", files={"file": ("clip.mp4", f, "video/mp4")}).json()["sessionId"]


def wait_job(client, job_id, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("succeeded", "failed", "cancelled"):
            return j
        time.sleep(0.02)
    raise TimeoutError


PTS = [{"x": 0.5, "y": 0.5, "label": 1}]


def test_health_reports_remote_without_calling_it(remote_client):
    h = remote_client.get("/api/health").json()
    assert h["remote"]["configured"] and h["remote"]["maxFrames"] == 300 and h["remote"]["gpu"] == "T4"
    assert h["remote"]["usage"]["dailyLimitS"] == 1800
    assert remote_client.worker.calls == []  # 상태 조회는 GPU 를 켜지 않음


def test_health_without_remote(client):
    assert client.get("/api/health").json()["remote"] == {"configured": False}


def test_remote_segment_and_track_feed_existing_preview_and_export(remote_client, clip_2s):
    from app.ffmpeg import find_ffmpeg

    c = remote_client
    sid = upload(c, clip_2s)
    r = c.post(f"/api/sessions/{sid}/segment", json={"frameIndex": 2, "points": PTS, "maskWidth": 320, "maskHeight": 180, "backend": "remote"}).json()
    assert r["metrics"]["backend"] == "remote" and r["metrics"]["deviceName"] == "Fake T4"
    m = rle_decode(r["mask"]["counts"], 320, 180)
    assert m[90, 160] == 1  # 원본 해상도 마스크 → 미리보기 크기로 축소

    job = c.post(f"/api/sessions/{sid}/propagate", json={"frameIndex": 2, "points": PTS, "maxFrames": 50, "backend": "remote"}).json()
    done = wait_job(c, job["id"])
    assert done["status"] == "succeeded" and done["stats"]["framesDone"] == 50 and done["stats"]["source"] == "remote-gpu"
    masks = c.get(f"/api/sessions/{sid}/tracks/{job['id']}/masks", params={"width": 320, "height": 180}).json()
    assert [f["index"] for f in masks["frames"]] == list(range(2, 52))

    if find_ffmpeg() is None:
        pytest.skip("FFmpeg not installed")
    fx = {"neon": {"enabled": False, "intensity": 0.8, "color": "#22d3ee", "thickness": 3}, "colorPop": {"enabled": True, "intensity": 1.0}, "blur": {"enabled": False, "intensity": 0.5}}
    exp = c.post(f"/api/sessions/{sid}/export", json={"trackJobId": job["id"], "effects": fx}).json()
    e = wait_job(c, exp["id"], timeout=120)
    assert e["status"] == "succeeded" and e["stats"]["frameCount"] == 50  # 원격 결과로 기존 내보내기 그대로


def test_auto_uses_remote_and_local_stays_available(remote_client, clip_2s, monkeypatch):
    c = remote_client
    sid = upload(c, clip_2s)
    r = c.post(f"/api/sessions/{sid}/segment", json={"frameIndex": 0, "points": PTS, "maskWidth": 64, "maskHeight": 36}).json()
    assert r["metrics"]["backend"] == "remote"
    # 로컬 CPU 선택: 원격을 부르지 않고 로컬 모델 경로 사용
    from app.inference.sam2_image import SegmentOutput

    monkeypatch.setattr(c.app.state.model, "segment", lambda *a, **k: SegmentOutput(np.ones((360, 640), np.uint8), 0.9, {"device": "cpu"}))
    n_calls = len(c.worker.calls)
    r = c.post(f"/api/sessions/{sid}/segment", json={"frameIndex": 0, "points": PTS, "maskWidth": 64, "maskHeight": 36, "backend": "local"}).json()
    assert r["metrics"]["backend"] == "local" and len(c.worker.calls) == n_calls


def test_remote_requested_but_not_configured(client, clip_2s):
    sid = upload(client, clip_2s)
    r = client.post(f"/api/sessions/{sid}/segment", json={"frameIndex": 0, "points": PTS, "maskWidth": 64, "maskHeight": 36, "backend": "remote"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "remote_unavailable"
    r = client.post(f"/api/sessions/{sid}/propagate", json={"frameIndex": 0, "points": PTS, "maxFrames": 10, "backend": "remote"})
    assert r.status_code == 409


def test_remote_track_cancel_via_api(remote_client, clip_2s):
    c = remote_client
    c.worker.frame_delay = 0.05
    sid = upload(c, clip_2s)
    job = c.post(f"/api/sessions/{sid}/propagate", json={"frameIndex": 0, "points": PTS, "maxFrames": 60, "backend": "remote"}).json()
    assert c.worker.started.wait(10)
    time.sleep(0.2)
    c.post(f"/api/jobs/{job['id']}/cancel")
    done = wait_job(c, job["id"])
    assert done["status"] == "cancelled" and 0 < done["stats"]["framesDone"] < 60
    masks = c.get(f"/api/sessions/{sid}/tracks/{job['id']}/masks", params={"width": 64, "height": 36}).json()
    assert len(masks["frames"]) == done["stats"]["framesDone"]  # 취소 전 프레임은 그대로 사용 가능


def test_remote_limits_via_api(remote_client, clip_2s):
    sid = upload(remote_client, clip_2s)
    r = remote_client.post(f"/api/sessions/{sid}/propagate", json={"frameIndex": 0, "points": PTS, "maxFrames": 301, "backend": "remote"})
    assert r.status_code == 422
    remote_client.app.state.remote.ledger.daily_limit_s = 1
    r = remote_client.post(f"/api/sessions/{sid}/propagate", json={"frameIndex": 0, "points": PTS, "maxFrames": 10, "backend": "remote"})
    assert r.status_code == 429 and r.json()["error"]["code"] == "budget_exceeded"


def test_warmup_and_session_delete_cleanup(remote_client, clip_2s):
    c = remote_client
    st = c.post("/api/remote/warmup").json()
    assert st["state"] == "ready" and st["worker"]["gpuName"] == "Fake T4"
    sid = upload(c, clip_2s)
    c.post(f"/api/sessions/{sid}/segment", json={"frameIndex": 0, "points": PTS, "maskWidth": 64, "maskHeight": 36, "backend": "remote"})
    assert c.worker.videos
    assert c.delete(f"/api/sessions/{sid}").status_code == 204
    assert not c.worker.videos  # 원격 임시 파일도 삭제
