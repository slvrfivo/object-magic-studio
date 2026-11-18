import threading
import time

import pytest

from app.errors import AppError
from app.inference.sam2_video import classify_frames
from app.jobs import JobManager


def wait_terminal(mgr, job_id, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = mgr.get(job_id)
        if job.status in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.01)
    raise TimeoutError


def test_success_with_progress():
    mgr = JobManager()

    def work(job):
        for i in range(3):
            job.update((i + 1) / 3, f"step {i}", {"framesDone": i + 1})
        return {"ok": True}

    job = wait_terminal(mgr, mgr.submit("propagate", "s", work).id)
    assert job.status == "succeeded" and job.progress == 1.0
    assert job.result == {"ok": True} and job.stats == {"framesDone": 3}


def test_cooperative_cancel_keeps_partial_stats():
    mgr = JobManager()
    started = threading.Event()

    def work(job):
        n = 0
        while not job.cancelled:
            n += 1
            job.update(stats={"framesDone": n})
            started.set()
            time.sleep(0.01)
        return {"framesDone": n}

    job = mgr.submit("propagate", "s", work)
    started.wait(2)
    mgr.cancel(job.id)
    job = wait_terminal(mgr, job.id)
    assert job.status == "cancelled"
    assert job.stats["framesDone"] >= 1


def test_queued_job_cancelled_before_start():
    mgr = JobManager()
    gate = threading.Event()
    first = mgr.submit("propagate", "s", lambda job: gate.wait(2))
    second = mgr.submit("propagate", "s", lambda job: {"ran": True})
    assert mgr.get(second.id).status == "queued"
    mgr.cancel(second.id)
    gate.set()
    wait_terminal(mgr, first.id)
    assert wait_terminal(mgr, second.id).status == "cancelled"
    assert mgr.get(second.id).result is None


@pytest.mark.parametrize(
    "exc,code",
    [(AppError("validation", "bad", 422), "validation"), (RuntimeError("boom"), "inference_failed")],
)
def test_errors_are_reported(exc, code):
    mgr = JobManager()

    def work(job):
        raise exc

    job = wait_terminal(mgr, mgr.submit("propagate", "s", work).id)
    assert job.status == "failed" and job.error["code"] == code


def test_oom_is_reported_as_gpu_oom():
    mgr = JobManager()

    class FakeOom(Exception):
        pass

    def work(job):
        raise FakeOom()

    job = wait_terminal(mgr, mgr.submit("propagate", "s", work, is_oom=lambda e: isinstance(e, FakeOom)).id)
    assert job.error["code"] == "gpu_oom"


def test_cancel_session_only_affects_that_session():
    mgr = JobManager()
    gate = threading.Event()
    a = mgr.submit("propagate", "A", lambda job: gate.wait(2))
    b = mgr.submit("propagate", "B", lambda job: None)
    mgr.cancel_session("B")
    gate.set()
    assert wait_terminal(mgr, a.id).status == "succeeded"
    assert wait_terminal(mgr, b.id).status == "cancelled"


def test_classify_frames_lost_and_unstable():
    lost, unstable = classify_frames([0.04, 0.041, 0.0, 0.0, 0.039, 0.12, 0.04], lost_threshold=0.0002, jump_ratio=2.0)
    assert lost == [2, 3]
    assert unstable == [5, 6]  # 0.039 → 0.12 (x3.1), 0.12 → 0.04 (x3)
