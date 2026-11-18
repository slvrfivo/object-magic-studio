"""백그라운드 작업 관리자.

큐마다 단일 워커 스레드를 둡니다.
  - "model": 모델을 쓰는 작업 (추적). 모델이 하나뿐이므로 순서대로 실행
  - "export": 모델을 쓰지 않는 작업 (MP4 내보내기). 긴 추적 작업 뒤에서 기다리지 않음
작업 함수는 job.update() 로 진행률을 알리고, job.cancelled 를 주기적으로 확인해 협조적으로 취소합니다.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .errors import AppError, not_found

log = logging.getLogger(__name__)

TERMINAL = {"succeeded", "failed", "cancelled"}


@dataclass
class Job:
    id: str
    kind: str
    session_id: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    error: dict | None = None
    stats: dict = field(default_factory=dict)
    result: dict | None = None
    result_url: str | None = None
    created_at: float = field(default_factory=time.time)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def update(self, progress: float | None = None, message: str | None = None, stats: dict | None = None) -> None:
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))
        if message is not None:
            self.message = message
        if stats is not None:
            self.stats = stats

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "error": self.error,
            "stats": self.stats,
            "resultUrl": self.result_url if self.status == "succeeded" else None,
        }


class JobManager:
    def __init__(self):
        self._executors: dict[str, ThreadPoolExecutor] = {}
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def _executor(self, queue: str) -> ThreadPoolExecutor:
        with self._lock:
            if queue not in self._executors:
                self._executors[queue] = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"job-{queue}")
            return self._executors[queue]

    def submit(
        self,
        kind: str,
        session_id: str,
        fn: Callable[[Job], dict | None],
        is_oom=lambda e: False,
        queue: str = "model",
    ) -> Job:
        job = Job(id=uuid.uuid4().hex, kind=kind, session_id=session_id)
        with self._lock:
            self._jobs[job.id] = job
        self._executor(queue).submit(self._run, job, fn, is_oom)
        return job

    def _run(self, job: Job, fn, is_oom) -> None:
        if job.cancelled:
            job.status, job.message = "cancelled", "Cancelled before start"
            return
        job.status, job.message = "running", "Starting…"
        try:
            job.result = fn(job)
            if job.cancelled:
                job.status, job.message = "cancelled", job.message or "Cancelled"
            else:
                job.status, job.progress = "succeeded", 1.0
        except AppError as e:
            job.status, job.error = "failed", {"code": e.code, "message": e.message}
        except Exception as e:  # noqa: BLE001
            log.exception("Job %s failed", job.id)
            if is_oom(e):
                job.error = {"code": "gpu_oom", "message": "GPU out of memory during tracking."}
            else:
                job.error = {"code": "inference_failed", "message": f"{type(e).__name__}: {e}"}
            job.status = "failed"
        if job.status == "failed":
            job.message = "Failed"

    def register(self, kind: str, session_id: str, stats: dict, message: str, result: dict | None = None) -> Job:
        """외부에서 완료된 작업(예: Colab GPU 추적 결과 가져오기)을 성공 상태로 등록."""
        job = Job(id=uuid.uuid4().hex, kind=kind, session_id=session_id, status="succeeded", progress=1.0, message=message, stats=stats, result=result)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise not_found("Job not found.")
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status not in TERMINAL:
            job._cancel.set()
            if job.status == "queued":
                job.status, job.message = "cancelled", "Cancelled"
            else:
                job.message = "Cancelling…"
        return job

    def cancel_session(self, session_id: str) -> None:
        with self._lock:
            jobs = [j for j in self._jobs.values() if j.session_id == session_id]
        for j in jobs:
            if j.status not in TERMINAL:
                self.cancel(j.id)

    def shutdown(self) -> None:
        with self._lock:
            for j in self._jobs.values():
                j._cancel.set()
            executors = list(self._executors.values())
        for ex in executors:
            ex.shutdown(wait=False, cancel_futures=True)
