"""Background jobs, because checking a capture takes most of a minute.

A capture is allowed 60 seconds on a laptop CPU, which is far too long to hold
an HTTP request open: the browser, and anything between it and the server, is
entitled to give up long before then. So an upload returns a job id straight
away and the browser asks for progress.

Only one job runs at a time. The pipeline is CPU-bound and the 60-second budget
is measured on a machine that is not doing anything else, so running two at
once would not make either finish sooner and would make both miss the budget.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from capturecheck.utils.logging import get_logger

log = get_logger(__name__)

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"


@dataclass
class Job:
    """One uploaded capture, on its way through the pipeline."""

    job_id: str
    filename: str
    status: str = QUEUED
    stage: str = "waiting to start"
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    result: dict | None = None
    error: str | None = None
    #: Where the upload was written; removed once the job ends.
    workdir: Path | None = None

    @property
    def elapsed_s(self) -> float:
        if self.started is None:
            return 0.0
        return (self.finished or time.time()) - self.started

    def as_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "status": self.status,
            "stage": self.stage,
            "elapsed_s": round(self.elapsed_s, 1),
            "queued_s": round((self.started or time.time()) - self.created, 1),
            "result": self.result,
            "error": self.error,
        }


class JobStore:
    """Runs capture jobs one at a time and remembers what they said.

    In memory on purpose: a result is worth keeping only as long as the browser
    that asked for it is still open, and a restart should not resurrect a job
    whose uploaded file is long gone.
    """

    def __init__(self, max_jobs: int = 50) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="capturecheck")
        self._max_jobs = max_jobs

    # ------------------------------------------------------------------ #
    def submit(
        self,
        filename: str,
        workdir: Path,
        work: Callable[[Callable[[str], None]], dict],
    ) -> Job:
        """Queue ``work``, which is handed a function for reporting its stage."""
        job = Job(job_id=uuid.uuid4().hex[:12], filename=filename, workdir=workdir)
        with self._lock:
            self._jobs[job.job_id] = job
            self._forget_old()
        self._pool.submit(self._run, job, work)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: -j.created)

    def forget(self, job_id: str) -> None:
        """Drop one job from memory."""
        with self._lock:
            self._jobs.pop(job_id, None)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ #
    def _run(self, job: Job, work: Callable[[Callable[[str], None]], dict]) -> None:
        def stage(text: str) -> None:
            job.stage = text
            log.info("job %s: %s", job.job_id, text)

        job.status = RUNNING
        job.started = time.time()
        stage("reading the capture")

        outcome, stage_text = DONE, "done"
        try:
            job.result = work(stage)
        except Exception as exc:  # the browser gets a sentence, not a traceback
            log.exception("job %s failed", job.job_id)
            outcome, stage_text = FAILED, "failed"
            job.error = str(exc) or exc.__class__.__name__
        finally:
            job.finished = time.time()
            self._cleanup(job)
            # The terminal status goes last, so a client that polls and sees
            # `done` sees a job that is done *including* its tidying up: the
            # uploaded file is already gone by the time anyone is told.
            job.stage = stage_text
            job.status = outcome
            # Pruning only at submit time would never drop anything: every job
            # is still running then, and only finished jobs may be forgotten.
            with self._lock:
                self._forget_old()

    def _cleanup(self, job: Job) -> None:
        """Delete the uploaded file. It is not ours to keep."""
        if job.workdir is None:
            return
        import shutil

        shutil.rmtree(job.workdir, ignore_errors=True)
        job.workdir = None

    def _forget_old(self) -> None:
        """Keep the newest ``max_jobs``, dropping the oldest finished ones.

        Call with ``_lock`` held. A job still queued or running is never
        dropped however old it is: the browser is still waiting on it.
        """
        over = len(self._jobs) - self._max_jobs
        if over <= 0:
            return
        oldest_first = sorted(self._jobs.values(), key=lambda j: j.created)
        for job in oldest_first:
            if over <= 0:
                break
            if job.status in (DONE, FAILED):
                self._jobs.pop(job.job_id, None)
                over -= 1


def wait_for(store: JobStore, job_id: str, timeout: float = 120.0) -> Job | None:
    """Block until a job finishes. For tests and scripts, never for a request."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = store.get(job_id)
        if job is None:
            return None
        if job.status in (DONE, FAILED):
            return job
        time.sleep(0.02)
    return store.get(job_id)


__all__ = ["DONE", "FAILED", "QUEUED", "RUNNING", "Job", "JobStore", "wait_for"]
