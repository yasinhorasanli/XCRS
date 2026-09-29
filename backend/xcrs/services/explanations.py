"""Explanation jobs in the background (ADR-0018).

The recommendation is returned at once; each role's explanation is a job. The rows in recommended_roles
are the queue's source of truth: the in-memory queue only says what to look at next, so on startup
every role still `pending` is queued again and a restart delays explanations instead of losing them.
"""

import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from typing import Protocol

from sqlalchemy.orm import Session

from xcrs.domain.results import ExplanationStatus
from xcrs.explain.base import Explainer, RoleContext
from xcrs.repository import activity

log = logging.getLogger(__name__)


class ExplanationQueue(Protocol):
    def submit(self, request_id: uuid.UUID, role_id: int) -> None: ...


class ExplanationWorker:
    """In-process worker threads. One by default: a CPU-bound LLM gains nothing from concurrency."""

    def __init__(
        self, explainer: Explainer, session_factory: Callable[[], Session], threads: int = 1, max_attempts: int = 3
    ):
        self._explainer = explainer
        self._session_factory = session_factory
        self._max_attempts = max_attempts
        self._queue: queue.Queue[tuple[uuid.UUID, int] | None] = queue.Queue()
        self._threads = [
            threading.Thread(target=self._run, name=f"explain-{i}", daemon=True) for i in range(max(1, threads))
        ]

    def start(self) -> None:
        for t in self._threads:
            t.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        """Ask the threads to finish. A job mid-generation can't be interrupted; its row stays `pending`
        and is picked up again at the next start."""
        for _ in self._threads:
            self._queue.put(None)
        for t in self._threads:
            t.join(timeout_s)

    def submit(self, request_id: uuid.UUID, role_id: int) -> None:
        self._queue.put((request_id, role_id))

    def requeue_pending(self) -> int:
        with self._session_factory() as session:
            jobs = activity.pending_roles(session)
        for job in jobs:
            self.submit(*job)
        return len(jobs)

    def wait_until_idle(self) -> None:
        self._queue.join()

    def process(self, request_id: uuid.UUID, role_id: int) -> ExplanationStatus | None:
        """Run one job. The LLM call happens outside any database session, so a slow generation
        doesn't hold a connection for minutes."""
        with self._session_factory() as session:
            data = activity.claim_role(session, request_id, role_id, self._max_attempts)
        if data is None:
            return None
        started = time.perf_counter()
        explanation = self._explainer.explain(RoleContext.from_dict(data))
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        with self._session_factory() as session:
            status = activity.save_explanation(session, request_id, role_id, explanation, elapsed_ms)
        log.info("explanation %s for request %s role %s in %d ms", status, request_id, role_id, elapsed_ms)
        return status

    def _run(self) -> None:
        while (job := self._queue.get()) is not None:
            try:
                self.process(*job)
            except Exception:
                # Keep the worker alive. The row stays `pending` and is retried on the next start,
                # up to max_attempts.
                log.exception("explanation job %s failed", job)
            finally:
                self._queue.task_done()
        self._queue.task_done()
