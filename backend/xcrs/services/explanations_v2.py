"""Background explanations for engine v2 roles (ADR-0018, ADR-0037).

Same shape as v1 (services/explanations.py): the explanations_v2 rows are the queue, the in-memory queue
only says what to look at next, startup re-queues `pending` rows, and the LLM call runs outside any
database session. One thread by default: a CPU-bound LLM gains nothing from concurrency.
"""

import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from xcrs.db.models import ExplanationV2, RecommendationV2
from xcrs.explain.v2 import PROMPT_VERSION_V2, RoleExplanationV2Out

log = logging.getLogger(__name__)


class ExplainerV2(Protocol):
    def explain(self, facts: dict) -> RoleExplanationV2Out | None: ...


def pending(session: Session) -> list[tuple[uuid.UUID, str]]:
    rows = session.execute(
        select(ExplanationV2.recommendation_id, ExplanationV2.role)
        .where(ExplanationV2.status == "pending")
        .order_by(ExplanationV2.created_at, ExplanationV2.rank)
    ).all()
    return [(r.recommendation_id, r.role) for r in rows]


def claim(session: Session, recommendation_id: uuid.UUID, role: str, max_attempts: int) -> dict | None:
    row = session.scalars(
        select(ExplanationV2)
        .where(ExplanationV2.recommendation_id == recommendation_id, ExplanationV2.role == role)
        .with_for_update()
    ).one_or_none()
    if row is None or row.status != "pending":
        return None
    if row.attempts >= max_attempts:
        row.status = "failed"
        session.commit()
        return None
    row.attempts += 1
    session.commit()
    return row.input


def save(session: Session, recommendation_id: uuid.UUID, role: str, out: RoleExplanationV2Out | None, ms: int) -> str:
    status = "done" if out else "failed"
    session.execute(
        update(ExplanationV2)
        .where(ExplanationV2.recommendation_id == recommendation_id, ExplanationV2.role == role)
        .values(
            status=status,
            explanation=out.explanation if out else None,
            next_step=out.next_step if out else None,
            prompt_version=PROMPT_VERSION_V2 if out else None,
            ms=ms,
            explained_at=datetime.now(UTC),
        )
    )
    session.commit()
    return status


def for_recommendation(session: Session, recommendation_id: uuid.UUID) -> dict[str, ExplanationV2]:
    rows = session.scalars(select(ExplanationV2).where(ExplanationV2.recommendation_id == recommendation_id))
    return {r.role: r for r in rows}


class ExplanationWorkerV2:
    def __init__(self, explainer: ExplainerV2, session_factory: Callable[[], Session], max_attempts: int = 3):
        self._explainer, self._session_factory, self._max_attempts = explainer, session_factory, max_attempts
        self._queue: queue.Queue[tuple[uuid.UUID, str] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="explain-v2", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._queue.put(None)
        self._thread.join(timeout_s)

    def submit(self, recommendation_id: uuid.UUID, role: str) -> None:
        self._queue.put((recommendation_id, role))

    def requeue_pending(self) -> int:
        with self._session_factory() as session:
            jobs = pending(session)
        for job in jobs:
            self.submit(*job)
        return len(jobs)

    def wait_until_idle(self) -> None:
        self._queue.join()

    def process(self, recommendation_id: uuid.UUID, role: str) -> str | None:
        with self._session_factory() as session:
            facts = claim(session, recommendation_id, role, self._max_attempts)
        if facts is None:
            return None
        started = time.perf_counter()
        out = self._explainer.explain(facts)
        ms = round((time.perf_counter() - started) * 1000)
        with self._session_factory() as session:
            status = save(session, recommendation_id, role, out, ms)
        log.info("v2 explanation %s for %s / %s in %d ms", status, recommendation_id, role, ms)
        return status

    def _run(self) -> None:
        while (job := self._queue.get()) is not None:
            try:
                self.process(*job)
            except Exception:
                log.exception("v2 explanation job %s failed", job)
            finally:
                self._queue.task_done()
        self._queue.task_done()


def exists(session: Session, recommendation_id: uuid.UUID) -> bool:
    return session.get(RecommendationV2, recommendation_id) is not None
