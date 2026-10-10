"""Ingestion plumbing for learning resources (ADR-0032): runs, versioned raw records, normalized upserts."""

import hashlib
import json
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from xcrs.db.models import IngestRun, LearningResource, RawRecord


def start_run(session: Session, source: str) -> IngestRun:
    run = IngestRun(source=source)
    session.add(run)
    session.flush()
    return run


def finish_run(session: Session, run: IngestRun, stats: dict, error: str | None = None) -> None:
    run.finished_at = datetime.now(UTC)
    run.status = "failed" if error else "ok"
    run.stats, run.error = stats, error
    session.flush()


def store_raw(session: Session, run: IngestRun, source: str, external_id: str, payload: dict) -> bool:
    """Keep the payload as received; a new version only when its content changed. True if stored."""
    content_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    stored = session.execute(
        insert(RawRecord)
        .values(source=source, external_id=external_id, content_hash=content_hash, payload=payload, run_id=run.id)
        .on_conflict_do_nothing()
        .returning(RawRecord.id)
    ).scalar()
    return stored is not None


def upsert_resource(session: Session, row: dict) -> int | None:
    """Insert or refresh a normalized resource by (source, external_id). A URL another source already
    serves (curated entries win, ADR-0033) is skipped: returns None."""
    owner = session.scalar(select(LearningResource.source).where(LearningResource.url == row["url"]))
    if owner is not None and owner != row["source"]:
        return None
    stmt = insert(LearningResource).values(**row, fetched_at=datetime.now(UTC))
    update = {c: stmt.excluded[c] for c in row if c not in ("source", "external_id")}
    update |= {"fetched_at": stmt.excluded.fetched_at, "updated_at": stmt.excluded.fetched_at, "is_active": True}
    return session.execute(
        stmt.on_conflict_do_update(index_elements=["source", "external_id"], set_=update).returning(LearningResource.id)
    ).scalar()
