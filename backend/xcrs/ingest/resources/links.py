"""Check that learning-resource links still work (ADR-0033); never run in CI (needs the network)."""

import concurrent.futures as cf
from datetime import UTC, datetime

import httpx
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from xcrs.db.models import LearningResource

# Some sites answer bots with 403; a browser-like agent gets the page a learner would get.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130 Safari/537.36 (XCRS link check)",
    "Accept": "text/html,*/*",
}


def status_of(url: str, timeout: float = 20.0) -> int:
    """HTTP status after redirects; 0 when the host can't be reached."""
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers=HEADERS) as client:
            return client.get(url).status_code
    except httpx.HTTPError:
        return 0


def check_links(session: Session, workers: int = 12) -> dict:
    """Record each active resource's status; returns the broken ones. The caller commits."""
    rows = session.execute(select(LearningResource.id, LearningResource.url).where(LearningResource.is_active)).all()
    with cf.ThreadPoolExecutor(workers) as pool:
        statuses = list(pool.map(lambda r: status_of(r.url), rows))
    now = datetime.now(UTC)
    for row, status in zip(rows, statuses, strict=True):
        session.execute(
            update(LearningResource)
            .where(LearningResource.id == row.id)
            .values(last_status=status, last_checked_at=now)
        )
    broken = [(row.url, status) for row, status in zip(rows, statuses, strict=True) if not 200 <= status < 400]
    return {"checked": len(rows), "broken": broken}
