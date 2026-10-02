"""YouTube playlists as learning resources (ADR-0026, ADR-0033), via the YouTube Data API v3.

Only `playlists.list` (1 quota unit per 50 playlists) for the playlist ids in catalog/sources/youtube.yaml:
no search calls (capped at 100 a day). The API terms require stored API data to be refreshed or deleted
within 30 days (`expire`), and YouTube to be shown as the source. Off until XCRS_YOUTUBE_API_KEY is set.
"""

from datetime import UTC, datetime, timedelta

import httpx
import yaml
from sqlalchemy import delete
from sqlalchemy.orm import Session

from xcrs.catalog.model import CATALOG_DIR
from xcrs.db.models import LearningResource, RawRecord
from xcrs.ingest.resources.base import finish_run, start_run, store_raw, upsert_resource

SOURCE = "youtube"
API = "https://www.googleapis.com/youtube/v3/playlists"
MAX_AGE = timedelta(days=30)


class NoApiKey(RuntimeError):
    pass


def configured_playlists(path=CATALOG_DIR / "sources" / "youtube.yaml") -> list[str]:
    data = yaml.safe_load(path.read_text()) if path.exists() else None
    return [str(p) for p in (data or {}).get("playlists") or []]


def normalize(item: dict) -> dict:
    snippet = item.get("snippet", {})
    return {
        "source": SOURCE,
        "external_id": item["id"],
        "type": "playlist",
        "provider": f"YouTube · {snippet.get('channelTitle', 'unknown channel')}",
        "url": f"https://www.youtube.com/playlist?list={item['id']}",
        "title": snippet.get("title", item["id"]),
        "description": (snippet.get("description") or "")[:1000] or None,
        "language": (snippet.get("defaultLanguage") or "en")[:2],
        "is_free": True,
        "quality": {"videos": item.get("contentDetails", {}).get("itemCount")},
    }


def ingest(session: Session, client: httpx.Client, api_key: str | None, playlists: list[str]) -> dict:
    if not api_key:
        raise NoApiKey("set XCRS_YOUTUBE_API_KEY to use the YouTube adapter (ADR-0033)")
    run = start_run(session, SOURCE)
    stats = {"requested": len(playlists), "upserted": 0}
    try:
        for i in range(0, len(playlists), 50):
            ids = ",".join(playlists[i : i + 50])
            response = client.get(
                API, params={"part": "snippet,contentDetails", "id": ids, "key": api_key, "maxResults": 50}
            )
            response.raise_for_status()
            for item in response.json().get("items", []):
                store_raw(session, run, SOURCE, item["id"], item)
                stats["upserted"] += upsert_resource(session, normalize(item)) is not None
        finish_run(session, run, stats)
        return stats
    except httpx.HTTPError as exc:
        finish_run(session, run, stats, error=str(exc))
        raise


def expire(session: Session, now: datetime | None = None) -> int:
    """Delete YouTube data not refreshed within 30 days (API terms). Returns resources deleted."""
    cutoff = (now or datetime.now(UTC)) - MAX_AGE
    deleted = session.execute(
        delete(LearningResource)
        .where(LearningResource.source == SOURCE, LearningResource.fetched_at < cutoff)
        .returning(LearningResource.id)
    ).all()
    session.execute(delete(RawRecord).where(RawRecord.source == SOURCE, RawRecord.fetched_at < cutoff))
    return len(deleted)
