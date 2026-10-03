"""YouTube playlists as learning resources (ADR-0026, ADR-0033), via the YouTube Data API v3.

Only `playlists.list` (1 quota unit per 50 playlists) for the playlist ids in catalog/sources/youtube.yaml:
no search calls (capped at 100 a day). The API terms require stored API data to be refreshed or deleted
within 30 days (`expire`), and YouTube to be shown as the source. Off until XCRS_YOUTUBE_API_KEY is set.
"""

from datetime import UTC, datetime, timedelta

import httpx
import yaml
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from xcrs.catalog.model import CATALOG_DIR
from xcrs.db.models import LearningResource, RawRecord, ResourceSkill, Skill
from xcrs.ingest.resources.base import finish_run, start_run, store_raw, upsert_resource

SOURCE = "youtube"
API = "https://www.googleapis.com/youtube/v3/playlists"
MAX_AGE = timedelta(days=30)


class NoApiKey(RuntimeError):
    pass


class QuotaExceeded(RuntimeError):
    """The daily quota (10,000 units) or a rate limit is used up; try again after midnight Pacific time."""


class YouTubeError(RuntimeError):
    pass


def api_get(client: httpx.Client, url: str, api_key: str, params: dict) -> dict:
    """GET a YouTube Data API endpoint. The key goes in a header, never the URL, so it can't leak into
    error messages, logs or the runs table; errors carry only the endpoint and status."""
    response = client.get(url, params=params, headers={"X-Goog-Api-Key": api_key})
    if response.status_code in (403, 429):
        reasons = (
            {e.get("reason") for e in response.json().get("error", {}).get("errors", [])} if response.content else set()
        )
        if response.status_code == 429 or reasons & {"quotaExceeded", "rateLimitExceeded", "dailyLimitExceeded"}:
            raise QuotaExceeded(f"{url.rsplit('/', 1)[-1]}: quota or rate limit reached ({response.status_code})")
    if response.is_error:
        raise YouTubeError(f"{url.rsplit('/', 1)[-1]}: HTTP {response.status_code}")
    return response.json()


def configured_playlists(path=CATALOG_DIR / "sources" / "youtube.yaml") -> list[tuple[str, str | None]]:
    """(playlist id, the skill it was approved for) from catalog/sources/youtube.yaml; plain ids are allowed."""
    data = yaml.safe_load(path.read_text()) if path.exists() else None
    out = []
    for p in (data or {}).get("playlists") or []:
        out.append((str(p["id"]), p.get("skill")) if isinstance(p, dict) else (str(p), None))
    return out


def tag_reviewed(session: Session, resource_id: int, skill: str) -> None:
    """The skill the decider approved the playlist for: always a tag, at working level."""
    skill_id = session.scalar(select(Skill.id).where(Skill.slug == skill))
    if skill_id is None:
        raise ValueError(f"catalog/sources/youtube.yaml: unknown skill {skill!r}")
    stmt = insert(ResourceSkill).values(
        resource_id=resource_id, skill_id=skill_id, relation="teaches", level=2, confidence=1.0, tagged_by="reviewed"
    )
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=["resource_id", "skill_id", "relation"],
            set_={"level": 2, "confidence": 1.0, "tagged_by": "reviewed"},
        )
    )


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


def ingest(
    session: Session, client: httpx.Client, api_key: str | None, playlists: list[str] | list[tuple[str, str | None]]
) -> dict:
    if not api_key:
        raise NoApiKey("set XCRS_YOUTUBE_API_KEY to use the YouTube adapter (ADR-0033)")
    approved = dict(p if isinstance(p, tuple) else (p, None) for p in playlists)
    playlists = list(approved)
    run = start_run(session, SOURCE)
    stats = {"requested": len(playlists), "upserted": 0}
    try:
        for i in range(0, len(playlists), 50):
            ids = ",".join(playlists[i : i + 50])
            page = api_get(client, API, api_key, {"part": "snippet,contentDetails", "id": ids, "maxResults": 50})
            for item in page.get("items", []):
                store_raw(session, run, SOURCE, item["id"], item)
                resource_id = upsert_resource(session, normalize(item))
                stats["upserted"] += resource_id is not None
                if resource_id is not None and approved.get(item["id"]):
                    tag_reviewed(session, resource_id, approved[item["id"]])
        finish_run(session, run, stats)
        return stats
    except (httpx.HTTPError, QuotaExceeded, YouTubeError) as exc:
        finish_run(
            session,
            run,
            stats,
            error=f"{type(exc).__name__}: {exc}" if not isinstance(exc, httpx.HTTPError) else type(exc).__name__,
        )
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
