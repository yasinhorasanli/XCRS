"""YouTube playlists and long videos as learning resources (ADR-0026, ADR-0033, ADR-0046), via the YouTube
Data API v3, for the ids approved in catalog/sources/youtube.yaml. No search calls here (100 units each).

Each run reads, at 1 quota unit per 50 ids or items: the playlists (`playlists.list`), their videos
(`playlistItems.list`), and every video's details (`videos.list`: duration, captions, views, likes, date,
description). From these it stores per resource a total duration and a few quality numbers, and its
**sections**: a playlist's videos, or a long video's chapters (the timestamps in its description), so a gap can
link to the part that teaches it. About 300 units a day for 140 playlists of ~40 videos.

The API terms require stored API data to be refreshed or deleted within 30 days (`expire`; sections go with
their resource), and YouTube to be shown as the source. Off until XCRS_YOUTUBE_API_KEY is set.
"""

import hashlib
import json
import re
import statistics
from datetime import UTC, datetime, timedelta
from itertools import pairwise

import httpx
import yaml
from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from xcrs.catalog.model import CATALOG_DIR
from xcrs.db.models import LearningResource, RawRecord, ResourceSection, ResourceSkill, Skill
from xcrs.ingest.resources.base import finish_run, start_run, store_raw, upsert_resource

SOURCE = "youtube"
API = "https://www.googleapis.com/youtube/v3/playlists"
ITEMS = "https://www.googleapis.com/youtube/v3/playlistItems"
VIDEOS = "https://www.googleapis.com/youtube/v3/videos"
MAX_AGE = timedelta(days=30)
MAX_ITEMS = 500  # videos read per playlist (10 quota units); longer playlists keep their first 500 as sections
UNAVAILABLE = {"Deleted video", "Private video"}


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


def _configured(kind: str, path) -> list[tuple[str, str | None]]:
    data = yaml.safe_load(path.read_text()) if path.exists() else None
    out = []
    for p in (data or {}).get(kind) or []:
        out.append((str(p["id"]), p.get("skill")) if isinstance(p, dict) else (str(p), None))
    return out


def configured_playlists(path=CATALOG_DIR / "sources" / "youtube.yaml") -> list[tuple[str, str | None]]:
    """(playlist id, the skill it was approved for) from catalog/sources/youtube.yaml; plain ids are allowed."""
    return _configured("playlists", path)


def configured_videos(path=CATALOG_DIR / "sources" / "youtube.yaml") -> list[tuple[str, str | None]]:
    """(video id, the skill it was approved for): single long videos, approved like playlists (ADR-0046)."""
    return _configured("videos", path)


def tag_reviewed(session: Session, resource_id: int, skill: str) -> None:
    """The skill the decider approved the playlist or video for: always a tag, at working level."""
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


DURATION = re.compile(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$")


def parse_duration(value: str | None) -> int | None:
    """ISO 8601 duration ("PT1H2M3S", "P1DT2H") to seconds; None if absent or not a duration."""
    match = DURATION.match(value or "")
    if not match or not any(match.groups()):
        return None
    days, hours, mins, secs = (int(g or 0) for g in match.groups())
    return ((days * 24 + hours) * 60 + mins) * 60 + secs


# "00:00 Intro", "(1:02:03) - Docker", "▶ 12:30 | Volumes": a timestamp at the start of a line, then the title.
TIMESTAMP = re.compile(r"^\W{0,3}[(\[]?((?:\d{1,2}:)?\d{1,2}:\d{2})[)\]]?\s*[-–—:|•]?\s*(.{2,200}?)\s*$")


def to_seconds(stamp: str) -> int:
    return sum(int(p) * 60**i for i, p in enumerate(reversed(stamp.split(":"))))


def parse_chapters(description: str | None, duration_s: int | None = None) -> list[tuple[int, str]]:
    """YouTube's own chapter rule: timestamps at the start of description lines, the first at 0:00, at least
    three, in increasing order (each at least 10 s after the last), within the video. [] if not chapters."""
    found: list[tuple[int, str]] = []
    for line in (description or "").splitlines():
        match = TIMESTAMP.match(line.strip())
        if not match:
            continue
        start, title = to_seconds(match.group(1)), match.group(2).strip(" -–—:|•")
        if found and start < found[-1][0] + 10:
            continue  # a stray time in a sentence, or a repeat
        if duration_s and start >= duration_s:
            break
        if title and not re.fullmatch(r"https?://\S+", title):
            found.append((start, title[:200]))
    if len(found) < 3 or found[0][0] != 0:
        return []
    return found


def fetch_videos(client: httpx.Client, api_key: str, ids: list[str]) -> dict[str, dict]:
    """Video details for many ids at 1 unit per 50; ids that are private or gone are missing from the result."""
    out: dict[str, dict] = {}
    unique = list(dict.fromkeys(ids))
    for i in range(0, len(unique), 50):
        params = {"part": "snippet,contentDetails,statistics,status", "id": ",".join(unique[i : i + 50])}
        out |= {v["id"]: v for v in api_get(client, VIDEOS, api_key, params | {"maxResults": 50}).get("items", [])}
    return out


def playlist_items(client: httpx.Client, api_key: str, playlist_id: str) -> list[dict]:
    """The playlist's entries in order (up to MAX_ITEMS), as {video, title}; 1 unit per 50."""
    items: list[dict] = []
    token = None
    while len(items) < MAX_ITEMS:
        params = {"part": "snippet,contentDetails", "playlistId": playlist_id, "maxResults": 50}
        if token:
            params["pageToken"] = token
        try:
            page = api_get(client, ITEMS, api_key, params)
        except YouTubeError:  # a playlist that turned private between the two calls
            break
        for item in page.get("items", []):
            video = item.get("contentDetails", {}).get("videoId")
            title = item.get("snippet", {}).get("title", "")
            if video and title not in UNAVAILABLE:
                items.append({"video": video, "title": title})
        token = page.get("nextPageToken")
        if not token:
            break
    return items[:MAX_ITEMS]


def video_facts(item: dict) -> dict:
    """The numbers kept for a video: duration, captions, views, likes and date."""
    counts, details = item.get("statistics", {}), item.get("contentDetails", {})
    return {
        "seconds": parse_duration(details.get("duration")),
        "captions": details.get("caption") == "true",
        "views": int(counts.get("viewCount", 0) or 0),
        "likes": int(counts.get("likeCount", 0) or 0),
        "published": item.get("snippet", {}).get("publishedAt", "")[:10],
    }


def summarize(facts: list[dict]) -> dict:
    """A playlist's quality numbers from its available videos (ADR-0046)."""
    if not facts:
        return {}
    views = sum(f["views"] for f in facts)
    dates = [f["published"] for f in facts if f["published"]]
    return {
        "available": len(facts),
        "captions": round(sum(f["captions"] for f in facts) / len(facts), 2),
        "median_views": int(statistics.median(f["views"] for f in facts)),
        "like_ratio": round(sum(f["likes"] for f in facts) / views, 4) if views else 0.0,
        "first": min(dates, default=""),
        "latest": max(dates, default=""),
    }


def to_minutes(seconds: int | None) -> int | None:
    return max(1, round(seconds / 60)) if seconds else None


def normalize(item: dict, videos: list[dict] | None = None) -> dict:
    """A playlist as a resource; `videos` (its available videos' facts) adds duration and quality numbers."""
    snippet = item.get("snippet", {})
    facts = videos or []
    return {
        "source": SOURCE,
        "external_id": item["id"],
        "type": "playlist",
        "provider": f"YouTube · {snippet.get('channelTitle', 'unknown channel')}",
        "url": f"https://www.youtube.com/playlist?list={item['id']}",
        "title": snippet.get("title", item["id"]),
        "description": (snippet.get("description") or "")[:1000] or None,
        "language": (snippet.get("defaultLanguage") or "en")[:2],
        "duration_minutes": to_minutes(sum(f["seconds"] or 0 for f in facts)),
        "is_free": True,
        "quality": {"videos": item.get("contentDetails", {}).get("itemCount")} | summarize(facts),
    }


def normalize_video(item: dict) -> dict:
    """A single long video as a resource (ADR-0046)."""
    snippet = item.get("snippet", {})
    facts = video_facts(item)
    return {
        "source": SOURCE,
        "external_id": item["id"],
        "type": "video",
        "provider": f"YouTube · {snippet.get('channelTitle', 'unknown channel')}",
        "url": f"https://www.youtube.com/watch?v={item['id']}",
        "title": snippet.get("title", item["id"]),
        # Chapter titles say what a long video covers; keep enough of them for the tagger to read.
        "description": (snippet.get("description") or "")[:2000] or None,
        "language": (snippet.get("defaultAudioLanguage") or snippet.get("defaultLanguage") or "en")[:2],
        "duration_minutes": to_minutes(facts["seconds"]),
        "is_free": True,
        "quality": {
            "views": facts["views"],
            "like_ratio": round(facts["likes"] / facts["views"], 4) if facts["views"] else 0.0,
            "captions": facts["captions"],
            "published": facts["published"],
            "chapters": len(parse_chapters(snippet.get("description"), facts["seconds"])),
        },
    }


def video_sections(item: dict) -> list[dict]:
    """A long video's chapters, each linking to its start."""
    seconds = video_facts(item)["seconds"]
    chapters = parse_chapters(item.get("snippet", {}).get("description"), seconds)
    out = []
    for i, (start, title) in enumerate(chapters):
        end = chapters[i + 1][0] if i + 1 < len(chapters) else seconds
        out.append(
            {
                "position": i,
                "title": title,
                "url": f"https://www.youtube.com/watch?v={item['id']}&t={start}s",
                "start_seconds": start,
                "duration_seconds": (end - start) if end else None,
            }
        )
    return out


def playlist_sections(playlist_id: str, entries: list[dict], videos: dict[str, dict]) -> list[dict]:
    """A playlist's available videos, in order, each linking to the video within the playlist."""
    available = [e for e in entries if e["video"] in videos]
    return [
        {
            "position": i,
            "title": entry["title"],
            "url": f"https://www.youtube.com/watch?v={entry['video']}&list={playlist_id}&index={i + 1}",
            "start_seconds": None,
            "duration_seconds": video_facts(videos[entry["video"]])["seconds"],
        }
        for i, entry in enumerate(available)
    ]


EPISODE = re.compile(
    r"(?:#|\b(?:part|session|lecture|lesson|day|episode|ep|video|chapter|module|class|tutorial)\s*[-:#.]?\s*)(\d{1,3})\b",
    re.IGNORECASE,
)


def first_episode(titles: list[str]) -> int | None:
    """The position of episode 1 when a playlist lists its numbered episodes newest first ("Session 56" on top,
    seen in the approved list); None when it reads in order or isn't numbered."""
    numbered = [(i, int(m.group(1))) for i, t in enumerate(titles) if (m := EPISODE.search(t))]
    if len(numbered) < max(3, 0.6 * len(titles)):
        return None
    steps = list(pairwise(numbered))
    if sum(b[1] < a[1] for a, b in steps) <= 0.7 * len(steps):
        return None
    return min(numbered, key=lambda x: x[1])[0]


def sections_hash(sections: list[dict]) -> str:
    return hashlib.sha256(json.dumps([(s["title"], s["url"]) for s in sections]).encode()).hexdigest()[:16]


def replace_sections(session: Session, resource_id: int, sections: list[dict], previous: str | None) -> bool:
    """Store the resource's sections. When titles and links are unchanged since the last run, keep the rows and
    their skills (tagging them again would cost embedding calls for nothing) and refresh only the durations.
    Returns True if the sections were replaced."""
    exists = session.scalar(select(ResourceSection.id).where(ResourceSection.resource_id == resource_id).limit(1))
    if previous == sections_hash(sections) and (exists or not sections):
        for s in sections:
            session.execute(
                update(ResourceSection)
                .where(ResourceSection.resource_id == resource_id, ResourceSection.position == s["position"])
                .values(duration_seconds=s["duration_seconds"])
            )
        return False
    session.execute(delete(ResourceSection).where(ResourceSection.resource_id == resource_id))
    if sections:
        session.execute(insert(ResourceSection), [{"resource_id": resource_id} | s for s in sections])
    return True


def without_counts(item: dict) -> dict:
    """The raw record keeps what describes the video, not its daily-changing counts (a version per change)."""
    return {k: v for k, v in item.items() if k != "statistics"}


def ingest(
    session: Session,
    client: httpx.Client,
    api_key: str | None,
    playlists: list[str] | list[tuple[str, str | None]],
    videos: list[tuple[str, str | None]] = (),
) -> dict:
    """Refresh the approved playlists and videos with their sections. A playlist or video another source
    already serves is skipped (curated wins, ADR-0033)."""
    if not api_key:
        raise NoApiKey("set XCRS_YOUTUBE_API_KEY to use the YouTube adapter (ADR-0033)")
    approved = dict(p if isinstance(p, tuple) else (p, None) for p in playlists)
    approved_videos = dict(videos)
    run = start_run(session, SOURCE)
    stats = {"requested": len(approved), "upserted": 0}
    stats |= {"videos_requested": len(approved_videos), "videos_upserted": 0, "sections": 0, "sections_changed": 0}
    previous = dict(
        session.execute(
            select(LearningResource.external_id, LearningResource.quality["sections"].astext).where(
                LearningResource.source == SOURCE
            )
        ).all()
    )

    def save(row: dict, sections: list[dict], skill: str | None) -> bool:
        row["quality"]["sections"] = sections_hash(sections)
        resource_id = upsert_resource(session, row)
        if resource_id is None:
            return False
        stats["sections_changed"] += replace_sections(session, resource_id, sections, previous.get(row["external_id"]))
        stats["sections"] += len(sections)
        if skill:
            tag_reviewed(session, resource_id, skill)
        return True

    try:
        ids = list(approved)
        for i in range(0, len(ids), 50):
            params = {"part": "snippet,contentDetails", "id": ",".join(ids[i : i + 50]), "maxResults": 50}
            for item in api_get(client, API, api_key, params).get("items", []):
                store_raw(session, run, SOURCE, item["id"], item)
                entries = playlist_items(client, api_key, item["id"])
                details = fetch_videos(client, api_key, [e["video"] for e in entries])
                row = normalize(item, [video_facts(v) for v in details.values()])
                sections = playlist_sections(item["id"], entries, details)
                if (start := first_episode([s["title"] for s in sections])) is not None:
                    row["quality"]["start"] = start  # listed newest first: open at episode 1 (ADR-0046)
                stats["upserted"] += save(row, sections, approved[item["id"]])
        for video_id, item in fetch_videos(client, api_key, list(approved_videos)).items():
            store_raw(session, run, SOURCE, video_id, without_counts(item))
            stats["videos_upserted"] += save(normalize_video(item), video_sections(item), approved_videos[video_id])
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
    """Delete YouTube data not refreshed within 30 days (API terms); sections and tags go with it."""
    cutoff = (now or datetime.now(UTC)) - MAX_AGE
    deleted = session.execute(
        delete(LearningResource)
        .where(LearningResource.source == SOURCE, LearningResource.fetched_at < cutoff)
        .returning(LearningResource.id)
    ).all()
    session.execute(delete(RawRecord).where(RawRecord.source == SOURCE, RawRecord.fetched_at < cutoff))
    return len(deleted)
