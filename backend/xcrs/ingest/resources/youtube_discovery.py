"""Find candidate YouTube playlists per skill (ADR-0033), for a human to approve before ingestion.

`search.list` costs 100 quota units of the free 10,000 a day (and searches are capped separately), so one
run searches at most `max_searches` skills, the ones most roles rely on first, and later runs continue
where the last stopped (a run that hits the quota keeps what it found). The key is sent in a header, never
in a URL. Candidates are ranked by title match, trusted channels and a quality score.

Storage follows the API terms: only playlist ids and the skill go into catalog/sources/youtube-candidates.yaml
(committed, reviewed); titles and channels go to a local review file that the next run overwrites.
Approved ids move into catalog/sources/youtube.yaml and are ingested by `xcrs resources ingest youtube`.
"""

import logging
import math
import re
import statistics
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

from xcrs.ingest.resources.youtube import QuotaExceeded, YouTubeError, api_get

log = logging.getLogger(__name__)

SEARCH = "https://www.googleapis.com/youtube/v3/search"
PLAYLISTS = "https://www.googleapis.com/youtube/v3/playlists"
ITEMS = "https://www.googleapis.com/youtube/v3/playlistItems"
VIDEOS = "https://www.googleapis.com/youtube/v3/videos"
CHANNELS = "https://www.googleapis.com/youtube/v3/channels"
SAMPLE_VIDEOS = 10  # videos per playlist whose statistics are read (1 quota unit per call)
ENRICH = 4  # candidates per skill that get statistics


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9+#]+", text.lower()) if len(w) > 1}


STOP = {"and", "the", "for", "of", "to", "in", "with", "on"}
MIN_OVERLAP = 0.5  # at least half of the skill's words must be in the playlist title


NON_LATIN = re.compile(r"[^\x00-\u024f\u2000-\u206f\u2100-\u214f\U0001f000-\U0001faff]")


OTHER_LANGUAGES = re.compile(
    r"\b(hindi|urdu|bangla|bengali|tamil|telugu|marathi|arabic|español|espanol|português|portugues|türkçe|turkce"
    r"|indonesia|bahasa|tagalog|vietnamese|russian|deutsch|français|francais|italiano)\b",
    re.IGNORECASE,
)


def english_looking(text: str) -> bool:
    """English only for now (ADR-0026): reject titles or channels in a non-Latin script, or that name
    another language ("... in Hindi")."""
    return not NON_LATIN.search(text) and not OTHER_LANGUAGES.search(text)


def rank(
    skill_name: str, candidates: list[dict], trusted: set[str] = frozenset(), blocked: set[str] = frozenset()
) -> list[dict]:
    """Keep playlists whose title names the skill, in a Latin script, from a channel not blocked; trusted
    channels first, then title match, then a course-like size (5-150 videos). The first test run kept an
    IELTS course for "Mentoring"."""
    wanted = words(re.sub(r"\(.*?\)", "", skill_name)) - STOP

    def overlap(c: dict) -> float:
        return len(wanted & words(c["title"])) / max(1, len(wanted))

    kept = [
        c
        for c in candidates
        if overlap(c) >= MIN_OVERLAP
        and english_looking(f"{c['title']} {c.get('channel', '')}")
        and c.get("channel", "").lower() not in blocked
    ]

    def key(c: dict) -> tuple:
        videos = c.get("videos") or 0
        return (c.get("channel", "").lower() in trusted, overlap(c), 5 <= videos <= 150, videos)

    return sorted(kept, key=key, reverse=True)


def add_quality(client: httpx.Client, api_key: str, candidates: list[dict]) -> None:
    """Views, likes, recency and channel size for each candidate (ADR-0026 decision 7), from its first
    videos. Costs about 2 units per candidate plus 2 per skill, against 100 for the search itself."""
    if not candidates:
        return
    videos_of: dict[str, list[str]] = {}
    for c in candidates:
        params = {"part": "contentDetails", "playlistId": c["id"], "maxResults": SAMPLE_VIDEOS}
        try:  # a playlist can be private or gone by now; a quota error still stops the run
            items = api_get(client, ITEMS, api_key, params).get("items", [])
        except YouTubeError:
            items = []
        videos_of[c["id"]] = [i["contentDetails"]["videoId"] for i in items]
    all_ids = [v for ids in videos_of.values() for v in ids][:50]
    stats: dict[str, dict] = {}
    if all_ids:
        r = api_get(client, VIDEOS, api_key, {"part": "statistics,snippet", "id": ",".join(all_ids)})
        for v in r.get("items", []):
            st = v.get("statistics", {})
            stats[v["id"]] = {
                "views": int(st.get("viewCount", 0)),
                "likes": int(st.get("likeCount", 0)),
                "published": v.get("snippet", {}).get("publishedAt", "")[:10],
            }
    channel_ids = sorted({c["channel_id"] for c in candidates if c.get("channel_id")})
    subscribers: dict[str, int] = {}
    if channel_ids:
        r = api_get(client, CHANNELS, api_key, {"part": "statistics", "id": ",".join(channel_ids)})
        for ch in r.get("items", []):
            subscribers[ch["id"]] = int(ch.get("statistics", {}).get("subscriberCount", 0) or 0)
    for c in candidates:
        sample = [stats[v] for v in videos_of.get(c["id"], []) if v in stats]
        views = sum(x["views"] for x in sample)
        c["median_views"] = int(statistics.median(x["views"] for x in sample)) if sample else 0
        c["like_ratio"] = round(sum(x["likes"] for x in sample) / views, 4) if views else 0.0
        c["latest"] = max((x["published"] for x in sample), default="")
        c["subscribers"] = subscribers.get(c.get("channel_id", ""), 0)
        c["quality"] = round(quality(c), 2)


def quality(c: dict) -> float:
    """A simple, explainable score: typical views (log), how liked (likes per view), channel size (log), and
    recency (videos from the last 4 years). Popularity is a weak proxy for teaching quality, so trusted
    channels still rank first and a human approves every playlist."""
    score = math.log10(1 + c.get("median_views", 0))
    score += min(c.get("like_ratio", 0.0), 0.08) * 25  # 2% likes per view -> +0.5, capped at 8%
    score += 0.5 * math.log10(1 + c.get("subscribers", 0))
    latest = c.get("latest", "")
    if latest and int(latest[:4]) >= datetime.now(UTC).year - 4:
        score += 1.0
    return score


def discover(
    client: httpx.Client,
    api_key: str,
    skills: list[tuple[str, str]],
    done: set[str],
    max_searches: int = 90,
    per_skill: int = 2,
    trusted: set[str] = frozenset(),
    blocked: set[str] = frozenset(),
) -> tuple[dict[str, list[dict]], int]:
    """Search playlists for skills not yet searched. Returns ({skill: ranked candidates}, searches used)."""
    found: dict[str, list[dict]] = {}
    used = 0
    for skill_id, name in skills:
        if skill_id in done:
            continue
        if used >= max_searches:
            break
        topic = re.sub(r"\(.*?\)", "", name).strip()  # "BI tools (Power BI, ...)" -> "BI tools"
        query = f"{topic} tutorial"
        params = {
            "part": "snippet",
            "q": query,
            "type": "playlist",
            "maxResults": 8,
            "relevanceLanguage": "en",
            "safeSearch": "strict",
        }
        try:
            result = api_get(client, SEARCH, api_key, params)
            used += 1
            ids = [item["id"]["playlistId"] for item in result.get("items", [])]
            details = []
            if ids:
                meta = api_get(client, PLAYLISTS, api_key, {"part": "snippet,contentDetails", "id": ",".join(ids)})
                details = [
                    {
                        "id": item["id"],
                        "title": item["snippet"]["title"],
                        "channel": item["snippet"].get("channelTitle", ""),
                        "videos": item.get("contentDetails", {}).get("itemCount"),
                        "channel_id": item["snippet"].get("channelId", ""),
                    }
                    for item in meta.get("items", [])
                ]
            kept = rank(name, details, trusted, blocked)[:ENRICH]
            add_quality(client, api_key, kept)
        except QuotaExceeded as exc:
            log.warning("stopping: %s; what was found so far is kept", exc)
            break
        kept.sort(key=lambda c: (c.get("channel", "").lower() in trusted, c.get("quality", 0)), reverse=True)
        found[skill_id] = kept[:per_skill]
    return found, used


def load_candidates(path: Path) -> dict:
    data = yaml.safe_load(path.read_text()) if path.exists() else None
    return data or {"searched": [], "candidates": []}


def save(path: Path, review: Path, data: dict, found: dict[str, list[dict]], names: dict[str, str]) -> None:
    data["searched"] = sorted(set(data["searched"]) | set(found))
    for skill_id, candidates in found.items():
        for c in candidates:
            data["candidates"].append({"skill": skill_id, "playlist": c["id"]})
    header = (
        "# YouTube playlist candidates per skill (ADR-0033), found by `xcrs resources youtube-discover`.\n"
        "# Only ids are kept here (API terms); titles are in the local review file. Approve a candidate by\n"
        "# moving its playlist id to youtube.yaml, then run `xcrs resources ingest youtube` and `xcrs resources tag`.\n"
    )
    path.write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    lines = [
        "# YouTube candidates to review (local; refreshed by each discovery run)",
        "",
        "| Skill | Playlist | Channel | Videos | Median views | Likes/view | Latest | Subscribers | Score | Link |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for skill_id, candidates in found.items():
        for c in candidates:
            lines.append(
                f"| {names[skill_id]} | {c['title']} | {c['channel']} | {c['videos']} | "
                f"{c.get('median_views', 0):,} | {c.get('like_ratio', 0):.1%} | {c.get('latest', '')[:4]} | "
                f"{c.get('subscribers', 0):,} | {c.get('quality', 0)} | "
                f"https://www.youtube.com/playlist?list={c['id']} |"
            )
    review.write_text("\n".join(lines) + "\n")
