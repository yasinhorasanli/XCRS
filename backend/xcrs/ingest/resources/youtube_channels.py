"""Discovery from trusted channels (ADR-0046): their new uploads and their playlists, matched to catalog skills
by title, for a human to approve like search results.

A search costs 100 quota units; reading a channel's uploads costs 1 per 50 videos (its uploads playlist, newest
first, read only back to the last scan), so watching channels scales with the catalog where searching doesn't.
The channels are `watch_channels` in catalog/sources/youtube.yaml (ids; reviewed like the rest).

Titles are matched to skills by name, not by the LLM: a skill's name (or the head of "X and Y"), never a generic
word ("Flow", "Performance"), and the one- and two-letter languages (Go, C, R) only when the title says
"programming", "language", "tutorial" or "course" right after them.
"""

import re
from collections import Counter

import httpx

from xcrs.domain.skill_matching import normalize
from xcrs.ingest.resources.youtube import api_get, fetch_videos
from xcrs.ingest.resources.youtube_discovery import (
    ENRICH,
    ITEMS,
    PLAYLISTS,
    VIDEOS_PER_SKILL,
    add_quality,
    add_subscribers,
    english_looking,
    keep_video,
    rank,
    video_candidate,
    video_order,
)

UPLOAD_PAGES = 20  # the first scan of a channel reads its last 1,000 uploads (20 units); later scans, new ones
PLAYLIST_PAGES = 4
PLAYLISTS_PER_SKILL = 2
MIN_PLAYLIST_VIDEOS = 5
# Names that are also everyday words in titles; these skills are found by search only.
GENERIC = {"flow", "manual", "identity", "performance", "hiring", "dom", "dx", "presales", "profiling", "caching"}
EXTRA = {"golang": "go", "k8s": "kubernetes", "postgres": "postgresql", "reactjs": "react", "vuejs": "vue"}
CONTEXT = {"programming", "language", "lang", "tutorial", "course", "crash", "for"}
PARENS = re.compile(r"\(([^)]*)\)")


class TitleMatcher:
    """Catalog skills named in a video or playlist title."""

    def __init__(self, skills: list[tuple[str, str]]):
        self.keys: dict[str, set[str]] = {}
        ids = {sid for sid, _ in skills}
        for sid, name in skills:
            outside = PARENS.sub(" ", name)
            head = re.split(r"\band\b|,", outside)[0]
            proper = [p for p in ",".join(PARENS.findall(name)).split(",") if p.strip()[:1].isupper()]
            for text in {outside, head, *proper}:
                key = normalize(text)
                if key and key not in GENERIC:
                    self.keys.setdefault(key, set()).add(sid)
        for key, sid in EXTRA.items():
            if sid in ids:
                self.keys.setdefault(key, set()).add(sid)

    def match(self, title: str) -> set[str]:
        words = normalize(title).split()
        found: set[str] = set()
        for n in (4, 3, 2, 1):
            for i in range(len(words) - n + 1):
                gram = " ".join(words[i : i + n])
                if gram not in self.keys:
                    continue
                if len(gram) <= 2 and (i + n >= len(words) or words[i + n] not in CONTEXT):
                    continue  # "Let's Go", "Part C": not the language
                found |= self.keys[gram]
        return found


def uploads(client: httpx.Client, api_key: str, channel_id: str, since: str | None) -> list[dict]:
    """The channel's uploads newer than `since` (a date), newest first, as {video, title, published}."""
    out: list[dict] = []
    token = None
    for _ in range(UPLOAD_PAGES):
        params = {"part": "snippet,contentDetails", "playlistId": "UU" + channel_id[2:], "maxResults": 50}
        if token:
            params["pageToken"] = token
        page = api_get(client, ITEMS, api_key, params)
        for item in page.get("items", []):
            published = item.get("contentDetails", {}).get("videoPublishedAt") or item["snippet"].get("publishedAt", "")
            if since and published[:10] < since:
                return out
            out.append(
                {"video": item["contentDetails"]["videoId"], "title": item["snippet"]["title"], "published": published}
            )
        token = page.get("nextPageToken")
        if not token:
            break
    return out


def playlists(client: httpx.Client, api_key: str, channel_id: str) -> list[dict]:
    """The channel's playlists, as discovery candidates."""
    out: list[dict] = []
    token = None
    for _ in range(PLAYLIST_PAGES):
        params = {"part": "snippet,contentDetails", "channelId": channel_id, "maxResults": 50}
        if token:
            params["pageToken"] = token
        page = api_get(client, PLAYLISTS, api_key, params)
        out += [
            {
                "id": item["id"],
                "title": item["snippet"]["title"],
                "channel": item["snippet"].get("channelTitle", ""),
                "channel_id": channel_id,
                "videos": item.get("contentDetails", {}).get("itemCount") or 0,
            }
            for item in page.get("items", [])
        ]
        token = page.get("nextPageToken")
        if not token:
            break
    return out


def scan(
    client: httpx.Client,
    api_key: str,
    channels: list[dict],
    state: dict[str, str],
    skills: list[tuple[str, str]],
    skip: set[str],
    have_playlist: set[str],
    video_count: Counter,
    trusted: set[str] = frozenset(),
    blocked: set[str] = frozenset(),
) -> dict[str, dict[str, list[dict]]]:
    """New candidates from the watched channels: {"playlists": {skill: […]}, "videos": {skill: […]}}.
    Playlists only for skills with no approved or proposed playlist; videos while a skill has fewer than
    VIDEOS_PER_SKILL approved or proposed. Updates `state` (channel id -> newest upload date seen)."""
    matcher = TitleMatcher(skills)
    names = dict(skills)
    video_pool: dict[str, list[dict]] = {}
    playlist_pool: dict[str, list[dict]] = {}
    for channel in channels:
        found = uploads(client, api_key, channel["id"], state.get(channel["id"]))
        if found:
            state[channel["id"]] = max(f["published"][:10] for f in found)
        matched = {f["video"]: matcher.match(f["title"]) for f in found if f["video"] not in skip}
        wanted = [v for v, s in matched.items() if any(video_count[k] < VIDEOS_PER_SKILL for k in s)]
        for item in fetch_videos(client, api_key, wanted).values():
            candidate = video_candidate(item)
            if keep_video(candidate, blocked):
                for skill in matched[item["id"]]:
                    video_pool.setdefault(skill, []).append(candidate)
        for p in playlists(client, api_key, channel["id"]):
            if p["id"] in skip or p["videos"] < MIN_PLAYLIST_VIDEOS or not english_looking(p["title"]):
                continue
            for skill in matcher.match(p["title"]) - have_playlist:
                playlist_pool.setdefault(skill, []).append(p)
    out: dict[str, dict[str, list[dict]]] = {"playlists": {}, "videos": {}}
    for skill, candidates in video_pool.items():
        room = VIDEOS_PER_SKILL - video_count[skill]
        if room > 0:
            unique = list({c["id"]: c for c in candidates}.values())
            add_subscribers(client, api_key, unique)
            out["videos"][skill] = sorted(unique, key=lambda c: video_order(c, trusted), reverse=True)[:room]
    for skill, candidates in playlist_pool.items():
        kept = rank(names[skill], list({c["id"]: c for c in candidates}.values()), trusted, blocked)[:ENRICH]
        add_quality(client, api_key, kept)
        kept.sort(key=lambda c: c.get("quality", 0), reverse=True)
        if kept:
            out["playlists"][skill] = kept[:PLAYLISTS_PER_SKILL]
    return out
