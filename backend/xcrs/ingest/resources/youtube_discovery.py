"""Find candidate YouTube playlists per skill (ADR-0033), for a human to approve before ingestion.

`search.list` costs 100 quota units of the free 10,000 a day (and searches are capped separately), so one
run searches at most `max_searches` skills, the ones most roles rely on first, and later runs continue
where the last stopped. Candidates are ranked by how well their title names the skill and by playlist size.

Storage follows the API terms: only playlist ids and the skill go into catalog/sources/youtube-candidates.yaml
(committed, reviewed); titles and channels go to a local review file that the next run overwrites.
Approved ids move into catalog/sources/youtube.yaml and are ingested by `xcrs resources ingest youtube`.
"""

import re
from pathlib import Path

import httpx
import yaml

SEARCH = "https://www.googleapis.com/youtube/v3/search"
PLAYLISTS = "https://www.googleapis.com/youtube/v3/playlists"


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9+#]+", text.lower()) if len(w) > 1}


def rank(skill_name: str, candidates: list[dict]) -> list[dict]:
    """Title naming the skill first, then playlists of a course-like size (5-150 videos)."""
    wanted = words(re.sub(r"\(.*?\)", "", skill_name))

    def key(c: dict) -> tuple:
        overlap = len(wanted & words(c["title"])) / max(1, len(wanted))
        videos = c.get("videos") or 0
        return (overlap, 5 <= videos <= 150, videos)

    return sorted(candidates, key=key, reverse=True)


def discover(
    client: httpx.Client,
    api_key: str,
    skills: list[tuple[str, str]],
    done: set[str],
    max_searches: int = 90,
    per_skill: int = 2,
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
        query = f"{topic} full course"
        response = client.get(
            SEARCH,
            params={
                "part": "snippet",
                "q": query,
                "type": "playlist",
                "maxResults": 8,
                "relevanceLanguage": "en",
                "safeSearch": "strict",
                "key": api_key,
            },
        )
        used += 1
        response.raise_for_status()
        ids = [item["id"]["playlistId"] for item in response.json().get("items", [])]
        details = []
        if ids:
            meta = client.get(PLAYLISTS, params={"part": "snippet,contentDetails", "id": ",".join(ids), "key": api_key})
            meta.raise_for_status()
            for item in meta.json().get("items", []):
                details.append(
                    {
                        "id": item["id"],
                        "title": item["snippet"]["title"],
                        "channel": item["snippet"].get("channelTitle", ""),
                        "videos": item.get("contentDetails", {}).get("itemCount"),
                    }
                )
        found[skill_id] = rank(name, details)[:per_skill]
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
        "| Skill | Playlist | Channel | Videos | Link |",
        "|---|---|---|---|---|",
    ]
    for skill_id, candidates in found.items():
        for c in candidates:
            lines.append(
                f"| {names[skill_id]} | {c['title']} | {c['channel']} | {c['videos']} | "
                f"https://www.youtube.com/playlist?list={c['id']} |"
            )
    review.write_text("\n".join(lines) + "\n")
