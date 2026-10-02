"""freeCodeCamp's open curriculum (BSD-3-Clause) as learning resources (ADR-0033).

One file, `intro.json`, lists every course ("superblock") with a title and an introduction. Current,
English programming courses become free `course` resources; legacy, beta, placeholder and spoken-language
courses are skipped. Skills are tagged afterwards by `tagging.tag_untagged` (ADR-0030's pipeline).
"""

import httpx
from sqlalchemy.orm import Session

from xcrs.ingest.resources.base import finish_run, start_run, store_raw, upsert_resource

SOURCE = "freecodecamp"
INTRO_URL = "https://raw.githubusercontent.com/freeCodeCamp/freeCodeCamp/main/client/i18n/locales/english/intro.json"
COURSE_URL = "https://www.freecodecamp.org/learn/{key}"
SKIP_WORDS = ("legacy", "beta", "english for developers", "spanish", "chinese", "playground")


def courses(intro: dict) -> list[dict]:
    """The superblocks worth offering, as normalized resource rows."""
    out = []
    for key, block in intro.items():
        if not isinstance(block, dict) or not {"title", "intro", "blocks"} <= block.keys():
            continue
        title = str(block["title"]).strip()
        text = " ".join(str(p) for p in block.get("intro") or []).strip()
        if (
            any(w in title.lower() for w in SKIP_WORDS)
            or "placeholder" in text.lower()
            or "to be added" in text.lower()
        ):
            continue
        out.append(
            {
                "source": SOURCE,
                "external_id": key,
                "type": "course",
                "provider": "freeCodeCamp",
                "url": COURSE_URL.format(key=key),
                "title": title,
                "description": text[:1000] or None,
                "is_free": True,
                "quality": {"lessons": len(block.get("blocks") or {})},
            }
        )
    return out


def ingest(session: Session, client: httpx.Client) -> dict:
    """Fetch, keep the raw file, upsert the courses. The caller commits."""
    run = start_run(session, SOURCE)
    try:
        response = client.get(INTRO_URL)
        response.raise_for_status()
        intro = response.json()
        new_raw = store_raw(session, run, SOURCE, "intro.json", intro)
        rows = courses(intro)
        stored = [upsert_resource(session, row) for row in rows]
        stats = {
            "courses": len(rows),
            "upserted": sum(1 for s in stored if s),
            "skipped_url_owned": stored.count(None),
            "raw_changed": new_raw,
        }
        finish_run(session, run, stats)
        return stats
    except (httpx.HTTPError, ValueError) as exc:
        finish_run(session, run, {}, error=str(exc))
        raise
