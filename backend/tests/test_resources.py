"""Learning resources (ADR-0033): curated list validation, the freeCodeCamp and YouTube adapters, tagging and
suggestions. Pure tests run anywhere; DB tests run in a rolled-back transaction."""

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError, ProgrammingError

from xcrs.catalog import model, validate
from xcrs.catalog.model import CuratedResource, Requirement
from xcrs.catalog.snapshot import snapshot_from_catalog
from xcrs.db.models import LearningResource, RawRecord, ResourceSkill
from xcrs.db.session import new_session
from xcrs.domain.role_scoring import Gap, ResourceRef, suggest_resources
from xcrs.ingest.resources import freecodecamp, tagging, youtube
from xcrs.ingest.resources.base import start_run, store_raw
from xcrs.repository import catalog_store

INTRO = {
    "python-v9": {"title": "Python Certification", "intro": ["Learn Python."], "blocks": {"a": {}, "b": {}}},
    "responsive-web-design": {"title": "Legacy Responsive Web Design", "intro": ["Old."], "blocks": {}},
    "a1-professional-spanish": {
        "title": "A1 Professional Spanish Certification (Beta)",
        "intro": ["Hola"],
        "blocks": {},
    },
    "full-stack-open": {"title": "Full-Stack Open", "intro": ["A good intro is to be added here."], "blocks": {}},
    "misc": {"title": "Not a course"},
}


def test_the_curated_list_is_valid_and_covers_every_skill():
    cat = model.load_catalog()
    report = validate.validate(cat)
    assert report.errors == [] and len(cat.resources) >= 200
    assert not any("no curated resource" in w for w in report.warnings)


def test_curated_resources_are_checked(tmp_path):
    cat = model.load_catalog()
    bad = CuratedResource("http://x.org", "", "P", "podcast", "expert", True, (Requirement(("cobol",), 2),))
    cat.resources = [bad, bad]
    errors = validate.validate(cat).errors
    for expected in (
        "URL must be https",
        "listed twice",
        "unknown type 'podcast'",
        "unknown level 'expert'",
        "needs a title",
        "unknown skill 'cobol'",
    ):
        assert any(expected in e for e in errors), expected


def test_freecodecamp_keeps_current_courses_only():
    rows = freecodecamp.courses(INTRO)
    assert [r["external_id"] for r in rows] == ["python-v9"]
    assert rows[0]["url"] == "https://www.freecodecamp.org/learn/python-v9" and rows[0]["quality"] == {"lessons": 2}


def test_youtube_needs_a_key_and_normalizes_playlists():
    with pytest.raises(youtube.NoApiKey):
        youtube.ingest(None, None, None, ["PL1"])
    row = youtube.normalize(
        {"id": "PL1", "snippet": {"title": "K8s", "channelTitle": "TechWorld"}, "contentDetails": {"itemCount": 12}}
    )
    assert row["url"] == "https://www.youtube.com/playlist?list=PL1" and row["provider"] == "YouTube · TechWorld"


def test_suggestions_prefer_curated_free_on_topic_resources_for_the_earliest_gaps():
    snap = snapshot_from_catalog(model.load_catalog())
    snap.teaching = {}
    for r in [
        ResourceRef("cpp", "C++ OOP", "u1", "P", "course", None, True, True, (("oop", 2), ("cpp", 2))),
        ResourceRef("java", "Java OOP", "u2", "P", "course", None, True, True, (("oop", 2), ("java", 2))),
        ResourceRef("paid", "Paid Git", "u3", "P", "course", None, False, True, (("git", 3),)),
        ResourceRef("llm", "LLM Git", "u4", "P", "course", None, True, False, (("git", 2),)),
        ResourceRef("git", "Pro Git", "u5", "P", "book", None, True, True, (("git", 3),)),
    ]:
        for skill, _ in r.teaches:
            snap.teaching.setdefault(skill, []).append(r)
    gaps = [Gap(("oop",), 2, 0, "Basics"), Gap(("git",), 2, 0, "Basics"), Gap(("sql",), 2, 0, "Data")]
    picked = suggest_resources(snap, gaps, relevant={"oop", "git", "java"})
    assert [(r.id, hits) for r, hits in picked] == [("java", ["oop"]), ("git", ["git"])]


@pytest.fixture
def session():
    s = new_session()
    try:
        s.execute(text("SELECT 1 FROM ingest.raw_records LIMIT 1"))
    except (OperationalError, ProgrammingError):
        s.close()
        pytest.skip("needs the database at migration 0008 or later")
    cat = model.load_catalog()
    catalog_store.import_catalog(
        s, cat, git_commit="t", git_dirty=True, checksum="t", stats=validate.validate(cat).stats
    )
    yield s
    s.rollback()
    s.close()


def test_curated_import_loads_resources_and_tags(session):
    curated = session.scalar(
        select(text("count(*)")).select_from(LearningResource).where(LearningResource.source == "curated")
    )
    assert curated == len(model.load_catalog().resources)
    tags = session.scalar(
        select(text("count(*)")).select_from(ResourceSkill).where(ResourceSkill.tagged_by == "curated")
    )
    assert tags > curated


def test_freecodecamp_ingest_keeps_raw_versions_and_skips_curated_urls(session):
    intro = {
        "test-only-course": {"title": "A Test Course", "intro": ["x"], "blocks": {}},
        "learn-rag-mcp-fundamentals": {"title": "Learn RAG and MCP Fundamentals", "intro": ["x"], "blocks": {}},
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=json.dumps(intro)))
    with httpx.Client(transport=transport) as client:
        first = freecodecamp.ingest(session, client)
        second = freecodecamp.ingest(session, client)
    assert first == {"courses": 2, "upserted": 1, "skipped_url_owned": 1, "raw_changed": True}  # RAG is curated
    assert second["raw_changed"] is False  # same content: no new raw version
    raw = session.scalars(
        select(RawRecord).where(RawRecord.source == "freecodecamp", RawRecord.payload["test-only-course"].isnot(None))
    ).all()
    assert len(raw) == 1


def test_tagging_keeps_llm_picks_the_embedding_confirms(session):
    run = start_run(session, "test")
    assert store_raw(session, run, "test", "1", {"a": 1}) is True
    session.add(
        LearningResource(
            source="test",
            external_id="1",
            type="course",
            provider="T",
            url="https://t.example/1",
            title="Docker for beginners",
            is_free=True,
        )
    )
    session.flush()

    class Picker:
        prompt_version, model = "resource-1", "fake"

        def pick(self, text):
            return ["docker", "kubernetes"]

    stats = tagging.tag_untagged(session, Picker(), lambda t: {"docker": 0.7, "kubernetes": 0.2}, source="test")
    assert stats["tags"] == 1
    tags = session.execute(
        text("""SELECT s.slug, rs.tagged_by FROM catalog.resource_skills rs
        JOIN catalog.skills s ON s.id = rs.skill_id JOIN catalog.learning_resources r ON r.id = rs.resource_id
        WHERE r.url = 'https://t.example/1'""")
    ).all()
    assert [tuple(t) for t in tags] == [("docker", "llm")]


def test_youtube_data_older_than_30_days_is_deleted(session):
    old = datetime.now(UTC) - timedelta(days=31)
    session.add(
        LearningResource(
            source="youtube",
            external_id="PL1",
            type="playlist",
            provider="YouTube · x",
            url="https://www.youtube.com/playlist?list=PL1",
            title="t",
            is_free=True,
            fetched_at=old,
        )
    )
    session.flush()
    assert youtube.expire(session) == 1


def test_youtube_discovery_ranks_candidates_and_resumes(tmp_path):
    from xcrs.ingest.resources import youtube_discovery

    def handler(request):
        if request.url.path.endswith("/playlistItems"):
            return httpx.Response(200, json={"items": [{"contentDetails": {"videoId": "v1"}}]})
        if request.url.path.endswith("/videos"):
            stats = {"viewCount": "50000", "likeCount": "1500"}
            return httpx.Response(
                200,
                json={"items": [{"id": "v1", "statistics": stats, "snippet": {"publishedAt": "2025-01-01T00:00:00Z"}}]},
            )
        if request.url.path.endswith("/channels"):
            return httpx.Response(200, json={"items": [{"id": "UC1", "statistics": {"subscriberCount": "100000"}}]})
        if request.url.path.endswith("/search"):
            ids = ["PLgood", "PLnoise"] if "Docker" in request.url.params["q"] else []
            return httpx.Response(200, json={"items": [{"id": {"playlistId": i}} for i in ids]})
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "PLnoise",
                        "snippet": {"title": "Cooking tips", "channelTitle": "C"},
                        "contentDetails": {"itemCount": 40},
                    },
                    {
                        "id": "PLgood",
                        "snippet": {"title": "Docker full course", "channelTitle": "D"},
                        "contentDetails": {"itemCount": 30},
                    },
                ]
            },
        )

    skills = [("docker", "Docker and containers"), ("git", "Git"), ("linux", "Linux")]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        found, used = youtube_discovery.discover(client, "key", skills, done={"git"}, max_searches=1)
    assert used == 1 and list(found) == ["docker"]  # git was done; the budget stopped before linux
    assert [c["id"] for c in found["docker"]] == ["PLgood"]  # "Cooking tips" doesn't name the skill
    assert found["docker"][0]["median_views"] == 50000 and found["docker"][0]["like_ratio"] == 0.03
    path, review = tmp_path / "candidates.yaml", tmp_path / "review.md"
    data = youtube_discovery.load_candidates(path)
    youtube_discovery.save(path, review, data, found, {s: n for s, n in skills})
    saved = youtube_discovery.load_candidates(path)
    assert saved["searched"] == ["docker"] and saved["candidates"][0] == {"skill": "docker", "playlist": "PLgood"}
    assert "Docker full course" not in path.read_text() and "Docker full course" in review.read_text()


def test_discovery_drops_titles_that_dont_name_the_skill_and_prefers_trusted_channels():
    from xcrs.ingest.resources.youtube_discovery import rank

    candidates = [
        {"id": "a", "title": "IELTS Full Course", "channel": "X", "videos": 60},
        {"id": "b", "title": "Docker Tutorial for Beginners", "channel": "Random", "videos": 30},
        {"id": "c", "title": "Docker course", "channel": "TechWorld with Nana", "videos": 12},
    ]
    assert [c["id"] for c in rank("Docker and containers", candidates, {"techworld with nana"})] == ["c", "b"]
    arabic = [{"id": "d", "title": "Docker | دورة", "channel": "Y", "videos": 20}, *candidates]
    assert "d" not in [c["id"] for c in rank("Docker and containers", arabic)]
    hindi = [{"id": "e", "title": "Docker Tutorial in Hindi", "channel": "Z", "videos": 20}]
    assert rank("Docker and containers", hindi) == []
    assert [c["id"] for c in rank("Docker and containers", candidates, blocked={"random"})] == ["c"]


def test_quality_prefers_watched_liked_recent_playlists():
    from xcrs.ingest.resources.youtube_discovery import quality

    popular = {"median_views": 200_000, "like_ratio": 0.03, "subscribers": 1_000_000, "latest": "2025-05-01"}
    stale = {"median_views": 200_000, "like_ratio": 0.03, "subscribers": 1_000_000, "latest": "2012-05-01"}
    obscure = {"median_views": 300, "like_ratio": 0.01, "subscribers": 2_000, "latest": "2025-05-01"}
    assert quality(popular) > quality(stale) > quality(obscure)
