"""Learning resources (ADR-0033): curated list validation, the freeCodeCamp and YouTube adapters, tagging and
suggestions. Pure tests run anywhere; DB tests run in a rolled-back transaction."""

import json
from dataclasses import replace
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
from xcrs.domain.role_scoring import Gap, ResourceRef, SectionRef, section_for, suggest_resources
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


def test_a_choice_gap_follows_the_option_the_learner_already_has():
    snap = snapshot_from_catalog(model.load_catalog())
    snap.teaching = {}
    for r in [
        ResourceRef("go", "Go playlist", "u1", "YouTube · X", "playlist", None, True, False, (("go", 2),)),
        ResourceRef("py", "Python playlist", "u2", "YouTube · Y", "playlist", None, True, False, (("python", 2),)),
    ]:
        for skill, _ in r.teaches:
            snap.teaching.setdefault(skill, []).append(r)
    gaps = [Gap(("go", "python"), 2, 1, "Foundations")]
    assert [r.id for r, _ in suggest_resources(snap, gaps, known={"python"})] == ["py"]
    assert [r.id for r, _ in suggest_resources(snap, gaps, known={"go"})] == ["go"]
    assert len(suggest_resources(snap, gaps, known=set())) == 1  # no preference: either


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

    video = ResourceRef("yt", "OOP playlist", "u6", "YouTube · X", "playlist", None, True, False, (("oop", 2),))
    snap.teaching["oop"].append(video)
    picked = suggest_resources(snap, gaps, relevant={"oop", "git", "java"})
    assert [(r.id, hits) for r, hits in picked] == [("java", ["oop"]), ("git", ["git"]), ("yt", ["oop"])]
    assert [r.id for r, _ in suggest_resources(snap, gaps, relevant={"oop", "git", "java"}, limit=2)] == ["java", "yt"]

    # The learner's language comes before "curated": a Python learner gets Python OOP, not Java OOP.
    snap.languages = frozenset({"java", "python", "cpp"})
    py = ResourceRef("py", "OOP in Python", "u7", "P", "course", None, True, False, (("oop", 2), ("python", 2)))
    snap.teaching["oop"].append(py)
    picked = suggest_resources(snap, gaps, relevant={"oop", "git", "java", "python"}, known={"python"})
    assert picked[0][0].id == "py"


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


def test_approved_playlists_name_catalog_skills():
    skills = set(model.load_catalog().skills)
    approved = youtube.configured_playlists()
    assert approved and all(skill in skills for _, skill in approved)
    assert len({pid for pid, _ in approved}) == len(approved)


def test_youtube_ingest_tags_the_approved_skill_and_the_llm_adds_more(session):
    def handler(request):
        assert "key" not in request.url.params
        item = {"id": "PLx", "snippet": {"title": "LangChain Tutorials", "channelTitle": "C"}, "contentDetails": {}}
        return httpx.Response(200, json={"items": [item]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        youtube.ingest(session, client, "key", [("PLx", "llm-frameworks")])

    class Picker:
        prompt_version, model = "resource-1", "fake"

        def pick(self, text):
            return ["llm-frameworks", "rag"]  # the approved skill again, and one more

    tagging.tag_untagged(session, Picker(), lambda t: {"llm-frameworks": 0.7, "rag": 0.6}, source="youtube")
    tags = session.execute(
        text("""SELECT s.slug, rs.tagged_by, rs.level FROM catalog.resource_skills rs
        JOIN catalog.skills s ON s.id = rs.skill_id JOIN catalog.learning_resources r ON r.id = rs.resource_id
        WHERE r.external_id = 'PLx' ORDER BY 1""")
    ).all()
    assert [tuple(t) for t in tags] == [("llm-frameworks", "reviewed", 2), ("rag", "llm", None)]
    assert tagging.untagged(session, source="youtube") == []  # processed once, not again every day


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
        assert "key" not in request.url.params and request.headers["X-Goog-Api-Key"] == "key"  # never in a URL
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


def test_discovery_stops_at_the_quota_and_keeps_what_it_found():
    from xcrs.ingest.resources import youtube_discovery

    calls = []

    def handler(request):
        if request.url.path.endswith("/search"):
            calls.append(request.url.params["q"])
            if len(calls) > 1:
                return httpx.Response(403, json={"error": {"errors": [{"reason": "quotaExceeded"}]}})
            return httpx.Response(200, json={"items": []})
        return httpx.Response(200, json={"items": []})

    skills = [("docker", "Docker"), ("git", "Git"), ("linux", "Linux")]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        found, used = youtube_discovery.discover(client, "key", skills, done=set(), max_searches=10)
    assert used == 1 and list(found) == ["docker"] and len(calls) == 2


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


# --- Long videos, sections and channel discovery (ADR-0046) ---

CHAPTERS = """Learn Docker and Kubernetes.
Course contents:
0:00 Introduction
(12:30) - Docker basics
1:05:10 | Kubernetes pods
see 1:05:15 in the notes
2:01:00 Wrap-up https://example.com
"""


def video_item(vid="v1", description=CHAPTERS, duration="PT2H10M", title="Docker and Kubernetes Course", **extra):
    return {
        "id": vid,
        "snippet": {
            "title": title,
            "channelTitle": "Nana",
            "channelId": "UC1",
            "description": description,
            "publishedAt": "2025-03-01T00:00:00Z",
            "defaultAudioLanguage": "en",
        },
        "contentDetails": {"duration": duration, "caption": "false"},
        "statistics": {"viewCount": "100000", "likeCount": "3000"},
    } | extra


def test_durations_and_chapters_follow_youtubes_rules():
    assert youtube.parse_duration("PT1H2M3S") == 3723 and youtube.parse_duration("P1DT1H") == 90000
    assert youtube.parse_duration("PT45S") == 45 and youtube.parse_duration("") is None
    assert youtube.parse_chapters(CHAPTERS, 7800) == [
        (0, "Introduction"),
        (750, "Docker basics"),
        (3910, "Kubernetes pods"),
        (7260, "Wrap-up https://example.com"),
    ]
    assert youtube.parse_chapters("1:00 A\n2:00 B\n3:00 C") == []  # must start at 0:00
    assert youtube.parse_chapters("0:00 A\n2:00 B") == []  # at least three
    assert youtube.parse_chapters("0:00 A\n0:05 B\n2:00 C\n3:00 D") == [(0, "A"), (120, "C"), (180, "D")]  # 10 s apart
    assert youtube.parse_chapters(CHAPTERS, 3000) == []  # timestamps past the end: only two remain


def test_a_long_video_becomes_a_resource_with_chapter_sections():
    row = youtube.normalize_video(video_item())
    assert row["type"] == "video" and row["url"] == "https://www.youtube.com/watch?v=v1"
    assert row["duration_minutes"] == 130 and row["quality"]["chapters"] == 4 and row["quality"]["like_ratio"] == 0.03
    sections = youtube.video_sections(video_item())
    assert [s["url"] for s in sections[:2]] == [
        "https://www.youtube.com/watch?v=v1&t=0s",
        "https://www.youtube.com/watch?v=v1&t=750s",
    ]
    assert sections[1]["duration_seconds"] == 3160 and sections[-1]["duration_seconds"] == 540
    assert "statistics" not in youtube.without_counts(video_item())  # raw versions don't change with the counts


def test_a_playlist_listed_newest_first_starts_at_episode_one():
    titles = [f"Session {n}: Selenium" for n in range(56, 0, -1)]
    assert youtube.first_episode(titles) == 55
    assert youtube.first_episode([f"Part {n}" for n in range(1, 9)]) is None  # in order
    assert youtube.first_episode(["Intro", "Docker volumes", "Compose"]) is None  # not numbered


def test_approved_videos_name_catalog_skills():
    skills = set(model.load_catalog().skills)
    approved = youtube.configured_videos()
    assert all(skill in skills for _, skill in approved)
    assert len({vid for vid, _ in approved}) == len(approved)


def youtube_api(playlist_items: list[dict], videos: dict[str, dict]):
    """A fake YouTube Data API: one playlist PLx with `playlist_items`, and `videos` by id."""

    def handler(request):
        assert "key" not in request.url.params and request.headers["X-Goog-Api-Key"] == "key"
        path = request.url.path
        if path.endswith("/playlists"):
            item = {"id": "PLx", "snippet": {"title": "DevOps Course", "channelTitle": "C"}, "contentDetails": {}}
            return httpx.Response(200, json={"items": [item]})
        if path.endswith("/playlistItems"):
            return httpx.Response(200, json={"items": playlist_items})
        ids = request.url.params["id"].split(",")
        return httpx.Response(200, json={"items": [videos[i] for i in ids if i in videos]})

    return httpx.Client(transport=httpx.MockTransport(handler))


def entry(vid, title):
    return {"snippet": {"title": title}, "contentDetails": {"videoId": vid}}


def test_youtube_ingest_stores_durations_and_sections_and_keeps_tags_while_unchanged(session):
    items = [entry("a", "Docker volumes"), entry("gone", "Private video"), entry("b", "Kubernetes pods")]
    videos = {"a": video_item("a", "", "PT10M"), "b": video_item("b", "", "PT20M"), "v1": video_item()}
    with youtube_api(items, videos) as client:
        stats = youtube.ingest(session, client, "key", [("PLx", "docker")], [("v1", "kubernetes")])
    assert stats["upserted"] == 1 and stats["videos_upserted"] == 1 and stats["sections"] == 2 + 4
    playlist = session.scalar(select(LearningResource).where(LearningResource.external_id == "PLx"))
    assert playlist.duration_minutes == 30 and playlist.quality["available"] == 2
    rows = session.execute(
        text("SELECT position, title, url FROM catalog.resource_sections WHERE resource_id = :r ORDER BY 1"),
        {"r": playlist.id},
    ).all()
    assert [tuple(r) for r in rows] == [
        (0, "Docker volumes", "https://www.youtube.com/watch?v=a&list=PLx&index=1"),
        (1, "Kubernetes pods", "https://www.youtube.com/watch?v=b&list=PLx&index=2"),
    ]
    video = session.scalar(select(LearningResource).where(LearningResource.external_id == "v1"))
    assert video.type == "video" and video.duration_minutes == 130
    tags = session.execute(
        text("""SELECT s.slug, rs.tagged_by FROM catalog.resource_skills rs JOIN catalog.skills s ON s.id = rs.skill_id
        WHERE rs.resource_id = :r"""),
        {"r": video.id},
    ).all()
    assert [tuple(t) for t in tags] == [("kubernetes", "reviewed")]

    session.execute(
        text("UPDATE catalog.resource_sections SET tagged_at = now() WHERE resource_id = :r"), {"r": playlist.id}
    )
    with youtube_api(items, videos) as client:
        again = youtube.ingest(session, client, "key", [("PLx", "docker")], [("v1", "kubernetes")])
    assert again["sections_changed"] == 0  # same titles: rows (and their tags) kept
    assert (
        session.scalar(
            text("SELECT count(*) FROM catalog.resource_sections WHERE resource_id = :r AND tagged_at IS NOT NULL"),
            {"r": playlist.id},
        )
        == 2
    )
    with youtube_api([*items, entry("c", "Helm charts")], videos | {"c": video_item("c", "", "PT5M")}) as client:
        changed = youtube.ingest(session, client, "key", [("PLx", "docker")])
    assert changed["sections_changed"] == 1 and changed["sections"] == 3


def test_sections_take_only_their_resources_skills_the_title_is_about():
    sims = {"docker": 0.62, "kubernetes": 0.6, "terraform": 0.9, "git": 0.3}
    assert tagging.section_skills(sims, {"docker", "kubernetes", "git"}) == ["docker", "kubernetes"]
    assert tagging.section_skills({"docker": 0.62, "kubernetes": 0.5}, {"docker", "kubernetes"}) == ["docker"]
    assert tagging.section_skills({"docker": 0.4}, {"docker"}) == []  # below the floor
    assert tagging.section_skills(sims, {"git"}) == []  # terraform isn't the resource's
    assert tagging.distinct_titles(
        ["Full React Tutorial #16 - Using JSON Server", "Full React Tutorial #29 - Forms"]
    ) == [
        "Using JSON Server",
        "Forms",
    ]
    titles = ["Course structure", "Introduction", "Setup", "Data structures", "Intro to Docker"]
    assert [t for t in titles if tagging.GENERIC_SECTION.match(t)] == ["Course structure", "Introduction", "Setup"]


def test_section_tagging_and_the_snapshot_link_a_gap_to_its_part(session):
    items = [entry(v, t) for v, t in [("a", "Git basics"), ("b", "Docker volumes"), ("c", "Git branching")]]
    items += [entry("d", "Git rebase")]
    videos = {v: video_item(v, "", "PT10M") for v in "abcd"}
    with youtube_api(items, videos) as client:
        youtube.ingest(session, client, "key", [("PLx", "git")])
    resource_id = session.scalar(select(LearningResource.id).where(LearningResource.external_id == "PLx"))
    session.execute(
        text("""INSERT INTO catalog.resource_skills (resource_id, skill_id, relation, confidence, tagged_by)
        SELECT :r, id, 'teaches', 0.7, 'llm' FROM catalog.skills WHERE slug = 'docker'"""),
        {"r": resource_id},
    )

    def similarities(texts):
        return [{"git": 0.7, "docker": 0.3} if "Git" in t else {"docker": 0.7, "git": 0.3} for t in texts]

    stats = tagging.tag_sections(session, similarities)
    assert stats == {"sections": 4, "section_tags": 4}
    assert tagging.tag_sections(session, similarities) == {"sections": 0, "section_tags": 0}  # once
    ref = next(r for r in catalog_store.load_snapshot(session).resources if r.id == str(resource_id))
    assert ref.section_count == 4 and ref.duration_minutes == 40
    assert section_for(ref, ["docker"]).url.endswith("v=b&list=PLx&index=2")  # 1 of 4 sections: worth linking
    assert section_for(ref, ["git"]) is None  # most of the playlist is Git: open it at the start


def test_section_for_links_a_side_topic_but_opens_the_main_subject_at_its_start():
    docker = SectionRef("Docker", "u#docker", 600, frozenset({"docker"}), position=3)
    basics = SectionRef("Variables", "u#vars", 900, frozenset({"programming-fundamentals"}), position=4)
    start = SectionRef("Session 1", "u#1", None)
    teaches = (("docker", 2), ("git", 2), ("programming-fundamentals", 1))
    ref = ResourceRef("1", "DevOps", "u", "YouTube", "video", None, True, False, teaches)
    ref = replace(ref, sections=(docker, basics), section_count=6, start=start, main=frozenset({"git"}))
    assert section_for(ref, ["docker"]) == docker  # a side topic: open its part
    assert section_for(ref, ["git", "docker"]) == start  # the main subject is a gap: from episode 1
    assert section_for(replace(ref, section_count=1), ["docker"]) == start  # it is mostly about Docker
    prerequisites = {"git": [(("programming-fundamentals",), 1)]}
    assert section_for(ref, ["programming-fundamentals"]) == basics
    assert section_for(ref, ["programming-fundamentals"], prerequisites) == start  # a basic of its subject
    first = replace(docker, position=0)
    assert section_for(replace(ref, sections=(first,)), ["docker"]) == start  # the first part is the start


def test_titles_name_skills_but_not_everyday_words_or_stray_letters():
    from xcrs.ingest.resources.youtube_channels import TitleMatcher

    matcher = TitleMatcher([(s.id, s.name) for s in model.load_catalog().skills.values()])
    assert {"docker", "kubernetes"} <= matcher.match("Docker and Kubernetes Full Course for Beginners")
    assert "go" in matcher.match("Go Programming – Golang Course with Bonus Projects")
    assert "go" in matcher.match("Learn Golang in 3 hours")
    assert "go" not in matcher.match("Let's go! Building a game") and "r" not in matcher.match("Part R of the series")
    assert matcher.match("Flow state and performance tips") == set()


def test_only_course_like_long_english_videos_are_proposed():
    from xcrs.ingest.resources.youtube_discovery import course_like, keep_video, video_candidate

    assert course_like("Docker Tutorial for Beginners [FULL COURSE in 3 Hours]")
    assert course_like("CS50P - Lecture 0 - Functions") is False  # no course word
    assert not course_like("CS50P Full Course - Lecture 8 - Object-Oriented Programming")  # one episode
    assert not course_like("How Fast is MySQL on HTTP/3?")  # a talk
    good = video_candidate(video_item())
    assert keep_video(good) and good["chapters"] == 4
    assert not keep_video(video_candidate(video_item(duration="PT19M")))  # a clip
    hindi = video_item()
    hindi["snippet"] = hindi["snippet"] | {"defaultAudioLanguage": "hi"}
    assert not keep_video(video_candidate(hindi))


def test_video_search_and_channel_scan_propose_candidates(tmp_path):
    from collections import Counter

    from xcrs.ingest.resources import youtube_channels, youtube_discovery

    uploads = [
        {
            "snippet": {"title": "Kubernetes Full Course", "publishedAt": "2026-10-01T00:00:00Z"},
            "contentDetails": {"videoId": "k1", "videoPublishedAt": "2026-10-01T00:00:00Z"},
        },
        {
            "snippet": {"title": "Old Git course", "publishedAt": "2026-01-01T00:00:00Z"},
            "contentDetails": {"videoId": "g1", "videoPublishedAt": "2026-01-01T00:00:00Z"},
        },
    ]
    videos = {
        "k1": video_item("k1", title="Kubernetes Full Course"),
        "g1": video_item("g1", title="Old Git course"),
        "d1": video_item("d1", title="Docker Full Course for Beginners"),
        "d2": video_item("d2", title="Cooking full course"),
    }

    def handler(request):
        path, params = request.url.path, request.url.params
        if path.endswith("/search"):
            assert params["type"] == "video" and params["videoDuration"] == "long"
            return httpx.Response(200, json={"items": [{"id": {"videoId": v}} for v in ("d1", "d2")]})
        if path.endswith("/playlistItems"):
            assert params["playlistId"] == "UU1"  # the channel's uploads
            return httpx.Response(200, json={"items": uploads})
        if path.endswith("/playlists"):
            return httpx.Response(200, json={"items": []})
        if path.endswith("/channels"):
            return httpx.Response(200, json={"items": [{"id": "UC1", "statistics": {"subscriberCount": "1000"}}]})
        ids = params["id"].split(",")
        return httpx.Response(200, json={"items": [videos[i] for i in ids if i in videos]})

    skills = [("docker", "Docker and containers"), ("kubernetes", "Kubernetes"), ("git", "Git")]
    state = {"UC1": "2026-06-01"}
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        channels = youtube_channels.scan(
            client,
            "key",
            [{"id": "UC1", "name": "Nana"}],
            state,
            skills,
            set(),
            {"docker", "kubernetes", "git"},
            Counter(),
        )
        found, used = youtube_discovery.discover_videos(client, "key", skills, {"kubernetes", "git"}, max_searches=5)
    assert [c["id"] for c in channels["videos"]["kubernetes"]] == ["k1"] and "git" not in channels["videos"]
    assert state["UC1"] == "2026-10-01"  # the next scan reads only newer uploads
    assert used == 1 and [c["id"] for c in found["docker"]] == ["d1"]  # "Cooking" doesn't name the skill
    path, review = tmp_path / "c.yaml", tmp_path / "r.md"
    data = youtube_discovery.load_candidates(path)
    youtube_discovery.save(path, review, data, {}, dict(skills), found, channels)
    saved = youtube_discovery.load_candidates(path)
    assert saved["searched_videos"] == ["docker"]
    assert saved["video_candidates"] == [{"skill": "docker", "video": "d1"}, {"skill": "kubernetes", "video": "k1"}]
    assert "Kubernetes Full Course" in review.read_text() and "Kubernetes Full Course" not in path.read_text()
