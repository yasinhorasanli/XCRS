"""Engine v2 recommendations through the API, against the local database; rolled back (the `client`
fixture is in conftest.py)."""

from sqlalchemy import text


def test_a_recommendation_is_made_stored_and_read_back(client):
    chips = [
        {"category": "liked", "skill": "java", "proficiency": 3},
        {"category": "liked", "text": "Spring Boot"},
        {"category": "liked", "text": "SQL"},
        {"category": "curious", "skill": "kafka"},
        {"category": "disliked", "text": "CSS"},
    ]
    created = client.post("/api/v2/recommendations", json={"chips": chips})
    assert created.status_code == 200
    body = created.json()
    assert body["status"] == "ok" and body["catalog_version"] == "test-v2"
    assert [m["method"] for m in body["matched"]] == ["picked", "lookup", "lookup", "picked", "lookup"]
    first = body["roles"][0]
    assert 0 < first["interest_in_role"] <= 1
    assert first["id"] == "backend-engineer" and first["target_level"]["id"] == "entry"
    assert first["title_for_you"] == "Java Backend Engineer (Spring Boot)"  # ADR-0044
    assert first["job_titles"][:2] == ["Backend Engineer", "Backend Developer"]
    assert {b["id"] for b in first["because"]} >= {"java", "spring-boot", "kafka"}
    assert first["gaps"] and all(g["have"] < g["need"] for g in first["gaps"])
    # The level bar: the next level (entry here) fills partly, later ones are empty; each gap knows its level.
    progress = [lv["progress"] for lv in first["levels"]]
    assert 0 < progress[0] < 1 and progress[1:] == [0.0] * (len(progress) - 1)
    assert {g["level"] for g in first["gaps"]} == {"entry"}
    again = client.get(f"/api/v2/recommendations/{body['id']}")
    assert again.status_code == 200 and again.json()["roles"] == body["roles"]
    feedback = client.post(
        f"/api/v2/recommendations/{body['id']}/feedback", json={"role": "backend-engineer", "rating": 1}
    )
    assert feedback.status_code == 201


def test_input_that_matches_no_skill_is_reported(client):
    body = client.post("/api/v2/recommendations", json={"chips": [{"category": "liked", "text": "knitting"}]}).json()
    assert body["status"] == "insufficient_input" and body["roles"] == []


def test_reached_levels_are_full_and_missing_skills_are_grouped_by_level(client):
    """A mid-level backend board: entry and mid reached (full), senior partly; the skills of those levels it
    didn't list ("basics") each carry their level, for the review under the level bar."""
    chips = [
        {"category": "liked", "skill": s, "proficiency": 3}
        for s in (
            "java",
            "spring-boot",
            "sql",
            "postgresql",
            "rest-api-design",
            "docker",
            "ci-cd",
            "database-design",
            "caching",
            "authentication-authorization",
            "web-security",
            "integration-testing",
            "design-patterns",
            "concurrency",
            "unit-testing",
            "http",
        )
    ]
    role = next(
        r
        for r in client.post("/api/v2/recommendations", json={"chips": chips}).json()["roles"]
        if r["id"] == "backend-engineer"
    )
    by_level = {lv["id"]: lv for lv in role["levels"]}
    assert role["level"]["id"] == "mid" and by_level["entry"]["progress"] == by_level["mid"]["progress"] == 1.0
    assert 0 <= by_level["senior"]["progress"] < 1 and by_level["staff"]["progress"] == 0.0
    assert role["basics"] and {b["level"] for b in role["basics"]} <= {"entry", "mid"}


def test_a_student_shows_little_of_mid_and_part_of_an_entry_not_reached_yet(client):
    """The level bar for the computer science student example: Embedded is reached at entry and shows only a little
    of mid (operating systems; working C and basic Linux are entry's, not partly mid's: engine v3.8), and Game
    Developer (entry not reached yet) shows part of entry, not zero."""
    liked = ["programming-fundamentals", "data-structures", "algorithms", "oop", "java", "python"]
    neutral = {
        "c": 2,
        "sql": 2,
        "relational-databases": 2,
        "operating-systems": 2,
        "computer-architecture": 1,
        "git": 1,
        "linux": 1,
        "testing-fundamentals": 1,
        "statistics": 1,
    }
    chips = [{"category": "liked", "skill": s, "proficiency": 2} for s in liked]
    chips += [{"category": "neutral", "skill": s, "proficiency": p} for s, p in neutral.items()]
    chips += [
        {"category": "disliked", "skill": "digital-electronics", "proficiency": 1},
        {"category": "disliked", "skill": "calculus", "proficiency": 2},
    ]
    chips += [
        {"category": "curious", "skill": s}
        for s in ("machine-learning-fundamentals", "computer-networking", "computer-graphics", "llm-fundamentals")
    ]
    roles = {
        r["id"]: r
        for r in client.post("/api/v2/recommendations", json={"chips": chips, "experience": "student"}).json()["roles"]
    }
    embedded = {lv["id"]: lv["progress"] for lv in roles["embedded-software-engineer"]["levels"]}
    assert roles["embedded-software-engineer"]["level"]["id"] == "entry"
    assert embedded["entry"] == 1.0 and 0 < embedded["mid"] < 0.1 and embedded["senior"] == embedded["staff"] == 0.0
    game = roles["game-developer"]
    assert game["level"] is None and 0 < game["levels"][0]["progress"] < 1


def test_a_board_that_singles_out_no_role_says_why(client):
    """ADR-0052: "Python" alone is recognised but used by most roles, so no role is shown, with a hint."""
    body = client.post("/api/v2/recommendations", json={"chips": [{"category": "liked", "skill": "python"}]}).json()
    assert body["status"] == "no_clear_match" and body["roles"] == []
    assert body["no_match"]["positive"] == 1 and body["no_match"]["broad"][0]["id"] == "python"
    assert body["no_match"]["broad"][0]["roles"] >= body["no_match"]["roles_total"] / 2


def test_health_answers_and_the_classic_api_is_gone(client):
    assert client.get("/api/v2/health").json() == {"status": "ok"}
    assert client.post("/api/v1/recommendations", json={"liked": ["Java"]}).status_code == 404


def test_a_chip_needs_exactly_one_of_skill_or_text(client):
    both = {"category": "liked", "skill": "java", "text": "Java"}
    assert client.post("/api/v2/recommendations", json={"chips": [both]}).status_code == 422
    assert client.get("/api/v2/recommendations/00000000-0000-0000-0000-000000000000").status_code == 404


def test_the_picker_finds_skills_and_suggests_per_family(client):
    found = client.get("/api/v2/skills", params={"q": "post"}).json()["skills"]
    assert found[0]["id"] == "postgresql"
    groups = client.get("/api/v2/skills/groups").json()["groups"]
    assert len(groups) == 8 and all(len(g["skills"]) == 12 for g in groups)


def test_feedback_is_one_changeable_vote_per_role(client):
    """Security audit 2026-10-26 (#1): every POST added a row, so one client could fill the disk."""
    chips = [
        {"category": "liked", "skill": "java", "proficiency": 3},
        {"category": "liked", "skill": "postgresql", "proficiency": 3},
        {"category": "curious", "skill": "kafka"},
    ]
    made = client.post("/api/v2/recommendations", json={"chips": chips}).json()
    url = f"/api/v2/recommendations/{made['id']}/feedback"
    first, second = (r["id"] for r in made["roles"][:2])
    for rating in (1, -1, 1, -1):
        assert client.post(url, json={"role": first, "rating": rating}).status_code == 201
    assert client.post(url, json={"role": second, "rating": 1}).status_code == 201
    assert client.post(url, json={}).status_code == 201
    assert client.post(url, json={}).status_code == 201
    # Only the result's own roles: any other string would make a new row each time.
    shown = {r["id"] for r in made["roles"]}
    assert client.post(url, json={"role": "not-a-role-1", "rating": 1}).status_code == 422
    other = next(s for s in ("ios-engineer", "game-developer", "penetration-tester") if s not in shown)
    assert client.post(url, json={"role": other, "rating": 1}).status_code == 422
    rows = client.db.execute(
        text("SELECT role, rating FROM feedback_v2 WHERE recommendation_id = :r"), {"r": made["id"]}
    ).all()
    assert sorted(rows, key=lambda r: r.role or "") == [(None, None), *sorted([(first, -1), (second, 1)])]
    assert client.post(url, json={"role": first, "comment": "x" * 501}).status_code == 422
