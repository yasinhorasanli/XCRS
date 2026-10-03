"""Engine v2 recommendations through the API, against the local database; rolled back (the `client`
fixture is in conftest.py)."""


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
    assert first["id"] == "backend-engineer" and first["target_level"]["id"] == "entry"
    assert {b["id"] for b in first["because"]} >= {"java", "spring-boot", "kafka"}
    assert first["gaps"] and all(g["have"] < g["need"] for g in first["gaps"])
    again = client.get(f"/api/v2/recommendations/{body['id']}")
    assert again.status_code == 200 and again.json()["roles"] == body["roles"]
    feedback = client.post(
        f"/api/v2/recommendations/{body['id']}/feedback", json={"role": "backend-engineer", "rating": 1}
    )
    assert feedback.status_code == 201


def test_input_that_matches_no_skill_is_reported(client):
    body = client.post("/api/v2/recommendations", json={"chips": [{"category": "liked", "text": "knitting"}]}).json()
    assert body["status"] == "insufficient_input" and body["roles"] == []


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
