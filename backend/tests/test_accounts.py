"""Accounts (ADR-0043): the signed identity header, sign-in with linking by verified email, saved boards and
results, sign out everywhere, export and deletion. API tests run on the local database, rolled back."""

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from xcrs.api import identity
from xcrs.config import get_settings

SECRET = "test-secret"
CHIPS = [
    {"category": "liked", "skill": "java", "proficiency": 3},
    {"category": "liked", "text": "Spring Boot"},
    {"category": "curious", "skill": "kafka"},
]


def test_a_signed_header_verifies_and_a_changed_one_does_not():
    user = uuid.uuid4()
    value = identity.sign(SECRET, user, 1_000, issued_at_ms=2_000)
    assert identity.verify(SECRET, value, now_ms=2_500) == identity.SignedUser(user, 1_000)
    assert identity.verify("other-secret", value, now_ms=2_500) is None
    forged = value.replace(str(user), str(uuid.uuid4()))
    assert identity.verify(SECRET, forged, now_ms=2_500) is None
    assert identity.verify(SECRET, value, now_ms=2_000 + identity.MAX_AGE_MS + 1) is None  # stale
    assert identity.verify(SECRET, "not.a.header", now_ms=2_500) is None
    assert identity.verify(SECRET, identity.sign(SECRET, "nope", 1, 2), now_ms=2) is None


@pytest.fixture
def api(client, monkeypatch):
    try:
        client.db.execute(text("SELECT 1 FROM users LIMIT 1"))
    except ProgrammingError:
        pytest.skip("needs the database at migration 0012 or later")
    monkeypatch.setattr(get_settings(), "internal_secret", SECRET)
    return client


def sign_in(api, subject="1", provider="github", email="ada@example.com", verified=True, name="Ada"):
    body = {"provider": provider, "subject": subject, "name": name, "email": email, "email_verified": verified}
    r = api.post("/internal/sign-in", json=body, headers={"X-XCRS-Internal-Secret": SECRET})
    assert r.status_code == 200, r.text
    return r.json()


def headers(signed: dict) -> dict:
    return {"X-XCRS-User": identity.sign(SECRET, signed["user_id"], signed["signed_in_at_ms"])}


def test_sign_in_is_only_for_the_nuxt_server(api):
    body = {"provider": "github", "subject": "1"}
    assert api.post("/internal/sign-in", json=body).status_code == 404
    assert api.post("/internal/sign-in", json=body, headers={"X-XCRS-Internal-Secret": "guess"}).status_code == 404


def test_providers_with_the_same_verified_email_are_one_account(api):
    github = sign_in(api, "1", "github", "Ada@Example.com")
    google = sign_in(api, "g-1", "google", "ada@example.com")
    linkedin = sign_in(api, "li-1", "linkedin", "ada@example.com", verified=False)
    again = sign_in(api, "1", "github", "ada@example.com")
    assert github["user_id"] == google["user_id"] == again["user_id"] != linkedin["user_id"]
    me = api.get("/api/v2/me", headers=headers(google)).json()
    assert me["email"] == "ada@example.com" and me["providers"] == ["github", "google"]
    assert api.get("/api/v2/me", headers=headers(linkedin)).json()["email"] is None  # unverified: not stored


def test_me_needs_a_valid_signed_header(api, monkeypatch):
    assert api.get("/api/v2/me").status_code == 401
    forged = api.get("/api/v2/me", headers={"X-XCRS-User": identity.sign("guess", uuid.uuid4(), 1)})
    assert forged.status_code == 401 and forged.headers["X-XCRS-Session"] == "revoked"
    monkeypatch.setattr(get_settings(), "internal_secret", None)
    assert api.get("/api/v2/me").status_code == 404  # accounts off


def test_a_signed_in_users_results_and_board_are_kept(api):
    user = headers(sign_in(api))
    made = api.post("/api/v2/recommendations", json={"chips": CHIPS, "experience": "0-2"}, headers=user).json()
    results = api.get("/api/v2/me/results", headers=user).json()
    assert [r["id"] for r in results] == [made["id"]] and results[0]["roles"][0]["id"] == made["roles"][0]["id"]
    board = api.get("/api/v2/me/board", headers=user).json()["board"]
    assert board["experience"] == "0-2"
    assert [(c["skill"], c["text"], c["name"]) for c in board["chips"]] == [
        ("java", None, "Java"),
        (None, "Spring Boot", "Spring Boot"),
        ("kafka", None, "Apache Kafka"),
    ]
    assert api.put("/api/v2/me/board", json={"chips": [CHIPS[0]]}, headers=user).status_code == 204
    assert len(api.get("/api/v2/me/board", headers=user).json()["board"]["chips"]) == 1
    both = {"category": "liked", "skill": "java", "text": "Java"}
    assert api.put("/api/v2/me/board", json={"chips": [both]}, headers=user).status_code == 422
    # still public by link, and the anonymous flow is unchanged
    assert api.get(f"/api/v2/recommendations/{made['id']}").status_code == 200
    assert api.post("/api/v2/recommendations", json={"chips": CHIPS}).status_code == 200


def test_an_anonymous_result_can_be_saved_once_and_only_while_fresh(api):
    ada, bob = headers(sign_in(api, "1")), headers(sign_in(api, "2", email="bob@example.com", name="Bob"))
    anonymous = api.post("/api/v2/recommendations", json={"chips": CHIPS}).json()["id"]
    assert api.get(f"/api/v2/recommendations/{anonymous}", headers=ada).json()["can_save"] is True
    assert api.post(f"/api/v2/me/results/{anonymous}", headers=ada).status_code == 200
    viewed = api.get(f"/api/v2/recommendations/{anonymous}", headers=ada).json()
    assert viewed["saved"] is True and viewed["can_save"] is False
    assert api.get(f"/api/v2/recommendations/{anonymous}", headers=bob).json()["saved"] is False
    assert api.post(f"/api/v2/me/results/{anonymous}", headers=ada).status_code == 200  # idempotent
    assert api.post(f"/api/v2/me/results/{anonymous}", headers=bob).status_code == 409
    assert api.get("/api/v2/me/board", headers=ada).json()["board"]["chips"]
    old = api.post("/api/v2/recommendations", json={"chips": CHIPS}).json()["id"]
    api.db.execute(
        text("UPDATE recommendations_v2 SET created_at = now() - interval '2 days' WHERE id = :id"), {"id": old}
    )
    assert api.post(f"/api/v2/me/results/{old}", headers=bob).status_code == 409
    assert api.post(f"/api/v2/me/results/{uuid.uuid4()}", headers=bob).status_code == 404
    assert api.delete(f"/api/v2/me/results/{anonymous}", headers=bob).status_code == 404
    assert api.delete(f"/api/v2/me/results/{anonymous}", headers=ada).status_code == 204
    assert api.get(f"/api/v2/recommendations/{anonymous}").status_code == 404


def test_sign_out_everywhere_ends_older_sessions(api):
    first = sign_in(api)
    assert api.post("/api/v2/me/sign-out-everywhere", headers=headers(first)).status_code == 204
    ended = api.get("/api/v2/me", headers=headers(first))
    assert ended.status_code == 401 and ended.headers["X-XCRS-Session"] == "revoked"
    # anonymous use goes on; the proxy is told to clear the cookie
    made = api.post("/api/v2/recommendations", json={"chips": CHIPS}, headers=headers(first))
    assert made.status_code == 200 and made.headers["X-XCRS-Session"] == "revoked"
    assert api.get("/api/v2/me/results", headers=headers(sign_in(api))).json() == []
    assert api.get("/api/v2/me", headers=headers(sign_in(api))).status_code == 200


def test_export_and_delete_cover_everything_linked(api):
    signed = sign_in(api)
    user = headers(signed)
    made = api.post("/api/v2/recommendations", json={"chips": CHIPS}, headers=user).json()["id"]
    api.post(f"/api/v2/recommendations/{made}/feedback", json={"role": "backend-engineer", "rating": 1})
    exported = api.get("/api/v2/me/export", headers=user)
    assert exported.headers["content-disposition"].startswith("attachment;")
    data = exported.json()
    assert data["user"]["email"] == "ada@example.com" and data["sign_in_accounts"][0]["provider"] == "github"
    assert data["results"][0]["id"] == made and data["results"][0]["feedback"][0]["rating"] == 1
    assert data["board"]["input"]["chips"]
    gone = api.delete("/api/v2/me", headers=user)
    assert gone.status_code == 204 and gone.headers["X-XCRS-Session"] == "revoked"
    assert api.get("/api/v2/me", headers=user).status_code == 401
    assert api.get(f"/api/v2/recommendations/{made}").status_code == 404
    left = api.db.execute(
        text("""SELECT (SELECT count(*) FROM user_identities WHERE user_id = :u)
                     + (SELECT count(*) FROM boards WHERE user_id = :u)
                     + (SELECT count(*) FROM feedback_v2 WHERE recommendation_id = :r)"""),
        {"u": signed["user_id"], "r": made},
    ).scalar()
    assert left == 0
