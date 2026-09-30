"""Explanations as background jobs (ADR-0018), against the local database with the imported catalog.

A fake embedder returns stored concept vectors, so recommendations are deterministic and need no Ollama;
a fake explainer stands in for the LLM.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, event, select
from sqlalchemy.exc import OperationalError

from xcrs.api.app import app, get_knowledge_units, get_service, get_session
from xcrs.db.models import Feedback, NodeEmbedding, RecommendationRequest, RecommendedRole, RoadmapNode
from xcrs.db.session import get_engine, new_session
from xcrs.domain.results import ExplanationStatus
from xcrs.domain.types import Category
from xcrs.explain.base import RoleExplanation
from xcrs.repository import activity
from xcrs.repository import catalog as catalog_repo
from xcrs.services.explanations import ExplanationWorker
from xcrs.services.knowledge_units import KnowledgeUnitService
from xcrs.services.recommend import RecommendationService


class ConceptVectorEmbedder:
    """Embeds each phrase as the stored vector of the concept with that name: an exact match."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def embed_query(self, texts):
        return [self.vectors[t] for t in texts]


class RecordingQueue:
    def __init__(self):
        self.jobs = []

    def submit(self, request_id, role_id):
        self.jobs.append((request_id, role_id))


class FakeExplainer:
    """Explains the role and every course, like a well-behaved LLM."""

    def explain(self, context):
        return RoleExplanation(
            role_explanation=f"Fits {context.role}.",
            course_explanations={c.course_id: f"Helps with {c.covers[0]}." for c in context.courses},
            prompt_version="test-v1",
        )


class BrokenExplainer:
    def explain(self, context):
        return RoleExplanation()  # what a failed LLM call degrades to


@pytest.fixture
def db():
    session = new_session()
    try:
        catalog_repo.active_model(session)
    except (OperationalError, RuntimeError):
        session.close()
        pytest.skip("needs the local database with an active, embedded model")
    created: list[uuid.UUID] = []
    yield session, created
    session.rollback()
    if created:
        session.execute(delete(RecommendationRequest).where(RecommendationRequest.id.in_(created)))
        session.commit()
    session.close()


@pytest.fixture
def embedder(db):
    session, _ = db
    model = catalog_repo.active_model(session)
    rows = session.execute(
        select(RoadmapNode.name, NodeEmbedding.embedding)
        .join(NodeEmbedding, NodeEmbedding.node_id == RoadmapNode.id)
        .where(RoadmapNode.type == "concept", NodeEmbedding.model_id == model.id)
        .order_by(RoadmapNode.id)
        .limit(400)
    ).all()
    if not rows:
        pytest.skip("no embedded concepts")
    return ConceptVectorEmbedder({name: list(vec) for name, vec in rows})


def recommend(db, embedder, queue, liked=3, curious=2):
    session, created = db
    names = list(embedder.vectors)
    user_input = {
        Category.LIKED: names[:liked],
        Category.NEUTRAL: [],
        Category.DISLIKED: [],
        Category.CURIOUS: names[-curious:],
    }
    service = RecommendationService(session, queue, embedder_factory=lambda _model: embedder)
    result = service.recommend(user_input)
    created.append(result.request_id)
    assert result.roles, "fixture phrases should always produce a recommendation"
    return result


def worker(explainer) -> ExplanationWorker:
    return ExplanationWorker(explainer, new_session)


def test_recommendation_returns_at_once_and_queues_one_job_per_role(db, embedder):
    queue = RecordingQueue()
    result = recommend(db, embedder, queue)

    assert all(r.explanation is None and r.explanation_status is ExplanationStatus.PENDING for r in result.roles)
    assert queue.jobs == [(result.request_id, r.role_id) for r in result.roles]
    session, _ = db
    stored = session.scalars(select(RecommendedRole).where(RecommendedRole.request_id == result.request_id)).all()
    assert all(r.explanation_input and r.explanation_input["role"] for r in stored)  # what the LLM will get


def test_worker_fills_in_role_and_course_explanations(db, embedder):
    result = recommend(db, embedder, RecordingQueue())
    w = worker(FakeExplainer())
    for role in result.roles:
        assert w.process(result.request_id, role.role_id) is ExplanationStatus.DONE

    session, _ = db
    loaded = activity.load_recommendation(session, result.request_id)
    for role in loaded.roles:
        assert role.explanation_status is ExplanationStatus.DONE
        assert role.explanation == f"Fits {role.role}."
        assert role.prompt_version == "test-v1"
        assert all(c.explanation and c.explanation.startswith("Helps with") for c in role.courses)
    timings = session.scalars(
        select(RecommendedRole.explanation_ms).where(RecommendedRole.request_id == result.request_id)
    ).all()
    assert len(timings) == len(result.roles) and all(ms is not None and ms >= 0 for ms in timings)


def test_a_failed_explanation_marks_the_role_failed_and_keeps_the_recommendation(db, embedder):
    result = recommend(db, embedder, RecordingQueue())
    role = result.roles[0]
    assert worker(BrokenExplainer()).process(result.request_id, role.role_id) is ExplanationStatus.FAILED

    session, _ = db
    loaded = activity.load_recommendation(session, result.request_id)
    assert loaded.roles[0].explanation_status is ExplanationStatus.FAILED
    assert loaded.roles[0].explanation is None
    assert [c.course_id for c in loaded.roles[0].courses] == [c.course_id for c in role.courses]  # still there


def test_a_job_that_keeps_crashing_gives_up_after_max_attempts(db, embedder):
    result = recommend(db, embedder, RecordingQueue())
    request_id, role_id = result.request_id, result.roles[0].role_id
    session, _ = db
    session.execute(
        RecommendedRole.__table__.update()
        .where(RecommendedRole.request_id == request_id, RecommendedRole.role_id == role_id)
        .values(explanation_attempts=3)
    )
    session.commit()

    assert worker(FakeExplainer()).process(request_id, role_id) is None
    session.expire_all()
    loaded = activity.load_recommendation(session, request_id)
    assert loaded.roles[0].explanation_status is ExplanationStatus.FAILED


def test_restart_requeues_pending_roles_and_the_threads_finish_them(db, embedder):
    """The rows are the source of truth: jobs queued before a 'restart' are found again."""
    result = recommend(db, embedder, RecordingQueue())  # queued in memory only, never run
    w = worker(FakeExplainer())
    w.start()
    try:
        assert w.requeue_pending() >= len(result.roles)
        w.wait_until_idle()
    finally:
        w.stop()

    session, _ = db
    loaded = activity.load_recommendation(session, result.request_id)
    assert {r.explanation_status for r in loaded.roles} == {ExplanationStatus.DONE}


def test_disabled_explanations_store_no_job(db, embedder):
    result = recommend(db, embedder, queue=None)
    assert {r.explanation_status for r in result.roles} == {ExplanationStatus.DISABLED}
    session, _ = db
    assert session.scalars(
        select(RecommendedRole.explanation_input).where(RecommendedRole.request_id == result.request_id)
    ).all() == [None] * len(result.roles)


def test_reading_a_result_takes_a_fixed_number_of_queries(db, embedder):
    """ADR-0012: no N+1. Loading a result is 4 queries, however many roles and courses it has."""
    result = recommend(db, embedder, RecordingQueue(), liked=12, curious=6)
    statements = []

    def count(*_args):
        statements.append(1)

    event.listen(get_engine(), "before_cursor_execute", count)
    try:
        with new_session() as session:
            loaded = activity.load_recommendation(session, result.request_id)
    finally:
        event.remove(get_engine(), "before_cursor_execute", count)

    assert sum(len(r.courses) for r in loaded.roles) > 1
    assert len(statements) == 4


def test_api_returns_the_recommendation_then_serves_it_by_id(db, embedder):
    session, created = db
    queue = RecordingQueue()
    app.dependency_overrides[get_service] = lambda: RecommendationService(
        session, queue, embedder_factory=lambda _model: embedder
    )
    try:
        client = TestClient(app)
        names = list(embedder.vectors)[:3]
        posted = client.post("/api/v1/recommendations", json={"liked": names}).json()
        created.append(uuid.UUID(posted["request_id"]))
        assert {r["explanation_status"] for r in posted["roles"]} == {"pending"}
        assert all(r["next_to_learn"] is not None for r in posted["roles"])

        fetched = client.get(f"/api/v1/recommendations/{posted['request_id']}").json()
        assert [r["role"] for r in fetched["roles"]] == [r["role"] for r in posted["roles"]]
        assert [c["course_id"] for r in fetched["roles"] for c in r["courses"]] == [
            c["course_id"] for r in posted["roles"] for c in r["courses"]
        ]
        assert client.get(f"/api/v1/recommendations/{uuid.uuid4()}").status_code == 404
    finally:
        app.dependency_overrides.clear()


# --- Feedback and suggestions endpoints (ADR-0023) ---


def client_with(session, embedder, queue=None) -> TestClient:
    app.dependency_overrides[get_service] = lambda: RecommendationService(
        session, queue or RecordingQueue(), embedder_factory=lambda _model: embedder
    )
    app.dependency_overrides[get_knowledge_units] = lambda: KnowledgeUnitService(
        session, embedder_factory=lambda _model: embedder
    )
    app.dependency_overrides[get_session] = lambda: session
    return TestClient(app)


def test_result_echoes_the_input_and_accepts_feedback(db, embedder):
    session, created = db
    try:
        client = client_with(session, embedder)
        names = list(embedder.vectors)[:3]
        posted = client.post("/api/v1/recommendations", json={"liked": names, "curious": []}).json()
        request_id = posted["request_id"]
        created.append(uuid.UUID(request_id))
        assert posted["input"]["liked"] == names
        assert client.get(f"/api/v1/recommendations/{request_id}").json()["input"]["liked"] == names

        role = posted["roles"][0]
        course = role["courses"][0]
        url = f"/api/v1/recommendations/{request_id}/feedback"
        assert client.post(url, json={"role_id": role["role_id"], "rating": 1}).status_code == 201
        assert (
            client.post(
                url, json={"role_id": role["role_id"], "course_id": course["course_id"], "rating": -1}
            ).status_code
            == 201
        )
        assert client.post(url, json={"rating": 1, "comment": "useful"}).status_code == 201
        assert client.post(url, json={"role_id": 999, "rating": 1}).status_code == 404  # not shown in this result
        assert client.post(url, json={"rating": 5}).status_code == 422
        assert session.scalars(select(Feedback.rating).where(Feedback.request_id == uuid.UUID(request_id))).all() == [
            1,
            -1,
            1,
        ]
    finally:
        app.dependency_overrides.clear()


def test_knowledge_unit_endpoints(db, embedder):
    session, _ = db
    try:
        client = client_with(session, embedder)
        groups = client.get("/api/v1/knowledge-units/groups").json()["groups"]
        assert groups and all(g["units"] for g in groups)

        found = client.get("/api/v1/knowledge-units", params={"q": "python"}).json()["units"]
        assert found[0]["label"] == "Python" and len(found[0]["roles"]) >= 2

        phrase = next(iter(embedder.vectors))
        related_units = client.post("/api/v1/knowledge-units/related", json={"phrases": [phrase]}).json()["units"]
        assert related_units and all(u["because"] == phrase for u in related_units)
        assert all(u["label"].lower() != phrase.lower() for u in related_units)
    finally:
        app.dependency_overrides.clear()
