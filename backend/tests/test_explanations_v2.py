"""Engine v2 explanations (ADR-0037): grounded facts, the job flow, and the API's view of it."""

import contextlib

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, ProgrammingError

from xcrs.api.app import app, get_service_v2
from xcrs.catalog import model, validate
from xcrs.db.models import EmbeddingModel
from xcrs.db.session import new_session
from xcrs.domain.role_scoring import Category
from xcrs.explain.v2 import RoleExplanationV2Out, build_facts, system_prompt
from xcrs.repository import catalog_store
from xcrs.services.explanations_v2 import ExplanationWorkerV2
from xcrs.services.recommend_v2 import Chip, RecommendationServiceV2
from xcrs.services.skill_matching import DatabaseMatchStore, SkillMatcher, lexical_index

ROLE = {
    "id": "data-engineer",
    "name": "Data Engineer",
    "coverage": 0.3,
    "level": None,
    "target_level": {"id": "entry", "title": None, "coverage": 0.3},
    "because": [
        {"id": "airflow", "name": "Apache Airflow", "category": "curious"},
        {"id": "css", "name": "CSS", "category": "disliked"},
    ],
    "gaps": [{"skills": [{"id": "sql", "name": "SQL"}], "need": 2, "have": 0, "stage": "Foundations"}],
    "resources": [],
}
MATCHED = [
    {
        "text": "data pipelines",
        "category": "curious",
        "proficiency": None,
        "method": "llm",
        "skills": [{"id": "airflow", "name": "Apache Airflow"}],
    }
]


def test_facts_keep_the_learners_words_and_feelings_and_drop_empty_fields():
    facts = build_facts(ROLE, MATCHED, "Builds data pipelines.", {"entry": "entry level"})
    assert facts["because"][0] == {"user_said": "data pipelines", "skill": "Apache Airflow", "feeling": "curious about"}
    assert facts["because"][1]["feeling"] == "did not enjoy"
    assert facts["level"] == "start at entry level" and facts["covers"] == "some of it"
    assert "resources" not in facts and "resources" not in system_prompt(facts).split("Rules:")[0]


class FakeExplainer:
    def explain(self, facts):
        return RoleExplanationV2Out(explanation=f"{facts['role']} fits.", next_step="Start with the basics.")


class Queue:
    def __init__(self):
        self.jobs = []

    def submit(self, recommendation_id, role):
        self.jobs.append((recommendation_id, role))


@pytest.fixture
def session():
    s = new_session()
    try:
        s.execute(text("SELECT 1 FROM explanations_v2 LIMIT 1"))
        embedding_model = s.get(EmbeddingModel, 1)
    except (OperationalError, ProgrammingError):
        s.close()
        pytest.skip("needs the database at migration 0009 or later")
    if embedding_model is None:
        s.close()
        pytest.skip("embedding model 1 not registered")
    cat = model.load_catalog()
    catalog_store.import_catalog(
        s, cat, git_commit="t", git_dirty=True, checksum="t-expl", stats=validate.validate(cat).stats
    )
    s.execute(text("DELETE FROM catalog.skill_embeddings"))
    s.commit = s.flush
    yield s, embedding_model
    s.rollback()
    s.close()


def test_a_recommendation_queues_one_job_per_role_and_the_worker_writes_them(session):
    s, embedding_model = session

    class NoVectors:
        def embed_query(self, texts):
            return [[1.0, 0.0, 0.0]]

    matcher = SkillMatcher(lexical_index(s), DatabaseMatchStore(s, embedding_model), NoVectors(), picker=None)
    jobs = Queue()
    service = RecommendationServiceV2(s, matcher, jobs)
    row = service.recommend(
        [Chip(Category.LIKED, skill="python", proficiency=3), Chip(Category.CURIOUS, text="Airflow")]
    )
    assert len(jobs.jobs) == len(row.result["roles"]) == 3

    app.dependency_overrides[get_service_v2] = lambda: service
    try:
        client = TestClient(app)
        before = client.get(f"/api/v2/recommendations/{row.id}").json()
        assert {r["explanation_status"] for r in before["roles"]} == {"pending"}

        @contextlib.contextmanager
        def same_session():
            yield s

        worker = ExplanationWorkerV2(FakeExplainer(), same_session)
        assert [worker.process(*job) for job in jobs.jobs] == ["done"] * 3
        assert worker.process(*jobs.jobs[0]) is None  # already done: nothing to do
        after = client.get(f"/api/v2/recommendations/{row.id}").json()
    finally:
        app.dependency_overrides.clear()
    first = after["roles"][0]
    assert first["explanation_status"] == "done" and first["explanation"] == f"{first['name']} fits."
    assert first["next_step"] == "Start with the basics."
