"""Skill vectors, the phrase cache and the v2 match endpoint against the local database. Every test runs in
a transaction that is rolled back."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, ProgrammingError

from xcrs.api.app import app, get_session, get_skill_matcher
from xcrs.catalog import model, validate
from xcrs.db.models import EmbeddingModel, SkillEmbedding
from xcrs.db.session import new_session
from xcrs.repository import catalog_store, vectors
from xcrs.services.skill_matching import DatabaseMatchStore, SkillMatcher, lexical_index


@pytest.fixture
def session():
    s = new_session()
    try:
        s.execute(text("SELECT 1 FROM catalog.phrase_matches LIMIT 1"))
        embedding_model = s.get(EmbeddingModel, 1)
    except (OperationalError, ProgrammingError):
        s.close()
        pytest.skip("needs the database at migration 0006 or later")
    if embedding_model is None:
        s.close()
        pytest.skip("embedding model 1 not registered")
    cat = model.load_catalog()
    catalog_store.import_catalog(
        s, cat, git_commit="test", git_dirty=True, checksum="test-checksum", stats=validate.validate(cat).stats
    )
    s.execute(text("DELETE FROM catalog.skill_embeddings WHERE model_id = 1"))
    ids = dict(
        s.execute(text("SELECT slug, id FROM catalog.skills WHERE slug IN ('docker', 'kubernetes', 'git')")).all()
    )
    for slug, vec in {"docker": [1, 0, 0], "kubernetes": [0.8, 0.6, 0], "git": [0, 0, 1]}.items():
        s.add(SkillEmbedding(skill_id=ids[slug], model_id=1, embedding=vec, content_hash="x"))
    s.flush()
    s.commit = s.flush  # keep everything inside the test transaction
    yield s, embedding_model
    s.rollback()
    s.close()


def test_similarities_to_every_embedded_skill(session):
    s, embedding_model = session
    sims = vectors.skill_similarities(s, embedding_model, [1, 0, 0])
    assert sims["docker"] == pytest.approx(1.0) and sims["kubernetes"] == pytest.approx(0.8)
    assert sims["git"] == pytest.approx(0.0)


def test_the_cache_is_keyed_by_catalog_version_prompt_and_model(session):
    s, embedding_model = session
    store = DatabaseMatchStore(s, embedding_model)
    assert store.catalog_checksum == "test-checksum"

    class Picker:
        prompt_version, model = "pick-2", "m1"

    store.save("jira", Picker, ["agile-scrum"], ["agile-scrum", "jenkins"], 1800)
    store.save("jira", Picker, ["other"], [], 1)  # a second writer loses quietly
    assert store.cached("jira", Picker) == ["agile-scrum"]
    Picker.model = "m2"
    assert store.cached("jira", Picker) is None


def test_the_match_endpoint_names_the_skills(session):
    s, embedding_model = session

    class Embedder:
        def embed_query(self, texts):
            return [[0.9, 0.1, 0]]

    matcher = SkillMatcher(lexical_index(s), DatabaseMatchStore(s, embedding_model), Embedder(), picker=None)
    app.dependency_overrides[get_session] = lambda: s
    app.dependency_overrides[get_skill_matcher] = lambda: matcher
    try:
        response = TestClient(app).post("/api/v2/skills/match", json={"phrases": ["Dokcer", "container stuff"]})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    first, second = response.json()["matches"]
    assert first == {
        "phrase": "Dokcer",
        "skills": [{"id": "docker", "name": "Docker and containers"}],
        "method": "lookup",
    }
    assert second["method"] == "embedding" and second["skills"][0]["id"] == "docker"
