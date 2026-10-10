"""Shared test setup: rate limits (ADR-0035) are off for API tests, which all come from one client;
tests/test_limits.py exercises the limiter directly. `client` is the API on the local database, inside one
transaction that is rolled back (skips without the database)."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, ProgrammingError

from xcrs.api import limits
from xcrs.api.app import app, get_session, get_skill_matcher
from xcrs.catalog import model, validate
from xcrs.db.models import EmbeddingModel
from xcrs.db.session import new_session
from xcrs.repository import catalog_store
from xcrs.services.skill_matching import DatabaseMatchStore, SkillMatcher, lexical_index


@pytest.fixture(autouse=True)
def no_rate_limits():
    limiter = limits.limiter()
    enabled, limiter.enabled = limiter.enabled, False
    yield
    limiter.enabled = enabled


class NoVectors:
    def embed_query(self, texts):
        return [[1.0, 0.0, 0.0]]


@pytest.fixture
def client():
    s = new_session()
    try:
        s.execute(text("SELECT 1 FROM recommendations_v2 LIMIT 1"))
        embedding_model = s.get(EmbeddingModel, 1)
    except (OperationalError, ProgrammingError):
        s.close()
        pytest.skip("needs the database at migration 0007 or later")
    if embedding_model is None:
        s.close()
        pytest.skip("embedding model 1 not registered")
    cat = model.load_catalog()
    catalog_store.import_catalog(
        s, cat, git_commit="test", git_dirty=True, checksum="test-v2", stats=validate.validate(cat).stats
    )
    s.execute(text("DELETE FROM catalog.skill_embeddings"))  # the fake embedder's vectors are 3-dimensional
    s.commit = s.flush  # keep everything inside the test transaction
    matcher = SkillMatcher(lexical_index(s), DatabaseMatchStore(s, embedding_model), NoVectors(), picker=None)
    app.dependency_overrides[get_session] = lambda: s
    app.dependency_overrides[get_skill_matcher] = lambda: matcher
    try:
        test_client = TestClient(app)
        test_client.db = s  # for tests that look at the rows
        yield test_client
    finally:
        app.dependency_overrides.clear()
        s.rollback()
        s.close()
