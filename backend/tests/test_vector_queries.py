"""Integration tests against the local database (docker compose up postgres + imported data)."""

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError

from xcrs.db.models import Course, CourseEmbedding, EmbeddingModel
from xcrs.db.session import new_session
from xcrs.repository.vectors import nearest_courses_sql
from xcrs.retrieval import CourseRetriever


@pytest.fixture
def session_and_model():
    session = new_session()
    try:
        model = session.scalars(select(EmbeddingModel).where(EmbeddingModel.id == 1)).one_or_none()
    except OperationalError:
        pytest.skip("database not reachable")
    if model is None:
        pytest.skip("model 1 not registered")
    yield session, model
    session.close()


def _plan(session, sql: str, model: EmbeddingModel) -> str:
    # At today's size the planner prefers "scan, then sort", so sorting is disabled: the only way to
    # return rows in distance order is then an index that matches the query's expression exactly.
    session.execute(text("SET LOCAL enable_sort = off"))
    query = "[" + ",".join(["0"] * (model.dimensions - 1) + ["1"]) + "]"
    return "\n".join(session.execute(text("EXPLAIN " + sql), {"query": query, "k": 5}).scalars())


def test_nearest_courses_query_matches_the_per_model_hnsw_index(session_and_model):
    """ADR-0009: if the query's cast or filter drifts from the index definition, Postgres silently
    falls back to a full scan instead of using the per-model HNSW index."""
    session, model = session_and_model
    assert "course_emb_m1_hnsw" in _plan(session, nearest_courses_sql(model), model)


def test_query_without_the_cast_cannot_use_the_index(session_and_model):
    """Negative control: proves the test above really detects a mismatched query shape."""
    session, model = session_and_model
    mismatched = nearest_courses_sql(model).replace(f"embedding::vector({model.dimensions})", "embedding")
    assert "course_emb_m1_hnsw" not in _plan(session, mismatched, model)


def test_course_retriever_finds_the_course_whose_own_vector_is_the_query(session_and_model):
    """The LangChain retriever wraps our pgvector SQL (ADR-0019). Querying with a course's stored vector
    must return that course first, with its metadata."""
    session, model = session_and_model
    course = session.scalars(select(Course).where(Course.is_active).order_by(Course.id).limit(1)).one()
    stored = session.scalars(
        select(CourseEmbedding.embedding).where(CourseEmbedding.course_id == course.id, CourseEmbedding.model_id == 1)
    ).one()

    class FixedEmbedder:
        model_id, dimensions = model.name, model.dimensions

        def embed_query(self, texts):
            return [list(stored)]

    docs = CourseRetriever(session=session, model=model, embedder=FixedEmbedder(), k=3).invoke("anything")
    assert docs[0].metadata["course_id"] == course.id
    assert docs[0].metadata["similarity"] > 0.99
    assert docs[0].page_content.startswith(course.title)
