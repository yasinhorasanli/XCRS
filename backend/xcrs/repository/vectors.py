"""Vector queries. The only place that writes pgvector SQL (ADR-0009, ADR-0012)."""

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from xcrs.db.models import EmbeddingModel


def _vector_literal(vector) -> str:
    return "[" + ",".join(f"{float(x):.8g}" for x in vector) + "]"


@dataclass(frozen=True)
class ConceptMatch:
    phrase_index: int  # 0-based position in the input list
    concept_id: int
    similarity: float


def concepts_above_threshold(
    session: Session, model: EmbeddingModel, phrase_vectors: list, threshold: float
) -> list[ConceptMatch]:
    """All (phrase, concept) pairs above `threshold`, via an exact scan (ADR-0010).

    Every phrase of a request is handled in one query.
    """
    rows = session.execute(
        text("""
            WITH phrases AS MATERIALIZED (   -- parse each phrase vector once, not once per concept
                SELECT idx - 1 AS phrase_index, vec::vector AS vec
                FROM   unnest(CAST(:vectors AS text[])) WITH ORDINALITY AS p(vec, idx)
            ),
            scored AS (
                SELECT p.phrase_index, e.node_id AS concept_id, 1 - (e.embedding <=> p.vec) AS similarity
                FROM   phrases p
                JOIN   node_embeddings e ON e.model_id = :model_id
            )
            SELECT s.phrase_index, s.concept_id, s.similarity
            FROM   scored s
            JOIN   roadmap_nodes n ON n.id = s.concept_id AND n.type = 'concept'
            WHERE  s.similarity > :threshold
            ORDER  BY s.phrase_index, s.similarity DESC
        """),
        {"vectors": [_vector_literal(v) for v in phrase_vectors], "model_id": model.id, "threshold": threshold},
    ).all()
    return [ConceptMatch(r.phrase_index, r.concept_id, r.similarity) for r in rows]


def nearest_courses_sql(model: EmbeddingModel) -> str:
    """k-NN over course vectors, shaped to match the per-model partial HNSW index.

    The cast and the model filter must be exactly `embedding::vector(<dims>)` and
    `model_id = <id>`, otherwise Postgres silently falls back to a full scan (ADR-0009).
    Both values come from the registry as integers, never from user input.
    """
    dims, model_id = int(model.dimensions), int(model.id)
    vector_type = "vector" if dims <= 2000 else "halfvec"
    return f"""
        SELECT course_id, 1 - (embedding::{vector_type}({dims}) <=> CAST(:query AS {vector_type}({dims}))) AS similarity
        FROM   course_embeddings
        WHERE  model_id = {model_id}
        ORDER  BY embedding::{vector_type}({dims}) <=> CAST(:query AS {vector_type}({dims}))
        LIMIT  :k
    """


def nearest_courses(session: Session, model: EmbeddingModel, vector, k: int = 20) -> list[tuple[int, float]]:
    rows = session.execute(text(nearest_courses_sql(model)), {"query": _vector_literal(vector), "k": k}).all()
    return [(r.course_id, r.similarity) for r in rows]


def candidate_courses(session: Session, model: EmbeddingModel, concept_ids: list[int]) -> list[tuple[int, int, float]]:
    """(concept_id, course_id, similarity) from the precomputed top-k matches (ADR-0009)."""
    if not concept_ids:
        return []
    rows = session.execute(
        text("""
            SELECT concept_id, course_id, similarity
            FROM   concept_course_matches
            WHERE  model_id = :model_id AND concept_id = ANY(:concept_ids)
            ORDER  BY concept_id, rank
        """),
        {"model_id": model.id, "concept_ids": concept_ids},
    ).all()
    return [(r.concept_id, r.course_id, r.similarity) for r in rows]


def courses_similar_to(
    session: Session, model: EmbeddingModel, phrase_vectors: list, course_ids: list[int], threshold: float
) -> set[int]:
    """Which of the given (candidate) courses are above `threshold` for any phrase.

    Exact, and limited to candidates, so no request scans the whole catalog (ADR-0010).
    """
    if not phrase_vectors or not course_ids:
        return set()
    rows = session.execute(
        text("""
            WITH phrases AS MATERIALIZED (
                SELECT vec::vector AS vec FROM unnest(CAST(:vectors AS text[])) AS p(vec)
            )
            SELECT DISTINCT e.course_id
            FROM   phrases p
            JOIN   course_embeddings e ON e.model_id = :model_id AND e.course_id = ANY(:course_ids)
            WHERE  1 - (e.embedding <=> p.vec) > :threshold
        """),
        {
            "vectors": [_vector_literal(v) for v in phrase_vectors],
            "model_id": model.id,
            "course_ids": course_ids,
            "threshold": threshold,
        },
    ).all()
    return {r.course_id for r in rows}


def nearest_concepts(session: Session, model: EmbeddingModel, phrase_vectors: list, k: int) -> list[ConceptMatch]:
    """The k nearest concepts to each phrase (exact; concepts are few, ADR-0010). For suggestions."""
    rows = session.execute(
        text("""
            WITH phrases AS MATERIALIZED (
                SELECT idx - 1 AS phrase_index, vec::vector AS vec
                FROM   unnest(CAST(:vectors AS text[])) WITH ORDINALITY AS p(vec, idx)
            )
            SELECT p.phrase_index, x.concept_id, x.similarity
            FROM   phrases p
            CROSS  JOIN LATERAL (
                SELECT e.node_id AS concept_id, 1 - (e.embedding <=> p.vec) AS similarity
                FROM   node_embeddings e
                JOIN   roadmap_nodes n ON n.id = e.node_id AND n.type = 'concept'
                WHERE  e.model_id = :model_id
                ORDER  BY e.embedding <=> p.vec
                LIMIT  :k
            ) x
            ORDER  BY p.phrase_index, x.similarity DESC
        """),
        {"vectors": [_vector_literal(v) for v in phrase_vectors], "model_id": model.id, "k": k},
    ).all()
    return [ConceptMatch(r.phrase_index, r.concept_id, r.similarity) for r in rows]
