"""Embed the catalog with a registered model and derive the precomputed data (ADR-0008, ADR-0009).

Steps:
  1. embed concepts and courses that have no vector for this model yet, or whose text changed
     (content_hash mismatch);
  2. threshold statistics: mean and std of all course x concept cosine similarities;
  3. concept_course_matches: the top-k active courses per concept.

Steps 2 and 3 use exact in-memory math, which is fine at the current size (453 x 869). Once the
catalog grows, step 2 samples and step 3 uses the per-model HNSW index incrementally (ADR-0009).
"""

import time
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import and_, delete, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from xcrs.db.models import (
    ConceptCourseMatch,
    Course,
    CourseEmbedding,
    EmbeddingModel,
    NodeEmbedding,
    RoadmapNode,
)
from xcrs.embeddings import embedder_for

TOP_K = 20


def get_model(session: Session, name: str) -> EmbeddingModel:
    model = session.scalars(select(EmbeddingModel).where(EmbeddingModel.name == name)).one_or_none()
    if model is None:
        raise SystemExit(f"model {name!r} is not registered; run `register-model` first")
    return model


def embed_pending(session: Session, model: EmbeddingModel) -> dict[str, int]:
    embedder = embedder_for(model)

    concepts = session.execute(
        select(RoadmapNode.id, RoadmapNode.content, RoadmapNode.content_hash)
        .outerjoin(
            NodeEmbedding,
            and_(
                NodeEmbedding.node_id == RoadmapNode.id,
                NodeEmbedding.model_id == model.id,
            ),
        )
        .where(RoadmapNode.type == "concept")
        .where(
            or_(
                NodeEmbedding.node_id.is_(None),
                NodeEmbedding.content_hash != RoadmapNode.content_hash,
            )
        )
    ).all()
    courses = session.execute(
        select(Course.id, Course.embed_text, Course.content_hash)
        .outerjoin(
            CourseEmbedding,
            and_(
                CourseEmbedding.course_id == Course.id,
                CourseEmbedding.model_id == model.id,
            ),
        )
        .where(
            or_(
                CourseEmbedding.course_id.is_(None),
                CourseEmbedding.content_hash != Course.content_hash,
            )
        )
    ).all()

    for table, key, rows in (
        (NodeEmbedding, "node_id", concepts),
        (CourseEmbedding, "course_id", courses),
    ):
        if not rows:
            continue
        vectors = embedder.embed_documents([text for _, text, _ in rows])
        values = [
            {
                key: row_id,
                "model_id": model.id,
                "embedding": vec,
                "content_hash": content_hash,
            }
            for (row_id, _, content_hash), vec in zip(rows, vectors, strict=True)
        ]
        stmt = insert(table).values(values)
        session.execute(
            stmt.on_conflict_do_update(
                index_elements=[key, "model_id"],
                set_={
                    "embedding": stmt.excluded.embedding,
                    "content_hash": stmt.excluded.content_hash,
                },
            )
        )
    session.commit()
    return {"concepts_embedded": len(concepts), "courses_embedded": len(courses)}


def _load_matrix(session: Session, stmt) -> tuple[np.ndarray, np.ndarray, list]:
    """Rows of (id, vector, *extra) → (ids, L2-normalized matrix, rows)."""
    rows = session.execute(stmt).all()
    ids = np.array([r[0] for r in rows])
    matrix = np.vstack([np.asarray(r[1], dtype=np.float32) for r in rows])
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return ids, matrix, rows


def compute_stats_and_matches(session: Session, model: EmbeddingModel, k: int = TOP_K) -> dict[str, float]:
    concept_ids, concepts, _ = _load_matrix(
        session,
        select(NodeEmbedding.node_id, NodeEmbedding.embedding)
        .join(RoadmapNode, RoadmapNode.id == NodeEmbedding.node_id)
        .where(NodeEmbedding.model_id == model.id, RoadmapNode.type == "concept")
        .order_by(NodeEmbedding.node_id),
    )
    course_ids, courses, course_rows = _load_matrix(
        session,
        select(CourseEmbedding.course_id, CourseEmbedding.embedding, Course.is_active)
        .join(Course, Course.id == CourseEmbedding.course_id)
        .where(CourseEmbedding.model_id == model.id)
        .order_by(CourseEmbedding.course_id),
    )
    active = np.array([r.is_active for r in course_rows])

    similarity = concepts @ courses.T  # concepts x courses, cosine

    # Threshold statistics over all pairs, like the prototype (util.calculate_threshold).
    model.sim_mean = float(similarity.mean())
    model.sim_std = float(similarity.std())
    model.stats_computed_at = datetime.now(UTC)

    # Top-k active courses per concept.
    ranked = np.where(active[None, :], similarity, -np.inf)
    top = np.argsort(-ranked, axis=1)[:, :k]
    session.execute(delete(ConceptCourseMatch).where(ConceptCourseMatch.model_id == model.id))
    values = [
        {
            "model_id": model.id,
            "concept_id": int(concept_ids[i]),
            "course_id": int(course_ids[j]),
            "similarity": float(similarity[i, j]),
            "rank": rank,
        }
        for i in range(len(concept_ids))
        for rank, j in enumerate(top[i], start=1)
        if np.isfinite(ranked[i, j])
    ]
    for start in range(0, len(values), 5000):
        session.execute(insert(ConceptCourseMatch).values(values[start : start + 5000]))
    session.commit()
    return {
        "sim_mean": model.sim_mean,
        "sim_std": model.sim_std,
        "matches": len(values),
    }


def run(session: Session, model_name: str) -> dict[str, float]:
    model = get_model(session, model_name)
    t0 = time.perf_counter()
    result: dict[str, float] = dict(embed_pending(session, model))
    result["embed_seconds"] = round(time.perf_counter() - t0, 2)
    t1 = time.perf_counter()
    result.update(compute_stats_and_matches(session, model))
    result["stats_matches_seconds"] = round(time.perf_counter() - t1, 2)
    return result
