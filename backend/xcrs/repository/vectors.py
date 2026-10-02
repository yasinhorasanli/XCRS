"""Vector queries. The only place that writes pgvector SQL (ADR-0009, ADR-0012)."""

from sqlalchemy import text
from sqlalchemy.orm import Session

from xcrs.db.models import EmbeddingModel


def _vector_literal(vector) -> str:
    return "[" + ",".join(f"{float(x):.8g}" for x in vector) + "]"


def skill_similarities(session: Session, model: EmbeddingModel, vector) -> dict[str, float]:
    """Cosine similarity of one vector to every catalog skill (ADR-0030). An exact scan: 257 skills today;
    add a per-model index and a top-k query when the catalog grows past ~10,000 skills."""
    rows = session.execute(
        text("""
            SELECT s.slug, 1 - (e.embedding <=> CAST(:vector AS vector)) AS similarity
            FROM   catalog.skill_embeddings e
            JOIN   catalog.skills s ON s.id = e.skill_id
            WHERE  e.model_id = :model_id
        """),
        {"vector": _vector_literal(vector), "model_id": model.id},
    ).all()
    return {r.slug: float(r.similarity) for r in rows}
