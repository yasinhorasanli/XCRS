"""Embed catalog skills for a registered model (ADR-0030): "name: description", only new or changed ones."""

import hashlib

from sqlalchemy import and_, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from xcrs.db.models import EmbeddingModel, Skill, SkillEmbedding
from xcrs.embeddings import embedder_for


def skill_text(name: str, description: str) -> str:
    return f"{name}: {description}"


def embed_skills(session: Session, model: EmbeddingModel) -> int:
    """Returns how many skills were (re-)embedded. The caller commits."""
    skills = session.execute(
        select(Skill.id, Skill.name, Skill.description, SkillEmbedding.content_hash).outerjoin(
            SkillEmbedding, and_(SkillEmbedding.skill_id == Skill.id, SkillEmbedding.model_id == model.id)
        )
    ).all()
    pending = []
    for skill_id, name, description, stored_hash in skills:
        text = skill_text(name, description)
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        if stored_hash != content_hash:
            pending.append((skill_id, text, content_hash))
    if not pending:
        return 0
    vectors = embedder_for(model).embed_documents([text for _, text, _ in pending])
    stmt = insert(SkillEmbedding).values(
        [
            {"skill_id": sid, "model_id": model.id, "embedding": vec, "content_hash": h}
            for (sid, _, h), vec in zip(pending, vectors, strict=True)
        ]
    )
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=["skill_id", "model_id"],
            set_={"embedding": stmt.excluded.embedding, "content_hash": stmt.excluded.content_hash},
            where=or_(SkillEmbedding.content_hash != stmt.excluded.content_hash),
        )
    )
    return len(pending)
