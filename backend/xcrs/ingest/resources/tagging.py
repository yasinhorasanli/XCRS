"""Tag learning resources with the skills they teach (ADR-0033), reusing skill matching (ADR-0030): the LLM
picks skills from the catalog for the resource's title and description, and each pick is kept only if the
embedding of that text is similar enough to the skill. Curated resources are tagged by hand and skipped."""

import logging
from collections.abc import Callable

import httpx
import openai
from sqlalchemy import and_, exists, select
from sqlalchemy.orm import Session

from xcrs.db.models import LearningResource, ResourceSkill, Skill
from xcrs.services.skill_matching import CONFIRM_FLOOR, Picker

log = logging.getLogger(__name__)


def resource_text(title: str, description: str | None) -> str:
    return f"{title}: {description}" if description else title


def untagged(session: Session, limit: int | None = None, source: str | None = None) -> list[LearningResource]:
    tagged = exists().where(and_(ResourceSkill.resource_id == LearningResource.id))
    stmt = select(LearningResource).where(LearningResource.source != "curated", LearningResource.is_active, ~tagged)
    if source:
        stmt = stmt.where(LearningResource.source == source)
    return list(session.scalars(stmt.order_by(LearningResource.id).limit(limit)))


def tag_untagged(
    session: Session,
    picker: Picker,
    similarities: Callable[[str], dict[str, float]],
    limit: int | None = None,
    source: str | None = None,
) -> dict:
    """Tag resources that have no skills yet. `similarities(text)` gives the text's similarity to every skill.
    The caller commits."""
    skill_ids = dict(session.execute(select(Skill.slug, Skill.id)).all())
    stats = {"resources": 0, "tags": 0, "no_skill": 0, "failed": 0}
    for resource in untagged(session, limit, source):
        text = resource_text(resource.title, resource.description)
        try:
            picked = picker.pick(text)
        except (openai.OpenAIError, httpx.HTTPError, ValueError) as exc:
            log.warning("tagging failed for %s: %s", resource.url, exc)
            stats["failed"] += 1
            continue
        sims = similarities(text)
        kept = [s for s in picked if s in skill_ids and sims.get(s, 0.0) >= CONFIRM_FLOOR]
        stats["resources"] += 1
        stats["no_skill"] += not kept
        for skill in kept:
            session.add(
                ResourceSkill(
                    resource_id=resource.id,
                    skill_id=skill_ids[skill],
                    relation="teaches",
                    confidence=round(sims[skill], 3),
                    tagged_by="llm",
                )
            )
            stats["tags"] += 1
        session.flush()
    return stats
