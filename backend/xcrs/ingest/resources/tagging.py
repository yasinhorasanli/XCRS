"""Tag learning resources with the skills they teach (ADR-0033), reusing skill matching (ADR-0030): the LLM
picks skills from the catalog for the resource's title and description, and each pick is kept only if the
embedding of that text is similar enough to the skill. Curated resources are tagged by hand and skipped; a
resource with only a reviewed tag (the skill a YouTube playlist was approved for) still gets the LLM's tags.

Sections (a long video's chapters, a playlist's videos; ADR-0046) are tagged without the LLM: a section may only
take skills its resource teaches, the one its title is most similar to (and any within a hair of it), if similar
enough. Thousands of short titles, one embedding each, and the resource's tags keep them on topic."""

import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import openai
from sqlalchemy import select
from sqlalchemy.orm import Session

from xcrs.db.models import LearningResource, ResourceSection, ResourceSectionSkill, ResourceSkill, Skill
from xcrs.services.skill_matching import CONFIRM_FLOOR, Picker

log = logging.getLogger(__name__)


def resource_text(title: str, description: str | None) -> str:
    return f"{title}: {description}" if description else title


def untagged(session: Session, limit: int | None = None, source: str | None = None) -> list[LearningResource]:
    stmt = select(LearningResource).where(
        LearningResource.source != "curated", LearningResource.is_active, LearningResource.tagged_at.is_(None)
    )
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
        have = set(session.scalars(select(ResourceSkill.skill_id).where(ResourceSkill.resource_id == resource.id)))
        kept = [s for s in picked if s in skill_ids and skill_ids[s] not in have and sims.get(s, 0.0) >= CONFIRM_FLOOR]
        stats["resources"] += 1
        resource.tagged_at = datetime.now(UTC)  # processed, even if nothing matched: not retried daily
        stats["no_skill"] += not kept and not have
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


SECTION_FLOOR = 0.44  # a section's title must be at least this similar to the skill (FALLBACK_FLOOR, ADR-0030)
SECTION_MARGIN = 0.03  # other skills of the resource kept with the best one when this close to it
SECTION_BATCH = 64


def section_skills(sims: dict[str, float], candidates: set[str]) -> list[str]:
    """The resource's skills a section's title is about: the most similar one if above the floor, and any
    other within the margin of it ("Docker Compose" in a DevOps course: Docker, not Kubernetes)."""
    ranked = sorted(((sims.get(s, 0.0), s) for s in candidates), reverse=True)
    if not ranked or ranked[0][0] < SECTION_FLOOR:
        return []
    best = ranked[0][0]
    return [s for sim, s in ranked if sim >= SECTION_FLOOR and sim >= best - SECTION_MARGIN]


EPISODE_MARK = re.compile(
    r"^\W*(?:#\s*\d+|\d{1,3}(?=\s*[-–—:|.)])|(?:part|session|lecture|lesson|day|episode|ep|video|chapter|module|class|tutorial)\s*[-:#.]?\s*\d+)"
    r"\s*[-–—:|.]?\s*",
    re.IGNORECASE,
)


# Chapters about the course itself, not a topic ("Course structure" is not Data structures).
GENERIC_SECTION = re.compile(
    r"^(?:(?:course|section|module|chapter|video|lesson|class)\s+)?(?:intro(?:duction)?|outro|overview|welcome|"
    r"conclusion|summary|recap|wrap[- ]?up|final (?:thoughts|words)|next steps|what'?s next|thank you|thanks|q ?& ?a|"
    r"questions|bonus|setup|set ?up|installation|install(?:ing)?|requirements|prerequisites|getting started|"
    r"structure|outline|agenda|resources|about (?:me|this course|the course|the instructor)|"
    r"course (?:structure|outline|overview|contents|resources)|sponsor(?:ed)?(?: segment)?|ad)\W*$",
    re.IGNORECASE,
)


def distinct_titles(titles: list[str]) -> list[str]:
    """What each section's title says beyond its series: the words every title shares at the start or end
    ("Full React Tutorial #16 - Using JSON Server" -> "Using JSON Server") and the episode mark go, so the
    series name doesn't drown the topic in the embedding. A title left empty keeps its full text."""
    split = [t.split() for t in titles]
    if len(split) < 2:
        return titles

    def shared(lists: list[list[str]]) -> int:
        n = 0
        while all(len(w) > n for w in lists) and len({w[n].lower() for w in lists}) == 1:
            n += 1
        return n

    head = shared(split)
    tail = shared([w[::-1] for w in split])
    out = []
    for title, words in zip(titles, split, strict=True):
        core = " ".join(words[head : len(words) - tail if tail else None])
        core = EPISODE_MARK.sub("", core).strip(" -–—:|.")
        out.append(core if len(core) >= 3 else title)
    return out


def tag_sections(
    session: Session, similarities: Callable[[list[str]], list[dict[str, float]]], limit: int | None = None
) -> dict:
    """Tag untagged sections of active resources that already have skills. `similarities(texts)` gives each
    text's similarity to every skill. The caller commits."""
    slugs = dict(session.execute(select(Skill.id, Skill.slug)).all())
    teaches: dict[int, set[str]] = {}
    for resource_id, skill_id in session.execute(
        select(ResourceSkill.resource_id, ResourceSkill.skill_id).where(ResourceSkill.relation == "teaches")
    ):
        teaches.setdefault(resource_id, set()).add(slugs[skill_id])
    sections = list(
        session.scalars(
            select(ResourceSection)
            .join(LearningResource, LearningResource.id == ResourceSection.resource_id)
            .where(
                ResourceSection.tagged_at.is_(None),
                LearningResource.is_active,
                ResourceSection.resource_id.in_(list(teaches)),
            )
            .order_by(ResourceSection.resource_id, ResourceSection.position)
            .limit(limit)
        )
    )
    ids = {v: k for k, v in slugs.items()}
    siblings: dict[int, list[str]] = {}
    for resource_id, title in session.execute(
        select(ResourceSection.resource_id, ResourceSection.title)
        .where(ResourceSection.resource_id.in_({s.resource_id for s in sections}))
        .order_by(ResourceSection.resource_id, ResourceSection.position)
    ):
        siblings.setdefault(resource_id, []).append(title)
    core = {rid: dict(zip(titles, distinct_titles(titles), strict=True)) for rid, titles in siblings.items()}
    stats = {"sections": 0, "section_tags": 0}
    for i in range(0, len(sections), SECTION_BATCH):
        batch = sections[i : i + SECTION_BATCH]
        texts = [core[s.resource_id].get(s.title, s.title) for s in batch]
        for section, text, sims in zip(batch, texts, similarities(texts), strict=True):
            for skill in (
                [] if GENERIC_SECTION.match(text.strip()) else section_skills(sims, teaches[section.resource_id])
            ):
                session.add(
                    ResourceSectionSkill(section_id=section.id, skill_id=ids[skill], confidence=round(sims[skill], 3))
                )
                stats["section_tags"] += 1
            section.tagged_at = datetime.now(UTC)
            stats["sections"] += 1
        session.flush()
    return stats
