"""Knowledge-unit suggestions for the input page (ADR-0023). Pure functions."""

from collections.abc import Iterable
from dataclasses import dataclass, field

from xcrs.domain.labels import display_label


@dataclass
class KnowledgeUnit:
    label: str
    source: str  # "curated" (starter list) or "roadmap" (a roadmap concept)
    roles: list[str] = field(default_factory=list)  # roadmaps that contain it


@dataclass(frozen=True)
class RelatedUnit:
    label: str
    because: str  # the learner's phrase it is close to
    similarity: float


def merge_units(curated: Iterable[str], concepts: Iterable[tuple[str, str]]) -> list[KnowledgeUnit]:
    """Curated labels plus roadmap concepts (name, role), one unit per case-insensitive label.
    A curated label keeps its spelling and gains the roles of a concept with the same name."""
    units: dict[str, KnowledgeUnit] = {}
    for label in curated:
        units.setdefault(label.lower(), KnowledgeUnit(label, "curated"))
    for name, role in concepts:
        label = display_label(name)
        unit = units.setdefault(label.lower(), KnowledgeUnit(label, "roadmap"))
        if role not in unit.roles:
            unit.roles.append(role)
    return list(units.values())


def search(units: Iterable[KnowledgeUnit], query: str, limit: int) -> list[KnowledgeUnit]:
    """Exact match first, then prefix, then word prefix, then substring; curated before roadmap-only,
    then units found in more roadmaps."""
    q = " ".join(query.lower().split())
    if not q:
        return []

    def quality(label: str) -> int | None:
        text = label.lower()
        if text == q:
            return 0
        if text.startswith(q):
            return 1
        if any(word.startswith(q) for word in text.replace("/", " ").replace("-", " ").split()):
            return 2
        return 3 if q in text else None

    ranked = [(quality(u.label), u) for u in units]
    hits = [(r, u) for r, u in ranked if r is not None]
    hits.sort(key=lambda h: (h[0], h[1].source != "curated", -len(h[1].roles), len(h[1].label), h[1].label))
    return [u for _, u in hits[:limit]]


def related(
    hits: Iterable[tuple[str, str, float]], entered: Iterable[str], limit: int, min_similarity: float
) -> list[RelatedUnit]:
    """Suggestions from (phrase, concept name, similarity) hits: the best hit per label, leaving out
    what the learner already entered and anything too far from it."""
    taken = {e.strip().lower() for e in entered}
    best: dict[str, RelatedUnit] = {}
    for phrase, name, similarity in hits:
        label = display_label(name)
        key = label.lower()
        if key in taken or similarity < min_similarity:
            continue
        if key not in best or similarity > best[key].similarity:
            best[key] = RelatedUnit(label, phrase, round(similarity, 3))
    return sorted(best.values(), key=lambda u: -u.similarity)[:limit]
