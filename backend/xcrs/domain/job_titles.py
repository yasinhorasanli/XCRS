"""Job titles for a recommended role (ADR-0044). Pure; no I/O.

A role has market titles (`also_called` in roles.yaml: "Backend Developer", "API Developer") and title skills:
the languages and frameworks job ads put in the title. The learner's strongest title skills make a title for
them: a language goes in front ("Java Backend Engineer"), a framework or platform in brackets ("Backend Engineer
(Spring Boot)"), both together when they have both ("Python Backend Engineer (Django)").

A skill counts when the learner enjoyed it (any rating) or is neutral about it at a rating of 2 or more: titles
are what someone would apply for, so not skills they are only curious about or didn't enjoy.
"""

from collections.abc import Iterable

from xcrs.domain.role_scoring import Category, Mention, RoleSnapshot

MIN_NEUTRAL_RATING = 2


def _strength(m: Mention, order: int) -> tuple[int, bool, int]:
    """Rating first (unrated counts as 2), then enjoyed over neutral, then the order roles.yaml lists them in."""
    return (m.proficiency or 2, m.category == Category.LIKED, -order)


def title_for(
    role: RoleSnapshot, mentions: Iterable[Mention], names: dict[str, str], languages: frozenset[str]
) -> str | None:
    """The learner's own title for the role, or None when no title skill counts."""
    strength = {}
    for m in mentions:
        if m.skill not in role.title_skills:
            continue
        if m.category == Category.LIKED or (
            m.category == Category.NEUTRAL and (m.proficiency or 0) >= MIN_NEUTRAL_RATING
        ):
            strength[m.skill] = _strength(m, role.title_skills.index(m.skill))
    language = max((s for s in strength if s in languages), key=strength.__getitem__, default=None)
    other = max((s for s in strength if s not in languages), key=strength.__getitem__, default=None)
    name = role.name
    if language:
        name = f"{names[language]} {name}"
    if other:
        name = f"{name} ({names[other]})"
    return name if language or other else None


def market_titles(role: RoleSnapshot) -> list[str]:
    """The role's own name and the other titles job ads use for it."""
    return [role.name, *role.market_titles]
