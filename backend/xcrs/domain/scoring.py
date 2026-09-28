"""Role scoring, ported from the prototype (backend/src/recom.py:recommend_role, util.custom_activation)."""

import math
from collections.abc import Iterable, Mapping

from xcrs.domain.types import Category, PhraseConceptMatch, RoleScore

WEIGHTS = {Category.CURIOUS: 1.0, Category.LIKED: 0.75, Category.NEUTRAL: 0.5, Category.DISLIKED: -0.25}

# When a concept is matched by phrases of several categories, the highest-priority one counts.
PRIORITY = {Category.CURIOUS: 4, Category.LIKED: 3, Category.DISLIKED: 2, Category.NEUTRAL: 1}

# custom_activation(0) = 0.67, i.e. "no matched concepts"; roles must score above it.
MIN_SCORE = 0.68
MAX_ROLES = 3


def activation(x: float) -> float:
    """Shifted, scaled sigmoid mapping weighted coverage (%) to a 0–100 score."""
    return round(100 / (1 + math.exp(-0.2 * (x - 25))), 2)


def concept_categories(matches: Iterable[PhraseConceptMatch]) -> dict[int, Category]:
    """The deciding category for each matched concept."""
    best: dict[int, Category] = {}
    for m in matches:
        current = best.get(m.concept_id)
        if current is None or PRIORITY[m.phrase.category] > PRIORITY[current]:
            best[m.concept_id] = m.phrase.category
    return best


def score_roles(
    matches: Iterable[PhraseConceptMatch],
    concept_roles: Mapping[int, int],
    concepts_per_role: Mapping[int, int],
) -> list[RoleScore]:
    """Top roles by weighted coverage of their roadmap concepts.

    Deliberate change from the prototype: when fewer than 3 roles qualified, the prototype ordered
    them by id (`sorted(filtered_keys, reverse=True)`), intended as an early-stage-first ordering.
    Here roles are always ordered by score. Learning order is kept at concept level instead
    (`roadmap_nodes.sequence`, used by `selection.concepts_to_learn`).
    """
    totals = dict.fromkeys(concepts_per_role, 0.0)
    for concept_id, category in concept_categories(matches).items():
        totals[concept_roles[concept_id]] += WEIGHTS[category]

    scores = [
        RoleScore(role_id, activation(totals[role_id] * 100 / count)) for role_id, count in concepts_per_role.items()
    ]
    qualifying = [s for s in scores if s.score > MIN_SCORE]
    return sorted(qualifying, key=lambda s: s.score, reverse=True)[:MAX_ROLES]
