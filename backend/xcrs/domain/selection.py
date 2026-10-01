"""Course selection and topic coverage, ported from the research prototype
(on `main`: backend/src/recom.py:recommend_courses, util.top_n_courses_for_concept, util.calculate_topic_coverage)."""

from collections.abc import Iterable, Mapping, Sequence

from xcrs.domain.types import CourseCandidate, CoursePick

# If the user already covers this share of a role's concepts, recommend from the END of the
# roadmap (the more advanced part) instead of from all remaining concepts.
ADVANCED_SHARE = 0.3
COURSES_PER_CONCEPT = 3
COURSES_PER_ROLE = 3
TOPIC_COVERAGE_MIN = 0.4
DISLIKED_PENALTY = 0.5


def concepts_to_learn(role_concepts_in_order: Sequence[int], known: set[int]) -> list[int]:
    """The concepts a role's courses should target, in roadmap (learning) order."""
    if not role_concepts_in_order:
        return []
    known_in_role = known.intersection(role_concepts_in_order)
    if len(known_in_role) / len(role_concepts_in_order) >= ADVANCED_SHARE:
        remaining = len(role_concepts_in_order) - len(known_in_role)
        return list(role_concepts_in_order[len(role_concepts_in_order) - remaining :])
    return [c for c in role_concepts_in_order if c not in known_in_role]


def pick_courses(
    concept_ids: Sequence[int],
    candidates: Mapping[int, Sequence[CourseCandidate]],
    penalized: set[int],
) -> list[CoursePick]:
    """Pick the courses that best serve the target concepts.

    Per concept: rank its candidate courses with the disliked penalty applied (similarity halved
    for courses similar to something the user disliked) and keep the top 3. Across concepts:
    prefer courses picked for many concepts, then by similarity. Keep 3, ordered by similarity.
    Reported similarities are the unpenalized ones, as in the prototype.
    """
    picks: dict[int, CoursePick] = {}
    counts: dict[int, int] = {}
    for concept_id in concept_ids:
        ranked = sorted(
            candidates.get(concept_id, ()),
            key=lambda c: c.similarity * (DISLIKED_PENALTY if c.course_id in penalized else 1.0),
            reverse=True,
        )[:COURSES_PER_CONCEPT]
        for c in ranked:
            pick = picks.setdefault(c.course_id, CoursePick(c.course_id, c.similarity))
            pick.similarity = max(pick.similarity, c.similarity)
            pick.concept_ids.append(concept_id)
            counts[c.course_id] = counts.get(c.course_id, 0) + 1

    best = sorted(picks.values(), key=lambda p: (counts[p.course_id], p.similarity), reverse=True)
    return sorted(best[:COURSES_PER_ROLE], key=lambda p: p.similarity, reverse=True)


def covered_topics(
    concept_ids: Iterable[int],
    concept_ancestors: Mapping[int, Sequence[int]],
    concepts_per_topic: Mapping[int, int],
) -> list[int]:
    """Topics of which the given concepts cover at least 40%, most covered first.

    Used to summarize "what the user knows" at topic level ("Relational Databases") rather than
    listing every single concept.
    """
    covered: dict[int, int] = {}
    for concept_id in set(concept_ids):
        for topic_id in concept_ancestors.get(concept_id, ()):
            covered[topic_id] = covered.get(topic_id, 0) + 1
    shares = {t: n / concepts_per_topic[t] for t, n in covered.items() if concepts_per_topic.get(t)}
    return [t for t, share in sorted(shares.items(), key=lambda kv: kv[1], reverse=True) if share >= TOPIC_COVERAGE_MIN]
