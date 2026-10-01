"""Pure domain logic: no database, no models (ADR-0017). Expected values come from the prototype."""

import pytest

from xcrs.domain import matching
from xcrs.domain.scoring import activation, concept_categories, score_roles
from xcrs.domain.selection import concepts_to_learn, covered_topics, pick_courses
from xcrs.domain.types import Category, CourseCandidate, Phrase, PhraseConceptMatch


def match(concept_id: int, role_id: int, category: Category, text: str = "x") -> PhraseConceptMatch:
    return PhraseConceptMatch(Phrase(text, category), concept_id, role_id, 0.6)


@pytest.mark.parametrize(
    ("x", "expected"),
    # from the comment in the research prototype (on `main`: backend/src/recom.py:recommend_role)
    [(-25, 0.0), (0, 0.67), (10.67, 5.39), (23.645, 43.27), (32.5, 81.76), (50, 99.33), (75, 100.0)],
)
def test_activation_matches_prototype(x, expected):
    assert activation(x) == expected


def test_highest_priority_category_decides_a_concept():
    matches = [match(1, 1, Category.NEUTRAL), match(1, 1, Category.CURIOUS), match(1, 1, Category.LIKED)]
    assert concept_categories(matches) == {1: Category.CURIOUS}


def test_roles_ranked_by_weighted_coverage():
    concept_roles = {1: 1, 2: 1, 3: 2, 4: 3}
    concepts_per_role = {1: 2, 2: 1, 3: 1}
    matches = [match(1, 1, Category.LIKED), match(3, 2, Category.CURIOUS), match(4, 3, Category.DISLIKED)]
    scores = score_roles(matches, concept_roles, concepts_per_role)
    # role 2: 100% curious coverage; role 1: 0.75/2 = 37.5%; role 3: negative → excluded
    assert [s.role_id for s in scores] == [2, 1]


def test_fewer_than_three_roles_are_sorted_by_score_not_id():
    """The prototype ordered by role id in this case (early-stage-first intent); deliberately changed."""
    concept_roles = {1: 1, 2: 9}
    scores = score_roles([match(1, 1, Category.NEUTRAL), match(2, 9, Category.CURIOUS)], concept_roles, {1: 1, 9: 1})
    assert [s.role_id for s in scores] == [9, 1]


def test_concepts_to_learn_beginner_gets_all_remaining_in_order():
    assert concepts_to_learn([10, 11, 12, 13, 14], known={11}) == [10, 12, 13, 14]


def test_concepts_to_learn_advanced_user_gets_the_end_of_the_roadmap():
    # 2 of 5 known (40% ≥ 30%): recommend the last 3 concepts in roadmap order
    assert concepts_to_learn([10, 11, 12, 13, 14], known={10, 13}) == [12, 13, 14]


def test_pick_courses_prefers_courses_serving_many_concepts():
    candidates = {
        1: [CourseCandidate(1, 100, 0.9), CourseCandidate(1, 200, 0.8)],
        2: [CourseCandidate(2, 200, 0.7), CourseCandidate(2, 300, 0.95)],
    }
    picks = pick_courses([1, 2], candidates, penalized=set())
    assert {p.course_id for p in picks} == {100, 200, 300}
    assert next(p for p in picks if p.course_id == 200).concept_ids == [1, 2]


def test_disliked_penalty_halves_ranking_score_but_reports_original_similarity():
    candidates = {1: [CourseCandidate(1, c, s) for c, s in [(100, 0.9), (200, 0.8), (300, 0.7), (400, 0.6)]]}
    picks = pick_courses([1], candidates, penalized={100})  # 0.9 → ranks as 0.45
    assert [p.course_id for p in picks] == [200, 300, 400]
    picks = pick_courses([1], candidates, penalized=set())
    assert picks[0].course_id == 100 and picks[0].similarity == 0.9


def test_covered_topics_needs_forty_percent():
    ancestors = {1: [10], 2: [10], 3: [10], 4: [20], 5: [20]}
    per_topic = {10: 3, 20: 5}
    assert covered_topics([1, 2, 4], ancestors, per_topic) == [10]  # 67% of topic 10, 20% of topic 20


# --- matching (ADR-0022) ---


def _m(phrase, concept_id, similarity):
    return PhraseConceptMatch(phrase, concept_id, 1, similarity)


def test_matches_above_the_threshold_all_count_and_weaker_ones_are_ignored():
    java = Phrase("Java", Category.LIKED)
    selected = matching.select_matches([_m(java, 1, 0.7), _m(java, 2, 0.6), _m(java, 3, 0.45)], 0.5, 0.05)
    assert [m.concept_id for m in selected] == [1, 2]


def test_a_phrase_with_no_match_above_the_threshold_keeps_its_near_best_candidates():
    """Regression: at 2.5 sigma "Python" matched nothing and was silently dropped. Its candidates are the
    "python" concepts of several roadmaps, nearly tied; all of them count, not an arbitrary top-k."""
    python = Phrase("Python", Category.LIKED)
    candidates = [_m(python, 1, 0.485), _m(python, 2, 0.483), _m(python, 3, 0.454), _m(python, 4, 0.41)]
    assert [m.concept_id for m in matching.select_matches(candidates, 0.5, 0.05)] == [1, 2, 3]


def test_fallback_is_per_phrase():
    java, python = Phrase("Java", Category.LIKED), Phrase("Python", Category.CURIOUS)
    selected = matching.select_matches([_m(java, 1, 0.7), _m(python, 2, 0.45), _m(java, 3, 0.45)], 0.5, 0.05)
    assert {(m.phrase.text, m.concept_id) for m in selected} == {("Java", 1), ("Python", 2)}
