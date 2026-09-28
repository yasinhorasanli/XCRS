"""Prompt construction for explanations: pure functions, no LLM call."""

from xcrs.explain.base import CourseContext, KnownItem, RoleContext
from xcrs.explain.llm import build_payload, build_system_prompt, parse_response


def context(curious: list[KnownItem]) -> RoleContext:
    return RoleContext(
        role="AI Data Scientist",
        score=1.6,
        known=[KnownItem("SQL", "sql", "liked")],
        curious=curious,
        covered_topics=[],
        next_to_learn=["statistics clt"],
        courses=[CourseContext(7, "Stats 101", None, None, ["statistics clt"])],
    )


def test_empty_fields_and_score_are_not_sent():
    payload = build_payload(context(curious=[]))
    assert set(payload) == {"role", "known", "concepts_not_yet_covered", "courses"}


def test_prompt_never_asks_about_curiosity_the_learner_did_not_state():
    """Regression: a fixed prompt made the model invent "you are curious about ..." (5/5 runs)."""
    prompt = build_system_prompt(build_payload(context(curious=[])))
    assert "curious" not in prompt


def test_prompt_asks_about_curiosity_when_stated():
    prompt = build_system_prompt(build_payload(context(curious=[KnownItem("Docker", "docker", "curious")])))
    assert "what they know and what they are curious about" in prompt


def test_parse_response_matches_courses_by_id_and_drops_unknown_ids():
    content = '{"role_explanation": "Fits.", "courses": [{"course_id": 7, "explanation": "Good."}, {"course_id": 99, "explanation": "?"}]}'
    result = parse_response(content, context(curious=[]))
    assert result.role_explanation == "Fits."
    assert result.course_explanations == {7: "Good."}
