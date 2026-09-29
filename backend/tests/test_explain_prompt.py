"""Explanation prompt and chain: no real LLM call."""

import openai
from langchain_core.runnables import RunnableLambda

from xcrs.explain.base import CourseContext, KnownItem, RoleContext
from xcrs.explain.llm import CourseExplanationOut, LangChainExplainer, RoleExplanationOut, to_explanation
from xcrs.explain.prompts import EXPLAIN_PROMPT, PROMPT_VERSION, build_payload, build_system_prompt


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


def explainer_returning(fn) -> LangChainExplainer:
    """A real explainer whose chain is replaced, so the test never reaches a model server."""
    explainer = LangChainExplainer(base_url="http://localhost:1", model="test")
    explainer._chain = RunnableLambda(fn)
    return explainer


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


def test_template_passes_json_payload_through_unchanged():
    """The payload is JSON; its braces must not be read as template placeholders."""
    payload = '{"role": "Backend Developer", "known": [{"user_said": "Java"}]}'
    messages = EXPLAIN_PROMPT.format_messages(system="sys {not a placeholder}", payload=payload)
    assert [m.content for m in messages] == ["sys {not a placeholder}", payload]


def test_courses_are_matched_by_id_and_unknown_ids_dropped():
    out = RoleExplanationOut(
        role_explanation=" Fits. ",
        courses=[
            CourseExplanationOut(course_id=7, explanation="Good."),
            CourseExplanationOut(course_id=99, explanation="?"),
        ],
    )
    result = to_explanation(out, context(curious=[]))
    assert result.role_explanation == "Fits."
    assert result.course_explanations == {7: "Good."}
    assert result.prompt_version == PROMPT_VERSION


def test_chain_receives_the_dynamic_system_prompt_and_json_payload():
    seen = {}

    def fake_chain(inputs: dict) -> RoleExplanationOut:
        seen.update(inputs)
        return RoleExplanationOut(role_explanation="Fits.", courses=[])

    result = explainer_returning(fake_chain).explain(context(curious=[]))
    assert result.role_explanation == "Fits."
    assert "curious" not in seen["system"]
    assert '"user_said": "SQL"' in seen["payload"]


def test_model_failure_degrades_to_no_explanation():
    def failing_chain(_inputs):
        raise openai.APIConnectionError(request=None)

    result = explainer_returning(failing_chain).explain(context(curious=[]))
    assert result.role_explanation is None
    assert result.course_explanations == {}
    assert result.prompt_version is None


def test_invalid_model_output_degrades_to_no_explanation():
    def invalid_output(_inputs):
        raise ValueError("model returned JSON that doesn't match the schema")

    assert explainer_returning(invalid_output).explain(context(curious=[])).role_explanation is None
