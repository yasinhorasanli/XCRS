"""LLM explanations as a LangChain chain: prompt template | chat model with structured output (ADR-0019).

Replaces the prototype's free-text arrays matched to items by position: the model gets the algorithm's
actual reasons as JSON and answers in a schema whose courses are matched by course_id.
"""

import json
import logging

import openai
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from xcrs.explain.base import RoleContext, RoleExplanation
from xcrs.explain.prompts import EXPLAIN_PROMPT, PROMPT_VERSION, build_payload, build_system_prompt

log = logging.getLogger(__name__)


class CourseExplanationOut(BaseModel):
    course_id: int
    explanation: str


class RoleExplanationOut(BaseModel):
    """The answer schema. The model only explains; everything it explains was decided by the algorithm."""

    role_explanation: str
    courses: list[CourseExplanationOut]


def to_explanation(out: RoleExplanationOut, context: RoleContext) -> RoleExplanation:
    """Keep only well-formed parts: non-empty text, and courses that were actually recommended."""
    known_ids = {c.course_id for c in context.courses}
    explanation = RoleExplanation(role_explanation=out.role_explanation.strip() or None, prompt_version=PROMPT_VERSION)
    for item in out.courses:
        text = item.explanation.strip()
        if item.course_id in known_ids and text:
            explanation.course_explanations[item.course_id] = text
    return explanation


class LangChainExplainer:
    """Explains one role per call through any OpenAI-compatible chat server (Ollama now; vLLM or hosted later)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 120.0,
        disable_thinking: bool = True,
        api_key: str | None = None,
    ):
        llm = ChatOpenAI(
            base_url=base_url,
            model=model,
            api_key=api_key or "not-needed",  # local servers ignore it; the client requires a value
            temperature=0.1,
            timeout=timeout_s,
            max_retries=0,  # a retry doubles a slow CPU generation; failures degrade instead
            reasoning_effort="none" if disable_thinking else None,  # reasoning models: answer directly
            use_responses_api=False,  # plain chat completions, which every OpenAI-compatible server has
        )
        self._chain = EXPLAIN_PROMPT | llm.with_structured_output(RoleExplanationOut, method="json_schema")

    def explain(self, context: RoleContext) -> RoleExplanation:
        payload = build_payload(context)
        try:
            out = self._chain.invoke(
                {"system": build_system_prompt(payload), "payload": json.dumps(payload, ensure_ascii=False)}
            )
            return to_explanation(out, context)
        except (openai.APIError, ValueError) as exc:  # ValueError covers invalid JSON and schema mismatches
            # Explanations enrich a recommendation; a failure must not lose the recommendation.
            log.warning("explanation failed for role %s: %s", context.role, exc)
            return RoleExplanation()
