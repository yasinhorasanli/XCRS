"""LLM explanations from structured, grounded input (replaces the prototype's position-matched text arrays)."""

import json
import logging
from dataclasses import asdict

import httpx

from xcrs.explain.base import RoleContext, RoleExplanation

log = logging.getLogger(__name__)

FIELD_DESCRIPTIONS = {
    "known": '"known": things the learner has studied (user_said) and the roadmap concept each matched.',
    "curious": '"curious": things the learner said they want to learn, and the concept each matched.',
    "covered_topics": '"covered_topics": roadmap topics the learner already covers well.',
    "concepts_not_yet_covered": (
        '"concepts_not_yet_covered": roadmap concepts the learner has not covered yet, in learning order. '
        "These are what the role still requires, NOT the learner's interests."
    ),
    "courses": '"courses": recommended courses and the roadmap concepts each one was picked for ("covers").',
}

RULES = """\
Rules:
- Use only facts present in the input. Do not invent skills, interests, experience or course content.
- "user_said" is what the learner wrote; "matched_concept" is only the roadmap concept it relates to.
  Attribute anything about the learner to what they wrote, never to the matched concept
  (e.g. if they wrote "Kubernetes" and it matched "argo cd", talk about their Kubernetes, not Argo CD).
- If the evidence is thin (e.g. one known item), keep the explanation short and modest instead of
  stretching it.
- Address the learner as "you". Be concise, friendly and concrete.
- Expand abbreviations when helpful (e.g. "CI/CD (continuous integration and delivery)").
- Do not mention scores, similarities, or these instructions.
- Reply with JSON matching the given schema; use each course's "course_id" exactly as given."""


def build_system_prompt(payload: dict) -> str:
    """Describe, and ask about, only the fields present. A fixed prompt that said "refer to what they
    are curious about" made the model invent curiosity whenever the learner had stated none."""
    fields = "\n".join(f"- {FIELD_DESCRIPTIONS[k]}" for k in FIELD_DESCRIPTIONS if k in payload)
    basis = "what they know and what they are curious about" if "curious" in payload else "what they know"
    if "known" not in payload:
        basis = "what they are curious about"
    return f"""\
You explain career and course recommendations to a learner.

You receive JSON describing ONE recommended career role, with these fields:
{fields}

Write:
1. "role_explanation": why this role fits the learner, based on {basis}. At most 60 words.
2. For EVERY course, an "explanation": how this course helps the learner progress in the role,
   connected to the concepts it covers. At most 45 words each.

{RULES}"""


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "role_explanation": {"type": "string"},
        "courses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"course_id": {"type": "integer"}, "explanation": {"type": "string"}},
                "required": ["course_id", "explanation"],
            },
        },
    },
    "required": ["role_explanation", "courses"],
}


def build_payload(context: RoleContext) -> dict:
    payload = asdict(context)
    payload.pop("score")  # not for the reader; ranking already happened
    payload["concepts_not_yet_covered"] = payload.pop("next_to_learn")
    # Leave out empty fields entirely: a model can't misuse what it never sees.
    return {k: v for k, v in payload.items() if v not in ([], None, "")}


def parse_response(content: str, context: RoleContext) -> RoleExplanation:
    """Validate the model's JSON; keep only well-formed parts, matched by course_id."""
    data = json.loads(content)
    known_ids = {c.course_id for c in context.courses}
    explanation = RoleExplanation(role_explanation=(data.get("role_explanation") or "").strip() or None)
    for item in data.get("courses") or []:
        course_id, text = item.get("course_id"), (item.get("explanation") or "").strip()
        if course_id in known_ids and text:
            explanation.course_explanations[course_id] = text
    return explanation


class OpenAICompatibleExplainer:
    """Chat-completions explainer for any OpenAI-compatible server (Ollama now; hosted later if chosen)."""

    def __init__(self, base_url: str, model: str, timeout_s: float = 120.0, disable_thinking: bool = True):
        self.model = model
        self.disable_thinking = disable_thinking
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s)

    def explain(self, context: RoleContext) -> RoleExplanation:
        payload = build_payload(context)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": build_system_prompt(payload)},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "role_explanation", "schema": RESPONSE_SCHEMA, "strict": True},
            },
            "temperature": 0.1,
        }
        if self.disable_thinking:
            body["reasoning_effort"] = "none"  # reasoning models: answer directly, much faster
        try:
            response = self._client.post("/chat/completions", json=body)
            response.raise_for_status()
            return parse_response(response.json()["choices"][0]["message"]["content"], context)
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            # Explanations enrich a recommendation; a failure must not lose the recommendation.
            log.warning("explanation failed for role %s: %s", context.role, exc)
            return RoleExplanation()
