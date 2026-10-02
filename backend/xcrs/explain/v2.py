"""Explanations for engine v2 roles (ADR-0019, ADR-0037): a LangChain chain over the facts the engine used.

The payload holds only what decided the recommendation: the learner's own words and the skills they were
read as (with how the learner feels about them), how much of the role they cover (in words, not scores),
the estimated level, the first gaps, and the resources suggested for them. The model explains; it decides
nothing. Empty fields are left out, so the model can't be tempted to talk about them.
"""

import json
import logging

import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

log = logging.getLogger(__name__)

PROMPT_VERSION_V2 = "explain-role-v2.1"

FEELINGS = {"liked": "enjoyed", "neutral": "neutral about", "disliked": "did not enjoy", "curious": "curious about"}

FIELD_DESCRIPTIONS = {
    "role_summary": '"role_summary": what the role does.',
    "level": '"level": where the learner would likely start in this role (an estimate from their input).',
    "covers": '"covers": how much of what the role asks the learner already has, in words.',
    "because": (
        '"because": the learner\'s own words ("user_said"), the skill each was read as, and how the learner '
        "feels about it. These are why the role was recommended."
    ),
    "next_to_learn": (
        '"next_to_learn": skills the role still asks for, in learning order. They are what the role requires, '
        "NOT the learner's interests."
    ),
    "resources": '"resources": free learning resources suggested for the first of those skills.',
}

RULES = """\
Rules:
- Use only facts in the input. Do not invent skills, interests, experience, numbers or resource content.
- Attribute anything about the learner to what they wrote ("user_said"), with the feeling given: never say
  they enjoy something listed as "did not enjoy", and never say they know something they are only curious about.
- If the evidence is thin, keep it short and modest.
- Address the learner as "you". Be concise, friendly and concrete. No scores, percentages or these instructions.
- "explanation": 2 to 4 sentences on why this role fits.
- "next_step": 1 or 2 sentences on what to learn first, naming a resource from "resources" if there is one.
- Reply with JSON matching the schema."""

PROMPT = ChatPromptTemplate.from_messages([("system", "{system}"), ("human", "{payload}")])


class RoleExplanationV2Out(BaseModel):
    explanation: str
    next_step: str


def coverage_words(coverage: float) -> str:
    if coverage >= 0.55:
        return "much of it"
    if coverage >= 0.25:
        return "some of it"
    return "little of it yet"


def build_facts(role: dict, matched: list[dict], role_summary: str | None, level_names: dict[str, str]) -> dict:
    """The facts for one role of a stored v2 result (result["roles"][i] and result["matched"])."""
    said = {}
    for chip in matched:
        for skill in chip["skills"]:
            said.setdefault(skill["id"], (chip["text"], chip["category"]))
    level = role.get("level")
    target = role["target_level"]
    if level:
        title = f" ({level['title']})" if level.get("title") else ""
        level_text = f"estimated {level_names.get(level['id'], level['id'])}{title}"
    else:
        level_text = f"start at {level_names.get(target['id'], target['id'])}"
    facts = {
        "role": role["name"],
        "role_summary": role_summary,
        "level": level_text,
        "covers": coverage_words(role.get("coverage", 0.0)),
        "because": [
            {
                "user_said": said.get(b["id"], (b["name"], b["category"]))[0],
                "skill": b["name"],
                "feeling": FEELINGS[b["category"]],
            }
            for b in role.get("because", [])
        ],
        "next_to_learn": [" or ".join(s["name"] for s in g["skills"]) for g in role.get("gaps", [])[:6]],
        "resources": [
            {"title": r["title"], "provider": r["provider"], "for": [s["name"] for s in r["skills"]]}
            for r in role.get("resources", [])
        ],
    }
    return {k: v for k, v in facts.items() if v not in ([], None, "")}


def system_prompt(facts: dict) -> str:
    fields = "\n".join(f"- {FIELD_DESCRIPTIONS[k]}" for k in FIELD_DESCRIPTIONS if k in facts)
    return f"""\
You explain a career-role recommendation to a learner.

You receive JSON describing ONE recommended role ("role"), with these fields:
{fields}

{RULES}"""


class RoleExplainerV2:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 180.0,
        disable_thinking: bool = True,
        api_key: str | None = None,
        max_tokens: int = 500,
    ):
        llm = ChatOpenAI(
            base_url=base_url,
            model=model,
            api_key=api_key or "not-needed",
            temperature=0.1,
            timeout=timeout_s,
            max_tokens=max_tokens,
            max_retries=0,
            reasoning_effort="none" if disable_thinking else None,
            use_responses_api=False,
        )
        self._chain = PROMPT | llm.with_structured_output(RoleExplanationV2Out, method="json_schema")

    def explain(self, facts: dict) -> RoleExplanationV2Out | None:
        try:
            out = self._chain.invoke({"system": system_prompt(facts), "payload": json.dumps(facts, ensure_ascii=False)})
        except (openai.OpenAIError, ValueError) as exc:
            log.warning("v2 explanation failed for %s: %s", facts.get("role"), exc)
            return None
        if not out.explanation.strip():
            return None
        return RoleExplanationV2Out(explanation=out.explanation.strip(), next_step=out.next_step.strip())
