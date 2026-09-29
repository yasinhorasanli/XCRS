"""Explanation prompt (ADR-0019): built only from the facts the algorithm used, as a LangChain template."""

from dataclasses import asdict

from langchain_core.prompts import ChatPromptTemplate

from xcrs.explain.base import RoleContext

# Stored with every explanation (like ALGORITHM_VERSION), so a past explanation can be traced to its prompt.
# Bump it whenever the wording, the rules or the payload fields change.
PROMPT_VERSION = "explain-role-v1"

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

# Both messages are variables, not template text: the system prompt is assembled per role (below) and the
# payload is JSON, whose braces would otherwise be parsed as template placeholders.
EXPLAIN_PROMPT = ChatPromptTemplate.from_messages([("system", "{system}"), ("human", "{payload}")])


def build_payload(context: RoleContext) -> dict:
    payload = asdict(context)
    payload.pop("score")  # not for the reader; ranking already happened
    payload["concepts_not_yet_covered"] = payload.pop("next_to_learn")
    # Leave out empty fields entirely: a model can't misuse what it never sees.
    return {k: v for k, v in payload.items() if v not in ([], None, "")}


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
