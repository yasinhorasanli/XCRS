"""Shared helpers for the evaluation scripts in backend/eval/ (not part of the served app)."""

import json
import re
from pathlib import Path

from xcrs.domain.types import Category
from xcrs.explain.base import RoleContext
from xcrs.explain.llm import RoleExplanationOut
from xcrs.explain.prompts import build_payload

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"


def load_profiles() -> list[dict]:
    return json.loads((EVAL_DIR / "profiles.json").read_text())["profiles"]


def to_user_input(profile: dict) -> dict[Category, list[str]]:
    return {c: list(profile.get(c.value, [])) for c in Category}


# --- Grounding checks ----------------------------------------------------------------------------
# Heuristics for the failure modes seen in practice (Story 12). A flag means "read this one", not proof.

CURIOSITY_WORDS = re.compile(r"\bcurio|\binterest(ed|s)? in\b|\bwant(s)? to learn\b|\beager\b|\bexcited about\b", re.I)
META_WORDS = re.compile(r"\bscore|\bsimilarit|\binstruction", re.I)
ATTRIBUTION = re.compile(
    r"\byou(r|'ve|'re)?\b.*\b(know|knowledge|experience|curio|interest|familiar|background|skill|studied|learned)",
    re.I,
)


def words(text: str) -> int:
    return len(text.split())


def grounding_flags(context: RoleContext, out: RoleExplanationOut) -> list[str]:
    payload = build_payload(context)
    flags = []
    texts = [out.role_explanation] + [c.explanation for c in out.courses]
    everything = " ".join(texts)

    if "curious" not in payload and CURIOSITY_WORDS.search(out.role_explanation):
        flags.append("invented_curiosity")

    # A matched concept the learner never wrote, named in a sentence that describes the learner
    # ("your curiosity about ... Argo CD", "you already know ... Argo CD").
    said = " ".join(i.user_said for i in context.known + context.curious).lower()
    for sentence in re.split(r"(?<=[.!?])\s+", out.role_explanation):
        if not ATTRIBUTION.search(sentence):
            continue
        for item in context.known + context.curious:
            concept = item.matched_concept.lower()
            if len(concept) > 2 and concept not in said and re.search(rf"\b{re.escape(concept)}\b", sentence.lower()):
                flags.append(f"misattributed:{item.matched_concept}")

    expected = {c.course_id for c in context.courses}
    returned = {c.course_id for c in out.courses}
    if returned - expected:
        flags.append("unknown_course_ids")
    if expected - returned:
        flags.append("missing_courses")
    if words(out.role_explanation) > 75 or any(words(c.explanation) > 56 for c in out.courses):
        flags.append("too_long")  # limits are 60 / 45 words; 25% slack
    if META_WORDS.search(everything):
        flags.append("mentions_internals")
    return flags
