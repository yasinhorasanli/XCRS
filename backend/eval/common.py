"""Shared helpers for the evaluation scripts in backend/eval/ (not part of the served app)."""

import re
from pathlib import Path

from xcrs.explain.v2 import RoleExplanationV2Out

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"


# --- Grounding checks for v2 explanations (ADR-0037) ---------------------------------------------
# Heuristics for the failure modes seen in practice. A flag means "read this one", not proof.

CURIOSITY_WORDS = re.compile(r"\bcurio|\binterest(ed|s)? in\b|\bwant(s)? to learn\b|\beager\b|\bexcited about\b", re.I)
ENJOY_WORDS = re.compile(r"\benjoy|\blove|\bpassion|\blike(s|d)?\b", re.I)
META_WORDS = re.compile(r"\bscore|\bsimilarit|\binstruction|\d+\s*%|\bpercent", re.I)


def words(text: str) -> int:
    return len(text.split())


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def grounding_flags(facts: dict, out: RoleExplanationV2Out) -> list[str]:
    flags = []
    feelings = {b["feeling"] for b in facts.get("because", [])}
    if "curious about" not in feelings and CURIOSITY_WORDS.search(out.explanation):
        flags.append("invented_curiosity")
    # A skill the learner did not enjoy, named in a sentence that says they enjoy it.
    for b in facts.get("because", []):
        if b["feeling"] != "did not enjoy":
            continue
        for s in sentences(out.explanation):
            if ENJOY_WORDS.search(s) and not re.search(r"\bnot\b|n't", s) and b["user_said"].lower() in s.lower():
                flags.append(f"disliked_as_enjoyed:{b['user_said']}")
    # A resource is named in quotes that isn't among the suggested ones.
    titles = {r["title"].lower() for r in facts.get("resources", [])}
    for quoted in re.findall(r"[\"“]([^\"”]{6,})[\"”]", out.next_step):
        if titles and not any(quoted.lower() in t or t in quoted.lower() for t in titles):
            flags.append("unknown_resource")
    if len(sentences(out.explanation)) > 5 or words(out.explanation) > 110 or words(out.next_step) > 60:
        flags.append("too_long")  # limits are 4 sentences / 2 sentences, with slack
    if META_WORDS.search(f"{out.explanation} {out.next_step}"):
        flags.append("mentions_internals")
    return flags
