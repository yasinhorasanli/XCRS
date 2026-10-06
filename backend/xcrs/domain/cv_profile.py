"""Pure rules for a CV import (ADR-0045): catalog names found in the text, whether an LLM's evidence is really in
the text, how long each skill was used, the suggested level, and the experience band (ADR-0041). No I/O.
"""

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from xcrs.domain.skill_matching import LexicalIndex, normalize

Month = tuple[int, int]  # (year, month)
JobKind = Literal["tech", "internship", "other"]

# Levels from years of use: never expert (4) automatically; the learner decides.
ADVANCED_MONTHS = 60
WORKING_MONTHS = 24
STALE_MONTHS = 60  # last used longer ago than this: one level lower
STUDENT_MAX_MONTHS = 6  # someone still studying with less tech work than this is a student
BANDS = [(24, "0-2"), (60, "2-5"), (120, "5-10")]  # tech months below the bound → band; otherwise "10+"

# The scan only adds what is unmistakable. Names that are ordinary words are left to the LLM, or count only
# written as a proper noun (capitalized, not at the start of a sentence: "React quickly to feedback" is not React).
NEVER_SCANNED = {"c", "r", "go", "dx", "dom", "flow", "manual", "performance", "identity"}
PROPER_NOUN_ONLY = {
    "react": "React",
    "spark": "Spark",
    "swift": "Swift",
    "rust": "Rust",
    "unity": "Unity",
    "dart": "Dart",
    "chef": "Chef",
    "puppet": "Puppet",
    "transformers": "Transformers",
    "flutter": "Flutter",
}
MAX_KEY_WORDS = 4

_PRESENT = {"", "present", "now", "current", "today", "halen", "günümüz", "gunumuz", "devam ediyor", "hâlâ"}


@dataclass(frozen=True)
class Job:
    start: Month | None
    end: Month | None  # None: current
    kind: JobKind = "tech"


@dataclass(frozen=True)
class Education:
    start_year: int | None
    end_year: int | None  # expected end while still studying


def parse_month(value: str | int | None, end: bool = False) -> Month | None:
    """'2022-03', '2022-3', '03/2022', '2022' (January, or December for an end). None if it can't be read."""
    if value is None:
        return None
    text = str(value).strip()
    if m := re.fullmatch(r"(\d{4})[-/.](\d{1,2})", text):
        year, month = int(m[1]), int(m[2])
    elif m := re.fullmatch(r"(\d{1,2})[-/.](\d{4})", text):
        year, month = int(m[2]), int(m[1])
    elif m := re.fullmatch(r"\d{4}", text):
        year, month = int(text), 12 if end else 1
    else:
        return None
    return (year, month) if 1 <= month <= 12 and 1950 <= year <= 2100 else None


def dated_in_text(start: str | int | None, text: str) -> bool:
    """Whether a job's start year is written in the CV. Real positions always show their year ("Nisan 2024",
    "03/2025"); a position the LLM made up (from a headline, say) usually has an invented date."""
    month = parse_month(start)
    return month is not None and re.search(rf"(?<!\d){month[0]}(?!\d)", text) is not None


def is_present(value: str | None) -> bool:
    return value is None or value.strip().casefold() in _PRESENT


def _index(month: Month) -> int:
    return month[0] * 12 + month[1] - 1


def _months(jobs: Iterable[Job], today: Month) -> int:
    """Months covered by the jobs, counting overlaps once and both ends."""
    covered: set[int] = set()
    for job in jobs:
        if job.start is None:
            continue
        last = min(_index(job.end or today), _index(today))
        covered.update(range(_index(job.start), last + 1))
    return len(covered)


def months_used(jobs: list[Job], indexes: Iterable[int], today: Month) -> int:
    return _months((jobs[i] for i in set(indexes) if 0 <= i < len(jobs)), today)


def last_used(jobs: list[Job], indexes: Iterable[int], today: Month) -> Month | None:
    ends = [jobs[i].end or today for i in set(indexes) if 0 <= i < len(jobs) and jobs[i].start is not None]
    return max(ends, default=None)


def suggest_level(months: int, last: Month | None, today: Month) -> int | None:
    """1–3 from months of use; one lower if not used for STALE_MONTHS. None without dated use."""
    if months <= 0 or last is None:
        return None
    level = 3 if months >= ADVANCED_MONTHS else 2 if months >= WORKING_MONTHS else 1
    if _index(today) - _index(last) > STALE_MONTHS:
        level = max(1, level - 1)
    return level


def experience_band(jobs: list[Job], education: list[Education], today: Month) -> str | None:
    """The board's experience band (ADR-0041) from tech positions (internships don't count); "student" while
    studying with little tech work; None when the CV says nothing about either."""
    tech = _months((j for j in jobs if j.kind == "tech"), today)
    studying = any(
        e.end_year is not None
        and (e.end_year > today[0] or (e.end_year == today[0] and today[1] <= 6))
        and (e.start_year is None or e.start_year <= today[0])
        for e in education
    )
    if studying and tech < STUDENT_MAX_MONTHS:
        return "student"
    if not any(j.start for j in jobs):
        return None
    return next((band for bound, band in BANDS if tech < bound), "10+")


def _proper_noun(line: str, word: str) -> bool:
    for m in re.finditer(rf"\b{re.escape(word)}\b", line):
        before = line[: m.start()].rstrip()
        if before and before[-1] not in ".!?:-•*–—":
            return True
    return False


def _scan_line(index: LexicalIndex, line: str, min_words: int = 1) -> dict[str, str]:
    found: dict[str, str] = {}
    tokens = [t.strip(".") for t in normalize(line).split()]
    tokens = [t for t in tokens if t]
    for n in range(min_words, MAX_KEY_WORDS + 1):
        for i in range(len(tokens) - n + 1):
            key = " ".join(tokens[i : i + n])
            if key in NEVER_SCANNED or key not in index.keys:
                continue
            if key in PROPER_NOUN_ONLY and not _proper_noun(line, PROPER_NOUN_ONLY[key]):
                continue
            for skill in index.keys[key]:
                found.setdefault(skill, line.strip())
    return found


def scan(index: LexicalIndex, text: str) -> dict[str, str]:
    """Catalog skills named in the text (exact names, aliases, O*NET names) → the first line naming them. Names
    of several words also match across a line break, as PDFs wrap lines ("feature" / "engineering on Spark")."""
    lines = text.split("\n")
    found: dict[str, str] = {}
    for line in lines:
        for skill, evidence in _scan_line(index, line).items():
            found.setdefault(skill, evidence)
    for first, second in pairwise(lines):
        for skill, evidence in _scan_line(index, f"{first} {second}", min_words=2).items():
            found.setdefault(skill, evidence)
    return found


def words(text: str) -> str:
    """Casefolded words without accents, for comparing an LLM's quote with the text (any language)."""
    decomposed = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", text))
    plain = "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()
    return " ".join(re.findall(r"[^\W_]+", plain))


def evidence_found(evidence: str, text_words: str, min_share: float = 0.8) -> bool:
    """Whether a quote is really in the text (`text_words` = words(text)): verbatim, or nearly all its words."""
    quote = words(evidence)
    if not quote:
        return False
    if f" {quote} " in f" {text_words} ":
        return True
    tokens = [t for t in quote.split() if len(t) >= 2]
    if not tokens:
        return False
    vocabulary = set(text_words.split())
    return sum(t in vocabulary for t in tokens) / len(tokens) >= min_share
