"""Prompts for matching free text to catalog skills (ADR-0029, ADR-0030).

`PICK_SYSTEM_2` is the one in use: the LLM picks catalog skills from the full list (ranked, at most 5),
and each pick is then confirmed by embedding similarity. The others are kept so the benchmark
(eval/bench_skill_matching.py) can be reproduced: `PICK_SYSTEM` (prompt 1, over-picked for vague input)
and `DEFINE_SYSTEM` (an LLM definition to embed, which measured no better than the phrase itself).
"""

from pydantic import BaseModel, Field

DEFINE_PROMPT_VERSION = "define-1"

DEFINE_SYSTEM = """\
A learner typed a few words about their skills on a career-guidance site for software engineering.
Say what they mean, in the style of a skills catalog.

- If it is a software, data, IT, security or engineering-practice skill, tool, technology, language, \
framework, platform or topic (even misspelled, abbreviated, described in other words, or a course title), \
set is_software_skill to true and write ONE sentence of at most 30 words: "<proper name>: <what it is \
and what it is used for>". Expand abbreviations and fix typos. If several skills are mentioned, name each.
- If it is anything else (a hobby, another profession, everyday words, gibberish), set is_software_skill \
to false and leave definition empty.

Answer with JSON only."""

QUERY_INSTRUCTION = (
    "Instruct: Given a skill, technology, tool or topic that a learner mentions, "
    "retrieve the matching software skills\nQuery:"
)


class PhraseDefinition(BaseModel):
    is_software_skill: bool
    definition: str = Field(default="", description="one sentence, '<name>: <what it is>', or empty")


PICK_PROMPT_VERSION = "pick-1"

PICK_SYSTEM = """\
A learner typed a few words about their skills on a career-guidance site for software engineering.
Choose the catalog skills they mean, from the list below (id: name). Choose only skills the words \
name or clearly describe, usually one to three; choose none if they don't mean a software skill.
Answer with JSON only: {{"skills": [ids]}}.

{catalog}"""


PICK_PROMPT_VERSION_2 = "pick-2"

# pick-1 picked far too many skills for vague input ("ML": 28) and nothing for tools missing from the
# catalog (Jira, BigQuery); pick-2 ranks, caps, and maps tools to their skill (benchmark 2026-10-02).
PICK_SYSTEM_2 = """\
A learner typed a few words about their skills on a career-guidance site for software engineering.
Choose the catalog skills they mean from the list below (id: name), most relevant first, at most 5:
- a skill in the list, named or described (even misspelled or abbreviated) -> that skill;
- a tool, library or product that is not in the list -> the skill it belongs to (for example, \
Sass -> css);
- several skills -> each of them;
- a broad area -> only its 1 to 3 most central skills;
- not a software, data, IT, security or engineering-practice skill -> none. Engineering practices such \
as mentoring, technical writing or public speaking are skills.
Answer with JSON only: {{"skills": [ids]}}.

{catalog}"""

PICK_LIMIT = 5


class SkillPick(BaseModel):
    skills: list[str]
