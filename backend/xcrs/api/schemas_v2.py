"""HTTP request/response models for /api/v2: engine v2 on the new catalog (ADR-0029, ADR-0030)."""

from typing import Literal

from pydantic import BaseModel, Field


class SkillMatchRequestV2(BaseModel):
    phrases: list[str] = Field(min_length=1, max_length=30)

    model_config = {"json_schema_extra": {"examples": [{"phrases": ["k8s", "Jira", "teaching computers to learn"]}]}}


class MatchedSkillV2(BaseModel):
    id: str
    name: str


class PhraseMatchV2(BaseModel):
    phrase: str
    skills: list[MatchedSkillV2]
    method: Literal["lookup", "llm", "cache", "embedding", "none"]


class SkillMatchResponseV2(BaseModel):
    matches: list[PhraseMatchV2]


Category = Literal["liked", "neutral", "disliked", "curious"]


class ChipV2(BaseModel):
    """A skill on the board: a catalog skill id (picked) or typed text (matched), at most one of each."""

    category: Category
    skill: str | None = Field(default=None, max_length=80)
    text: str | None = Field(default=None, max_length=100)
    proficiency: int | None = Field(default=None, ge=1, le=4)


Experience = Literal["student", "0-2", "2-5", "5-10", "10+"]


class RecommendationRequestV2(BaseModel):
    chips: list[ChipV2] = Field(min_length=1, max_length=60)
    experience: Experience | None = None  # years in software, optional (ADR-0041)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "chips": [
                        {"category": "liked", "skill": "python", "proficiency": 3},
                        {"category": "liked", "text": "building REST APIs with Django"},
                        {"category": "curious", "skill": "kubernetes"},
                        {"category": "disliked", "text": "CSS"},
                    ]
                }
            ]
        }
    }


class SkillRefV2(BaseModel):
    id: str
    name: str


class LevelV2(BaseModel):
    id: str
    title: str | None
    coverage: float  # cumulative: the share of everything up to this level the board meets
    evidence: float | None = None  # what this level adds, met (the level is reached at 0.45); results since v3.7
    progress: float | None = None  # the level bar: 1 = reached, the next level partly, later ones 0; since v3.7


class GapV2(BaseModel):
    skills: list[SkillRefV2]  # several = any one of them
    need: int
    have: int
    stage: str
    level: str | None = None  # the level that asks for it at this proficiency (results since v3.7)


class BecauseV2(SkillRefV2):
    category: Category


class SectionV2(BaseModel):
    """The part of a resource to open (ADR-0046)."""

    title: str
    url: str
    start_seconds: int | None = None
    kind: Literal["chapter", "video", "start"]  # a long video's chapter, a playlist's video, or episode 1


class ResourceV2(BaseModel):
    id: str
    title: str
    url: str
    provider: str
    type: str
    level: str | None
    free: bool
    curated: bool
    skills: list[SkillRefV2]  # the role's gaps this resource covers
    duration_minutes: int | None = None
    section: SectionV2 | None = None


class RoleV2(BaseModel):
    id: str
    name: str
    family: str
    title_for_you: str | None = None  # built from the learner's strongest title skills (ADR-0044)
    job_titles: list[str] = []  # the role's name and other market titles, to search job ads for
    score: float
    interest: float  # ranks roles: the share of the learner's input on this role
    interest_in_role: float | None = None  # shown: share of the role's skills enjoyed or curious (ADR-0052; v3.3+)
    coverage: float
    level: LevelV2 | None  # None: start at the role's first level
    target_level: LevelV2
    levels: list[LevelV2]
    because: list[BecauseV2]
    gaps: list[GapV2]  # what the next level adds (ADR-0041)
    gaps_total: int
    basics: list[GapV2] = []  # unlisted skills of the levels reached: assumed, for the learner to check
    resources: list[ResourceV2] = []
    explanation_status: Literal["pending", "done", "failed", "disabled"] = "disabled"
    explanation: str | None = None  # written by the local LLM in the background (ADR-0037)
    next_step: str | None = None  # results explained before prompt explain-role-v2.2 only


class MatchedChipV2(BaseModel):
    text: str | None
    category: Category
    proficiency: int | None
    method: str
    skills: list[SkillRefV2]


class BroadSkillV2(SkillRefV2):
    roles: int  # how many roles use it


class NoMatchV2(BaseModel):
    """Why no role was shown (ADR-0052)."""

    positive: int  # skills on the board that aren't "didn't enjoy"
    roles_total: int
    broad: list[BroadSkillV2]  # of those, the ones most roles use


class RecommendationResponseV2(BaseModel):
    id: str
    created_at: str
    status: Literal["ok", "insufficient_input", "no_clear_match"]
    algorithm_version: str
    catalog_version: str
    matched: list[MatchedChipV2]
    roles: list[RoleV2]
    no_match: NoMatchV2 | None = None  # status "no_clear_match" only
    experience: Experience | None = None  # as given on the board (ADR-0041)
    saved: bool = False  # in the signed-in viewer's account (ADR-0043)
    can_save: bool = False  # anonymous and recent: "Save to my account" can attach it


class FeedbackV2Request(BaseModel):
    role: str | None = Field(default=None, max_length=80)
    rating: Literal[-1, 1] | None = None
    comment: str | None = Field(default=None, max_length=2000)


class SkillSuggestionV2(SkillRefV2):
    kind: str


class SkillSearchV2(BaseModel):
    skills: list[SkillSuggestionV2]


class SkillGroupV2(BaseModel):
    family: str
    name: str
    skills: list[SkillSuggestionV2]


class SkillGroupsV2(BaseModel):
    groups: list[SkillGroupV2]
