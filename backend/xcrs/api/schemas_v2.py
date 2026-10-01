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
