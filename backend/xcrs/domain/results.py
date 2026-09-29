"""What a recommendation returns. Plain data, shared by the service, the repository and the API."""

import uuid
from dataclasses import dataclass, field
from enum import StrEnum


class ExplanationStatus(StrEnum):
    PENDING = "pending"  # queued or being generated (ADR-0018)
    DONE = "done"
    FAILED = "failed"  # the LLM failed or returned nothing usable; the recommendation stands
    DISABLED = "disabled"  # explanations are switched off


@dataclass
class CourseResult:
    course_id: int
    title: str
    url: str
    similarity: float
    concepts: list[str]  # the roadmap concepts this course was picked for
    explanation: str | None = None
    concept_ids: list[int] = field(default_factory=list)


@dataclass
class RoleResult:
    role_id: int
    role: str
    score: float
    explanation: str | None = None
    explanation_status: ExplanationStatus = ExplanationStatus.PENDING
    prompt_version: str | None = None
    next_to_learn: list[str] = field(default_factory=list)  # decided by the algorithm, not the LLM
    next_concept_ids: list[int] = field(default_factory=list)
    courses: list[CourseResult] = field(default_factory=list)


@dataclass
class RecommendationResult:
    request_id: uuid.UUID
    status: str  # ok | insufficient_input
    model: str
    roles: list[RoleResult]
    latency_ms: int  # the response only; explanation time is per role (explanation_ms)
