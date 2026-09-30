"""HTTP request/response models for /api/v1 (ADR-0016)."""

import uuid
from typing import Literal

from pydantic import BaseModel, Field

from xcrs.domain.labels import display_label
from xcrs.domain.results import ExplanationStatus, RecommendationResult
from xcrs.domain.types import Category

Phrases = list[str]


class RecommendationRequestV1(BaseModel):
    liked: Phrases = Field(default_factory=list, max_length=30)
    neutral: Phrases = Field(default_factory=list, max_length=30)
    disliked: Phrases = Field(default_factory=list, max_length=30)
    curious: Phrases = Field(default_factory=list, max_length=30)

    model_config = {
        "json_schema_extra": {
            "examples": [{"liked": ["Java", "SQL"], "neutral": ["HTML"], "disliked": ["PHP"], "curious": ["Docker"]}]
        }
    }

    def to_user_input(self) -> dict[Category, list[str]]:
        return {
            Category.LIKED: [p[:100] for p in self.liked],
            Category.NEUTRAL: [p[:100] for p in self.neutral],
            Category.DISLIKED: [p[:100] for p in self.disliked],
            Category.CURIOUS: [p[:100] for p in self.curious],
        }


class CourseV1(BaseModel):
    course_id: int
    title: str
    url: str
    explanation: str | None
    concepts: list[str]
    similarity: float


class RoleV1(BaseModel):
    role_id: int
    role: str
    score: float
    explanation: str | None
    # pending: poll GET /api/v1/recommendations/{request_id}; done | failed | disabled are final (ADR-0018)
    explanation_status: ExplanationStatus
    next_to_learn: list[str]  # the role's first uncovered roadmap concepts, in learning order (not LLM-made)
    courses: list[CourseV1]


class RecommendationResponseV1(BaseModel):
    request_id: uuid.UUID
    status: str
    model: str
    latency_ms: int
    input: dict[str, list[str]]  # what the learner entered, by category
    roles: list[RoleV1]

    @classmethod
    def from_result(cls, r: RecommendationResult) -> "RecommendationResponseV1":
        return cls(
            request_id=r.request_id,
            status=r.status,
            model=r.model,
            latency_ms=r.latency_ms,
            input=r.input,
            roles=[
                RoleV1(
                    role_id=role.role_id,
                    role=role.role,
                    score=role.score,
                    explanation=role.explanation,
                    explanation_status=role.explanation_status,
                    next_to_learn=[display_label(n) for n in role.next_to_learn],
                    courses=[
                        CourseV1(
                            course_id=c.course_id,
                            title=c.title,
                            url=c.url,
                            explanation=c.explanation,
                            concepts=[display_label(n) for n in c.concepts],
                            similarity=round(c.similarity, 4),
                        )
                        for c in role.courses
                    ],
                )
                for role in r.roles
            ],
        )


class FeedbackV1(BaseModel):
    """Thumbs up (1) or down (-1) on a role, a course, or the whole result."""

    role_id: int | None = None
    course_id: int | None = None
    rating: Literal[-1, 1]
    comment: str | None = Field(default=None, max_length=1000)


class KnowledgeUnitGroupV1(BaseModel):
    name: str
    units: list[str]


class KnowledgeUnitGroupsV1(BaseModel):
    groups: list[KnowledgeUnitGroupV1]


class KnowledgeUnitV1(BaseModel):
    label: str
    source: Literal["curated", "roadmap"]
    roles: list[str]  # roadmaps that contain it


class KnowledgeUnitsV1(BaseModel):
    units: list[KnowledgeUnitV1]


class RelatedRequestV1(BaseModel):
    phrases: list[str] = Field(max_length=120)
    limit: int = Field(default=12, ge=1, le=30)


class RelatedUnitV1(BaseModel):
    label: str
    because: str  # the learner's phrase it is close to
    similarity: float


class RelatedUnitsV1(BaseModel):
    units: list[RelatedUnitV1]
