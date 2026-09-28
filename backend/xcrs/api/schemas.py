"""HTTP request/response models for /api/v1 (ADR-0016)."""

import uuid

from pydantic import BaseModel, Field

from xcrs.domain.types import Category
from xcrs.services.recommend import RecommendationResult

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
    role: str
    score: float
    explanation: str | None
    courses: list[CourseV1]


class RecommendationResponseV1(BaseModel):
    request_id: uuid.UUID
    status: str
    model: str
    latency_ms: int
    roles: list[RoleV1]

    @classmethod
    def from_result(cls, r: RecommendationResult) -> "RecommendationResponseV1":
        return cls(
            request_id=r.request_id,
            status=r.status,
            model=r.model,
            latency_ms=r.latency_ms,
            roles=[
                RoleV1(
                    role=role.role,
                    score=role.score,
                    explanation=role.explanation,
                    courses=[
                        CourseV1(
                            course_id=c.course_id, title=c.title, url=c.url, explanation=c.explanation,
                            concepts=c.concepts, similarity=round(c.similarity, 4),
                        )
                        for c in role.courses
                    ],
                )
                for role in r.roles
            ],
        )
