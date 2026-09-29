from dataclasses import asdict, dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class KnownItem:
    user_said: str
    matched_concept: str
    category: str  # liked | neutral | disliked | curious


@dataclass(frozen=True)
class CourseContext:
    course_id: int
    title: str
    headline: str | None
    what_you_learn: str | None
    covers: list[str]  # roadmap concepts the course was picked for


@dataclass(frozen=True)
class RoleContext:
    """Everything the algorithm actually used for one recommended role: the facts an explanation
    may draw on, and nothing else."""

    role: str
    score: float
    known: list[KnownItem]
    curious: list[KnownItem]
    covered_topics: list[str]
    next_to_learn: list[str]
    courses: list[CourseContext]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RoleContext":
        """Rebuild a stored context (recommended_roles.explanation_input) for a background job."""
        return cls(
            role=data["role"],
            score=data["score"],
            known=[KnownItem(**k) for k in data["known"]],
            curious=[KnownItem(**k) for k in data["curious"]],
            covered_topics=list(data["covered_topics"]),
            next_to_learn=list(data["next_to_learn"]),
            courses=[CourseContext(**c) for c in data["courses"]],
        )


@dataclass
class RoleExplanation:
    role_explanation: str | None = None
    course_explanations: dict[int, str] = field(default_factory=dict)
    prompt_version: str | None = None  # set only when an explanation was actually produced


class Explainer(Protocol):
    def explain(self, context: RoleContext) -> RoleExplanation: ...


class NoExplainer:
    """Used when explanations are disabled; the recommendation itself still works."""

    def explain(self, context: RoleContext) -> RoleExplanation:
        return RoleExplanation()
