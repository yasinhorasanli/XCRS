"""Plain data types shared by the domain functions. No I/O, no framework imports (ADR-0017)."""

from dataclasses import dataclass, field
from enum import StrEnum


class Category(StrEnum):
    LIKED = "liked"
    NEUTRAL = "neutral"
    DISLIKED = "disliked"
    CURIOUS = "curious"

    @property
    def is_taken(self) -> bool:
        """Liked, neutral and disliked all mean "the user has studied this"."""
        return self is not Category.CURIOUS


@dataclass(frozen=True)
class Phrase:
    text: str
    category: Category


@dataclass(frozen=True)
class PhraseConceptMatch:
    """A user phrase matched to a roadmap concept above the similarity threshold."""

    phrase: Phrase
    concept_id: int
    role_id: int
    similarity: float


@dataclass(frozen=True)
class RoleScore:
    role_id: int
    score: float  # 0–100, after the sigmoid activation


@dataclass(frozen=True)
class CourseCandidate:
    """A precomputed concept → course match (concept_course_matches)."""

    concept_id: int
    course_id: int
    similarity: float


@dataclass
class CoursePick:
    course_id: int
    similarity: float  # best similarity to any of the concepts it was picked for
    concept_ids: list[int] = field(default_factory=list)  # concepts it was picked for, in roadmap order
