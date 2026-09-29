"""Recommendation activity: what was shown, reading it back, and the explanation job rows.

ADR-0013 (activity tables) and ADR-0018 (recommended_roles doubles as the explanation queue).
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from xcrs.db.models import (
    Course,
    EmbeddingModel,
    RecommendationRequest,
    RecommendedCourse,
    RecommendedRole,
    RoadmapNode,
    Role,
)
from xcrs.domain.results import CourseResult, ExplanationStatus, RecommendationResult, RoleResult
from xcrs.explain.base import RoleExplanation


def save_recommendation(
    session: Session,
    result: RecommendationResult,
    *,
    model_id: int,
    algorithm_version: str,
    threshold: float,
    user_input: dict,
    explanation_inputs: dict[int, dict],
) -> None:
    """Save a request with everything it showed, and its explanation jobs, in one transaction."""
    session.add(
        RecommendationRequest(
            id=result.request_id,
            model_id=model_id,
            algorithm_version=algorithm_version,
            threshold_used=threshold,
            input=user_input,
            status=result.status,
            latency_ms=result.latency_ms,
        )
    )
    session.flush()  # the request row must exist before its roles (foreign key)
    session.add_all(
        RecommendedRole(
            request_id=result.request_id,
            rank=rank,
            role_id=role.role_id,
            score=role.score,
            explanation=role.explanation,
            prompt_version=role.prompt_version,
            explanation_status=role.explanation_status.value,
            explanation_input=explanation_inputs.get(role.role_id),
            next_concept_ids=role.next_concept_ids,
        )
        for rank, role in enumerate(result.roles, start=1)
    )
    session.flush()  # roles before their courses (composite foreign key)
    session.add_all(
        RecommendedCourse(
            request_id=result.request_id,
            role_id=role.role_id,
            rank=rank,
            course_id=c.course_id,
            similarity=c.similarity,
            explanation=c.explanation,
            concept_ids=c.concept_ids,
        )
        for role in result.roles
        for rank, c in enumerate(role.courses, start=1)
    )
    session.commit()


def load_recommendation(session: Session, request_id: uuid.UUID) -> RecommendationResult | None:
    """Rebuild a saved result. A fixed number of queries (4), however many roles and courses (ADR-0012)."""
    head = session.execute(
        select(RecommendationRequest, EmbeddingModel.name)
        .join(EmbeddingModel, EmbeddingModel.id == RecommendationRequest.model_id)
        .where(RecommendationRequest.id == request_id)
    ).one_or_none()
    if head is None:
        return None
    request, model_name = head

    roles = session.execute(
        select(RecommendedRole, Role.name)
        .join(Role, Role.id == RecommendedRole.role_id)
        .where(RecommendedRole.request_id == request_id)
        .order_by(RecommendedRole.rank)
    ).all()
    courses = session.execute(
        select(RecommendedCourse, Course.title, Course.url)
        .join(Course, Course.id == RecommendedCourse.course_id)
        .where(RecommendedCourse.request_id == request_id)
        .order_by(RecommendedCourse.role_id, RecommendedCourse.rank)
    ).all()

    concept_ids = {i for r, _ in roles for i in r.next_concept_ids} | {i for c, _, _ in courses for i in c.concept_ids}
    names = (
        dict(session.execute(select(RoadmapNode.id, RoadmapNode.name).where(RoadmapNode.id.in_(concept_ids))).all())
        if concept_ids
        else {}
    )

    courses_by_role: dict[int, list[CourseResult]] = {}
    for c, title, url in courses:
        courses_by_role.setdefault(c.role_id, []).append(
            CourseResult(
                course_id=c.course_id,
                title=title,
                url=url,
                similarity=c.similarity,
                concepts=[names[i] for i in c.concept_ids if i in names],
                explanation=c.explanation,
                concept_ids=list(c.concept_ids),
            )
        )
    return RecommendationResult(
        request_id=request.id,
        status=request.status,
        model=model_name,
        latency_ms=request.latency_ms or 0,
        roles=[
            RoleResult(
                role_id=r.role_id,
                role=role_name,
                score=r.score,
                explanation=r.explanation,
                explanation_status=ExplanationStatus(r.explanation_status),
                prompt_version=r.prompt_version,
                next_to_learn=[names[i] for i in r.next_concept_ids if i in names],
                next_concept_ids=list(r.next_concept_ids),
                courses=courses_by_role.get(r.role_id, []),
            )
            for r, role_name in roles
        ],
    )


# --- Explanation jobs (ADR-0018) -----------------------------------------------------------------


def pending_roles(session: Session) -> list[tuple[uuid.UUID, int]]:
    """Roles still waiting for an explanation, oldest request first (re-queued on startup)."""
    rows = session.execute(
        select(RecommendedRole.request_id, RecommendedRole.role_id)
        .join(RecommendationRequest, RecommendationRequest.id == RecommendedRole.request_id)
        .where(RecommendedRole.explanation_status == ExplanationStatus.PENDING.value)
        .order_by(RecommendationRequest.created_at, RecommendedRole.rank)
    ).all()
    return [(r.request_id, r.role_id) for r in rows]


def claim_role(session: Session, request_id: uuid.UUID, role_id: int, max_attempts: int) -> dict | None:
    """Start a job: return its stored input, or None if there is nothing to do.

    Counting attempts bounds a job that keeps crashing the process: after `max_attempts` it fails.
    """
    role = session.scalars(
        select(RecommendedRole)
        .where(RecommendedRole.request_id == request_id, RecommendedRole.role_id == role_id)
        .with_for_update()
    ).one_or_none()
    if role is None or role.explanation_status != ExplanationStatus.PENDING.value:
        return None
    if role.explanation_attempts >= max_attempts or role.explanation_input is None:
        role.explanation_status = ExplanationStatus.FAILED.value
        session.commit()
        return None
    role.explanation_attempts += 1
    session.commit()
    return role.explanation_input


def save_explanation(
    session: Session, request_id: uuid.UUID, role_id: int, explanation: RoleExplanation, elapsed_ms: int
) -> ExplanationStatus:
    status = ExplanationStatus.DONE if explanation.prompt_version else ExplanationStatus.FAILED
    session.execute(
        update(RecommendedRole)
        .where(RecommendedRole.request_id == request_id, RecommendedRole.role_id == role_id)
        .values(
            explanation=explanation.role_explanation,
            prompt_version=explanation.prompt_version,
            explanation_status=status.value,
            explanation_ms=elapsed_ms,
            explained_at=datetime.now(UTC),
        )
    )
    for course_id, text in explanation.course_explanations.items():
        session.execute(
            update(RecommendedCourse)
            .where(
                RecommendedCourse.request_id == request_id,
                RecommendedCourse.role_id == role_id,
                RecommendedCourse.course_id == course_id,
            )
            .values(explanation=text)
        )
    session.commit()
    return status
