"""FastAPI application. HTTP only: validation, dependency wiring, status codes (ADR-0017).

uv run uvicorn xcrs.api.app:app --reload
"""

import logging
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from sqlalchemy.orm import Session

from xcrs.api.schemas import (
    FeedbackV1,
    KnowledgeUnitGroupsV1,
    KnowledgeUnitsV1,
    KnowledgeUnitV1,
    RecommendationRequestV1,
    RecommendationResponseV1,
    RelatedRequestV1,
    RelatedUnitsV1,
    RelatedUnitV1,
)
from xcrs.config import get_settings
from xcrs.db.session import new_session
from xcrs.explain import get_explainer
from xcrs.repository import activity
from xcrs.repository import catalog as catalog_repo
from xcrs.services.explanations import ExplanationWorker
from xcrs.services.knowledge_units import KnowledgeUnitService
from xcrs.services.recommend import RecommendationService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Explanations run in a background worker (ADR-0018); roles left `pending` by a restart are re-queued."""
    settings = get_settings()
    worker = None
    if settings.llm_enabled:
        worker = ExplanationWorker(get_explainer(), new_session, threads=settings.explain_threads)
        worker.start()
        log.info("explanation worker started; %d pending roles re-queued", worker.requeue_pending())
    app.state.explanations = worker
    yield
    if worker:
        worker.stop()


app = FastAPI(
    title="XCRS API", version="1.1.0", description="Explainable Course Recommendation System", lifespan=lifespan
)


def get_session() -> Iterator[Session]:
    with new_session() as session:
        yield session


def get_service(request: Request, session: Session = Depends(get_session)) -> RecommendationService:
    return RecommendationService(session, request.app.state.explanations)


def get_knowledge_units(session: Session = Depends(get_session)) -> KnowledgeUnitService:
    return KnowledgeUnitService(session)


@app.get("/api/v1/health")
def health(session: Session = Depends(get_session)) -> dict:
    catalog_repo.ping(session)
    return {"status": "ok"}


# Synchronous on purpose (ADR-0015): FastAPI runs `def` endpoints in a thread pool.
@app.post("/api/v1/recommendations", response_model=RecommendationResponseV1)
def create_recommendation(
    body: RecommendationRequestV1, service: RecommendationService = Depends(get_service)
) -> RecommendationResponseV1:
    """Returns roles and courses at once; explanations arrive later (poll the GET endpoint while any role
    has `explanation_status: pending`)."""
    return RecommendationResponseV1.from_result(service.recommend(body.to_user_input()))


@app.get("/api/v1/recommendations/{request_id}", response_model=RecommendationResponseV1)
def get_recommendation(
    request_id: uuid.UUID, service: RecommendationService = Depends(get_service)
) -> RecommendationResponseV1:
    result = service.get(request_id)
    if result is None:
        raise HTTPException(status_code=404, detail="recommendation not found")
    return RecommendationResponseV1.from_result(result)


@app.post("/api/v1/recommendations/{request_id}/feedback", status_code=201)
def create_feedback(request_id: uuid.UUID, body: FeedbackV1, session: Session = Depends(get_session)) -> dict:
    if not activity.feedback_target_exists(session, request_id, body.role_id, body.course_id):
        raise HTTPException(status_code=404, detail="no such recommendation, role or course")
    activity.save_feedback(session, request_id, body.role_id, body.course_id, body.rating, body.comment)
    return {"status": "ok"}


# --- Knowledge-unit suggestions for the input page (ADR-0023) ------------------------------------


@app.get("/api/v1/knowledge-units/groups", response_model=KnowledgeUnitGroupsV1)
def knowledge_unit_groups(service: KnowledgeUnitService = Depends(get_knowledge_units)) -> KnowledgeUnitGroupsV1:
    return KnowledgeUnitGroupsV1(groups=service.groups())


@app.get("/api/v1/knowledge-units", response_model=KnowledgeUnitsV1)
def search_knowledge_units(
    q: str = Query(min_length=1, max_length=100),
    limit: int = Query(default=20, ge=1, le=50),
    service: KnowledgeUnitService = Depends(get_knowledge_units),
) -> KnowledgeUnitsV1:
    return KnowledgeUnitsV1(units=[KnowledgeUnitV1(**vars(u)) for u in service.search(q, limit)])


@app.post("/api/v1/knowledge-units/related", response_model=RelatedUnitsV1)
def related_knowledge_units(
    body: RelatedRequestV1, service: KnowledgeUnitService = Depends(get_knowledge_units)
) -> RelatedUnitsV1:
    """Suggestions close to what the learner enjoyed or is curious about, away from what they didn't enjoy
    (one embedding call)."""
    units = service.related(
        [p[:100] for p in body.phrases], [a[:100] for a in body.avoid], [e[:100] for e in body.exclude], body.limit
    )
    return RelatedUnitsV1(units=[RelatedUnitV1(**vars(u)) for u in units])
