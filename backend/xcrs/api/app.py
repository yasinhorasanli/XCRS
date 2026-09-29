"""FastAPI application. HTTP only: validation, dependency wiring, status codes (ADR-0017).

uv run uvicorn xcrs.api.app:app --reload
"""

import logging
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from sqlalchemy.orm import Session

from xcrs.api.schemas import RecommendationRequestV1, RecommendationResponseV1
from xcrs.config import get_settings
from xcrs.db.session import new_session
from xcrs.explain import get_explainer
from xcrs.repository import catalog as catalog_repo
from xcrs.services.explanations import ExplanationWorker
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
