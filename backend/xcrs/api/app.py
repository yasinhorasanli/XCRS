"""FastAPI application. HTTP only: validation, dependency wiring, status codes (ADR-0017).

uv run uvicorn xcrs.api.app:app --reload
"""

import logging
from collections.abc import Iterator

from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.orm import Session

from xcrs.api.schemas import RecommendationRequestV1, RecommendationResponseV1
from xcrs.db.session import new_session
from xcrs.explain import get_explainer
from xcrs.services.recommend import RecommendationService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="XCRS API", version="1.0.0", description="Explainable Course Recommendation System")


def get_session() -> Iterator[Session]:
    with new_session() as session:
        yield session


def get_service(session: Session = Depends(get_session)) -> RecommendationService:
    return RecommendationService(session, get_explainer())


@app.get("/api/v1/health")
def health(session: Session = Depends(get_session)) -> dict:
    session.execute(text("SELECT 1"))
    return {"status": "ok"}


# Synchronous on purpose (ADR-0015): FastAPI runs `def` endpoints in a thread pool.
@app.post("/api/v1/recommendations", response_model=RecommendationResponseV1)
def create_recommendation(
    body: RecommendationRequestV1, service: RecommendationService = Depends(get_service)
) -> RecommendationResponseV1:
    return RecommendationResponseV1.from_result(service.recommend(body.to_user_input()))
