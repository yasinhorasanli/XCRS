"""FastAPI application. HTTP only: validation, dependency wiring, status codes (ADR-0017).

uv run uvicorn xcrs.api.app:app --reload
"""

import logging
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from sqlalchemy.orm import Session

from xcrs.api.limits import heavy, light
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
from xcrs.api.schemas_v2 import (
    FeedbackV2Request,
    MatchedSkillV2,
    PhraseMatchV2,
    RecommendationRequestV2,
    RecommendationResponseV2,
    SkillGroupsV2,
    SkillGroupV2,
    SkillMatchRequestV2,
    SkillMatchResponseV2,
    SkillSearchV2,
    SkillSuggestionV2,
)
from xcrs.config import get_settings
from xcrs.db.session import new_session
from xcrs.domain.role_scoring import Category as CategoryV2
from xcrs.explain import get_explainer, get_explainer_v2
from xcrs.repository import activity, catalog_store
from xcrs.repository import catalog as catalog_repo
from xcrs.services import explanations_v2
from xcrs.services.explanations import ExplanationWorker
from xcrs.services.explanations_v2 import ExplanationWorkerV2
from xcrs.services.knowledge_units import KnowledgeUnitService
from xcrs.services.recommend import RecommendationService
from xcrs.services.recommend_v2 import Chip, RecommendationServiceV2
from xcrs.services.skill_matching import SkillMatcher, build_matcher

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
    worker_v2 = None
    if settings.llm_enabled:
        worker_v2 = ExplanationWorkerV2(get_explainer_v2(), new_session)
        worker_v2.start()
        log.info("v2 explanation worker started; %d pending roles re-queued", worker_v2.requeue_pending())
    app.state.explanations_v2 = worker_v2
    yield
    if worker:
        worker.stop()
    if worker_v2:
        worker_v2.stop()


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


def get_skill_matcher(session: Session = Depends(get_session)) -> SkillMatcher:
    return build_matcher(session)


def get_service_v2(
    request: Request, session: Session = Depends(get_session), matcher: SkillMatcher = Depends(get_skill_matcher)
) -> RecommendationServiceV2:
    return RecommendationServiceV2(session, matcher, getattr(request.app.state, "explanations_v2", None))


@app.get("/api/v1/health")
def health(session: Session = Depends(get_session)) -> dict:
    catalog_repo.ping(session)
    return {"status": "ok"}


# Synchronous on purpose (ADR-0015): FastAPI runs `def` endpoints in a thread pool.
@app.post("/api/v1/recommendations", response_model=RecommendationResponseV1, dependencies=[Depends(heavy)])
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


@app.post("/api/v1/knowledge-units/related", response_model=RelatedUnitsV1, dependencies=[Depends(light)])
def related_knowledge_units(
    body: RelatedRequestV1, service: KnowledgeUnitService = Depends(get_knowledge_units)
) -> RelatedUnitsV1:
    """Suggestions close to what the learner enjoyed or is curious about, away from what they didn't enjoy
    (one embedding call)."""
    units = service.related(
        [p[:100] for p in body.phrases], [a[:100] for a in body.avoid], [e[:100] for e in body.exclude], body.limit
    )
    return RelatedUnitsV1(units=[RelatedUnitV1(**vars(u)) for u in units])


# --- /api/v2: engine v2 on the new catalog (ADR-0029); /api/v1 serves the legacy engine until the switch-over.


@app.post("/api/v2/skills/match", response_model=SkillMatchResponseV2, dependencies=[Depends(heavy)])
def match_skills(
    body: SkillMatchRequestV2,
    matcher: SkillMatcher = Depends(get_skill_matcher),
    session: Session = Depends(get_session),
) -> SkillMatchResponseV2:
    """Catalog skills for typed phrases (ADR-0030). Clients call it as each chip is added, so the LLM step
    (seconds on a CPU, once per phrase) is usually done before "Recommend"."""
    results = matcher.match([p.strip()[:100] for p in body.phrases])
    names = catalog_store.skill_display_names(session, [s for r in results for s in r.skills])
    return SkillMatchResponseV2(
        matches=[
            PhraseMatchV2(
                phrase=r.phrase, skills=[MatchedSkillV2(id=s, name=names[s]) for s in r.skills], method=r.method
            )
            for r in results
        ]
    )


def _response_v2(row, session: Session) -> RecommendationResponseV2:
    """The stored result, with each role's explanation as far as it has been written."""
    explained = explanations_v2.for_recommendation(session, row.id)
    roles = [
        {
            **role,
            "explanation_status": explained[role["id"]].status if role["id"] in explained else "disabled",
            "explanation": explained[role["id"]].explanation if role["id"] in explained else None,
            "next_step": explained[role["id"]].next_step if role["id"] in explained else None,
        }
        for role in row.result["roles"]
    ]
    return RecommendationResponseV2(
        id=str(row.id),
        created_at=row.created_at.isoformat(),
        status=row.status,
        algorithm_version=row.algorithm_version,
        catalog_version=row.catalog_checksum[:12],
        matched=row.result["matched"],
        roles=roles,
    )


@app.post("/api/v2/recommendations", response_model=RecommendationResponseV2, dependencies=[Depends(heavy)])
def create_recommendation_v2(
    body: RecommendationRequestV2, service: RecommendationServiceV2 = Depends(get_service_v2)
) -> RecommendationResponseV2:
    """Roles for the board's chips (ADR-0031): score, estimated level, coverage per level, the skills that
    count most, and the gaps to the next level in learning order."""
    chips = []
    for c in body.chips:
        if bool(c.skill) == bool(c.text and c.text.strip()):
            raise HTTPException(422, "each chip needs exactly one of `skill` (picked) or `text` (typed)")
        chips.append(Chip(CategoryV2(c.category), c.skill, c.text.strip()[:100] if c.text else None, c.proficiency))
    return _response_v2(service.recommend(chips), service.session)


@app.get("/api/v2/recommendations/{recommendation_id}", response_model=RecommendationResponseV2)
def get_recommendation_v2(
    recommendation_id: uuid.UUID, service: RecommendationServiceV2 = Depends(get_service_v2)
) -> RecommendationResponseV2:
    row = service.get(recommendation_id)
    if row is None:
        raise HTTPException(404, "recommendation not found")
    return _response_v2(row, service.session)


@app.post("/api/v2/recommendations/{recommendation_id}/feedback", status_code=201)
def create_feedback_v2(
    recommendation_id: uuid.UUID, body: FeedbackV2Request, service: RecommendationServiceV2 = Depends(get_service_v2)
) -> dict:
    if service.get(recommendation_id) is None:
        raise HTTPException(404, "recommendation not found")
    service.feedback(recommendation_id, body.role, body.rating, body.comment)
    return {"status": "saved"}


@app.get("/api/v2/skills", response_model=SkillSearchV2)
def search_skills_v2(
    q: str = Query(min_length=1, max_length=100),
    limit: int = Query(default=20, ge=1, le=50),
    session: Session = Depends(get_session),
) -> SkillSearchV2:
    """Catalog skills for the board's picker."""
    return SkillSearchV2(
        skills=[SkillSuggestionV2(id=s, name=n, kind=k) for s, n, k in catalog_store.search_skills(session, q, limit)]
    )


@app.get("/api/v2/skills/groups", response_model=SkillGroupsV2)
def skill_groups_v2(session: Session = Depends(get_session)) -> SkillGroupsV2:
    """Suggested skills per role family, for a quick start on the board."""
    return SkillGroupsV2(
        groups=[
            SkillGroupV2(family=f, name=n, skills=[SkillSuggestionV2(id=s, name=sn, kind=k) for s, sn, k in skills])
            for f, n, skills in catalog_store.skill_groups(session)
        ]
    )
