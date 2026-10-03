"""FastAPI application. HTTP only: validation, dependency wiring, status codes (ADR-0017).

uv run uvicorn xcrs.api.app:app --reload
"""

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from xcrs.api import accounts, dev
from xcrs.api.deps import get_session
from xcrs.api.identity import accounts_on, optional_user
from xcrs.api.limits import heavy
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
from xcrs.db.models import User
from xcrs.db.session import new_session
from xcrs.domain.role_scoring import Category as CategoryV2
from xcrs.explain import get_explainer_v2
from xcrs.repository import catalog_store
from xcrs.services import explanations_v2
from xcrs.services.accounts import AccountService, board_input, claimable
from xcrs.services.explanations_v2 import ExplanationWorkerV2
from xcrs.services.recommend_v2 import Chip, RecommendationServiceV2
from xcrs.services.skill_matching import SkillMatcher, build_matcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Explanations run in a background worker (ADR-0018, ADR-0037); roles left `pending` by a restart are
    re-queued."""
    worker = None
    if get_settings().llm_enabled:
        worker = ExplanationWorkerV2(get_explainer_v2(), new_session)
        worker.start()
        log.info("explanation worker started; %d pending roles re-queued", worker.requeue_pending())
    app.state.explanations_v2 = worker
    yield
    if worker:
        worker.stop()


app = FastAPI(
    title="XCRS API", version="2.0.0", description="Explainable Course Recommendation System", lifespan=lifespan
)


app.include_router(dev.router)
app.include_router(accounts.router)
app.include_router(accounts.internal)


def get_skill_matcher(session: Session = Depends(get_session)) -> SkillMatcher:
    return build_matcher(session)


def get_service_v2(
    request: Request, session: Session = Depends(get_session), matcher: SkillMatcher = Depends(get_skill_matcher)
) -> RecommendationServiceV2:
    return RecommendationServiceV2(session, matcher, getattr(request.app.state, "explanations_v2", None))


@app.get("/api/v2/health")
def health(session: Session = Depends(get_session)) -> dict:
    session.execute(select(1))
    return {"status": "ok"}


# Synchronous on purpose (ADR-0015): FastAPI runs `def` endpoints in a thread pool.
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


def _response_v2(row, session: Session, user: User | None = None) -> RecommendationResponseV2:
    """The stored result, with each role's explanation as far as it has been written, and whether it is in the
    viewer's account or could be saved to one (ADR-0043)."""
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
        experience=row.input.get("experience"),
        saved=user is not None and row.user_id == user.id,
        can_save=accounts_on() and row.user_id is None and claimable(row.created_at, session),
    )


@app.post("/api/v2/recommendations", response_model=RecommendationResponseV2, dependencies=[Depends(heavy)])
def create_recommendation_v2(
    body: RecommendationRequestV2,
    service: RecommendationServiceV2 = Depends(get_service_v2),
    user: User | None = Depends(optional_user),
) -> RecommendationResponseV2:
    """Roles for the board's chips (ADR-0031): score, estimated level, coverage per level, the skills that
    count most, and the gaps to the next level in learning order. Signed in (ADR-0043), the result belongs to
    the account and its board becomes the saved board."""
    chips = []
    for c in body.chips:
        if bool(c.skill) == bool(c.text and c.text.strip()):
            raise HTTPException(422, "each chip needs exactly one of `skill` (picked) or `text` (typed)")
        chips.append(Chip(CategoryV2(c.category), c.skill, c.text.strip()[:100] if c.text else None, c.proficiency))
    row = service.recommend(chips, body.experience, user_id=user.id if user else None)
    if user:
        AccountService(service.session).save_board(user, board_input(row.input))
    return _response_v2(row, service.session, user)


@app.get("/api/v2/recommendations/{recommendation_id}", response_model=RecommendationResponseV2)
def get_recommendation_v2(
    recommendation_id: uuid.UUID,
    service: RecommendationServiceV2 = Depends(get_service_v2),
    user: User | None = Depends(optional_user),
) -> RecommendationResponseV2:
    row = service.get(recommendation_id)
    if row is None:
        raise HTTPException(404, "recommendation not found")
    return _response_v2(row, service.session, user)


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


@app.post("/api/v2/dev/profiles/{profile_id}/run", dependencies=[Depends(dev.dev_tools_on)])
def run_dev_profile(profile_id: str, service: RecommendationServiceV2 = Depends(get_service_v2)) -> dict:
    """A real, stored recommendation for a test profile (marked `test`), with explanations, for the results page."""
    profile = next((p for p in dev.load_profiles() if p.id == profile_id), None)
    if profile is None:
        raise HTTPException(404, "no such test profile")
    chips = [Chip(CategoryV2(c.category), c.skill, None, c.proficiency) for c in profile.chips]
    return {"id": str(service.recommend(chips, profile.experience, test=True).id)}
