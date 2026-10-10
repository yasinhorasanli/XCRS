"""Account endpoints (ADR-0043). `/internal/sign-in` is called by the Nuxt server after a provider has signed
the user in (it is outside /api/v2, so the proxy never forwards it from a browser); `/api/v2/me/*` act for the
user named in the proxy's signed X-XCRS-User header (xcrs/api/identity.py)."""

import json
import uuid
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from xcrs.api.deps import get_session
from xcrs.api.identity import REVOKED, internal_only, required_user
from xcrs.api.limits import light
from xcrs.api.schemas_v2 import ChipV2, Experience
from xcrs.db.models import RecommendationV2, User
from xcrs.services.accounts import AccountService, ClaimError, SignIn, ms

Provider = Literal["github", "google", "linkedin"]


class SignInRequest(BaseModel):
    provider: Provider
    subject: str = Field(min_length=1, max_length=200)
    name: str | None = Field(default=None, max_length=300)
    email: str | None = Field(default=None, max_length=320)
    email_verified: bool = False


class SignInResponse(BaseModel):
    user_id: str
    display_name: str | None
    signed_in_at_ms: int  # goes into the session cookie and every signed header (revocation, ADR-0043)


class MeV2(BaseModel):
    id: str
    display_name: str | None
    email: str | None
    providers: list[Provider]
    created_at: str


class BoardChipV2(ChipV2):
    name: str | None = None  # display name: the typed text or the picked skill's name


class BoardV2(BaseModel):
    chips: list[BoardChipV2]
    experience: Experience | None = None
    updated_at: str


class SavedBoardV2(BaseModel):
    board: BoardV2 | None


class BoardInputV2(BaseModel):
    chips: list[ChipV2] = Field(max_length=60)
    experience: Experience | None = None


class ResultRoleV2(BaseModel):
    id: str
    name: str
    level: str | None


class ResultSummaryV2(BaseModel):
    id: str
    created_at: str
    status: str
    roles: list[ResultRoleV2]


def get_accounts(session: Session = Depends(get_session)) -> AccountService:
    return AccountService(session)


internal = APIRouter(prefix="/internal", dependencies=[Depends(internal_only)], include_in_schema=False)
router = APIRouter(prefix="/api/v2/me", dependencies=[Depends(light)])


@internal.post("/sign-in", response_model=SignInResponse)
def sign_in(body: SignInRequest, accounts: AccountService = Depends(get_accounts)) -> SignInResponse:
    user = accounts.sign_in(SignIn(body.provider, body.subject, body.name, body.email, body.email_verified))
    return SignInResponse(
        user_id=str(user.id), display_name=user.display_name, signed_in_at_ms=ms(user.last_sign_in_at)
    )


@router.get("", response_model=MeV2)
def me(user: User = Depends(required_user), accounts: AccountService = Depends(get_accounts)) -> MeV2:
    return MeV2(
        id=str(user.id),
        display_name=user.display_name,
        email=user.email,
        providers=[i.provider for i in accounts.identities(user)],
        created_at=user.created_at.isoformat(),
    )


@router.delete("", status_code=204)
def delete_me(user: User = Depends(required_user), accounts: AccountService = Depends(get_accounts)) -> Response:
    """Delete the account and everything linked to it (ADR-0043); the proxy then clears the session."""
    accounts.delete(user)
    return Response(status_code=204, headers=REVOKED)


@router.post("/sign-out-everywhere", status_code=204)
def sign_out_everywhere(
    user: User = Depends(required_user), accounts: AccountService = Depends(get_accounts)
) -> Response:
    """End every session of this account, this one included."""
    accounts.sign_out_everywhere(user)
    return Response(status_code=204, headers=REVOKED)


@router.get("/board", response_model=SavedBoardV2)
def get_board(user: User = Depends(required_user), accounts: AccountService = Depends(get_accounts)) -> SavedBoardV2:
    board = accounts.board(user)
    if board is None:
        return SavedBoardV2(board=None)
    return SavedBoardV2(
        board=BoardV2(
            chips=[BoardChipV2(**c) for c in accounts.board_chips(board)],
            experience=board.input.get("experience"),
            updated_at=board.updated_at.isoformat(),
        )
    )


@router.put("/board", status_code=204)
def put_board(
    body: BoardInputV2, user: User = Depends(required_user), accounts: AccountService = Depends(get_accounts)
) -> Response:
    chips = [c.model_dump(exclude_none=True) for c in body.chips]
    for c in chips:
        if bool(c.get("skill")) == bool(c.get("text", "").strip()):
            raise HTTPException(422, "each chip needs exactly one of `skill` (picked) or `text` (typed)")
    accounts.save_board(user, {"chips": chips, **({"experience": body.experience} if body.experience else {})})
    return Response(status_code=204)


def summary(row: RecommendationV2) -> ResultSummaryV2:
    return ResultSummaryV2(
        id=str(row.id),
        created_at=row.created_at.isoformat(),
        status=row.status,
        roles=[
            ResultRoleV2(id=r["id"], name=r["name"], level=(r.get("level") or {}).get("id"))
            for r in row.result.get("roles", [])
        ],
    )


@router.get("/results", response_model=list[ResultSummaryV2])
def results(
    user: User = Depends(required_user), accounts: AccountService = Depends(get_accounts)
) -> list[ResultSummaryV2]:
    return [summary(r) for r in accounts.results(user)]


@router.post("/results/{recommendation_id}", response_model=ResultSummaryV2)
def save_result(
    recommendation_id: uuid.UUID, user: User = Depends(required_user), accounts: AccountService = Depends(get_accounts)
) -> ResultSummaryV2:
    """Save an anonymous result (at most a day old) to the account; its board becomes the saved board."""
    try:
        return summary(accounts.claim(user, recommendation_id))
    except LookupError:
        raise HTTPException(404, "recommendation not found") from None
    except ClaimError as e:
        raise HTTPException(409, str(e)) from None


@router.delete("/results/{recommendation_id}", status_code=204)
def delete_result(
    recommendation_id: uuid.UUID, user: User = Depends(required_user), accounts: AccountService = Depends(get_accounts)
) -> Response:
    if not accounts.forget_result(user, recommendation_id):
        raise HTTPException(404, "no such result in this account")
    return Response(status_code=204)


@router.get("/export")
def export(user: User = Depends(required_user), accounts: AccountService = Depends(get_accounts)) -> Response:
    """Everything stored about the user, as a JSON download."""
    day = datetime.now().strftime("%Y-%m-%d")
    return Response(
        json.dumps(accounts.export(user), ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="xcrs-my-data-{day}.json"'},
    )
