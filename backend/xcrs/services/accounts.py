"""Accounts (ADR-0043): sign-in with linking by verified email, the saved board, the user's results, export and
deletion. Signing in itself (OAuth) happens in the Nuxt server; this is what it reports and what the API keeps.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from xcrs.db.models import Board, ExplanationV2, FeedbackV2, RecommendationV2, User, UserIdentity
from xcrs.repository import catalog_store

PROVIDERS = ("github", "google", "linkedin")
CLAIM_WINDOW = timedelta(days=1)  # an anonymous result can be saved to an account this long after it was made
RESULTS_LISTED = 100


@dataclass(frozen=True)
class SignIn:
    provider: str
    subject: str  # the provider's stable user id
    name: str | None
    email: str | None
    email_verified: bool


class ClaimError(Exception):
    pass


def ms(t: datetime) -> int:
    return int(t.timestamp() * 1000)


class AccountService:
    def __init__(self, session: Session):
        self.session = session

    def sign_in(self, s: SignIn) -> User:
        """The user for this provider account: known; else the user with the same verified email (linked);
        else a new one. Only verified emails are stored."""
        email = s.email.strip().lower() if s.email and s.email_verified else None
        name = (s.name or "").strip()[:100] or None
        identity = self.session.get(UserIdentity, (s.provider, s.subject))
        if identity is not None:
            user = self.session.get_one(User, identity.user_id)
        else:
            user = self.session.scalar(select(User).where(User.email == email)) if email else None
            if user is None:
                user = User(display_name=name, email=email)
                self.session.add(user)
                self.session.flush()
            identity = UserIdentity(provider=s.provider, subject=s.subject, user_id=user.id)
            self.session.add(identity)
        identity.email = email
        identity.last_sign_in_at = func.clock_timestamp()
        user.last_sign_in_at = func.clock_timestamp()  # not now(): that is the transaction's start
        user.display_name = user.display_name or name
        if email and user.email is None and self.session.scalar(select(User.id).where(User.email == email)) is None:
            user.email = email
        self.session.commit()
        self.session.refresh(user)
        return user

    def identities(self, user: User) -> list[UserIdentity]:
        return list(
            self.session.scalars(
                select(UserIdentity).where(UserIdentity.user_id == user.id).order_by(UserIdentity.created_at)
            )
        )

    def sign_out_everywhere(self, user: User) -> None:
        user.sessions_valid_after = func.clock_timestamp()
        self.session.commit()

    def delete(self, user: User) -> None:
        """The user and everything linked: identities, board, results and (by cascade) their feedback and
        explanations."""
        self.session.execute(delete(User).where(User.id == user.id))
        self.session.commit()

    # --- the board -------------------------------------------------------------------------------

    def board(self, user: User) -> Board | None:
        return self.session.scalar(select(Board).where(Board.user_id == user.id))

    def save_board(self, user: User, board_input: dict) -> None:
        board = self.board(user)
        if board is None:
            self.session.add(Board(user_id=user.id, input=board_input))
        else:
            board.input, board.updated_at = board_input, func.now()
        self.session.commit()

    def board_chips(self, board: Board) -> list[dict]:
        """The saved chips, with display names for picked skills."""
        chips = board.input.get("chips", [])
        names = catalog_store.skill_display_names(self.session, [c["skill"] for c in chips if c.get("skill")])
        return [{**c, "name": c.get("text") or names.get(c.get("skill") or "", c.get("skill"))} for c in chips]

    # --- results ---------------------------------------------------------------------------------

    def results(self, user: User) -> list[RecommendationV2]:
        return list(
            self.session.scalars(
                select(RecommendationV2)
                .where(RecommendationV2.user_id == user.id)
                .order_by(RecommendationV2.created_at.desc())
                .limit(RESULTS_LISTED)
            )
        )

    def claim(self, user: User, recommendation_id: uuid.UUID) -> RecommendationV2:
        """Save an anonymous result to the account (and its board as the saved board)."""
        row = self.session.get(RecommendationV2, recommendation_id)
        if row is None:
            raise LookupError(recommendation_id)
        if row.user_id == user.id:
            return row
        if row.user_id is not None:
            raise ClaimError("these results belong to another account")
        if not claimable(row.created_at, self.session):
            raise ClaimError("these results are too old to save; make them again while signed in")
        row.user_id = user.id
        self.save_board(user, board_input(row.input))
        return row

    def forget_result(self, user: User, recommendation_id: uuid.UUID) -> bool:
        """Delete one of the user's results (with its feedback and explanations)."""
        done = self.session.execute(
            delete(RecommendationV2).where(
                RecommendationV2.id == recommendation_id, RecommendationV2.user_id == user.id
            )
        )
        self.session.commit()
        return bool(done.rowcount)

    def export(self, user: User) -> dict:
        """Everything stored about the user, as JSON (the right of access and portability)."""
        results = list(
            self.session.scalars(
                select(RecommendationV2)
                .where(RecommendationV2.user_id == user.id)
                .order_by(RecommendationV2.created_at)
            )
        )
        ids = [r.id for r in results]
        feedback = self.session.scalars(select(FeedbackV2).where(FeedbackV2.recommendation_id.in_(ids))).all()
        explanations = self.session.scalars(
            select(ExplanationV2).where(ExplanationV2.recommendation_id.in_(ids)).order_by(ExplanationV2.rank)
        ).all()
        board = self.board(user)
        return {
            "exported_at": self.session.scalar(select(func.now())).isoformat(),
            "user": {
                "id": str(user.id),
                "display_name": user.display_name,
                "email": user.email,
                "created_at": user.created_at.isoformat(),
                "last_sign_in_at": user.last_sign_in_at.isoformat(),
            },
            "sign_in_accounts": [
                {"provider": i.provider, "email": i.email, "linked_at": i.created_at.isoformat()}
                for i in self.identities(user)
            ],
            "board": {"input": board.input, "updated_at": board.updated_at.isoformat()} if board else None,
            "results": [
                {
                    "id": str(r.id),
                    "created_at": r.created_at.isoformat(),
                    "input": r.input,
                    "result": r.result,
                    "explanations": [
                        {"role": e.role, "explanation": e.explanation, "next_step": e.next_step}
                        for e in explanations
                        if e.recommendation_id == r.id
                    ],
                    "feedback": [
                        {
                            "role": f.role,
                            "rating": f.rating,
                            "comment": f.comment,
                            "created_at": f.created_at.isoformat(),
                        }
                        for f in feedback
                        if f.recommendation_id == r.id
                    ],
                }
                for r in results
            ],
        }


def board_input(recommendation_input: dict) -> dict:
    """The board part of a recommendation's input (without markers such as `test`)."""
    return {k: recommendation_input[k] for k in ("chips", "experience") if k in recommendation_input}


def claimable(created_at: datetime, session: Session) -> bool:
    """Whether an anonymous result is recent enough to be saved to an account."""
    return created_at >= session.scalar(select(func.now())) - CLAIM_WINDOW
