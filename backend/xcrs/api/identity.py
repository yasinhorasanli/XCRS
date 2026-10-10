"""Who is signed in (ADR-0043). The Nuxt server handles sign-in and keeps the session in a sealed cookie; for a
signed-in user its proxy adds `X-XCRS-User: <user id>.<signed in at ms>.<issued at ms>.<HMAC-SHA256 hex>`,
signed with XCRS_INTERNAL_SECRET, after removing any X-XCRS-* header the browser sent. The API accepts the
header only with a valid, fresh signature from a user who still exists and has not signed out everywhere since.

A rejected header answers 401 (or, where signing in is optional, is ignored) with `X-XCRS-Session: revoked`,
which tells the proxy to clear the cookie.
"""

import hashlib
import hmac
import time
import uuid
from dataclasses import dataclass
from typing import Literal

from fastapi import Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session

from xcrs.api.deps import get_session
from xcrs.config import get_settings
from xcrs.db.models import User

HEADER = "x-xcrs-user"
REVOKED = {"X-XCRS-Session": "revoked"}
MAX_AGE_MS = 5 * 60 * 1000  # a signature is made per request; this only bounds replay of a logged header


@dataclass(frozen=True)
class SignedUser:
    user_id: uuid.UUID
    signed_in_at_ms: int


def sign(secret: str, user_id: uuid.UUID | str, signed_in_at_ms: int, issued_at_ms: int | None = None) -> str:
    """The header value (the Nuxt server does the same in TypeScript; tests use this)."""
    issued = int(time.time() * 1000) if issued_at_ms is None else issued_at_ms
    payload = f"{user_id}.{signed_in_at_ms}.{issued}"
    return f"{payload}.{hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()}"


def verify(secret: str, value: str, now_ms: int | None = None) -> SignedUser | None:
    """The signed user, or None for a malformed, forged or stale header."""
    parts = value.split(".")
    if len(parts) != 4:
        return None
    user, signed_in, issued, signature = parts
    expected = hmac.new(secret.encode(), f"{user}.{signed_in}.{issued}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        return None
    try:
        user_id, signed_in_ms, issued_ms = uuid.UUID(user), int(signed_in), int(issued)
    except ValueError:
        return None
    now = int(time.time() * 1000) if now_ms is None else now_ms
    if not -MAX_AGE_MS < now - issued_ms < MAX_AGE_MS:
        return None
    return SignedUser(user_id, signed_in_ms)


def accounts_on() -> bool:
    return bool(get_settings().internal_secret)


def resolve(request: Request, session: Session) -> User | Literal[False] | None:
    """The signed-in user; None when nobody is signed in; False when a header was sent but is not valid."""
    value = request.headers.get(HEADER)
    secret = get_settings().internal_secret
    if not value or not secret:
        return None
    signed = verify(secret, value)
    if signed is None:
        return False
    user = session.get(User, signed.user_id)
    if user is None:  # the account was deleted
        return False
    cut = user.sessions_valid_after
    if cut is not None and signed.signed_in_at_ms <= int(cut.timestamp() * 1000):  # a tie: the revocation wins
        return False
    return user


def optional_user(request: Request, response: Response, session: Session = Depends(get_session)) -> User | None:
    """For endpoints that work signed in or not: a rejected session counts as anonymous (and is cleared)."""
    user = resolve(request, session)
    if user is False:
        response.headers.update(REVOKED)
        return None
    return user


def required_user(request: Request, session: Session = Depends(get_session)) -> User:
    if not accounts_on():
        raise HTTPException(404, "accounts are not enabled")
    user = resolve(request, session)
    if user is None:
        raise HTTPException(401, "sign in first")
    if user is False:
        raise HTTPException(401, "your session has ended; please sign in again", headers=REVOKED)
    return user


def internal_only(request: Request) -> None:
    """For calls from the Nuxt server only (they are outside /api/v2, so its proxy never forwards them)."""
    secret = get_settings().internal_secret
    given = request.headers.get("x-xcrs-internal-secret", "")
    if not secret or not hmac.compare_digest(given.encode(), secret.encode()):
        raise HTTPException(404, "Not Found")
