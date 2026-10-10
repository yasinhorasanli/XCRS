"""Dev tools for the decider's manual checks: test profiles (xcrs/data/test_profiles.yaml) and what the engine
makes of them. Off unless XCRS_DEV_TOOLS=true (404 otherwise); on the VMs Caddy also hides /api/v2/dev from
public Funnel requests. A preview scores without storing anything or calling the LLM; a run makes a real,
stored recommendation marked `test`, with explanations, to open on the results page."""

from functools import cache
from pathlib import Path

import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from xcrs.config import get_settings
from xcrs.db.session import new_session
from xcrs.domain.role_scoring import Category, Mention, score_roles
from xcrs.repository import catalog_store

PROFILES = Path(__file__).resolve().parent.parent / "data" / "test_profiles.yaml"
CATEGORIES = ("liked", "neutral", "disliked", "curious")


def dev_tools_on() -> None:
    if not get_settings().dev_tools:
        raise HTTPException(404, "Not Found")


router = APIRouter(prefix="/api/v2/dev", dependencies=[Depends(dev_tools_on)])


class DevChip(BaseModel):
    category: str
    skill: str
    name: str = ""
    proficiency: int | None = None


class DevProfile(BaseModel):
    id: str
    title: str
    background: str
    expect: list[str]
    level: str
    experience: str | None = None
    chips: list[DevChip]


class DevRole(BaseModel):
    role: str
    name: str
    score: float
    level: str | None
    target_level: str | None
    coverage: float
    interest: float


class DevPreview(BaseModel):
    id: str
    roles: list[DevRole]


@cache
def load_profiles(path: Path = PROFILES) -> tuple[DevProfile, ...]:
    out = []
    for p in yaml.safe_load(path.read_text())["profiles"]:
        chips = []
        for category in CATEGORIES:
            for chip in p.get(category, []):
                skill, _, level = str(chip).partition(":")
                chips.append(DevChip(category=category, skill=skill, proficiency=int(level) if level else None))
        fields = {k: p[k] for k in ("id", "title", "background", "expect", "level")}
        out.append(DevProfile(**fields, experience=p.get("experience"), chips=chips))
    return tuple(out)


def mentions(profile: DevProfile) -> list[Mention]:
    return [Mention(c.skill, Category(c.category), c.proficiency) for c in profile.chips]


def get_session():
    with new_session() as session:
        yield session


@router.get("/profiles", response_model=list[DevProfile])
def profiles(session: Session = Depends(get_session)) -> list[DevProfile]:
    """The test profiles, with skill names for the board."""
    names = catalog_store.skill_display_names(session, [c.skill for p in load_profiles() for c in p.chips])
    return [
        p.model_copy(update={"chips": [c.model_copy(update={"name": names.get(c.skill, c.skill)}) for c in p.chips]})
        for p in load_profiles()
    ]


@router.get("/profiles/preview", response_model=list[DevPreview])
def preview(session: Session = Depends(get_session)) -> list[DevPreview]:
    """The top five roles per profile, scored in memory (milliseconds; nothing stored, no LLM)."""
    snapshot = catalog_store.load_snapshot(session)
    out = []
    for p in load_profiles():
        known = [m for m in mentions(p) if m.skill in snapshot.skill_names]
        roles = score_roles(snapshot, known, experience=p.experience)[:5]
        out.append(
            DevPreview(
                id=p.id,
                roles=[
                    DevRole(
                        role=r.role,
                        name=snapshot.roles[r.role].name,
                        score=round(r.score, 4),
                        level=r.level,
                        target_level=r.target_level,
                        coverage=round(r.coverage, 3),
                        interest=round(r.interest, 3),
                    )
                    for r in roles
                ],
            )
        )
    return out
