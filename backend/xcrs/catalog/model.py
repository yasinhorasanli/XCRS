"""Catalog-as-code (ADR-0028): skills, roles, transitions and roadmaps, loaded from `catalog/*.yaml`.

The YAML is the reviewed source of truth (changes arrive as pull requests); `validate.py` checks it and an
import loads it into the database. Plain data and parsing only; no database access here.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from xcrs.config import REPO_ROOT

CATALOG_DIR = REPO_ROOT / "catalog"

LADDER = ("entry", "mid", "senior", "staff")
SKILL_KINDS = {"language", "framework", "library", "tool", "platform", "concept", "practice"}
PATH_KINDS = {"broaden", "specialize", "pivot", "lead"}
PROFICIENCY = {1: "basic", 2: "working", 3: "advanced", 4: "expert"}

_REQUIREMENT = re.compile(r"^(?P<options>[a-z0-9-]+(\|[a-z0-9-]+)*):(?P<level>[1-4])$")
_ROLE_LEVEL = re.compile(r"^(?P<role>[a-z0-9-]+)@(?P<level>[a-z]+)$")


class CatalogError(ValueError):
    """The YAML doesn't have the expected shape (as opposed to a content problem, see validate.py)."""


@dataclass(frozen=True)
class Requirement:
    """One skill at a proficiency, or a choice: any one of `options` at that proficiency ("python|go:2")."""

    options: tuple[str, ...]
    level: int

    @classmethod
    def parse(cls, text: str) -> "Requirement":
        match = _REQUIREMENT.match(str(text).strip())
        if not match:
            raise CatalogError(f"bad skill reference {text!r}: expected 'skill:1-4' or 'a|b:1-4'")
        return cls(tuple(match["options"].split("|")), int(match["level"]))

    @property
    def is_choice(self) -> bool:
        return len(self.options) > 1

    def __str__(self) -> str:
        return f"{'|'.join(self.options)}:{self.level}"


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    kind: str
    description: str
    requires: tuple[Requirement, ...] = ()
    onet: tuple[str, ...] = ()


@dataclass(frozen=True)
class Role:
    id: str
    name: str
    family: str
    summary: str
    onet: str
    levels: tuple[str, ...]


@dataclass(frozen=True)
class RoleLevel:
    role: str
    level: str

    @classmethod
    def parse(cls, text: str) -> "RoleLevel":
        match = _ROLE_LEVEL.match(str(text).strip())
        if not match:
            raise CatalogError(f"bad role level {text!r}: expected 'role@level'")
        return cls(match["role"], match["level"])

    def __str__(self) -> str:
        return f"{self.role}@{self.level}"


@dataclass(frozen=True)
class CommonPath:
    """A move between roles that people commonly make (ADR-0027). Any move is possible; these are evidence."""

    source: RoleLevel
    target: RoleLevel
    kind: str
    typical_years: str | None = None


@dataclass(frozen=True)
class Stage:
    name: str
    items: tuple[Requirement, ...]
    optional: bool = False  # "good to know": never a prerequisite for required skills


@dataclass(frozen=True)
class RoadmapLevel:
    level: str
    summary: str
    stages: tuple[Stage, ...]
    title: str | None = None  # a per-role title for the level, e.g. "Senior Engineering Manager"


@dataclass(frozen=True)
class Roadmap:
    role: str
    levels: tuple[RoadmapLevel, ...]  # in ladder order


@dataclass
class Catalog:
    levels: dict[str, dict]
    families: dict[str, str]
    skills: dict[str, Skill]
    roles: dict[str, Role]
    common_paths: list[CommonPath]
    legacy_roles: dict[str, str | None]
    roadmaps: dict[str, Roadmap] = field(default_factory=dict)
    roadmap_files: dict[str, str] = field(default_factory=dict)  # role id -> file name, to check naming


def _read(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CatalogError(f"{path.name}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise CatalogError(f"{path.name}: expected a mapping at the top level")
    return data


def _skill(skill_id: str, raw: dict) -> Skill:
    try:
        return Skill(
            id=skill_id,
            name=raw["name"],
            kind=raw["kind"],
            description=raw["description"],
            requires=tuple(Requirement.parse(r) for r in raw.get("requires") or ()),
            onet=tuple(raw.get("onet") or ()),
        )
    except KeyError as exc:
        raise CatalogError(f"skills.yaml: skill {skill_id!r} lacks {exc}") from exc


def _roadmap(path: Path) -> Roadmap:
    raw = _read(path)
    levels = []
    for level, body in (raw.get("levels") or {}).items():
        body = body or {}
        stages = tuple(
            Stage(
                stage["name"],
                tuple(Requirement.parse(item) for item in stage.get("skills") or ()),
                bool(stage.get("optional", False)),
            )
            for stage in body.get("stages") or ()
        )
        levels.append(RoadmapLevel(level, body.get("summary", ""), stages, body.get("title")))
    levels.sort(key=lambda lv: LADDER.index(lv.level) if lv.level in LADDER else len(LADDER))
    return Roadmap(raw.get("role", ""), tuple(levels))


def load_catalog(directory: Path = CATALOG_DIR) -> Catalog:
    skills_raw = _read(directory / "skills.yaml").get("skills") or {}
    roles_raw = _read(directory / "roles.yaml")
    catalog = Catalog(
        levels=roles_raw.get("levels") or {},
        families=roles_raw.get("families") or {},
        skills={sid: _skill(sid, body) for sid, body in skills_raw.items()},
        roles={
            rid: Role(rid, body["name"], body["family"], body["summary"], str(body["onet"]), tuple(body["levels"]))
            for rid, body in (roles_raw.get("roles") or {}).items()
        },
        common_paths=[
            CommonPath(RoleLevel.parse(t["from"]), RoleLevel.parse(t["to"]), t["kind"], t.get("typical_years"))
            for t in roles_raw.get("common_paths") or ()
        ],
        legacy_roles=roles_raw.get("legacy_roles") or {},
    )
    for path in sorted((directory / "roadmaps").glob("*.yaml")):
        roadmap = _roadmap(path)
        catalog.roadmaps[roadmap.role] = roadmap
        catalog.roadmap_files[roadmap.role] = path.stem
    return catalog
