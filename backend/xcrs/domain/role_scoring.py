"""Engine v2 role scoring (ADR-0029, ADR-0031). Pure functions over a catalog snapshot; no I/O.

For each role:
- **coverage** of a level: the share of that level's requirements the learner meets, each weighted by how
  distinctive the skill is and by the proficiency asked; partial proficiency counts partly. Skills the
  learner enjoyed, felt neutral about or didn't enjoy count as known (at their 1-4 rating, unrated = 1);
  curious ones don't, since that's what they want to learn.
- **interest**: the share of the learner's attention that falls on this role's skills. Every skill they
  mentioned carries a category weight (curious, liked, neutral, disliked) times its distinctiveness;
  a role collects the weights of the mentioned skills it requires, over the total of all mentioned.
- **score** = a * interest + (1 - a) * coverage, with `a` and the category weights calibrated on labeled
  learner profiles (eval/bench_role_scoring.py).
- **level**: the highest level whose own additions, and every lower level's, are at least
  STARTING_COVERAGE met (the bar of `xcrs catalog moves`); None means "start at the role's first level".
- **gaps**: what the next level asks for that the learner doesn't have yet, in roadmap order.

Known skills imply their prerequisites (the skills graph, ADR-0028): someone who knows Django knows Python
at working level, even if they didn't list it. Learners name about ten skills, not the forty a level asks for.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

LADDER = ("entry", "mid", "senior", "staff")
STARTING_COVERAGE = 0.55


class Category(StrEnum):
    LIKED = "liked"
    NEUTRAL = "neutral"
    DISLIKED = "disliked"
    CURIOUS = "curious"


KNOWN = {Category.LIKED, Category.NEUTRAL, Category.DISLIKED}


@dataclass(frozen=True)
class Weights:
    """The defaults are the values calibrated on eval/learner_profiles.yaml (ADR-0031): held-out top-1 96%,
    top-3 100%, MRR 0.98; a between 0.3 and 0.5 performs alike, 0.4 is the middle."""

    interest: float = 0.4  # a: the share of the score that is interest (the rest is coverage)
    curious: float = 1.25
    liked: float = 0.75
    neutral: float = 0.0  # neutral skills count for coverage, not for interest
    disliked: float = -0.5
    coverage_at: str = "mean"  # which coverage ranks roles: "first" level, "top" level, or "mean" of levels

    def of(self, category: Category) -> float:
        return getattr(self, Category(category).value)


DEFAULT_WEIGHTS = Weights()


@dataclass(frozen=True)
class Mention:
    skill: str
    category: Category
    proficiency: int | None = None  # 1-4, None = not rated


@dataclass(frozen=True)
class Requirement:
    options: tuple[str, ...]
    level: int
    stage: str
    order: int  # position in the roadmap, for showing gaps in learning order


@dataclass
class RoleSnapshot:
    id: str
    name: str
    family: str
    levels: list[str]  # ladder levels the role has, in order
    titles: dict[str, str | None]  # level -> title (e.g. "Principal Scientist")
    requirements: dict[str, list[Requirement]]  # level -> cumulative required items, roadmap order


@dataclass
class CatalogSnapshot:
    roles: dict[str, RoleSnapshot]
    skill_names: dict[str, str]
    prerequisites: dict[str, list[tuple[tuple[str, ...], int]]] = field(default_factory=dict)  # skill -> requires
    weights: dict[str, float] = field(default_factory=dict)  # distinctiveness per skill

    def __post_init__(self) -> None:
        if not self.weights:
            self.weights = distinctiveness(self.roles.values(), self.skill_names)


def distinctiveness(roles: Iterable[RoleSnapshot], skills: Iterable[str]) -> dict[str, float]:
    """Inverse role frequency, as `xcrs.catalog.validate.skill_weights`: rare across roles weighs more."""
    roles = list(roles)
    used: dict[str, int] = {}
    for role in roles:
        top = role.requirements[role.levels[-1]]
        for option in {o for r in top for o in r.options}:
            used[option] = used.get(option, 0) + 1
    n = len(roles)
    return {s: math.log((1 + n) / (1 + used.get(s, 0))) + 1 for s in skills}


@dataclass
class Gap:
    options: tuple[str, ...]
    need: int
    have: int
    stage: str


@dataclass
class RoleScore:
    role: str
    score: float
    interest: float
    coverage: float  # the one used for ranking (Weights.coverage_at)
    level_coverage: dict[str, float]
    level: str | None
    target_level: str  # the level the gaps lead to
    gaps: list[Gap]
    because: list[str]  # the learner's skills that count most for this role, strongest first


def with_prerequisites(have: dict[str, int], prerequisites: dict[str, list[tuple[tuple[str, ...], int]]]):
    """Knowing a skill implies its prerequisites at the proficiency they ask for (Django:2 needs Python:2),
    recursively. A choice ("python|go:2") implies nothing unless the learner named one of its options."""
    implied = dict(have)
    stack = list(have)
    while stack:
        skill = stack.pop()
        for options, level in prerequisites.get(skill, ()):
            if len(options) == 1:
                target = options[0]
            else:
                named = [o for o in options if o in have]
                if not named:
                    continue
                target = max(named, key=lambda o: have[o])
            if implied.get(target, 0) < level:
                implied[target] = level
                stack.append(target)
    return implied


def learner_skills(mentions: Iterable[Mention]) -> tuple[dict[str, int], dict[str, Category]]:
    """Known proficiency per skill, and one category per skill. A skill mentioned twice keeps its
    highest rating, and the more telling category (liked > curious > neutral > disliked)."""
    rank = {Category.LIKED: 3, Category.CURIOUS: 2, Category.NEUTRAL: 1, Category.DISLIKED: 0}
    have: dict[str, int] = {}
    category: dict[str, Category] = {}
    for m in mentions:
        if m.category in KNOWN:
            have[m.skill] = max(have.get(m.skill, 0), m.proficiency or 1)
        if m.skill not in category or rank[m.category] > rank[category[m.skill]]:
            category[m.skill] = m.category
    return have, category


def _coverage(requirements: list[Requirement], have: dict[str, int], weights: dict[str, float]) -> float:
    total = met = 0.0
    for r in requirements:
        w = max(weights.get(o, 1.0) for o in r.options) * r.level
        total += w
        met += w * min(1.0, max(have.get(o, 0) for o in r.options) / r.level)
    return met / total if total else 0.0


def level_additions(role: RoleSnapshot) -> dict[str, list[Requirement]]:
    """What each level adds to the one below: new items, or items asked at a higher proficiency."""
    out, before = {}, {}
    for lv in role.levels:
        out[lv] = [r for r in role.requirements[lv] if before.get(r.options, 0) < r.level]
        before = {r.options: r.level for r in role.requirements[lv]}
    return out


def reached_level(role: RoleSnapshot, have: dict[str, int], weights: dict[str, float]) -> str | None:
    """The highest level whose own additions (and every lower level's) are at least STARTING_COVERAGE met.
    Cumulative coverage alone overshoots: meeting all of mid also covers most of staff's list."""
    reached = None
    for lv, added in level_additions(role).items():
        if _coverage(added, have, weights) < STARTING_COVERAGE:
            break
        reached = lv
    return reached


def score_roles(
    snapshot: CatalogSnapshot, mentions: Iterable[Mention], weights: Weights = DEFAULT_WEIGHTS
) -> list[RoleScore]:
    """Every role, best first."""
    mentions = list(mentions)
    have, category = learner_skills(mentions)
    have = with_prerequisites(have, snapshot.prerequisites)
    attention = sum(abs(weights.of(c)) * snapshot.weights.get(s, 1.0) for s, c in category.items()) or 1.0
    results = []
    for role in snapshot.roles.values():
        level_coverage = {lv: _coverage(role.requirements[lv], have, snapshot.weights) for lv in role.levels}
        if weights.coverage_at == "first":
            coverage = level_coverage[role.levels[0]]
        elif weights.coverage_at == "top":
            coverage = level_coverage[role.levels[-1]]
        else:
            coverage = sum(level_coverage.values()) / len(level_coverage)
        in_role = {o for r in role.requirements[role.levels[-1]] for o in r.options}
        contributions = {s: weights.of(c) * snapshot.weights.get(s, 1.0) for s, c in category.items() if s in in_role}
        interest = sum(contributions.values()) / attention
        level = reached_level(role, have, snapshot.weights)
        target = (
            role.levels[0] if level is None else role.levels[min(role.levels.index(level) + 1, len(role.levels) - 1)]
        )
        gaps = [
            Gap(r.options, r.level, max(have.get(o, 0) for o in r.options), r.stage)
            for r in role.requirements[target]
            if max(have.get(o, 0) for o in r.options) < r.level
        ]
        because = [s for s, v in sorted(contributions.items(), key=lambda kv: -kv[1]) if v > 0]
        score = weights.interest * interest + (1 - weights.interest) * coverage
        results.append(RoleScore(role.id, score, interest, coverage, level_coverage, level, target, gaps, because))
    return sorted(results, key=lambda r: -r.score)
