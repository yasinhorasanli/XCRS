"""Content checks for the catalog (ADR-0028). Run in CI on every pull request that touches `catalog/`.

Errors block a merge; warnings are for the reviewer. The central rule: a roadmap may only ask for a
skill once its required prerequisites are in place (earlier stage or level, at enough proficiency).
Optional ("good to know") stages must meet their own prerequisites but never count as one, since a
learner may skip them.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from xcrs.catalog.model import (
    LADDER,
    PATH_KINDS,
    RESOURCE_LEVELS,
    RESOURCE_TYPES,
    SKILL_KINDS,
    Alias,
    Catalog,
    Requirement,
    Role,
    RoleLevel,
)

_SLUG = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_ESCO = re.compile(r"^http://data\.europa\.eu/esco/occupation/[0-9a-f-]{36}$")
_ONET = re.compile(r"^\d{2}-\d{4}\.\d{2}$")


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _check_skills(cat: Catalog, report: Report) -> None:
    for skill in cat.skills.values():
        if not _SLUG.match(skill.id):
            report.errors.append(f"skill {skill.id!r}: id must be lowercase-kebab-case")
        if skill.kind not in SKILL_KINDS:
            report.errors.append(f"skill {skill.id}: unknown kind {skill.kind!r}")
        if not skill.description.strip():
            report.errors.append(f"skill {skill.id}: empty description")
        for req in skill.requires:
            for option in req.options:
                if option == skill.id:
                    report.errors.append(f"skill {skill.id}: requires itself")
                elif option not in cat.skills:
                    report.errors.append(f"skill {skill.id}: requires unknown skill {option!r}")
    names = Counter(s.name.lower() for s in cat.skills.values())
    report.errors += [f"skill name {n!r} is used by {c} skills" for n, c in names.items() if c > 1]

    # Prerequisites must form a DAG (every option counts as an edge).
    state: dict[str, int] = {}  # 1 = on the current path, 2 = done

    def visit(node: str, path: list[str]) -> None:
        state[node] = 1
        for req in cat.skills[node].requires:
            for nxt in req.options:
                if nxt not in cat.skills:
                    continue
                if state.get(nxt) == 1:
                    cycle = [*path[path.index(nxt) :], nxt] if nxt in path else [node, nxt]
                    report.errors.append(f"prerequisite cycle: {' -> '.join(cycle)}")
                elif state.get(nxt) is None:
                    visit(nxt, [*path, nxt])
        state[node] = 2

    for sid in cat.skills:
        if sid not in state:
            visit(sid, [sid])


def _check_roles(cat: Catalog, report: Report) -> None:
    if tuple(cat.levels) != LADDER:
        report.errors.append(f"roles.yaml: levels must be {', '.join(LADDER)} in that order")
    for role in cat.roles.values():
        where = f"role {role.id}"
        if not _SLUG.match(role.id):
            report.errors.append(f"{where}: id must be lowercase-kebab-case")
        if role.family not in cat.families:
            report.errors.append(f"{where}: unknown family {role.family!r}")
        if role.esco is not None and not _ESCO.match(role.esco):
            report.errors.append(f"{where}: ESCO {role.esco!r} must be an ESCO occupation URI")
        if not _ONET.match(role.onet):
            report.errors.append(f"{where}: O*NET code {role.onet!r} must look like 15-1252.00")
        positions = [LADDER.index(lv) for lv in role.levels if lv in LADDER]
        if (
            not positions
            or len(positions) != len(role.levels)
            or positions != list(range(positions[0], positions[0] + len(positions)))
        ):
            report.errors.append(f"{where}: levels must be consecutive ladder levels, got {list(role.levels)}")

    def valid(rl: RoleLevel, where: str) -> bool:
        if rl.role not in cat.roles:
            report.errors.append(f"{where}: unknown role {rl.role!r}")
            return False
        if rl.level not in cat.roles[rl.role].levels:
            report.errors.append(f"{where}: {rl.role} has no level {rl.level!r}")
            return False
        return True

    seen = set()
    for t in cat.common_paths:
        where = f"common path {t.source} -> {t.target}"
        valid(t.source, where), valid(t.target, where)
        if t.kind not in PATH_KINDS:
            report.errors.append(f"{where}: unknown kind {t.kind!r}")
        if t.source.role == t.target.role:
            report.errors.append(f"{where}: moving up inside a role is implied; list only cross-role moves")
        key = (str(t.source), str(t.target))
        if key in seen:
            report.errors.append(f"{where}: listed twice")
        seen.add(key)
    for legacy, new in cat.legacy_roles.items():
        if new is not None and new not in cat.roles:
            report.errors.append(f"legacy role {legacy}: maps to unknown role {new!r}")
    titles: dict[str, str] = {}
    for role in cat.roles.values():
        for title in [role.name, *(a.title for a in role.also_called)]:
            if title.lower() in titles:
                report.errors.append(f"role {role.id}: title {title!r} is already used by {titles[title.lower()]}")
            titles[title.lower()] = role.id
        for alias in role.also_called:
            for req in alias.adds:
                for option in req.options:
                    if option not in cat.skills:
                        report.errors.append(f"role {role.id}, also called {alias.title!r}: unknown skill {option!r}")
        # Title skills (ADR-0044) name the role's job titles, so they must be skills its roadmap teaches.
        roadmap = cat.roadmaps.get(role.id)
        taught = (
            {o for lv in roadmap.levels for st in lv.stages for it in st.items for o in it.options}
            if roadmap
            else set()
        )
        for skill in role.title_skills:
            if skill not in cat.skills:
                report.errors.append(f"role {role.id}: unknown title skill {skill!r}")
            elif skill not in taught:
                report.errors.append(f"role {role.id}: title skill {skill!r} is not in its roadmap")
        if len(set(role.title_skills)) != len(role.title_skills):
            report.errors.append(f"role {role.id}: a title skill is listed twice")
    entry_roles = {r.id for r in cat.roles.values()}
    reachable = {t.target.role for t in cat.common_paths} | {r.id for r in cat.roles.values() if r.levels[0] == "entry"}
    for rid in sorted(entry_roles - reachable):
        report.warnings.append(f"role {rid}: starts above entry level but no common path leads to it")


def _satisfied(req: Requirement, have: dict[str, int]) -> bool:
    return any(have.get(option, 0) >= req.level for option in req.options)


def _check_roadmaps(cat: Catalog, report: Report) -> Counter:
    usage: Counter = Counter()
    for rid in cat.roles:
        if rid not in cat.roadmaps:
            report.errors.append(f"role {rid}: no roadmap file roadmaps/{rid}.yaml")
    for rid, roadmap in cat.roadmaps.items():
        if rid not in cat.roles:
            report.errors.append(f"roadmaps/{cat.roadmap_files.get(rid, rid)}.yaml: unknown role {rid!r}")
            continue
        if cat.roadmap_files.get(rid) != rid:
            report.errors.append(f"roadmap for {rid} must be in roadmaps/{rid}.yaml")
        role = cat.roles[rid]
        if tuple(lv.level for lv in roadmap.levels) != role.levels:
            report.errors.append(f"{rid}: roadmap levels {[lv.level for lv in roadmap.levels]} != role levels")
        have: dict[str, int] = {}  # cumulative: what someone at this point of the roadmap knows
        used_by_role: set[str] = set()
        for level in roadmap.levels:
            if not level.summary.strip():
                report.errors.append(f"{rid}@{level.level}: missing summary")
            if not level.stages or not any(stage.items for stage in level.stages):
                report.errors.append(f"{rid}@{level.level}: no skills")
            for stage in level.stages:
                where = f"{rid}@{level.level} [{stage.name}]"
                stage_have = dict(have)  # items of one stage may support each other (order-free)
                for item in stage.items:
                    for option in item.options:
                        if option not in cat.skills:
                            report.errors.append(f"{where}: unknown skill {option!r}")
                        stage_have[option] = max(stage_have.get(option, 0), item.level)
                for item in stage.items:
                    for option in item.options:
                        if option not in cat.skills:
                            continue
                        previous = have.get(option)
                        if previous is not None and not item.is_choice:
                            if item.level < previous:
                                report.errors.append(f"{where}: {option} drops from {previous} to {item.level}")
                            elif item.level == previous:
                                report.warnings.append(f"{where}: {option}:{item.level} repeats an earlier level")
                        for req in cat.skills[option].requires:
                            if not _satisfied(req, stage_have):
                                got = max((stage_have.get(o, 0) for o in req.options), default=0)
                                report.errors.append(
                                    f"{where}: {option} needs {req}" + (f" (roadmap has {got})" if got else "")
                                )
                        used_by_role.add(option)
                if not stage.optional:
                    have = stage_have
        usage.update(used_by_role)
    return usage


def _check_common_paths(cat: Catalog, report: Report) -> None:
    """A common path (other than to a leadership role, which is meant to need new skills) whose target is
    in the far half of the source's ranked moves gets a second look: maybe right, but a big gap."""
    ranked: dict[str, list[Move]] = {}
    for t in cat.common_paths:
        if t.source.role not in cat.roadmaps or t.target.role not in cat.roadmaps:
            continue
        if coverage(cat, t.source, t.target) >= 1.0:
            report.warnings.append(f"common path {t.source} -> {t.target}: nothing to learn; is it a real move?")
        if t.kind == "lead":
            continue
        ms = ranked.setdefault(str(t.source), moves(cat, t.source))
        rank = next(i for i, m in enumerate(ms, 1) if m.role == t.target.role)
        if rank > len(ms) / 2:
            report.warnings.append(
                f"common path {t.source} -> {t.target}: {t.target.role} is only #{rank} of {len(ms)} nearest roles; "
                "confirm the move is common"
            )


# A title stays an alias while the role covers at least this share of what the title asks for; below it,
# the title is a different job and needs its own role and roadmap (ADR-0027).
ALIAS_MIN_COVERAGE = 0.8


def alias_level(role: Role) -> str:
    """Titles are compared where most hiring happens: mid level, or the role's first level."""
    return "mid" if "mid" in role.levels else role.levels[0]


def alias_coverage(cat: Catalog, role: Role, alias: Alias, weights: dict[str, float] | None = None) -> float:
    """Share of the alias's requirements (the role's, plus `adds`) that the role itself covers."""
    at = RoleLevel(role.id, alias_level(role))
    target = requirements(cat, at)
    for req in alias.adds:
        target[req.options] = max(target.get(req.options, 0), req.level)
    return _covered(requirements(cat, at), target, weights or skill_weights(cat))


def _check_aliases(cat: Catalog, report: Report) -> None:
    weights = skill_weights(cat)
    for role in cat.roles.values():
        for alias in role.also_called:
            if not alias.adds:
                continue
            have = {
                o: lv for opts, lv in requirements(cat, RoleLevel(role.id, alias_level(role))).items() for o in opts
            }
            for req in alias.adds:
                have.update({o: max(have.get(o, 0), req.level) for o in req.options})
            for req in alias.adds:
                for option in req.options:
                    for pre in cat.skills[option].requires if option in cat.skills else ():
                        if not _satisfied(pre, have):
                            report.errors.append(f"role {role.id}, also called {alias.title!r}: {option} needs {pre}")
            covered = alias_coverage(cat, role, alias, weights)
            if covered < ALIAS_MIN_COVERAGE:
                report.errors.append(
                    f"role {role.id}, also called {alias.title!r}: the role covers only {covered:.0%} of it "
                    f"(< {ALIAS_MIN_COVERAGE:.0%}); make it a role of its own"
                )


def requirements(cat: Catalog, at: RoleLevel) -> dict[tuple[str, ...], int]:
    """Required skills up to `at` in a role's roadmap, keyed by options (a choice stays one requirement)."""
    need: dict[tuple[str, ...], int] = {}
    for level in cat.roadmaps[at.role].levels:
        for stage in level.stages:
            if stage.optional:
                continue
            for item in stage.items:
                need[item.options] = max(need.get(item.options, 0), item.level)
        if level.level == at.level:
            break
    return need


def bridge(cat: Catalog, source: RoleLevel, target: RoleLevel) -> dict[str, tuple[int, int]]:
    """What moving from `source` to `target` asks for: "skill" or "a|b" -> (have, need), gaps only.
    Someone who reached `source` is assumed to know any option of a choice they went through."""
    have: dict[str, int] = {}
    for options, level in requirements(cat, source).items():
        for option in options:
            have[option] = max(have.get(option, 0), level)
    gap = {}
    for options, level in requirements(cat, target).items():
        best = max(have.get(o, 0) for o in options)
        if best < level:
            gap["|".join(options)] = (best, level)
    return gap


# --- Distance between roles (ADR-0027): any move is possible; this measures how far it is -----------

# Someone is taken to start a role at the highest level whose requirements they already cover this much.
STARTING_COVERAGE = 0.55


def skill_weights(cat: Catalog) -> dict[str, float]:
    """How defining a skill is: rare across roles weighs more than shared basics (inverse role frequency).
    Git and testing say little about whether someone is a data engineer; dbt and Airflow say a lot."""
    used_by: Counter = Counter()
    for rid, role in cat.roles.items():
        if rid in cat.roadmaps:
            used_by.update({o for options in requirements(cat, RoleLevel(rid, role.levels[-1])) for o in options})
    n = len(cat.roles)
    return {s: math.log((1 + n) / (1 + used_by.get(s, 0))) + 1 for s in cat.skills}


def coverage(cat: Catalog, source: RoleLevel, target: RoleLevel, weights: dict[str, float] | None = None) -> float:
    """Share of the target's requirements (weighted by distinctiveness and proficiency) that someone at
    `source` already meets; partial proficiency counts partly. 1.0 means nothing left to learn."""
    return _covered(requirements(cat, source), requirements(cat, target), weights or skill_weights(cat))


def _covered(
    source: dict[tuple[str, ...], int], target: dict[tuple[str, ...], int], weights: dict[str, float]
) -> float:
    have: dict[str, int] = {}
    for options, level in source.items():
        for option in options:
            have[option] = max(have.get(option, 0), level)
    total = met = 0.0
    for options, level in target.items():
        weight = max(weights[o] for o in options) * level
        total += weight
        met += weight * min(1.0, max(have.get(o, 0) for o in options) / level)
    return met / total if total else 1.0


@dataclass(frozen=True)
class Move:
    role: str
    coverage: float  # of the role's first level
    starting_level: str | None  # highest level, up to the source's, covered >= STARTING_COVERAGE, if any
    starting_coverage: float | None
    common: tuple[str, ...]  # target levels listed as common paths from the source


def moves(cat: Catalog, source: RoleLevel) -> list[Move]:
    """Every other role, nearest first, with the level the source would likely start at: the highest level
    covered enough, never above the source's own level (changing roles doesn't promote anyone)."""
    weights = skill_weights(cat)
    result = []
    for rid, role in cat.roles.items():
        if rid == source.role or rid not in cat.roadmaps:
            continue
        start, start_cov = None, None
        for level in role.levels:
            if source.level in LADDER and LADDER.index(level) > LADDER.index(source.level):
                break
            cov = coverage(cat, source, RoleLevel(rid, level), weights)
            if cov >= STARTING_COVERAGE:
                start, start_cov = level, cov
        common = tuple(str(p.target.level) for p in cat.common_paths if p.source == source and p.target.role == rid)
        first = coverage(cat, source, RoleLevel(rid, role.levels[0]), weights)
        result.append(Move(rid, first, start, start_cov, common))
    return sorted(result, key=lambda m: -m.coverage)


def _check_resources(cat: Catalog, report: Report) -> None:
    seen: set[str] = set()
    for r in cat.resources:
        where = f"resource {r.url}"
        if not r.url.startswith("https://"):
            report.errors.append(f"{where}: URL must be https")
        if r.url in seen:
            report.errors.append(f"{where}: listed twice")
        seen.add(r.url)
        if r.type not in RESOURCE_TYPES:
            report.errors.append(f"{where}: unknown type {r.type!r}")
        if r.level is not None and r.level not in RESOURCE_LEVELS:
            report.errors.append(f"{where}: unknown level {r.level!r}")
        if not r.title.strip() or not r.provider.strip():
            report.errors.append(f"{where}: needs a title and a provider")
        if not r.teaches:
            report.errors.append(f"{where}: teaches no skill")
        for req in r.teaches:
            if req.is_choice:
                report.errors.append(f"{where}: teaches a choice {req}; list the skills separately")
            for option in req.options:
                if option not in cat.skills:
                    report.errors.append(f"{where}: unknown skill {option!r}")
    taught = {o for r in cat.resources for req in r.teaches for o in req.options}
    missing = sorted(set(cat.skills) - taught)
    if cat.resources and missing:
        report.warnings.append(f"{len(missing)} skills have no curated resource yet: {', '.join(missing)}")


def validate(cat: Catalog) -> Report:
    report = Report()
    _check_skills(cat, report)
    _check_resources(cat, report)
    _check_roles(cat, report)
    usage = _check_roadmaps(cat, report)
    if not report.errors:
        _check_common_paths(cat, report)
        _check_aliases(cat, report)
    in_aliases = {o for r in cat.roles.values() for a in r.also_called for req in a.adds for o in req.options}
    unused = sorted(set(cat.skills) - set(usage) - in_aliases)
    if unused:
        report.warnings.append(f"{len(unused)} skills are in no roadmap or title: {', '.join(unused)}")
    report.stats = {
        "skills": len(cat.skills),
        "skills_by_kind": dict(Counter(s.kind for s in cat.skills.values()).most_common()),
        "prerequisite_edges": sum(len(r.options) for s in cat.skills.values() for r in s.requires),
        "roles": len(cat.roles),
        "common_paths": len(cat.common_paths),
        "titles": len(cat.roles) + sum(len(r.also_called) for r in cat.roles.values()),
        "resources": len(cat.resources),
        "roadmap_items": sum(len(st.items) for rm in cat.roadmaps.values() for lv in rm.levels for st in lv.stages),
        "most_shared_skills": [f"{s} ({n} roles)" for s, n in usage.most_common(10)],
    }
    return report
