"""Content checks for the catalog (ADR-0028). Run in CI on every pull request that touches `catalog/`.

Errors block a merge; warnings are for the reviewer. The central rule: a roadmap may only ask for a
skill once its required prerequisites are in place (earlier stage or level, at enough proficiency).
Optional ("good to know") stages must meet their own prerequisites but never count as one, since a
learner may skip them.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

from xcrs.catalog.model import LADDER, SKILL_KINDS, TRANSITION_KINDS, Catalog, Requirement, RoleLevel

_SLUG = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
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
    for t in cat.transitions:
        where = f"transition {t.source} -> {t.target}"
        valid(t.source, where), valid(t.target, where)
        if t.kind not in TRANSITION_KINDS:
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
    entry_roles = {r.id for r in cat.roles.values()}
    reachable = {t.target.role for t in cat.transitions} | {r.id for r in cat.roles.values() if r.levels[0] == "entry"}
    for rid in sorted(entry_roles - reachable):
        report.warnings.append(f"role {rid}: starts above entry level but no transition leads to it")


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


def _check_transitions_bridge(cat: Catalog, report: Report) -> None:
    for t in cat.transitions:
        if t.source.role in cat.roadmaps and t.target.role in cat.roadmaps:
            gap = bridge(cat, t.source, t.target)
            if not gap:
                report.warnings.append(f"transition {t.source} -> {t.target}: nothing to learn; is it a real move?")


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


def validate(cat: Catalog) -> Report:
    report = Report()
    _check_skills(cat, report)
    _check_roles(cat, report)
    usage = _check_roadmaps(cat, report)
    if not report.errors:
        _check_transitions_bridge(cat, report)
    unused = sorted(set(cat.skills) - set(usage))
    if unused:
        report.warnings.append(f"{len(unused)} skills are in no roadmap: {', '.join(unused)}")
    report.stats = {
        "skills": len(cat.skills),
        "skills_by_kind": dict(Counter(s.kind for s in cat.skills.values()).most_common()),
        "prerequisite_edges": sum(len(r.options) for s in cat.skills.values() for r in s.requires),
        "roles": len(cat.roles),
        "transitions": len(cat.transitions),
        "roadmap_items": sum(len(st.items) for rm in cat.roadmaps.values() for lv in rm.levels for st in lv.stages),
        "most_shared_skills": [f"{s} ({n} roles)" for s, n in usage.most_common(10)],
    }
    return report
