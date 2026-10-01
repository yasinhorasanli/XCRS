"""The catalog v2 in the database (ADR-0028): import from the YAML model, and read requirements back.

Entities (levels, families, skills, roles, role levels) are upserted by their natural key, so their ids
stay stable across imports and future tables (resources, activity) can refer to them. Their parts
(prerequisites, titles, stages, items, common paths, legacy mapping) are replaced on every import:
nothing outside the catalog refers to them. Entities no longer in the YAML are deleted; a delete that
something else still refers to fails the import (and rolls it back) instead of silently losing data.
The caller owns the transaction.
"""

from collections.abc import Iterable, Sequence

from sqlalchemy import Table, delete, func, literal_column, or_, select, tuple_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from xcrs.catalog.model import LADDER, Catalog, Requirement
from xcrs.catalog.snapshot import cumulative_requirements, optional_skills
from xcrs.db.models import (
    CareerRole,
    CareerRoleLevel,
    CatalogImport,
    CommonPath,
    Family,
    LegacyRoleMap,
    Level,
    RoadmapItem,
    RoadmapStage,
    RoleTitle,
    RoleTitleSkill,
    Skill,
    SkillPrerequisite,
)
from xcrs.domain.role_scoring import CatalogSnapshot, RoleSnapshot


def _upsert(session: Session, table: Table, key: Sequence[str], rows: list[dict]) -> tuple[list, list]:
    """Insert or update rows by `key`; rows whose values didn't change aren't touched.
    Returns the keys of (inserted, updated) rows."""
    if not rows:
        return [], []
    stmt = insert(table).values(rows)
    columns = [c for c in rows[0] if c not in key]
    changed = or_(*(table.c[c].is_distinct_from(stmt.excluded[c]) for c in columns))
    updates = {c: stmt.excluded[c] for c in columns}
    if "updated_at" in table.c:
        updates["updated_at"] = literal_column("now()")
    stmt = stmt.on_conflict_do_update(index_elements=list(key), set_=updates, where=changed).returning(
        *(table.c[k] for k in key), literal_column("xmax = 0").label("inserted")
    )
    inserted, updated = [], []
    for row in session.execute(stmt):
        (inserted if row.inserted else updated).append(row[0] if len(key) == 1 else tuple(row[: len(key)]))
    return inserted, updated


def _remove_absent(session: Session, table: Table, column: str, keep: Iterable) -> list:
    return list(session.scalars(delete(table).where(table.c[column].not_in(list(keep))).returning(table.c[column])))


def _requirement_rows(owner: dict, requirements: Iterable[Requirement], skill_ids: dict[str, int], no: str):
    """One row per option; rows sharing the number are alternatives of one requirement."""
    return [
        {**owner, no: n, "option_skill_id": skill_ids[option], "min_level": req.level}
        for n, req in enumerate(requirements, 1)
        for option in req.options
    ]


def import_catalog(
    session: Session, cat: Catalog, *, git_commit: str | None, git_dirty: bool, checksum: str, stats: dict
) -> dict:
    """Make the `catalog` schema match `cat` (validated beforehand). Returns what changed."""
    changes: dict = {}

    def track(name: str, inserted: list, updated: list, removed: list) -> None:
        changes[name] = {"added": sorted(map(str, inserted)), "updated": sorted(map(str, updated))}
        changes[name]["removed"] = sorted(map(str, removed))

    # Entities, by natural key.
    levels = Level.__table__
    level_ids = {slug: i for i, slug in enumerate(LADDER, 1)}  # the id is the position on the ladder
    rows = [
        {"id": level_ids[slug], "slug": slug, **{k: str(v) for k, v in cat.levels[slug].items()}} for slug in LADDER
    ]
    track("levels", *_upsert(session, levels, ["id"], rows), [])

    families = Family.__table__
    rows = [{"slug": slug, "name": name} for slug, name in cat.families.items()]
    family_changes = _upsert(session, families, ["slug"], rows)
    family_ids = dict(session.execute(select(families.c.slug, families.c.id)).all())

    skills = Skill.__table__
    rows = [
        {"slug": s.id, "name": s.name, "kind": s.kind, "description": s.description, "onet": list(s.onet)}
        for s in cat.skills.values()
    ]
    skill_changes = _upsert(session, skills, ["slug"], rows)
    skill_ids = dict(session.execute(select(skills.c.slug, skills.c.id)).all())

    roles = CareerRole.__table__
    rows = [
        {"slug": r.id, "name": r.name, "family_id": family_ids[r.family], "summary": r.summary, "onet_code": r.onet}
        for r in cat.roles.values()
    ]
    role_changes = _upsert(session, roles, ["slug"], rows)
    role_ids = dict(session.execute(select(roles.c.slug, roles.c.id)).all())

    role_levels = CareerRoleLevel.__table__
    rows = [
        {"role_id": role_ids[rm.role], "level_id": level_ids[lv.level], "title": lv.title, "summary": lv.summary}
        for rm in cat.roadmaps.values()
        for lv in rm.levels
    ]
    keep = [(r["role_id"], r["level_id"]) for r in rows]
    _upsert(session, role_levels, ["role_id", "level_id"], rows)
    session.execute(delete(role_levels).where(tuple_(role_levels.c.role_id, role_levels.c.level_id).not_in(keep)))

    # Parts: replaced as a whole.
    for model in (SkillPrerequisite, RoleTitle, RoadmapStage, CommonPath, LegacyRoleMap):
        session.execute(delete(model))
    prerequisites = [
        row
        for s in cat.skills.values()
        for row in _requirement_rows({"skill_id": skill_ids[s.id]}, s.requires, skill_ids, "group_no")
    ]
    if prerequisites:
        session.execute(insert(SkillPrerequisite), prerequisites)

    titles = [(r, a) for r in cat.roles.values() for a in r.also_called]
    if titles:
        title_ids = session.scalars(
            insert(RoleTitle).returning(RoleTitle.id, sort_by_parameter_order=True),
            [{"role_id": role_ids[r.id], "title": a.title} for r, a in titles],
        ).all()
        title_skills = [
            row
            for title_id, (_, alias) in zip(title_ids, titles, strict=True)
            for row in _requirement_rows({"title_id": title_id}, alias.adds, skill_ids, "item_no")
        ]
        if title_skills:
            session.execute(insert(RoleTitleSkill), title_skills)

    stages = [
        (rm.role, lv.level, position, stage)
        for rm in cat.roadmaps.values()
        for lv in rm.levels
        for position, stage in enumerate(lv.stages, 1)
    ]
    stage_ids = session.scalars(
        insert(RoadmapStage).returning(RoadmapStage.id, sort_by_parameter_order=True),
        [
            {"role_id": role_ids[r], "level_id": level_ids[lv], "position": n, "name": st.name, "optional": st.optional}
            for r, lv, n, st in stages
        ],
    ).all()
    items = [
        row
        for stage_id, (*_, stage) in zip(stage_ids, stages, strict=True)
        for row in _requirement_rows({"stage_id": stage_id}, stage.items, skill_ids, "item_no")
    ]
    session.execute(insert(RoadmapItem), items)

    if cat.common_paths:
        session.execute(
            insert(CommonPath),
            [
                {
                    "from_role_id": role_ids[p.source.role],
                    "from_level_id": level_ids[p.source.level],
                    "to_role_id": role_ids[p.target.role],
                    "to_level_id": level_ids[p.target.level],
                    "kind": p.kind,
                    "typical_years": p.typical_years,
                }
                for p in cat.common_paths
            ],
        )
    if cat.legacy_roles:
        session.execute(
            insert(LegacyRoleMap),
            [{"legacy_slug": k, "role_id": role_ids.get(v) if v else None} for k, v in cat.legacy_roles.items()],
        )

    # Entities that left the YAML, once nothing in the catalog refers to them.
    track("roles", *role_changes, _remove_absent(session, roles, "slug", cat.roles))
    track("skills", *skill_changes, _remove_absent(session, skills, "slug", cat.skills))
    track("families", *family_changes, _remove_absent(session, families, "slug", cat.families))
    changes["parts"] = {
        "skill_prerequisites": len(prerequisites),
        "role_titles": len(titles),
        "roadmap_stages": len(stage_ids),
        "roadmap_items": len(items),
        "common_paths": len(cat.common_paths),
    }
    record = CatalogImport(git_commit=git_commit, git_dirty=git_dirty, checksum=checksum, stats=stats, changes=changes)
    session.add(record)
    session.flush()
    return changes


def last_import_checksum(session: Session) -> str | None:
    return session.scalar(select(CatalogImport.checksum).order_by(CatalogImport.id.desc()).limit(1))


def role_requirements(session: Session, role: str, level: str) -> dict[tuple[str, ...], int]:
    """Required skills up to `level` of a role's roadmap, keyed by options (sorted), as in
    `validate.requirements`: levels are cumulative, optional stages don't count."""
    option = Skill.__table__.alias("option")
    rows = session.execute(
        select(RoadmapItem.stage_id, RoadmapItem.item_no, RoadmapItem.min_level, option.c.slug)
        .join(RoadmapStage, RoadmapStage.id == RoadmapItem.stage_id)
        .join(CareerRole, CareerRole.id == RoadmapStage.role_id)
        .join(option, option.c.id == RoadmapItem.option_skill_id)
        .where(
            CareerRole.slug == role,
            RoadmapStage.level_id <= LADDER.index(level) + 1,
            RoadmapStage.optional.is_(False),
        )
    ).all()
    grouped: dict[tuple[int, int], tuple[list[str], int]] = {}
    for r in rows:
        grouped.setdefault((r.stage_id, r.item_no), ([], r.min_level))[0].append(r.slug)
    need: dict[tuple[str, ...], int] = {}
    for options, min_level in grouped.values():
        key = tuple(sorted(options))
        need[key] = max(need.get(key, 0), min_level)
    return need


def skill_display_names(session: Session, slugs: Iterable[str]) -> dict[str, str]:
    slugs = list(set(slugs))
    if not slugs:
        return {}
    return dict(session.execute(select(Skill.slug, Skill.name).where(Skill.slug.in_(slugs))).all())


_snapshots: dict[str, CatalogSnapshot] = {}


def load_snapshot(session: Session) -> CatalogSnapshot:
    """The imported catalog as the scoring engine sees it (xcrs.domain.role_scoring), cached per import
    checksum: a new `xcrs catalog import` is picked up on the next request."""
    checksum = last_import_checksum(session) or ""
    if checksum in _snapshots:
        return _snapshots[checksum]
    skills = dict(session.execute(select(Skill.id, Skill.slug)).all())
    names = dict(session.execute(select(Skill.slug, Skill.name)).all())
    prerequisites: dict[str, dict[int, tuple[list[str], int]]] = {}
    for row in session.execute(select(SkillPrerequisite)).scalars():
        group = prerequisites.setdefault(skills[row.skill_id], {}).setdefault(row.group_no, ([], row.min_level))
        group[0].append(skills[row.option_skill_id])
    level_slugs = dict(session.execute(select(Level.id, Level.slug)).all())
    items: dict[int, dict[int, tuple[list[str], int]]] = {}
    for row in session.execute(select(RoadmapItem)).scalars():
        item = items.setdefault(row.stage_id, {}).setdefault(row.item_no, ([], row.min_level))
        item[0].append(skills[row.option_skill_id])
    stages: dict[tuple[int, int], list] = {}
    for st in session.execute(select(RoadmapStage).order_by(RoadmapStage.position)).scalars():
        stage_items = [(tuple(sorted(o)), lv) for _, (o, lv) in sorted(items.get(st.id, {}).items())]
        stages.setdefault((st.role_id, st.level_id), []).append((st.name, st.optional, stage_items))
    role_rows = session.execute(
        select(CareerRole.id, CareerRole.slug, CareerRole.name, Family.slug).join(Family).order_by(CareerRole.id)
    ).all()
    role_levels = session.execute(
        select(CareerRoleLevel.role_id, CareerRoleLevel.level_id, CareerRoleLevel.title).order_by(
            CareerRoleLevel.role_id, CareerRoleLevel.level_id
        )
    ).all()
    roles = {}
    for role_id, slug, name, family in role_rows:
        levels = [(level_slugs[lv], title, lv) for rid, lv, title in role_levels if rid == role_id]
        roles[slug] = RoleSnapshot(
            id=slug,
            name=name,
            family=family,
            levels=[lv for lv, _, _ in levels],
            titles={lv: title for lv, title, _ in levels},
            requirements=cumulative_requirements([(lv, stages.get((role_id, lid), [])) for lv, _, lid in levels]),
            optional=optional_skills([(lv, stages.get((role_id, lid), [])) for lv, _, lid in levels]),
        )
    snapshot = CatalogSnapshot(
        roles=roles,
        skill_names=names,
        prerequisites={s: [(tuple(o), lv) for _, (o, lv) in sorted(g.items())] for s, g in prerequisites.items()},
    )
    _snapshots.clear()
    _snapshots[checksum] = snapshot
    return snapshot


def search_skills(session: Session, query: str, limit: int = 20) -> list[tuple[str, str, str]]:
    """(slug, name, kind) of skills whose name, slug or O*NET names contain the query; names that start
    with it first. For the board's picker."""
    q = query.strip().lower()
    if not q:
        return []
    like = f"%{q}%"
    rows = session.execute(
        select(Skill.slug, Skill.name, Skill.kind)
        .where(
            or_(
                func.lower(Skill.name).like(like),
                Skill.slug.like(like.replace(" ", "-")),
                func.array_to_string(Skill.onet, " ").ilike(like),
            )
        )
        .order_by(func.lower(Skill.name).like(f"{q}%").desc(), func.length(Skill.name), Skill.name)
        .limit(limit)
    ).all()
    return [tuple(r) for r in rows]


def skill_groups(session: Session, per_group: int = 12) -> list[tuple[str, str, list[tuple[str, str, str]]]]:
    """Suggested skills per role family: skills required at the first level of the family's roles, ranked by
    how many of its roles ask for them times how distinctive they are, so the board can offer a quick start."""
    snapshot = load_snapshot(session)
    families = dict(session.execute(select(Family.slug, Family.name).order_by(Family.id)).all())
    kinds = dict(session.execute(select(Skill.slug, Skill.kind)).all())
    groups = []
    for family, name in families.items():
        counts: dict[str, int] = {}
        for role in (r for r in snapshot.roles.values() if r.family == family):
            for req in role.requirements[role.levels[0]]:
                for option in req.options:
                    counts[option] = counts.get(option, 0) + 1
        # Common in the family and telling of it: Git is everywhere, dbt says "data".
        ranked = sorted(counts, key=lambda s: (-counts[s] * snapshot.weights[s] ** 2, s))[:per_group]
        groups.append((family, name, [(s, snapshot.skill_names[s], kinds[s]) for s in ranked]))
    return groups
