"""The catalog as the scoring engine sees it (xcrs.domain.role_scoring), built from the YAML model.
The served API builds the same snapshot from the database (repository.catalog_store.load_snapshot)."""

from xcrs.catalog.model import Catalog
from xcrs.domain.role_scoring import CatalogSnapshot, Requirement, ResourceRef, RoleSnapshot


def cumulative_requirements(levels) -> dict[str, list[Requirement]]:
    """Per level, everything required up to it (optional stages don't count). An item asked again at a
    higher proficiency keeps the later stage, which is where that proficiency is learned."""
    out: dict[str, list[Requirement]] = {}
    merged: dict[tuple[str, ...], Requirement] = {}
    order = 0
    for level, stages in levels:
        for stage_name, optional, items in stages:
            if optional:
                continue
            for options, proficiency in items:
                order += 1
                current = merged.get(options)
                if current is None or proficiency > current.level:
                    merged[options] = Requirement(options, proficiency, stage_name, order)
        out[level] = sorted(merged.values(), key=lambda r: r.order)
    return out


def optional_skills(levels) -> dict[str, int]:
    """Skills from "good to know" stages, at the highest proficiency mentioned."""
    out: dict[str, int] = {}
    for _, stages in levels:
        for _, optional, items in stages:
            if optional:
                for options, proficiency in items:
                    for o in options:
                        out[o] = max(out.get(o, 0), proficiency)
    return out


def snapshot_from_catalog(cat: Catalog) -> CatalogSnapshot:
    roles = {}
    for rid, role in cat.roles.items():
        roadmap = cat.roadmaps[rid]
        levels = [
            (lv.level, [(st.name, st.optional, [(it.options, it.level) for it in st.items]) for st in lv.stages])
            for lv in roadmap.levels
        ]
        roles[rid] = RoleSnapshot(
            id=rid,
            name=role.name,
            family=role.family,
            levels=[lv.level for lv in roadmap.levels],
            titles={lv.level: lv.title for lv in roadmap.levels},
            requirements=cumulative_requirements(levels),
            optional=optional_skills(levels),
        )
    return CatalogSnapshot(
        roles=roles,
        skill_names={s.id: s.name for s in cat.skills.values()},
        prerequisites={s.id: [(r.options, r.level) for r in s.requires] for s in cat.skills.values()},
        resources=[
            ResourceRef(
                r.url,
                r.title,
                r.url,
                r.provider,
                r.type,
                r.level,
                r.free,
                True,
                tuple((t.options[0], t.level) for t in r.teaches),
            )
            for r in cat.resources
        ],
    )
