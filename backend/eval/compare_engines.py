"""Engine v2 (new catalog) against the legacy engine (research catalog), before switching over.

1. **Labeled profiles** (eval/learner_profiles.yaml): both engines get the same learner. The legacy engine
   gets the skills' names as typed phrases (its only input); its 10 roles map to the new ones through
   `legacy_roles` in catalog/roles.yaml. Scored on the profiles whose expected role the legacy engine can
   name at all, and on all profiles.
2. **Phrase profiles** (eval/profiles.json, typed text): how often the engines agree, with v2 matching the
   text (ADR-0030; LLM answers come from the cache when seen before).

    uv run python eval/compare_engines.py            # needs the local DB, Ollama and the imported catalog
"""

import json
import platform
import statistics
import time
from datetime import datetime

from bench_role_scoring import load_profiles as load_labeled
from common import RESULTS_DIR, load_profiles, to_user_input
from sqlalchemy import select

from xcrs.catalog import model
from xcrs.db.models import Role
from xcrs.db.session import new_session
from xcrs.domain.role_scoring import Category as V2Category
from xcrs.domain.role_scoring import Mention, score_roles
from xcrs.domain.types import Category as V1Category
from xcrs.repository import catalog_store
from xcrs.services.recommend import RecommendationService
from xcrs.services.skill_matching import build_matcher


def legacy_top(service, legacy_map, user_input) -> tuple[list[str], float]:
    started = time.perf_counter()
    roles = service.compute(user_input).roles
    ms = (time.perf_counter() - started) * 1000
    return [legacy_map.get(r.role_id) for r in roles], ms


def score(top: list, p: dict) -> tuple[bool, bool]:
    allowed = set(p["expect"]) | set(p.get("accept", []))
    return (bool(top) and top[0] in allowed), p["expect"][0] in top[:3]


def main() -> None:
    cat = model.load_catalog()
    report: dict = {"date": datetime.now().isoformat(timespec="minutes"), "host": platform.node()}
    with new_session() as session:
        slugs = dict(session.execute(select(Role.id, Role.slug)).all())
        legacy_map = {rid: cat.legacy_roles.get(slug) for rid, slug in slugs.items()}
        representable = {v for v in legacy_map.values() if v}
        legacy = RecommendationService(session, explanations=None)
        snapshot = catalog_store.load_snapshot(session)

        # 1. Labeled profiles
        rows = []
        for p in load_labeled():
            user_input = {c: [] for c in V1Category}
            for m in p["mentions"]:
                user_input[V1Category(m.category.value)].append(cat.skills[m.skill].name)
            v1, v1_ms = legacy_top(legacy, legacy_map, user_input)
            started = time.perf_counter()
            v2 = [r.role for r in score_roles(snapshot, p["mentions"])[:3]]
            v2_ms = (time.perf_counter() - started) * 1000
            rows.append(
                {
                    "profile": p["id"],
                    "kind": p["kind"],
                    "expect": p["expect"],
                    "representable": p["expect"][0] in representable,
                    "legacy_top3": v1[:3],
                    "v2_top3": v2,
                    "legacy_hit": score(v1, p),
                    "v2_hit": score(v2, p),
                    "legacy_ms": v1_ms,
                    "v2_ms": v2_ms,
                }
            )

        def summary(subset):
            n = len(subset) or 1
            return {
                "profiles": len(subset),
                "legacy_hit@1": sum(r["legacy_hit"][0] for r in subset) / n,
                "legacy_hit@3": sum(r["legacy_hit"][1] for r in subset) / n,
                "v2_hit@1": sum(r["v2_hit"][0] for r in subset) / n,
                "v2_hit@3": sum(r["v2_hit"][1] for r in subset) / n,
            }

        report["labeled"] = {
            "legacy_can_name_the_role": summary([r for r in rows if r["representable"]]),
            "all": summary(rows),
            "latency_ms_median": {
                "legacy (embed + threshold + course matches)": statistics.median(r["legacy_ms"] for r in rows),
                "v2 scoring (skills given)": statistics.median(r["v2_ms"] for r in rows),
            },
            "rows": rows,
        }

        # 2. Phrase profiles: agreement
        matcher = build_matcher(session)
        agreement = []
        for p in load_profiles():
            user_input = to_user_input(p)
            v1, _ = legacy_top(legacy, legacy_map, user_input)
            mentions = []
            for c, texts in user_input.items():
                for r in matcher.match(texts):
                    mentions += [Mention(s, V2Category(c.value)) for s in r.skills]
            v2 = [r.role for r in score_roles(snapshot, mentions)[:3]] if mentions else []
            v1_named = [r for r in v1 if r]
            agreement.append(
                {
                    "profile": p["id"],
                    "legacy_top3": v1[:3],
                    "v2_top3": v2,
                    "same_top1": bool(v1_named and v2 and v1_named[0] == v2[0]),
                    "top3_overlap": len(set(v1_named[:3]) & set(v2)) / 3,
                }
            )
        report["phrase_profiles"] = {
            "same_top1": sum(a["same_top1"] for a in agreement) / len(agreement),
            "top3_overlap": statistics.mean(a["top3_overlap"] for a in agreement),
            "rows": agreement,
        }

    lab = report["labeled"]
    lines = [
        f"# Engine v2 vs legacy: {report['date']} on {report['host']}",
        "",
        "Labeled learner profiles (`eval/learner_profiles.yaml`). The legacy engine gets the skills' names as phrases;",
        "its roles are mapped to the new catalog (`legacy_roles`). hit@1 = first role expected or acceptable;",
        "hit@3 = the expected role among the first three.",
        "",
        "| Profiles | n | Legacy hit@1 | Legacy hit@3 | v2 hit@1 | v2 hit@3 |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in (
        ("expected role exists in the legacy catalog", lab["legacy_can_name_the_role"]),
        ("all", lab["all"]),
    ):
        lines.append(
            f"| {name} | {s['profiles']} | {s['legacy_hit@1']:.0%} | {s['legacy_hit@3']:.0%} | "
            f"**{s['v2_hit@1']:.0%}** | **{s['v2_hit@3']:.0%}** |"
        )
    lines += [
        "",
        "Median latency per profile (ms): " + ", ".join(f"{k} {v:.0f}" for k, v in lab["latency_ms_median"].items()),
    ]
    pp = report["phrase_profiles"]
    lines += [
        "",
        f"Phrase profiles (`eval/profiles.json`, {len(pp['rows'])}): same first role {pp['same_top1']:.0%}, "
        f"top-3 overlap {pp['top3_overlap']:.0%}.",
        "",
        "| Profile | Legacy top 3 (mapped) | v2 top 3 |",
        "|---|---|---|",
    ]
    lines += [
        f"| {a['profile']} | {', '.join(r or '(ux)' for r in a['legacy_top3'])} | {', '.join(a['v2_top3'])} |"
        for a in pp["rows"]
    ]
    misses = [r for r in lab["rows"] if r["representable"] and not r["v2_hit"][0]]
    if misses:
        lines += ["", "v2 first-role misses where the legacy engine could have named the role:", ""]
        lines += [
            f"- {r['profile']}: expected {r['expect'][0]}, v2 {r['v2_top3']}, legacy {r['legacy_top3']}" for r in misses
        ]
    stem = RESULTS_DIR / f"compare-engines-{datetime.now():%Y%m%d-%H%M}-{platform.node().split('.')[0]}"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=1, default=str))
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"saved {stem}.json / .md")


if __name__ == "__main__":
    main()
