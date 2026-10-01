"""Prototype-vs-new comparison and threshold diagnostics (ADR-0005, ADR-0010).

    uv run python eval/compare_prototype.py                                   # new system only
    uv run python eval/compare_prototype.py --prototype-url http://localhost:8001

The research prototype is no longer on this branch: run it from `main` (or the Zenodo release) with
`cd backend/src && python main.py`. It needs the five providers' API keys and their embedding CSVs; each
call also makes gpt-4o requests (a few cents for all profiles).

Pass rule: the five prototype providers don't agree with each other either. The new system passes
when its mean top-3 role overlap with the prototype providers is at least their mean pairwise overlap
minus 0.10, i.e. it sits inside the prototype's own spread.

Threshold diagnostics (the open issue in ADR-0010): how many roadmap concepts each profile phrase
matches at several sigma levels, and how many phrases match nothing (lost input).
"""

import argparse
import itertools
import json
import statistics
from datetime import UTC, datetime

import httpx
from common import RESULTS_DIR, load_profiles, to_user_input

from xcrs.db.session import new_session
from xcrs.domain.types import Category
from xcrs.embeddings import embedder_for
from xcrs.repository import catalog as catalog_repo
from xcrs.repository import vectors
from xcrs.services.recommend import THRESHOLD_SIGMA, RecommendationService, to_phrases

PROTOTYPE_MODELS = ["google", "voyage", "openai", "mistral", "cohere"]
PASS_MARGIN = 0.10
SIGMAS = [1.5, 2.0, 2.5, 3.0]


def overlap(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    return len(set(a) & set(b)) / max(len(a), len(b))


def new_system(session) -> dict[str, dict]:
    service = RecommendationService(session, explanations=None)
    out = {}
    for profile in load_profiles():
        computed = service.compute(to_user_input(profile))
        out[profile["id"]] = {
            "roles": [r.role for r in computed.roles],
            "courses": sorted({c.title for r in computed.roles for c in r.courses}),
        }
    return out


def prototype(url: str, model: str) -> dict[str, dict]:
    client = httpx.Client(base_url=url, timeout=600)
    out = {}
    for profile in load_profiles():
        body = {
            "took_and_liked": ", ".join(profile["liked"]),
            "took_and_neutral": ", ".join(profile["neutral"]),
            "took_and_disliked": ", ".join(profile["disliked"]),
            "curious": ", ".join(profile["curious"]),
        }
        response = client.post(f"/recommendations/{model}", json=body)
        response.raise_for_status()
        roles = response.json()["recommendations"][0]["roles"]
        out[profile["id"]] = {
            "roles": [r["role"] for r in roles],
            "courses": sorted({c["course"] for r in roles for c in r["courses"]}),
        }
    return out


def compare(new: dict, protos: dict[str, dict]) -> dict:
    per_model = {
        m: {
            "role_overlap": statistics.mean(overlap(new[p]["roles"], res[p]["roles"]) for p in new),
            "top1_agreement": statistics.mean(
                bool(new[p]["roles"] and res[p]["roles"] and new[p]["roles"][0] == res[p]["roles"][0]) for p in new
            ),
            "course_overlap": statistics.mean(overlap(new[p]["courses"], res[p]["courses"]) for p in new),
        }
        for m, res in protos.items()
    }
    pairwise = [
        statistics.mean(overlap(protos[a][p]["roles"], protos[b][p]["roles"]) for p in new)
        for a, b in itertools.combinations(protos, 2)
    ]
    new_vs_proto = statistics.mean(v["role_overlap"] for v in per_model.values())
    prototype_spread = statistics.mean(pairwise) if pairwise else None
    return {
        "per_prototype_model": per_model,
        "new_vs_prototype_role_overlap": new_vs_proto,
        "prototype_pairwise_role_overlap": prototype_spread,
        "passes": prototype_spread is not None and new_vs_proto >= prototype_spread - PASS_MARGIN,
    }


def threshold_diagnostics(session) -> dict:
    model = catalog_repo.active_model(session)
    phrases = sorted(
        {p.text for profile in load_profiles() for p in to_phrases(to_user_input(profile))}
        | {p.text for profile in load_profiles() for p in to_phrases({Category.CURIOUS: profile["curious"]})}
    )
    vecs = embedder_for(model).embed_query(phrases)
    lowest = model.sim_mean + min(SIGMAS) * model.sim_std
    matches = vectors.concepts_above_threshold(session, model, vecs, lowest)
    best = dict.fromkeys(range(len(phrases)), None)
    for m in matches:
        if best[m.phrase_index] is None or m.similarity > best[m.phrase_index]:
            best[m.phrase_index] = m.similarity
    rows = []
    for sigma in SIGMAS:
        t = model.sim_mean + sigma * model.sim_std
        counts = [sum(1 for m in matches if m.phrase_index == i and m.similarity > t) for i in range(len(phrases))]
        rows.append(
            {
                "sigma": sigma,
                "threshold": round(t, 3),
                "median_matches_per_phrase": statistics.median(counts),
                "max_matches": max(counts),
                "phrases_with_no_match": [phrases[i] for i, c in enumerate(counts) if c == 0],
            }
        )
    return {
        "model": model.name,
        "course_x_concept_mean": round(model.sim_mean, 3),
        "course_x_concept_std": round(model.sim_std, 3),
        "phrases": len(phrases),
        "sigmas": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prototype-url")
    parser.add_argument("--prototype-models", nargs="+", default=PROTOTYPE_MODELS)
    args = parser.parse_args()

    with new_session() as session:
        report = {"date": datetime.now(UTC).isoformat(), "new": new_system(session)}
        report["threshold"] = threshold_diagnostics(session)

    if args.prototype_url:
        protos = {m: prototype(args.prototype_url, m) for m in args.prototype_models}
        report["prototype"] = protos
        report["comparison"] = compare(report["new"], protos)
    else:
        report["comparison"] = None

    print("## New system (qwen3-embedding:0.6b)\n")
    for pid, r in report["new"].items():
        print(f"- **{pid}**: {', '.join(r['roles']) or '(no role)'}  ({len(r['courses'])} courses)")

    d = report["threshold"]
    print(
        f"\n## Threshold diagnostics: {d['phrases']} phrases; course x concept mean {d['course_x_concept_mean']}, "
        f"std {d['course_x_concept_std']}\n"
    )
    print("| sigma | threshold | median matches/phrase | max | phrases with no match |")
    print("|---|---|---|---|---|")
    for row in d["sigmas"]:
        lost = ", ".join(row["phrases_with_no_match"]) or "-"
        marker = " (current)" if row["sigma"] == THRESHOLD_SIGMA else ""
        print(
            f"| {row['sigma']}{marker} | {row['threshold']} | {row['median_matches_per_phrase']} | "
            f"{row['max_matches']} | {len(row['phrases_with_no_match'])}: {lost} |"
        )

    if report["comparison"]:
        c = report["comparison"]
        print("\n## Comparison with the prototype\n")
        for m, v in c["per_prototype_model"].items():
            print(
                f"- {m}: role overlap {v['role_overlap']:.2f}, top-1 {v['top1_agreement']:.2f}, "
                f"course overlap {v['course_overlap']:.2f}"
            )
        print(
            f"\nNew vs prototype {c['new_vs_prototype_role_overlap']:.2f}; prototype providers among themselves "
            f"{c['prototype_pairwise_role_overlap']:.2f} → {'PASS' if c['passes'] else 'FAIL'}"
        )
    else:
        print("\nPrototype side skipped: pass --prototype-url (needs the providers' API keys and embedding CSVs).")

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"compare-{datetime.now(UTC):%Y%m%d-%H%M}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved {path.relative_to(RESULTS_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
