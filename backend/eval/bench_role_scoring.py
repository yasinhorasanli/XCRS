"""Calibrate engine v2 role scoring (ADR-0031) on labeled learner profiles (eval/learner_profiles.yaml).

score = a * interest + (1 - a) * coverage. Searches a, the category weights and which coverage ranks
roles, plus the bar for estimating the level; reports cross-validated (held-out) results next to the
alternatives that were considered: interest only (S3), coverage only, and coverage x interest (S1).

    uv run python eval/bench_role_scoring.py
"""

import itertools
import json
import platform
import random
import statistics
from datetime import datetime

import numpy as np
import yaml
from common import EVAL_DIR, RESULTS_DIR

from xcrs.catalog import model
from xcrs.catalog.snapshot import snapshot_from_catalog
from xcrs.domain.role_scoring import Category, Mention, _coverage, learner_skills, level_additions, with_prerequisites

CATEGORIES = [Category.CURIOUS, Category.LIKED, Category.NEUTRAL, Category.DISLIKED]
GRID = {
    "interest": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "curious": [0.5, 0.75, 1.0, 1.25, 1.5],
    "liked": [0.5, 0.75, 1.0],
    "neutral": [0.0, 0.25, 0.5],
    "disliked": [-1.0, -0.5, -0.25, 0.0],
    "coverage_at": ["first", "top", "mean"],
}
LEVEL_BARS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
LEGACY = {"curious": 1.0, "liked": 0.75, "neutral": 0.5, "disliked": -0.25}


def parse(chip: str) -> tuple[str, int | None]:
    skill, _, level = str(chip).partition(":")
    return skill, int(level) if level else None


def load_profiles() -> list[dict]:
    profiles = yaml.safe_load((EVAL_DIR / "learner_profiles.yaml").read_text())["profiles"]
    for p in profiles:
        p["mentions"] = [
            Mention(*parse(chip)[:1], Category(c), parse(chip)[1]) for c in CATEGORIES for chip in p.get(c.value, [])
        ]
    return profiles


BARRIER = None  # per role: True if it starts above entry (set in main; entry_barrier in role_scoring)


def features(snapshot, profiles):
    """Per profile and role: coverage (first, top, mean level), the distinctiveness of mentioned skills the
    role requires per category, and the learner's total per category."""
    roles = list(snapshot.roles)
    cov = np.zeros((len(profiles), len(roles), 3))
    in_role = np.zeros((len(profiles), len(roles), 4))
    totals = np.zeros((len(profiles), 4))
    level_cov = []
    for i, p in enumerate(profiles):
        have, category = learner_skills(p["mentions"])
        have = with_prerequisites(have, snapshot.prerequisites)
        for c_idx, c in enumerate(CATEGORIES):
            totals[i, c_idx] = sum(snapshot.weights[s] for s, cat in category.items() if cat == c)
        per_role = {}
        for j, rid in enumerate(roles):
            role = snapshot.roles[rid]
            lc = {}
            for lv in role.levels:
                reqs = role.requirements[lv]
                total = met = 0.0
                for r in reqs:
                    w = max(snapshot.weights[o] for o in r.options) * r.level
                    total += w
                    met += w * min(1.0, max(have.get(o, 0) for o in r.options) / r.level)
                lc[lv] = met / total
            per_role[rid] = lc
            cov[i, j] = [lc[role.levels[0]], lc[role.levels[-1]], sum(lc.values()) / len(lc)]
            reliance = snapshot.reliance(rid)
            for c_idx, c in enumerate(CATEGORIES):
                in_role[i, j, c_idx] = sum(
                    snapshot.weights[s] * reliance[s] for s, cat in category.items() if cat == c and s in reliance
                )
        level_cov.append(per_role)
    return roles, cov, in_role, totals, level_cov


def scores(params, cov, in_role, totals, multiply=False):
    w = np.array([params[c.value] for c in CATEGORIES])
    attention = (totals * np.abs(w)).sum(axis=1, keepdims=True)
    attention[attention == 0] = 1.0
    interest = (in_role * w).sum(axis=2) / attention
    coverage = cov[:, :, ["first", "top", "mean"].index(params["coverage_at"])]
    if multiply:
        return coverage * np.clip(interest, 0, None)
    blend = params["interest"] * interest + (1 - params["interest"]) * coverage
    if BARRIER is None:
        return blend
    first = cov[:, :, 0]
    factor = np.where(BARRIER, 0.5 + 0.5 * np.minimum(1.0, first / 0.55), 1.0)
    return blend * factor


def evaluate(s, profiles, roles, idx):
    hit1 = hit3 = rr = 0.0
    for i in idx:
        p = profiles[i]
        order = [roles[j] for j in np.argsort(-s[i])]
        main = p["expect"][0]
        hit1 += order[0] in set(p["expect"]) | set(p.get("accept", []))
        hit3 += main in order[:3]
        rr += 1 / (order.index(main) + 1)
    n = len(idx)
    return {"hit@1": hit1 / n, "hit@3": hit3 / n, "mrr": rr / n}


def objective(m):
    return (m["hit@1"] + m["hit@3"], m["mrr"])


def tune(cov, in_role, totals, profiles, roles, idx, fixed=None):
    best, best_key = None, None
    keys = list(GRID)
    for values in itertools.product(*(GRID[k] if not fixed or k not in fixed else [fixed[k]] for k in keys)):
        params = dict(zip(keys, values, strict=True))
        m = evaluate(scores(params, cov, in_role, totals), profiles, roles, idx)
        if best_key is None or objective(m) > best_key:
            best, best_key = params, objective(m)
    return best


def folds(profiles, seed):
    rng = random.Random(seed)
    a, b = [], []
    for kind in sorted({p["kind"] for p in profiles}):
        members = [i for i, p in enumerate(profiles) if p["kind"] == kind]
        rng.shuffle(members)
        for n, i in enumerate(members):
            (a if n % 2 == 0 else b).append(i)
    return a, b


def cross_validate(cov, in_role, totals, profiles, roles, fixed=None, multiply=False):
    held = []
    for seed in range(5):
        a, b = folds(profiles, seed)
        for train, test in ((a, b), (b, a)):
            params = tune(cov, in_role, totals, profiles, roles, train, fixed)
            held.append(evaluate(scores(params, cov, in_role, totals, multiply), profiles, roles, test))
    return {k: statistics.mean(m[k] for m in held) for k in ("hit@1", "hit@3", "mrr")}


def level_accuracy_additions(profiles, snapshot, bar):
    """Level reached = the longest run of levels whose own additions are at least `bar` met."""
    right = 0
    for p in profiles:
        have, _ = learner_skills(p["mentions"])
        have = with_prerequisites(have, snapshot.prerequisites)
        role = snapshot.roles[p["expect"][0]]
        estimate = role.levels[0]
        for lv, added in level_additions(role).items():
            if _coverage(added, have, snapshot.weights) < bar:
                break
            estimate = lv
        right += estimate == p["level"]
    return right / len(profiles)


def level_accuracy(profiles, level_cov, bar):
    right = 0
    for p, per_role in zip(profiles, level_cov, strict=True):
        lc = per_role[p["expect"][0]]
        levels = list(lc)
        estimate = levels[0]
        for lv in levels:
            if lc[lv] >= bar:
                estimate = lv
        right += estimate == p["level"]
    return right / len(profiles)


def main() -> None:
    cat = model.load_catalog()
    snapshot = snapshot_from_catalog(cat)
    profiles = load_profiles()
    roles, cov, in_role, totals, level_cov = features(snapshot, profiles)
    global BARRIER
    BARRIER = np.array([snapshot.roles[r].levels[0] != "entry" for r in roles])
    every = list(range(len(profiles)))
    variants = {
        "blend a*interest + (1-a)*coverage (S2)": {},
        "interest only (S3), legacy category weights": {"interest": 1.0, **LEGACY, "coverage_at": "first"},
        "interest only (S3), tuned category weights": {"interest": 1.0},
        "coverage only": {"interest": 0.0},
        "coverage x interest (S1)": {"interest": 1.0},
    }
    report = {"date": datetime.now().isoformat(timespec="minutes"), "host": platform.node(), "profiles": len(profiles)}
    report["variants"] = {}
    for name, fixed in variants.items():
        multiply = name.startswith("coverage x")
        held = cross_validate(cov, in_role, totals, profiles, roles, fixed, multiply)
        params = tune(cov, in_role, totals, profiles, roles, every, fixed)
        full = evaluate(scores(params, cov, in_role, totals, multiply), profiles, roles, every)
        report["variants"][name] = {"held_out": held, "all_profiles": full, "params": params}
        print(f"{name:48} held-out hit@1 {held['hit@1']:.0%} hit@3 {held['hit@3']:.0%} mrr {held['mrr']:.2f}  {params}")
    best = report["variants"]["blend a*interest + (1-a)*coverage (S2)"]["params"]
    s = scores(best, cov, in_role, totals)
    report["misses"] = []
    for i, p in enumerate(profiles):
        order = [roles[j] for j in np.argsort(-s[i])]
        if order[0] not in set(p["expect"]) | set(p.get("accept", [])):
            report["misses"].append({"profile": p["id"], "expected": p["expect"], "top3": order[:3]})
    report["level_accuracy_cumulative"] = {f"{b:.2f}": level_accuracy(profiles, level_cov, b) for b in LEVEL_BARS}
    report["level_accuracy"] = {f"{b:.2f}": level_accuracy_additions(profiles, snapshot, b) for b in LEVEL_BARS}
    # How sensitive is the result to a? Held-out with the category weights of the best blend, a fixed.
    report["a_curve"] = {}
    for a in GRID["interest"]:
        fixed = {**{k: v for k, v in best.items() if k != "interest"}, "interest": a}
        held = cross_validate(cov, in_role, totals, profiles, roles, fixed)
        report["a_curve"][a] = held
        print(f"  a={a:.1f}  held-out hit@1 {held['hit@1']:.0%} hit@3 {held['hit@3']:.0%} mrr {held['mrr']:.3f}")
    hard = [i for i, p in enumerate(profiles) if p["kind"] == "neighbours"]
    report["neighbours_only"] = {}
    for a in GRID["interest"]:
        fixed = {**{k: v for k, v in best.items() if k != "interest"}, "interest": a}
        m = evaluate(scores(fixed, cov, in_role, totals), profiles, roles, hard)
        report["neighbours_only"][a] = m
        print(f"  neighbours only, a={a:.1f}: hit@1 {m['hit@1']:.0%} mrr {m['mrr']:.3f}")
    print("misses:", report["misses"])
    print("level accuracy, cumulative coverage:", report["level_accuracy_cumulative"])
    print("level accuracy, each level's additions:", report["level_accuracy"])
    stem = RESULTS_DIR / f"role-scoring-{datetime.now():%Y%m%d-%H%M}-{platform.node().split('.')[0]}"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(f"saved {stem}.json")


if __name__ == "__main__":
    main()
