"""Which roles to show (ADR-0052): the gate against "always the top three", on the labeled learner profiles, the
gate boards (thin and broad) and broad boards made by merging three labeled profiles, plus a sweep of the gate's
values to show where they sit.

    uv run python eval/bench_role_gate.py

Reports, per setting: labeled profiles that keep their expected role, how many roles they are shown (0/1/2/3),
the share of shown roles an adviser expected or accepted (expect + accept in learner_profiles.yaml), the gate
boards handled right (eval/gate_boards.yaml), and for the merged boards (three profiles, 5-10 years): how many show
no role (should be none), and the share of shown roles among the three profiles' expected or accepted ones.
Saves JSON to eval/results/.
"""

import itertools
import json
import platform
import random
from datetime import datetime

import yaml
from bench_role_scoring import load_profiles
from common import EVAL_DIR, RESULTS_DIR

from xcrs.catalog import model
from xcrs.catalog.snapshot import snapshot_from_catalog
from xcrs.domain import role_scoring
from xcrs.domain.role_scoring import Category, Mention, recommendable, score_roles

MERGED = 1500  # merged boards in the sample (of 23,426 triples)
GATE = ("MIN_EVIDENCE", "MIN_LIFT", "STRONG_EVIDENCE", "TAIL")


def gate_boards() -> list[dict]:
    boards = yaml.safe_load((EVAL_DIR / "gate_boards.yaml").read_text())["boards"]
    for b in boards:
        b["mentions"] = [
            Mention(skill, c, int(level) if level else None)
            for c in Category
            for skill, _, level in (str(chip).partition(":") for chip in b.get(c.value, []))
        ]
    return boards


def merged_boards(profiles: list[dict], n: int = MERGED, seed: int = 7) -> list[dict]:
    """Broad boards: three labeled learners on one board (a skill on two boards keeps its first category)."""
    random.seed(seed)
    out = []
    for trio in random.sample(list(itertools.combinations(profiles, 3)), n):
        mentions = {}
        for p in trio:
            for m in p["mentions"]:
                mentions.setdefault(m.skill, m)
        ok = set().union(*(set(p["expect"]) | set(p.get("accept", [])) for p in trio))
        out.append({"mentions": list(mentions.values()), "ok": ok})
    return out


def evaluate(ranked_profiles, profiles, ranked_boards, boards, ranked_merged, merged, pick) -> dict:
    kept, counts, good, shown_total = 0, [0, 0, 0, 0], 0, 0
    for p, ranked in zip(profiles, ranked_profiles, strict=True):
        shown = [r.role for r in pick(ranked)]
        kept += p["expect"][0] in shown
        counts[len(shown)] += 1
        good += sum(r in set(p["expect"]) | set(p.get("accept", [])) for r in shown)
        shown_total += len(shown)
    boards_right, wrong = 0, []
    for b, ranked in zip(boards, ranked_boards, strict=True):
        shown = [r.role for r in pick(ranked)]
        ok = set(shown) <= set(b["may_show"]) and bool(shown) == bool(b["may_show"])
        boards_right += ok
        if not ok:
            wrong.append({"board": b["id"], "shown": shown})
    empty = mgood = mtotal = 0
    for m, ranked in zip(merged, ranked_merged, strict=True):
        shown = [r.role for r in pick(ranked)]
        empty += not shown
        mgood += sum(r in m["ok"] for r in shown)
        mtotal += len(shown)
    return {
        "expected_kept": f"{kept}/{len(profiles)}",
        "shown_0_1_2_3": counts,
        "precision": round(good / shown_total, 3) if shown_total else None,
        "boards_right": f"{boards_right}/{len(boards)}",
        "boards_wrong": wrong,
        "merged_empty": f"{empty}/{len(merged)}",
        "merged_precision": round(mgood / mtotal, 3) if mtotal else None,
    }


def main() -> None:
    snapshot = snapshot_from_catalog(model.load_catalog())
    profiles, boards = load_profiles(), gate_boards()
    merged = merged_boards(profiles)
    # Score once; the gate only reads the ranked lists.
    ranked_profiles = [score_roles(snapshot, p["mentions"]) for p in profiles]
    ranked_boards = [score_roles(snapshot, b["mentions"], experience=b.get("experience")) for b in boards]
    ranked_merged = [score_roles(snapshot, m["mentions"], experience="5-10") for m in merged]
    data = (ranked_profiles, profiles, ranked_boards, boards, ranked_merged, merged)

    defaults = {k: getattr(role_scoring, k) for k in GATE}
    report = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "gate": {k.lower(): v for k, v in defaults.items()},
        "always_top_3": evaluate(*data, lambda ranked: ranked[:3]),
        "gate_defaults": evaluate(*data, recommendable),
        "sweep": [],
    }
    # Strong evidence 0: no lift test at all; 1e9: the lift test for every role (engine v3.3).
    for strong in (0.0, 2.0, 4.0, 6.0, 1e9):
        for lift in (1.5, 2.0, 2.5):
            for tail in (0.4, 0.5, 0.6):
                setting = {"MIN_EVIDENCE": defaults["MIN_EVIDENCE"], "MIN_LIFT": lift, "STRONG_EVIDENCE": strong}
                setting["TAIL"] = tail
                for k, v in setting.items():
                    setattr(role_scoring, k, v)
                report["sweep"].append({k.lower(): v for k, v in setting.items()} | evaluate(*data, recommendable))
    for k, v in defaults.items():
        setattr(role_scoring, k, v)

    def line(label: str, r: dict) -> str:
        counts = str(r["shown_0_1_2_3"])
        return (
            f"{label:30} {r['expected_kept']:>6} {counts:>16} {r['precision']:>5.0%} {r['boards_right']:>7}"
            f" {r['merged_empty']:>10} {r['merged_precision']:>6.0%}"
        )

    print(f"{'setting':30} {'kept':>6} {'shown 0/1/2/3':>16} {'prec':>5} {'boards':>7} {'no role':>10} {'prec':>6}")
    print(f"{'':30} {'':>6} {'(53 labeled)':>16} {'':>5} {'':>7} {'(merged)':>10} {'':>6}")
    for name in ("always_top_3", "gate_defaults"):
        print(line(name, report[name]))
    print("\nsweep (strong evidence, lift, tail):")
    for r in report["sweep"]:
        print(line(f"  {r['strong_evidence']:<5g} {r['min_lift']:<4} {r['tail']:<4}", r))
    path = RESULTS_DIR / f"role-gate-{datetime.now():%Y%m%d-%H%M}-{platform.node().split('.')[0]}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nsaved {path.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
