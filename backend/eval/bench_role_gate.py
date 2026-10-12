"""Which roles to show (ADR-0052): the gate against "always the top three", on the labeled learner profiles and
the thin boards, plus a sweep of the gate's three values to show where they sit.

    uv run python eval/bench_role_gate.py

Reports, per setting: labeled profiles that keep their expected role, how many roles they are shown (0/1/2/3),
the share of shown roles an adviser expected or accepted (expect + accept in learner_profiles.yaml), and the thin
boards handled right (eval/thin_boards.yaml). Saves JSON to eval/results/.
"""

import json
import platform
from datetime import datetime

import yaml
from bench_role_scoring import load_profiles
from common import EVAL_DIR, RESULTS_DIR

from xcrs.catalog import model
from xcrs.catalog.snapshot import snapshot_from_catalog
from xcrs.domain import role_scoring
from xcrs.domain.role_scoring import Category, Mention, recommendable, score_roles


def thin_boards() -> list[dict]:
    boards = yaml.safe_load((EVAL_DIR / "thin_boards.yaml").read_text())["boards"]
    for b in boards:
        b["mentions"] = [
            Mention(skill, c, int(level) if level else None)
            for c in Category
            for skill, _, level in (str(chip).partition(":") for chip in b.get(c.value, []))
        ]
    return boards


def evaluate(snapshot, profiles, boards, pick) -> dict:
    kept, counts, good, shown_total = 0, [0, 0, 0, 0], 0, 0
    for p in profiles:
        shown = [r.role for r in pick(score_roles(snapshot, p["mentions"]))]
        kept += p["expect"][0] in shown
        counts[len(shown)] += 1
        good += sum(r in set(p["expect"]) | set(p.get("accept", [])) for r in shown)
        shown_total += len(shown)
    thin_right, wrong = 0, []
    for b in boards:
        shown = [r.role for r in pick(score_roles(snapshot, b["mentions"]))]
        ok = set(shown) <= set(b["may_show"]) and bool(shown) == bool(b["may_show"])
        thin_right += ok
        if not ok:
            wrong.append({"board": b["id"], "shown": shown})
    return {
        "expected_kept": f"{kept}/{len(profiles)}",
        "shown_0_1_2_3": counts,
        "precision": round(good / shown_total, 3) if shown_total else None,
        "thin_right": f"{thin_right}/{len(boards)}",
        "thin_wrong": wrong,
    }


def main() -> None:
    snapshot = snapshot_from_catalog(model.load_catalog())
    profiles, boards = load_profiles(), thin_boards()
    report = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "gate": {
            "min_evidence": role_scoring.MIN_EVIDENCE,
            "min_lift": role_scoring.MIN_LIFT,
            "tail": role_scoring.TAIL,
        },
        "always_top_3": evaluate(snapshot, profiles, boards, lambda ranked: ranked[:3]),
        "gate_defaults": evaluate(snapshot, profiles, boards, recommendable),
        "sweep": [],
    }
    defaults = (role_scoring.MIN_EVIDENCE, role_scoring.MIN_LIFT, role_scoring.TAIL)
    for evidence in (0.0, 0.8, 1.0, 1.5):
        for lift in (1.5, 2.0, 2.5, 3.0):
            for tail in (0.4, 0.5, 0.6):
                role_scoring.MIN_EVIDENCE, role_scoring.MIN_LIFT, role_scoring.TAIL = evidence, lift, tail
                row = evaluate(snapshot, profiles, boards, recommendable)
                report["sweep"].append({"min_evidence": evidence, "min_lift": lift, "tail": tail, **row})
    role_scoring.MIN_EVIDENCE, role_scoring.MIN_LIFT, role_scoring.TAIL = defaults

    def line(label: str, r: dict) -> str:
        counts = str(r["shown_0_1_2_3"])
        return f"{label:34} {r['expected_kept']:>13} {counts:>16} {r['precision']:>9.0%} {r['thin_right']:>10}"

    print(f"{'setting':34} {'expected kept':>13} {'shown 0/1/2/3':>16} {'precision':>9} {'thin right':>10}")
    for name in ("always_top_3", "gate_defaults"):
        print(line(name, report[name]))
    print("\nsweep (evidence, lift, tail):")
    for r in report["sweep"]:
        print(line(f"  {r['min_evidence']:<4} {r['min_lift']:<4} {r['tail']:<4}", r))
    path = RESULTS_DIR / f"role-gate-{datetime.now():%Y%m%d-%H%M}-{platform.node().split('.')[0]}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nsaved {path.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
