"""The skill-matching evaluation set (ADR-0029) refers only to catalog skills and is well-formed."""

from collections import Counter
from pathlib import Path

import yaml

from xcrs.catalog import model

SET = Path(__file__).parents[1] / "eval" / "skill_matching.yaml"
KINDS = {"exact", "alias", "typo", "paraphrase", "tool", "multi", "broad", "course", "unrelated"}
FIELDS = {"phrase", "expect", "any", "accept", "kind"}


def test_the_evaluation_set_is_well_formed():
    cases = yaml.safe_load(SET.read_text())["cases"]
    skills = model.load_catalog().skills
    assert len(cases) >= 250
    duplicates = [p for p, n in Counter(c["phrase"].lower() for c in cases).items() if n > 1]
    assert duplicates == []
    for case in cases:
        where = case.get("phrase")
        assert set(case) <= FIELDS and "phrase" in case and case.get("kind") in KINDS, where
        assert "expect" in case or "any" in case, where
        listed = [*case.get("expect", []), *case.get("any", []), *case.get("accept", [])]
        assert [s for s in listed if s not in skills] == [], where
        assert len(listed) == len(set(listed)), f"{where}: a skill is listed twice"
        if case["kind"] == "unrelated":
            assert case.get("expect") == [] and "any" not in case, where
