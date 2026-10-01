"""Catalog-as-code (ADR-0027, ADR-0028): the real catalog is valid, and each validation rule bites."""

import textwrap
from pathlib import Path

import pytest

from xcrs.catalog import model, validate
from xcrs.catalog.model import CatalogError, Requirement, RoleLevel

ROLES = """
levels:
  entry: {name: Entry, typical_years: "0-2", scope: x}
  mid: {name: Mid, typical_years: "2-5", scope: x}
  senior: {name: Senior, typical_years: "5+", scope: x}
  staff: {name: Staff, typical_years: "8+", scope: x}
families: {eng: Engineering}
roles:
  dev: {name: Developer, family: eng, summary: x, onet: 15-1252.00, levels: [entry, mid]}
  ops: {name: Operator, family: eng, summary: x, onet: 15-1299.08, levels: [entry, mid]}
common_paths:
  - {from: dev@mid, to: ops@mid, kind: pivot}
legacy_roles: {old: dev}
"""

SKILLS = """
skills:
  basics: {name: Basics, kind: concept, description: x}
  python: {name: Python, kind: language, description: x, requires: [basics:1]}
  go: {name: Go, kind: language, description: x, requires: [basics:1]}
  docker: {name: Docker, kind: tool, description: x, requires: ["python|go:2"]}
  kubernetes: {name: Kubernetes, kind: platform, description: x, requires: [docker:2]}
"""


def roadmap(role: str, entry: str, mid: str, optional_mid: str = "") -> str:
    extra = f"\n      - {{name: Extra, optional: true, skills: [{optional_mid}]}}" if optional_mid else ""
    return (
        textwrap.dedent(f"""
        role: {role}
        levels:
          entry:
            summary: x
            stages:
              - {{name: One, skills: [{entry}]}}
          mid:
            summary: x
            stages:
              - {{name: Two, skills: [{mid}]}}""")
        + extra
    )


def make(tmp_path: Path, dev_entry="basics:1, python:2", dev_mid="docker:2", ops_mid="kubernetes:2", **kw) -> Path:
    (tmp_path / "roadmaps").mkdir()
    (tmp_path / "skills.yaml").write_text(kw.get("skills", SKILLS))
    (tmp_path / "roles.yaml").write_text(kw.get("roles", ROLES))
    (tmp_path / "roadmaps" / "dev.yaml").write_text(roadmap("dev", dev_entry, dev_mid, kw.get("optional", "")))
    (tmp_path / "roadmaps" / "ops.yaml").write_text(roadmap("ops", "basics:1, go:2, docker:2", ops_mid))
    return tmp_path


def errors(path: Path) -> list[str]:
    return validate.validate(model.load_catalog(path)).errors


def test_the_real_catalog_is_valid():
    report = validate.validate(model.load_catalog())
    assert report.errors == []
    assert report.stats["roles"] >= 20 and report.stats["skills"] >= 200


def test_requirement_syntax():
    assert Requirement.parse("python|go:2") == Requirement(("python", "go"), 2)
    assert str(Requirement.parse("docker:3")) == "docker:3"
    for bad in ["docker", "docker:5", "Docker:2", "a||b:1"]:
        with pytest.raises(CatalogError):
            Requirement.parse(bad)
    assert RoleLevel.parse("dev@mid") == RoleLevel("dev", "mid")


def test_a_valid_small_catalog_passes(tmp_path):
    assert errors(make(tmp_path)) == []


def test_a_skill_before_its_prerequisite_is_an_error(tmp_path):
    assert errors(make(tmp_path, dev_entry="basics:1, kubernetes:1", dev_mid="python:2")) == [
        "dev@entry [One]: kubernetes needs docker:2"
    ]


def test_too_little_proficiency_in_a_prerequisite_is_an_error(tmp_path):
    assert errors(make(tmp_path, dev_entry="basics:1, python:1", dev_mid="docker:2")) == [
        "dev@mid [Two]: docker needs python|go:2 (roadmap has 1)"
    ]


def test_any_option_of_a_choice_satisfies_a_prerequisite(tmp_path):
    assert errors(make(tmp_path, dev_entry="basics:1, python|go:2", dev_mid="docker:2")) == []


def test_proficiency_cannot_drop_at_a_higher_level(tmp_path):
    assert errors(make(tmp_path, dev_entry="basics:1, python:3", dev_mid="python:2, docker:2")) == [
        "dev@mid [Two]: python drops from 3 to 2"
    ]


def test_optional_stages_never_satisfy_required_skills(tmp_path):
    """Python is only "good to know" in entry, so Docker's need for Python isn't met in mid."""
    path = make(tmp_path)
    (path / "roadmaps" / "dev.yaml").write_text(
        textwrap.dedent("""
            role: dev
            levels:
              entry:
                summary: x
                stages:
                  - {name: Core, skills: [basics:1]}
                  - {name: Extra, optional: true, skills: [python:2]}
              mid:
                summary: x
                stages:
                  - {name: Two, skills: [docker:2]}""")
    )
    assert errors(path) == ["dev@mid [Two]: docker needs python|go:2"]


def test_optional_skills_must_meet_their_own_prerequisites(tmp_path):
    ok, broken = tmp_path / "ok", tmp_path / "broken"
    ok.mkdir(), broken.mkdir()
    assert errors(make(ok, optional="kubernetes:1")) == []  # docker:2 is required earlier in dev@mid
    assert errors(make(broken, dev_mid="python:2", optional="kubernetes:1")) == [
        "dev@mid [Extra]: kubernetes needs docker:2"
    ]


def test_prerequisite_cycles_are_reported(tmp_path):
    skills = SKILLS.replace("requires: [basics:1]}\n  go", "requires: [basics:1, kubernetes:1]}\n  go")
    assert any("prerequisite cycle" in e for e in errors(make(tmp_path, skills=skills)))


def test_unknown_references_and_missing_roadmaps_are_reported(tmp_path):
    path = make(tmp_path, dev_mid="docker:2, terraform:1")
    (path / "roadmaps" / "ops.yaml").unlink()
    found = errors(path)
    assert "dev@mid [Two]: unknown skill 'terraform'" in found
    assert "role ops: no roadmap file roadmaps/ops.yaml" in found


def test_common_paths_must_cross_roles_and_use_existing_levels(tmp_path):
    roles = ROLES.replace("{from: dev@mid, to: ops@mid, kind: pivot}", "{from: dev@mid, to: dev@staff, kind: pivot}")
    found = errors(make(tmp_path, roles=roles))
    assert "common path dev@mid -> dev@staff: dev has no level 'staff'" in found
    assert any("moving up inside a role is implied" in e for e in found)


def test_bridge_lists_gaps_and_treats_a_choice_as_one_requirement(tmp_path):
    cat = model.load_catalog(make(tmp_path, dev_entry="basics:1, python|go:2", dev_mid="docker:1"))
    gap = validate.bridge(cat, RoleLevel("dev", "mid"), RoleLevel("ops", "mid"))
    assert gap == {"docker": (1, 2), "kubernetes": (0, 2)}  # go:2 is covered by dev's python|go choice


def test_distinctive_skills_weigh_more_than_shared_ones():
    cat = model.load_catalog()
    weights = validate.skill_weights(cat)
    assert weights["dbt"] > weights["git"]  # dbt says "data engineer"; Git says little


def test_coverage_is_one_for_the_same_role_and_less_for_another():
    cat = model.load_catalog()
    backend = RoleLevel("backend-engineer", "mid")
    assert validate.coverage(cat, backend, backend) == 1.0
    assert 0 < validate.coverage(cat, backend, RoleLevel("data-engineer", "entry")) < 1


def test_any_move_is_ranked_and_nearest_roles_make_sense():
    """Any role can move to any other; the measure puts the common, close moves first."""
    cat = model.load_catalog()
    ranked = validate.moves(cat, RoleLevel("devops-engineer", "senior"))
    assert len(ranked) == len(cat.roles) - 1
    assert {m.role for m in ranked[:3]} == {"cloud-engineer", "platform-engineer", "site-reliability-engineer"}
    assert ranked[-1].coverage < 0.15  # e.g. data analysis is a long way from DevOps


def test_starting_level_is_the_highest_level_already_mostly_covered():
    cat = model.load_catalog()
    by_role = {m.role: m for m in validate.moves(cat, RoleLevel("devops-engineer", "senior"))}
    assert by_role["cloud-engineer"].starting_level == "senior"
    assert by_role["site-reliability-engineer"].starting_level == "mid"
    assert by_role["data-analyst"].starting_level is None  # would start from scratch


def test_changing_roles_never_starts_above_the_current_level():
    cat = model.load_catalog()
    for source in [RoleLevel("devops-engineer", "senior"), RoleLevel("data-scientist", "mid")]:
        for move in validate.moves(cat, source):
            if move.starting_level:
                assert model.LADDER.index(move.starting_level) <= model.LADDER.index(source.level), move


def test_far_common_paths_are_flagged_for_review(tmp_path):
    roles = ROLES.replace(
        "roles:\n",
        "roles:\n  ana: {name: Analyst, family: eng, summary: x, onet: 15-2051.01, levels: [entry, mid]}\n",
    ).replace("{from: dev@mid, to: ops@mid, kind: pivot}", "{from: dev@mid, to: ana@mid, kind: pivot}")
    skills = (
        SKILLS
        + "  sql: {name: SQL, kind: language, description: x}\n"
        + "  stats: {name: Statistics, kind: concept, description: x}\n"
    )
    path = make(tmp_path, roles=roles, skills=skills)
    (path / "roadmaps" / "ana.yaml").write_text(roadmap("ana", "sql:2", "stats:2"))
    report = validate.validate(model.load_catalog(path))
    assert report.errors == []
    assert any("common path dev@mid -> ana@mid: ana is only #2 of 2 nearest roles" in w for w in report.warnings)


def test_titles_that_add_little_stay_aliases_and_big_ones_must_become_roles(tmp_path):
    """ops at mid needs basics, go, docker, kubernetes. Adding one skill keeps a title an alias; adding a
    distinctive stack makes it a different job (ADR-0027: role covers < 80%)."""
    skills = SKILLS + "  helm: {name: Helm, kind: tool, description: x, requires: [kubernetes:2]}\n"
    skills += "  argo: {name: Argo, kind: tool, description: x, requires: [kubernetes:2]}\n"
    skills += "  istio: {name: Istio, kind: tool, description: x, requires: [kubernetes:2]}\n"
    small = ROLES.replace(
        "onet: 15-1299.08, levels: [entry, mid]}",
        "onet: 15-1299.08, levels: [entry, mid], also_called: [SysAdmin, {title: K8s Admin, adds: [helm:1]}]}",
    )
    big = small.replace("adds: [helm:1]", "adds: [helm:3, argo:3, istio:3]")
    ok, too_far = tmp_path / "ok", tmp_path / "far"
    ok.mkdir(), too_far.mkdir()
    cat = model.load_catalog(make(ok, roles=small, skills=skills))
    assert validate.validate(cat).errors == []
    assert [a.title for a in cat.roles["ops"].also_called] == ["SysAdmin", "K8s Admin"]
    found = errors(make(too_far, roles=big, skills=skills))
    assert len(found) == 1 and "also called 'K8s Admin': the role covers only" in found[0]


def test_alias_titles_must_be_unique_and_use_known_skills(tmp_path):
    roles = ROLES.replace(
        "onet: 15-1299.08, levels: [entry, mid]}",
        "onet: 15-1299.08, levels: [entry, mid], also_called: [Developer, {title: X, adds: [cobol:1]}]}",
    )
    found = errors(make(tmp_path, roles=roles))
    assert "role ops: title 'Developer' is already used by dev" in found
    assert "role ops, also called 'X': unknown skill 'cobol'" in found
