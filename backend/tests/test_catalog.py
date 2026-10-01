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
transitions:
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


def test_transitions_must_cross_roles_and_use_existing_levels(tmp_path):
    roles = ROLES.replace("{from: dev@mid, to: ops@mid, kind: pivot}", "{from: dev@mid, to: dev@staff, kind: pivot}")
    found = errors(make(tmp_path, roles=roles))
    assert "transition dev@mid -> dev@staff: dev has no level 'staff'" in found
    assert any("moving up inside a role is implied" in e for e in found)


def test_bridge_lists_gaps_and_treats_a_choice_as_one_requirement(tmp_path):
    cat = model.load_catalog(make(tmp_path, dev_entry="basics:1, python|go:2", dev_mid="docker:1"))
    gap = validate.bridge(cat, RoleLevel("dev", "mid"), RoleLevel("ops", "mid"))
    assert gap == {"docker": (1, 2), "kubernetes": (0, 2)}  # go:2 is covered by dev's python|go choice
