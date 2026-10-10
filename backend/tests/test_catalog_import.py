"""Catalog import (ADR-0028) against the local database. Every test runs inside a transaction that is
rolled back, so the database keeps whatever catalog it had."""

import dataclasses

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, ProgrammingError

from xcrs.catalog import model, validate
from xcrs.catalog.model import RoleLevel
from xcrs.db.session import new_session
from xcrs.repository import catalog_store


@pytest.fixture
def session():
    s = new_session()
    try:
        s.execute(text("SELECT 1 FROM catalog.imports LIMIT 1"))
    except (OperationalError, ProgrammingError):
        s.close()
        pytest.skip("needs the database at migration 0005 or later")
    yield s
    s.rollback()
    s.close()


def run_import(session, cat):
    return catalog_store.import_catalog(
        session, cat, git_commit="test", git_dirty=True, checksum="test", stats=validate.validate(cat).stats
    )


def test_the_database_holds_the_same_requirements_as_the_yaml(session):
    cat = model.load_catalog()
    run_import(session, cat)
    for role in cat.roles.values():
        for level in role.levels:
            expected = {tuple(sorted(k)): v for k, v in validate.requirements(cat, RoleLevel(role.id, level)).items()}
            assert catalog_store.role_requirements(session, role.id, level) == expected, f"{role.id}@{level}"


def test_reimporting_the_same_catalog_changes_nothing(session):
    cat = model.load_catalog()
    run_import(session, cat)
    changes = run_import(session, cat)
    for name in ("skills", "roles", "families", "levels"):
        assert changes[name] == {"added": [], "updated": [], "removed": []}, name


def test_edits_and_removals_are_applied_and_reported(session):
    cat = model.load_catalog()
    run_import(session, cat)
    skills = dict(cat.skills)
    skills["docker"] = dataclasses.replace(skills["docker"], description="Containers.")
    del skills["feature-stores"]  # in no roadmap, so the catalog stays valid
    changes = run_import(session, dataclasses.replace(cat, skills=skills))
    assert changes["skills"] == {"added": [], "updated": ["docker"], "removed": ["feature-stores"]}
    found = session.execute(text("SELECT description FROM catalog.skills WHERE slug = 'docker'")).scalar_one()
    assert found == "Containers."
    assert session.execute(text("SELECT count(*) FROM catalog.skills WHERE slug = 'feature-stores'")).scalar() == 0


def sequence_values(session) -> dict[str, int]:
    """Sequences advance outside the test's transaction, so they show what the import drew."""
    rows = session.execute(
        text(
            "SELECT sequencename, coalesce(last_value, 0) FROM pg_sequences WHERE schemaname = 'catalog' "
            "AND sequencename IN ('roles_id_seq', 'families_id_seq', 'skills_id_seq')"
        )
    )
    return dict(rows.all())


def test_reimporting_draws_no_ids_and_a_new_skill_draws_one(session):
    cat = model.load_catalog()
    run_import(session, cat)
    before = sequence_values(session)
    run_import(session, cat)
    assert sequence_values(session) == before  # INSERT ... ON CONFLICT used to draw one per row (ran out in dev)
    new = dataclasses.replace(cat.skills["docker"], id="test-new-skill", name="Test new skill", onet=())
    run_import(session, dataclasses.replace(cat, skills={**cat.skills, new.id: new}))
    after = sequence_values(session)
    assert after["skills_id_seq"] == before["skills_id_seq"] + 1
    assert after["roles_id_seq"] == before["roles_id_seq"]
