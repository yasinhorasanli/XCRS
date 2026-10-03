"""Dev tools (xcrs/api/dev.py): off by default; with them on, the test profiles load and preview."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from xcrs.api import dev
from xcrs.api.app import app
from xcrs.catalog import model
from xcrs.config import get_settings
from xcrs.db.session import new_session


def test_test_profiles_use_catalog_skills_and_roles():
    cat = model.load_catalog()
    profiles = dev.load_profiles()
    assert len(profiles) >= 10 and len({p.id for p in profiles}) == len(profiles)
    for p in profiles:
        assert all(c.skill in cat.skills for c in p.chips), p.id
        assert all(r in cat.roles for r in p.expect) and p.level in ("entry", "mid", "senior", "staff"), p.id


def test_dev_endpoints_are_off_unless_enabled(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(get_settings(), "dev_tools", False)
    assert client.get("/api/v2/dev/profiles").status_code == 404
    assert client.get("/api/v2/dev/profiles/preview").status_code == 404
    assert client.post("/api/v2/dev/profiles/cs-student/run").status_code == 404
    monkeypatch.setattr(get_settings(), "dev_tools", True)
    try:
        first = client.get("/api/v2/dev/profiles").json()[0]
    except OperationalError:
        pytest.skip("needs the database")
    assert first["id"] and first["chips"][0]["name"]


def test_preview_scores_every_profile(monkeypatch):
    try:
        with new_session() as s:
            s.execute(text("SELECT 1 FROM catalog.roles LIMIT 1"))
    except OperationalError:
        pytest.skip("needs the database")
    monkeypatch.setattr(get_settings(), "dev_tools", True)
    previews = TestClient(app).get("/api/v2/dev/profiles/preview").json()
    assert len(previews) == len(dev.load_profiles()) and all(p["roles"] for p in previews)
