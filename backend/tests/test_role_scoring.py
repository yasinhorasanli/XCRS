"""Engine v2 role scoring (ADR-0031) on the real catalog: ranking, levels, gaps, implied prerequisites,
and the calibrated weights still doing what the calibration measured."""

from pathlib import Path

import pytest
import yaml

from xcrs.catalog import model
from xcrs.catalog.snapshot import snapshot_from_catalog
from xcrs.domain.role_scoring import Category, Mention, learner_skills, score_roles, with_prerequisites

L, N, D, C = Category.LIKED, Category.NEUTRAL, Category.DISLIKED, Category.CURIOUS


@pytest.fixture(scope="module")
def snapshot():
    return snapshot_from_catalog(model.load_catalog())


def top(snapshot, mentions, n=1):
    return [r.role for r in score_roles(snapshot, mentions)[:n]]


def test_a_backend_profile_ranks_backend_first(snapshot):
    mentions = [Mention("java", L, 3), Mention("spring-boot", L, 3), Mention("sql", L, 3), Mention("css", D, 1)]
    assert top(snapshot, mentions) == ["backend-engineer"]


def test_curiosity_points_a_career_changer_to_the_new_role(snapshot):
    mentions = [Mention("python", L, 3), Mention("sql", L, 3)]
    mentions += [Mention(s, C) for s in ("airflow", "spark", "etl-pipelines", "data-warehousing")]
    assert top(snapshot, mentions) == ["data-engineer"]


def test_disliking_a_role_s_skills_lowers_it(snapshot):
    base = [Mention("python", L, 3), Mention("pandas", L, 3)]
    liked = {r.role: r.score for r in score_roles(snapshot, [*base, Mention("bi-tools", L, 3)])}
    disliked = {r.role: r.score for r in score_roles(snapshot, [*base, Mention("bi-tools", D, 3)])}
    assert disliked["data-analyst"] < liked["data-analyst"]


def test_known_skills_imply_their_prerequisites(snapshot):
    have = with_prerequisites({"django": 2}, snapshot.prerequisites)
    assert have["python"] >= 2  # Django requires Python at working level
    choice = with_prerequisites({"docker": 2}, snapshot.prerequisites)
    assert "python" not in choice and "go" not in choice  # "python|go:2" implies neither, unless named


def test_a_skill_mentioned_twice_keeps_its_best_rating_and_the_more_telling_category():
    have, category = learner_skills([Mention("sql", N, 2), Mention("sql", L, 3), Mention("sql", C)])
    assert have == {"sql": 3} and category == {"sql": L}


def test_meeting_a_level_estimates_it_and_gaps_lead_to_the_next(snapshot):
    role = snapshot.roles["data-analyst"]
    mentions = [Mention(r.options[0], L, r.level) for r in role.requirements["mid"]]
    [result] = [r for r in score_roles(snapshot, mentions) if r.role == "data-analyst"]
    assert result.level == "mid" and result.target_level == "senior"
    orders = {tuple(r.options): r.order for r in role.requirements["senior"]}
    assert [orders[g.options] for g in result.gaps] == sorted(orders[g.options] for g in result.gaps)
    assert all(g.have < g.need for g in result.gaps)


def test_too_little_input_means_starting_at_the_first_level(snapshot):
    [result] = [r for r in score_roles(snapshot, [Mention("html", L, 1)]) if r.role == "frontend-engineer"]
    assert result.level is None and result.target_level == "entry"


def test_the_calibrated_weights_keep_their_accuracy_on_the_profiles(snapshot):
    """Guard for ADR-0031: the defaults measured 96% held-out top-1 on these profiles; a change to scoring or
    the catalog that drops the in-sample top-1 below 90% needs a new calibration."""
    profiles = yaml.safe_load((Path(__file__).parents[1] / "eval" / "learner_profiles.yaml").read_text())["profiles"]
    hits = 0
    for p in profiles:
        mentions = []
        for c in (L, N, D, C):
            for chip in p.get(c.value, []):
                skill, _, level = str(chip).partition(":")
                mentions.append(Mention(skill, c, int(level) if level else None))
        hits += top(snapshot, mentions)[0] in set(p["expect"]) | set(p.get("accept", []))
    assert hits / len(profiles) >= 0.9


def test_the_learner_profiles_use_catalog_skills_and_roles():
    cat = model.load_catalog()
    profiles = yaml.safe_load((Path(__file__).parents[1] / "eval" / "learner_profiles.yaml").read_text())["profiles"]
    assert len({p["id"] for p in profiles}) == len(profiles) >= 40
    for p in profiles:
        chips = [str(c).partition(":")[0] for k in ("liked", "neutral", "disliked", "curious") for c in p.get(k, [])]
        assert [s for s in chips if s not in cat.skills] == [], p["id"]
        assert all(r in cat.roles for r in [*p["expect"], *p.get("accept", [])]), p["id"]
        assert p["level"] in cat.roles[p["expect"][0]].levels, p["id"]


def test_interest_follows_how_much_a_role_relies_on_a_skill(snapshot):
    """Loving SQL points to a role built on SQL (Data Analyst), not one that asks for basic SQL."""
    assert snapshot.reliance("data-analyst")["sql"] > snapshot.reliance("solutions-engineer")["sql"]
    assert top(snapshot, [Mention("sql", L)]) == ["data-analyst"]


def test_good_to_know_skills_count_for_interest_at_half(snapshot):
    role = snapshot.roles["backend-engineer"]
    assert "mongodb" in role.optional and 0 < snapshot.reliance("backend-engineer")["mongodb"] <= 0.5


def test_roles_entered_from_other_roles_rank_high_only_near_their_start(snapshot):
    """Software Architect starts at senior: generic fundamentals from a beginner don't make it the top role."""
    beginner = [Mention(s, L) for s in ("oop", "data-structures", "algorithms", "rest-api-design")]
    assert top(snapshot, beginner)[0] != "software-architect"
    senior = [Mention(s, L, 4) for s in ("system-design", "distributed-systems", "architecture-patterns")]
    senior += [Mention(s, L, 3) for s in ("architecture-decision-records", "domain-driven-design", "microservices")]
    assert top(snapshot, senior)[0] == "software-architect"
