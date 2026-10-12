"""Engine v2 role scoring (ADR-0031) on the real catalog: ranking, levels, gaps, implied prerequisites,
the calibrated weights still doing what the calibration measured, and which roles are shown (ADR-0052)."""

import itertools
import random
from pathlib import Path

import pytest
import yaml

from xcrs.catalog import model
from xcrs.catalog.snapshot import snapshot_from_catalog
from xcrs.domain.role_scoring import Category, Mention, learner_skills, recommendable, score_roles, with_prerequisites

L, N, D, C = Category.LIKED, Category.NEUTRAL, Category.DISLIKED, Category.CURIOUS


@pytest.fixture(scope="module")
def snapshot():
    return snapshot_from_catalog(model.load_catalog())


def top(snapshot, mentions, n=1):
    return [r.role for r in score_roles(snapshot, mentions)[:n]]


EVAL = Path(__file__).parents[1] / "eval"


def board(entry: dict) -> list[Mention]:
    """A board in the learner_profiles.yaml format ("skill" or "skill:proficiency" per category)."""
    mentions = []
    for c in (L, N, D, C):
        for chip in entry.get(c.value, []):
            skill, _, level = str(chip).partition(":")
            mentions.append(Mention(skill, c, int(level) if level else None))
    return mentions


BOARDS = yaml.safe_load((EVAL / "gate_boards.yaml").read_text())["boards"]


@pytest.mark.parametrize("entry", BOARDS, ids=[b["id"] for b in BOARDS])
def test_a_board_shows_only_roles_it_gives_evidence_for(snapshot, entry):
    """ADR-0052: "Python" alone or a board of dislikes singles out no role; one telling skill shows its roles;
    a broad board still shows its strongest roles."""
    ranked = score_roles(snapshot, board(entry), experience=entry.get("experience"))
    shown = [r.role for r in recommendable(ranked)]
    if entry["may_show"]:
        assert shown and set(shown) <= set(entry["may_show"])
    else:
        assert shown == []


def test_the_gate_keeps_every_labeled_profile_s_expected_role(snapshot):
    """ADR-0052 calibration: the gate drops weak and near-tied roles, never the role a profile expects."""
    profiles = yaml.safe_load((EVAL / "learner_profiles.yaml").read_text())["profiles"]
    missing = [
        p["id"]
        for p in profiles
        if p["expect"][0] not in {r.role for r in recommendable(score_roles(snapshot, board(p)))}
    ]
    assert missing == []


def test_a_broad_board_always_shows_a_role(snapshot):
    """Engine v3.4: three labeled learners on one board (and 5-10 years) touch many roles and lift the mean score;
    the gate must still show their strongest roles. v3.3 showed none for 4 of 1,500 such boards."""
    profiles = yaml.safe_load((EVAL / "learner_profiles.yaml").read_text())["profiles"]
    rng = random.Random(7)
    empty = []
    for trio in rng.sample(list(itertools.combinations(profiles, 3)), 300):
        mentions = {}
        for p in trio:
            for m in board(p):
                mentions.setdefault(m.skill, m)
        if not recommendable(score_roles(snapshot, list(mentions.values()), experience="5-10")):
            empty.append([p["id"] for p in trio])
    assert empty == []


def test_interest_shown_is_the_share_of_the_role_not_of_the_board(snapshot):
    """One generic skill is the whole board but a small part of any role (ADR-0052); `interest` (for ranking)
    reads 50% there, which is what learners found misleading."""
    python = {r.role: r for r in score_roles(snapshot, [Mention("python", L)])}["backend-engineer"]
    assert python.interest == pytest.approx(0.5) and python.interest_in_role < 0.05
    frontend = [Mention(s, L, 3) for s in ("html", "css", "javascript", "react", "typescript")]
    shown = {r.role: r for r in score_roles(snapshot, frontend)}["frontend-engineer"]
    assert shown.interest_in_role > 5 * python.interest_in_role


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
    # Python asks basic fundamentals of a beginner; knowing it well implies working fundamentals, not expert ones.
    assert with_prerequisites({"python": 1}, snapshot.prerequisites)["programming-fundamentals"] == 1
    assert with_prerequisites({"python": 4}, snapshot.prerequisites)["programming-fundamentals"] == 2


def test_class_based_languages_show_oop(snapshot):
    """ADR-0028 note: Java isn't learned after OOP (no prerequisite), but knowing it shows OOP, at most working."""
    implied = lambda have: with_prerequisites(have, snapshot.prerequisites, snapshot.implications)  # noqa: E731
    assert implied({"java": 4})["oop"] == 2 and implied({"java": 1})["oop"] == 1
    assert implied({"kotlin": 3})["oop"] == 2 and "oop" not in implied({"python": 4})  # OOP is optional in Python
    backend = {r.role: r for r in score_roles(snapshot, [Mention("java", L, 4), Mention("kubernetes", L)])}
    assert not any("oop" in g.options for g in backend["backend-engineer"].gaps)


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


# --- ADR-0041: evidence per level, experience, next-level gaps --------------------------------------------


def _profile(pid):
    from xcrs.api import dev

    return next(p for p in dev.load_profiles() if p.id == pid)


def _top(snap, profile, experience="as-profile"):
    from xcrs.api import dev

    exp = profile.experience if experience == "as-profile" else experience
    roles = score_roles(snap, dev.mentions(profile), experience=exp)
    return next(r for r in roles if r.role == profile.expect[0])


def test_unlisted_basics_dont_stop_a_senior_at_entry():
    snap = snapshot_from_catalog(model.load_catalog())
    r = _top(snap, _profile("senior-data-engineer"), experience=None)  # 84% of senior, 49% of entry skills
    assert r.level == "senior"
    shown = {o for g in r.gaps for o in g.options}
    assert not shown & {"git", "debugging", "data-structures", "programming-fundamentals"}  # basics aren't gaps
    assert any("git" in g.options for g in r.basics)  # they are listed to check instead


def test_experience_caps_and_lifts_the_level():
    snap = snapshot_from_catalog(model.load_catalog())
    staff = _profile("staff-platform")
    assert _top(snap, staff, experience="student").level is None  # a student isn't staff
    assert _top(snap, _profile("python-backend-4y")).level == "mid"  # 2-5 years lifts entry to mid
    designer = _profile("designer-to-frontend")  # 2-5 years, but in design: no lift in frontend
    assert _top(snap, designer).level == "entry"


def test_test_profiles_reach_their_expected_level():
    from xcrs.api import dev

    snap = snapshot_from_catalog(model.load_catalog())
    ladder = ("entry", "mid", "senior", "staff")
    exact = sum(_top(snap, p).level == p.level for p in dev.load_profiles())
    near = sum(
        abs((ladder.index(_top(snap, p).level) if _top(snap, p).level else -1) - ladder.index(p.level)) <= 1
        for p in dev.load_profiles()
    )
    assert exact >= 13 and near == len(dev.load_profiles())  # measured 14/15 and 15/15 (ADR-0041)


def test_curious_skills_below_the_level_stay_gaps_not_assumed_basics():
    snap = snapshot_from_catalog(model.load_catalog())
    r = _top(snap, _profile("staff-platform"))  # curious about service mesh, a senior platform skill
    assert any("service-mesh" in g.options for g in r.gaps)
    assert not any("service-mesh" in g.options for g in r.basics)
