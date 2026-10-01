"""Knowledge-unit suggestions (ADR-0023): pure functions, no database."""

from xcrs.domain.labels import display_label, skill_label
from xcrs.domain.suggestions import KnowledgeUnit, merge_units, related, search


def test_roadmap_slugs_get_their_usual_spelling():
    assert display_label("ci cd") == "CI/CD"
    assert display_label("csharp") == "C#"
    assert display_label("nosql databases") == "NoSQL databases"
    assert display_label("basic usage of git") == "Basic usage of Git"
    assert display_label("event sourcing") == "Event sourcing"
    assert display_label("npm") == "npm"  # a known spelling isn't re-capitalized
    assert display_label("ecs fargate") == "ECS fargate"
    assert display_label("ab testing") == "A/B testing"


def test_merge_keeps_curated_spelling_and_collects_roadmaps():
    units = merge_units(["Python", "Spring Boot"], [("python", "Backend Developer"), ("python", "DevOps Engineer")])
    python = next(u for u in units if u.label == "Python")
    assert python.source == "curated" and python.roles == ["Backend Developer", "DevOps Engineer"]
    assert {u.label for u in units} == {"Python", "Spring Boot"}


def test_search_ranks_exact_then_prefix_then_word_prefix_then_substring():
    units = [
        KnowledgeUnit("Docker compose", "roadmap", ["DevOps Engineer"]),
        KnowledgeUnit("Using docker", "roadmap"),
        KnowledgeUnit("Docker", "curated"),
        KnowledgeUnit("Dockerfile basics", "roadmap"),
        KnowledgeUnit("Unrelated", "curated"),
    ]
    assert [u.label for u in search(units, "docker", 10)] == [
        "Docker",
        "Docker compose",
        "Dockerfile basics",
        "Using docker",
    ]
    assert search(units, "  ", 10) == []


def test_related_leaves_out_what_was_entered_and_weak_hits():
    hits = [
        ("Docker", "docker", 0.95),  # the phrase itself
        ("Docker", "kubernetes", 0.61),
        ("Linux", "kubernetes", 0.40),  # same label, weaker: the best one wins
        ("Docker", "podman", 0.30),  # below the minimum
        ("Linux", "bash", 0.58),
    ]
    units = related(hits, [], ["Docker", "Linux"], limit=10, min_similarity=0.35)
    assert [(u.label, u.because) for u in units] == [("Kubernetes", "Docker"), ("Bash", "Linux")]


def test_related_steers_away_from_what_the_learner_did_not_enjoy():
    """Liked Python, disliked HTML: "Writing semantic HTML" is closer to the dislike, so it isn't suggested.
    A concept nearer to the like than to the dislike still is."""
    hits = [("Python", "writing semantic html", 0.40), ("Python", "django", 0.70), ("Python", "jinja", 0.50)]
    avoid_hits = [("HTML", "writing semantic html", 0.80), ("HTML", "jinja", 0.45)]
    units = related(hits, avoid_hits, ["Python", "HTML"], limit=10, min_similarity=0.35)
    assert [u.label for u in units] == ["Django", "Jinja"]


def test_related_drops_roadmap_filler_and_unwraps_phrased_skills():
    hits = [("HTML", "learn the basics", 0.9), ("HTML", "what is http", 0.8), ("SQL", "basics of kotlin", 0.4)]
    assert [u.label for u in related(hits, [], ["HTML", "SQL"], 10, 0.35)] == ["HTTP", "Kotlin"]


def test_skill_labels():
    assert skill_label("learn the basics") is None
    assert skill_label("checkpoint static websites") is None
    assert skill_label("how does the internet work") is None
    assert skill_label("what is http") == "HTTP"
    assert skill_label("learn dom manipulation") == "DOM manipulation"
    assert skill_label("basic usage of git") == "Git"
    assert skill_label("basic authentication") == "Basic authentication"  # a real concept, not a prefix
    assert display_label("ethers js") == "ethers.js"
    assert display_label("next js") == "Next.js"
    assert display_label("gke eks aks") == "GKE / EKS / AKS"
