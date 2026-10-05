"""Job titles for a recommended role (ADR-0044): which skills count, and how the title is built."""

from xcrs.domain.job_titles import market_titles, title_for
from xcrs.domain.role_scoring import Category, Mention, RoleSnapshot

ROLE = RoleSnapshot(
    id="backend-engineer",
    name="Backend Engineer",
    family="product-engineering",
    levels=["entry"],
    titles={"entry": None},
    requirements={"entry": []},
    market_titles=("Backend Developer", "API Developer"),
    title_skills=("java", "python", "go", "spring-boot", "django", "nodejs"),
)
NAMES = {"java": "Java", "python": "Python", "go": "Go", "spring-boot": "Spring Boot", "django": "Django"}
NAMES |= {"nodejs": "Node.js", "css": "CSS"}
LANGUAGES = frozenset({"java", "python", "go", "css"})
L, N, C, D = Category.LIKED, Category.NEUTRAL, Category.CURIOUS, Category.DISLIKED


def title(*mentions):
    return title_for(ROLE, [Mention(*m) for m in mentions], NAMES, LANGUAGES)


def test_a_language_goes_in_front_and_a_framework_in_brackets():
    assert title(("java", L, 3)) == "Java Backend Engineer"
    assert title(("spring-boot", L, 2)) == "Backend Engineer (Spring Boot)"
    assert title(("java", L, 3), ("spring-boot", L, None)) == "Java Backend Engineer (Spring Boot)"


def test_the_strongest_skill_wins():
    assert title(("python", L, 2), ("java", N, 4)) == "Java Backend Engineer"  # rating first
    assert title(("python", L, 3), ("java", N, 3)) == "Python Backend Engineer"  # then enjoyed over neutral
    assert title(("go", L, None), ("python", L, None)) == "Python Backend Engineer"  # then roles.yaml order


def test_curious_disliked_weak_and_unrelated_skills_dont_count():
    assert title(("java", C, None)) is None
    assert title(("java", D, 4)) is None
    assert title(("java", N, 1)) is None and title(("java", N, None)) is None
    assert title(("java", N, 2)) == "Java Backend Engineer"
    assert title(("css", L, 4)) is None  # not one of the role's title skills


def test_market_titles_start_with_the_role_name():
    assert market_titles(ROLE) == ["Backend Engineer", "Backend Developer", "API Developer"]
