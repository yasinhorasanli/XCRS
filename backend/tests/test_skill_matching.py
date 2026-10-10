"""Name lookup and threshold selection for skill matching (ADR-0029)."""

from xcrs.catalog import model
from xcrs.domain.skill_matching import LexicalIndex, normalize, select


def index() -> LexicalIndex:
    cat = model.load_catalog()
    return LexicalIndex.build((s.id, s.name, s.onet) for s in cat.skills.values())


def test_normalize_keeps_the_characters_of_names():
    assert normalize("  C++ / C#  ") == "c++ c#"
    assert normalize("Node.js") == "node.js"
    assert normalize("Tür-kçe_test!") == "tur kce test"


def test_lookup_finds_names_parts_and_typos():
    idx = index()
    assert idx.lookup("Kubernetes") == ({"kubernetes"}, True)
    assert idx.lookup("docker") == ({"docker"}, True)  # from "Docker and containers"
    assert idx.lookup("Tableau") == ({"bi-tools"}, True)  # from "BI tools (Power BI, Tableau, Looker)"
    assert idx.lookup("Pyhton") == ({"python"}, True)  # one swap
    assert idx.lookup("React and TypeScript") == ({"react", "typescript"}, True)
    assert idx.lookup("teaching computers to learn from data") == (set(), False)


def test_lookup_avoids_generic_words_and_letter_substitutions():
    idx = index()
    assert idx.lookup("testing") == (set(), False)  # only the tail of "Data quality and testing"
    assert idx.lookup("Grafana") == ({"prometheus-grafana"}, True)  # a proper name in the tail
    assert idx.lookup("hiking")[0] == set()  # not "hiring"
    assert idx.lookup("NestJS")[0] == set()  # not "Next.js"


def test_a_partly_named_phrase_is_left_for_the_next_layer():
    found, resolved = index().lookup("HTML CSS and JavaScript")
    assert "javascript" in found and not resolved


def test_select_applies_the_floor_and_the_window():
    sims = {"a": 0.80, "b": 0.77, "c": 0.60, "d": 0.40}
    assert select(sims, floor=0.5, window=0.05) == ["a", "b"]
    assert select(sims, floor=0.5, window=0.30) == ["a", "b", "c"]
    assert select(sims, floor=0.9, window=0.30) == []
    assert select({}, floor=0.5, window=0.1) == []
