"""Which phrase → concept matches count (ADR-0010, ADR-0022). Pure functions."""

from collections.abc import Iterable

from xcrs.domain.types import Phrase, PhraseConceptMatch


def select_matches(
    candidates: Iterable[PhraseConceptMatch], threshold: float, fallback_margin: float
) -> list[PhraseConceptMatch]:
    """Every match above `threshold` counts, as in the prototype. A phrase with none keeps its best
    candidate and every other candidate within `fallback_margin` of it (the candidates are already
    above the lower fallback floor), so a phrase like "Python" is not silently dropped because it is
    short and generic.

    A margin, not a fixed top-k: several roadmaps have a concept of the same name ("python" is in
    five), and near-tied candidates must be treated alike rather than cut at an arbitrary rank.
    Phrases with no candidate at all still match nothing.
    """
    by_phrase: dict[Phrase, list[PhraseConceptMatch]] = {}
    for m in candidates:
        by_phrase.setdefault(m.phrase, []).append(m)

    selected: list[PhraseConceptMatch] = []
    for matches in by_phrase.values():
        strong = [m for m in matches if m.similarity > threshold]
        if strong:
            selected.extend(strong)
        else:
            best = max(m.similarity for m in matches)
            selected.extend(m for m in matches if m.similarity >= best - fallback_margin)
    return selected
