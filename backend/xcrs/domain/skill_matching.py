"""Matching what a learner types to catalog skills (ADR-0029). Pure functions, no I/O.

Layer 1 is a name lookup: catalog skill names (and their parts: "Docker and containers" -> "docker",
"BI tools (Power BI, Tableau, Looker)" -> "tableau"), slugs and O*NET technology names, after
normalization, with a small edit distance for typos. A phrase that names several skills ("React and
TypeScript") is split and each part looked up. Whatever the lookup can't resolve goes to the embedding
layer, whose similarities `select` turns into matches with an absolute floor and a window below the best.
"""

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field

_KEEP = re.compile(r"[^a-z0-9+#. ]+")
_SPACES = re.compile(r"\s+")
_PARENS = re.compile(r"\(([^)]*)\)")
# Ways people list several things in one phrase. Tried only when the whole phrase isn't a known name.
_SEPARATORS = re.compile(r"\s*(?:,|;|&|\+(?=\s)|\band\b|\bwith\b|\bplus\b|\bor\b)\s*")


def normalize(text: str) -> str:
    """Lowercase ASCII; separators become spaces; keeps the characters of names like c++, c#, .net, node.js."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    text = text.replace("_", " ").replace("-", " ").replace("/", " ")
    text = _SPACES.sub(" ", _KEEP.sub(" ", text)).strip()
    return text.strip(".").strip()


def _edit_distance(a: str, b: str, limit: int, substitution: int = 1) -> int:
    """Optimal string alignment distance (a swap of neighbours counts once), cut off above `limit`.
    `substitution` is the cost of replacing one letter with another."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous2: list[int] = []
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else substitution
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                current[j] = min(current[j], previous2[j - 2] + 1)
        if min(current) > limit:
            return limit + 1
        previous2, previous = previous, current
    return previous[-1]


def _typo_limit(length: int) -> int:
    return 0 if length < 5 else 1 if length < 9 else 2


def _substitution_cost(length: int) -> int:
    """In short words a different letter usually means a different word (hiking/hiring, NestJS/Next.js),
    so there only dropped, extra or swapped letters count as typos."""
    return 2 if length < 9 else 1


@dataclass
class LexicalIndex:
    keys: dict[str, set[str]] = field(default_factory=dict)  # normalized name -> skill ids
    exact_only: set[str] = field(default_factory=set)  # keys never matched with typos (product names)

    def add(self, name: str, skill_id: str, exact: bool = False) -> None:
        key = normalize(name)
        if key:
            self.keys.setdefault(key, set()).add(skill_id)
            if exact:
                self.exact_only.add(key)
            else:  # also a skill's own name: typos allowed after all
                self.exact_only.discard(key)

    @classmethod
    def build(cls, skills: Iterable[tuple[str, str, Iterable[str]]]) -> "LexicalIndex":
        """`skills`: (id, name, other names such as O*NET technology names)."""
        index = cls()
        for skill_id, name, others in skills:
            # Other names (O*NET products such as "TestNG") match only exactly: one dropped letter turns
            # "testing" into "testng", and those names are not what learners misspell.
            for other in others:
                index.add(other, skill_id, exact=True)
            for text in (name, skill_id):
                index.add(text, skill_id)
                inside = _PARENS.findall(text)
                outside = _PARENS.sub(" ", text)
                index.add(outside, skill_id)
                # The head of "X and Y" names the skill; a later part only if it is a proper name
                # ("Prometheus and Grafana"), not a generic word ("Data quality and testing").
                head, *rest = re.split(r"\band\b|,", outside)
                for part in [head, *(r for r in rest if r.strip()[:1].isupper()), *",".join(inside).split(",")]:
                    if len(normalize(part)) >= 2:
                        index.add(part, skill_id)
        return index

    def find(self, text: str) -> set[str]:
        """Skills whose name is `text`, allowing a typo or two in longer words."""
        key = normalize(text)
        if not key:
            return set()
        if key in self.keys:
            return set(self.keys[key])
        limit = _typo_limit(len(key))
        if not limit:
            return set()
        best, found = limit + 1, set()
        for candidate, ids in self.keys.items():
            if len(candidate) < 5 or candidate in self.exact_only:
                continue
            distance = _edit_distance(key, candidate, limit, _substitution_cost(len(key)))
            if distance < best:
                best, found = distance, set(ids)
            elif distance == best:
                found |= ids
        return found if best <= limit else set()

    def lookup(self, phrase: str) -> tuple[set[str], bool]:
        """(skills found, whether the whole phrase is accounted for). A phrase that isn't a name is split
        into listed parts; parts that aren't names leave the phrase unresolved for the next layer."""
        whole = self.find(phrase)
        if whole:
            return whole, True
        parts = [p for p in _SEPARATORS.split(phrase) if p and p.strip()]
        if len(parts) < 2:
            return set(), False
        found: set[str] = set()
        resolved = True
        for part in parts:
            ids = self.find(part)
            found |= ids
            resolved = resolved and bool(ids)
        return found, resolved


def select(similarities: dict[str, float], floor: float, window: float, limit: int = 5) -> list[str]:
    """Skills at least `floor` similar and within `window` of the best one, best first (at most `limit`).
    The floor rejects input that matches nothing well; the window keeps near-ties (several skills in one
    phrase) and drops the long tail."""
    if not similarities:
        return []
    best = max(similarities.values())
    ranked = sorted(similarities.items(), key=lambda kv: -kv[1])
    return [s for s, sim in ranked if sim >= floor and sim >= best - window][:limit]
