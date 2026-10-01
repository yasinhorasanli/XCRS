"""Knowledge-unit suggestions for the input page (ADR-0023): browse, search, and "related to what you entered"."""

import json
from collections.abc import Callable
from functools import cache
from importlib.resources import files

from sqlalchemy.orm import Session

from xcrs.db.models import EmbeddingModel
from xcrs.domain import suggestions
from xcrs.domain.suggestions import KnowledgeUnit, RelatedUnit
from xcrs.embeddings import embedder_for
from xcrs.embeddings.base import Embedder
from xcrs.repository import catalog as catalog_repo
from xcrs.repository import vectors

RELATED_PER_PHRASE = 8
# Below this, a "related" concept is usually noise. Just above the threshold statistics' mean + 1σ
# for qwen3-embedding:0.6b (0.234 + 0.105); re-check when the model changes.
RELATED_MIN_SIMILARITY = 0.35


@cache
def curated_groups() -> list[dict]:
    return json.loads(files("xcrs").joinpath("data/knowledge_units.json").read_text())["groups"]


class KnowledgeUnitService:
    def __init__(self, session: Session, embedder_factory: Callable[[EmbeddingModel], Embedder] = embedder_for):
        self.session = session
        self.embedder_factory = embedder_factory

    def groups(self) -> list[dict]:
        return curated_groups()

    def search(self, query: str, limit: int) -> list[KnowledgeUnit]:
        curated = [u for g in curated_groups() for u in g["units"]]
        units = suggestions.merge_units(curated, catalog_repo.concept_names(self.session))
        return suggestions.search(units, query, limit)

    def related(self, phrases: list[str], avoid: list[str], exclude: list[str], limit: int) -> list[RelatedUnit]:
        """Suggestions near `phrases` (what the learner enjoyed or is curious about), steering away from
        `avoid` (what they didn't enjoy) and never repeating anything in `exclude` (everything entered)."""
        phrases = [p.strip() for p in phrases if p.strip()]
        avoid = [a.strip() for a in avoid if a.strip()]
        if not phrases:
            return []
        model = catalog_repo.active_model(self.session)
        texts = phrases + avoid  # one embedding call for both
        vecs = self.embedder_factory(model).embed_query(texts)
        matches = vectors.nearest_concepts(self.session, model, vecs, RELATED_PER_PHRASE)
        catalog = catalog_repo.load_roadmap_catalog(self.session)
        hits = [(texts[m.phrase_index], catalog.node_names[m.concept_id], m.similarity) for m in matches]
        positive = [h for h, m in zip(hits, matches, strict=True) if m.phrase_index < len(phrases)]
        negative = [h for h, m in zip(hits, matches, strict=True) if m.phrase_index >= len(phrases)]
        return suggestions.related(positive, negative, [*phrases, *avoid, *exclude], limit, RELATED_MIN_SIMILARITY)
