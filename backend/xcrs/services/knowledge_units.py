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

    def related(self, phrases: list[str], limit: int) -> list[RelatedUnit]:
        phrases = [p.strip() for p in phrases if p.strip()]
        if not phrases:
            return []
        model = catalog_repo.active_model(self.session)
        vecs = self.embedder_factory(model).embed_query(phrases)
        matches = vectors.nearest_concepts(self.session, model, vecs, RELATED_PER_PHRASE)
        catalog = catalog_repo.load_roadmap_catalog(self.session)
        hits = [(phrases[m.phrase_index], catalog.node_names[m.concept_id], m.similarity) for m in matches]
        return suggestions.related(hits, phrases, limit, RELATED_MIN_SIMILARITY)
