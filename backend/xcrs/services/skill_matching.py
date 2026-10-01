"""Matching what a learner types to catalog skills (ADR-0030).

Order: name lookup → cached LLM answer → LLM picks from the catalog, each pick confirmed by embedding
similarity → (LLM unavailable) the most similar skill if similar enough. Picked chips from the catalog
never come here: they already are skill ids.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Literal, Protocol

import httpx
import openai
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from xcrs.config import get_settings
from xcrs.db.models import EmbeddingModel, PhraseMatch, Skill
from xcrs.domain.skill_matching import LexicalIndex, normalize
from xcrs.embeddings import embedder_for
from xcrs.embeddings.base import Embedder
from xcrs.matching.picker import LLMSkillPicker
from xcrs.matching.prompts import QUERY_INSTRUCTION
from xcrs.repository import catalog_store, vectors

log = logging.getLogger(__name__)

# Measured on eval/skill_matching.yaml (ADR-0030); re-tune with eval/bench_skill_matching.py when a model changes.
CONFIRM_FLOOR = 0.40  # an LLM pick is kept only if the phrase is at least this similar to the skill
FALLBACK_FLOOR = 0.44  # without the LLM: the most similar skill, if at least this similar

Method = Literal["lookup", "llm", "cache", "embedding", "none"]


@dataclass
class PhraseResult:
    phrase: str
    skills: list[str] = field(default_factory=list)
    method: Method = "none"


class Picker(Protocol):
    prompt_version: str
    model: str

    def pick(self, phrase: str) -> list[str]: ...


class MatchStore(Protocol):
    """What matching needs from the database (a fake in unit tests)."""

    catalog_checksum: str

    def similarities(self, vector) -> dict[str, float]: ...
    def cached(self, key: str, picker: Picker) -> list[str] | None: ...
    def save(self, key: str, picker: Picker, skills: list[str], picked: list[str], llm_ms: int) -> None: ...


class SkillMatcher:
    def __init__(self, index: LexicalIndex, store: MatchStore, embedder: Embedder, picker: Picker | None):
        self.index, self.store, self.embedder, self.picker = index, store, embedder, picker

    def match(self, phrases: list[str]) -> list[PhraseResult]:
        return [self._match(p) for p in phrases]

    def _match(self, phrase: str) -> PhraseResult:
        key = normalize(phrase)
        if not key:
            return PhraseResult(phrase)
        found, resolved = self.index.lookup(phrase)
        if resolved:
            return PhraseResult(phrase, sorted(found), "lookup")

        def with_found(skills: list[str], method: Method) -> PhraseResult:
            merged = [*sorted(found), *(s for s in skills if s not in found)]
            return PhraseResult(phrase, merged, method if merged else "none")

        if self.picker is not None:
            cached = self.store.cached(key, self.picker)
            if cached is not None:
                return with_found(cached, "cache")
            try:
                started = time.perf_counter()
                picked = self.picker.pick(phrase)
                llm_ms = int((time.perf_counter() - started) * 1000)
            except (openai.OpenAIError, httpx.HTTPError, ValueError) as exc:
                log.warning("skill picker failed for %r, falling back to embeddings: %s", phrase, exc)
            else:
                sims = self._similarities(phrase)
                confirmed = [s for s in picked if sims.get(s, 0.0) >= CONFIRM_FLOOR]
                self.store.save(key, self.picker, confirmed, picked, llm_ms)
                return with_found(confirmed, "llm")

        sims = self._similarities(phrase)
        best = max(sims.items(), key=lambda kv: kv[1], default=None)
        return with_found([best[0]] if best and best[1] >= FALLBACK_FLOOR else [], "embedding")

    def _similarities(self, phrase: str) -> dict[str, float]:
        return self.store.similarities(self.embedder.embed_query([phrase])[0])


class DatabaseMatchStore:
    def __init__(self, session: Session, model: EmbeddingModel):
        self.session, self.model = session, model
        self.catalog_checksum = catalog_store.last_import_checksum(session) or "none"

    def similarities(self, vector) -> dict[str, float]:
        return vectors.skill_similarities(self.session, self.model, vector)

    def cached(self, key: str, picker: Picker) -> list[str] | None:
        return self.session.scalar(
            select(PhraseMatch.skills).where(
                PhraseMatch.phrase_key == key,
                PhraseMatch.catalog_checksum == self.catalog_checksum,
                PhraseMatch.prompt_version == picker.prompt_version,
                PhraseMatch.llm_model == picker.model,
            )
        )

    def save(self, key: str, picker: Picker, skills: list[str], picked: list[str], llm_ms: int) -> None:
        self.session.execute(
            insert(PhraseMatch)
            .values(
                phrase_key=key,
                catalog_checksum=self.catalog_checksum,
                prompt_version=picker.prompt_version,
                llm_model=picker.model,
                skills=skills,
                picked=picked,
                llm_ms=llm_ms,
            )
            .on_conflict_do_nothing()
        )
        self.session.commit()


def lexical_index(session: Session) -> LexicalIndex:
    rows = session.execute(select(Skill.slug, Skill.name, Skill.onet)).all()
    return LexicalIndex.build((slug, name, onet or ()) for slug, name, onet in rows)


def skill_names(session: Session) -> list[tuple[str, str]]:
    """(slug, name) in catalog order, for the picker's prompt."""
    return [tuple(r) for r in session.execute(select(Skill.slug, Skill.name).order_by(Skill.id)).all()]


def build_matcher(session: Session, use_llm: bool | None = None) -> SkillMatcher:
    """A matcher over the imported catalog, the active embedding model and (if enabled) the LLM."""
    settings = get_settings()
    model = session.scalars(select(EmbeddingModel).where(EmbeddingModel.status == "active")).one_or_none()
    if model is None:
        raise RuntimeError("no active embedding model; run `xcrs register-model`")
    picker = None
    if settings.llm_enabled if use_llm is None else use_llm:
        picker = LLMSkillPicker(
            settings.llm_base_url,
            settings.llm_model,
            skill_names(session),
            timeout_s=settings.match_llm_timeout_s,
            disable_thinking=settings.llm_disable_thinking,
            api_key=settings.llm_api_key,
        )
    embedder = embedder_for(model, query_prefix=QUERY_INSTRUCTION)  # the instruction measured for skills
    return SkillMatcher(lexical_index(session), DatabaseMatchStore(session, model), embedder, picker)
