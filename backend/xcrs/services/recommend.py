"""The recommendation use case: orchestrates I/O adapters and pure domain functions (ADR-0017).

Explanations don't happen here any more (ADR-0018): the service saves each role's explanation input
and queues a job; the caller gets the recommendation immediately and reads explanations later.
"""

import logging
import time
import uuid
from collections.abc import Callable

from sqlalchemy.orm import Session

from xcrs.db.models import EmbeddingModel
from xcrs.domain import scoring, selection
from xcrs.domain.results import CourseResult, ExplanationStatus, RecommendationResult, RoleResult
from xcrs.domain.types import Category, CourseCandidate, Phrase, PhraseConceptMatch
from xcrs.embeddings import embedder_for
from xcrs.embeddings.base import Embedder
from xcrs.explain.base import CourseContext, KnownItem, RoleContext
from xcrs.repository import activity, vectors
from xcrs.repository import catalog as catalog_repo
from xcrs.services.explanations import ExplanationQueue

log = logging.getLogger(__name__)

ALGORITHM_VERSION = "1.0.0"  # prototype algorithm, ported (see domain/)
THRESHOLD_SIGMA = 2.5
EXPLAIN_MAX_ITEMS = 12
NEXT_CONCEPTS = 8  # uncovered concepts returned per role, and given to the explanation LLM

__all__ = ["CourseResult", "RecommendationResult", "RecommendationService", "RoleResult"]


def to_phrases(user_input: dict[Category, list[str]]) -> list[Phrase]:
    """Trimmed, non-empty, de-duplicated per category (as the prototype did), order preserved."""
    phrases: list[Phrase] = []
    for category, texts in user_input.items():
        seen: set[str] = set()
        for t in (t.strip() for t in texts):
            if t and t.lower() not in seen:
                seen.add(t.lower())
                phrases.append(Phrase(t, category))
    return phrases


class RecommendationService:
    def __init__(
        self,
        session: Session,
        explanations: ExplanationQueue | None,
        embedder_factory: Callable[[EmbeddingModel], Embedder] = embedder_for,
    ):
        self.session = session
        self.explanations = explanations  # None: explanations are disabled
        self.embedder_factory = embedder_factory

    def recommend(self, user_input: dict[Category, list[str]]) -> RecommendationResult:
        started = time.perf_counter()
        model = catalog_repo.active_model(self.session)
        threshold = model.sim_mean + THRESHOLD_SIGMA * model.sim_std
        phrases = to_phrases(user_input)

        roles: list[RoleResult] = []
        contexts: dict[int, RoleContext] = {}
        if phrases:
            roles, contexts = self._recommend(model, threshold, phrases)

        status = ExplanationStatus.PENDING if self.explanations else ExplanationStatus.DISABLED
        for role in roles:
            role.explanation_status = status

        result = RecommendationResult(
            request_id=uuid.uuid4(),
            status="ok" if roles else "insufficient_input",
            model=model.name,
            roles=roles,
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
        activity.save_recommendation(
            self.session,
            result,
            model_id=model.id,
            algorithm_version=ALGORITHM_VERSION,
            threshold=threshold,
            user_input={category.value: texts for category, texts in user_input.items()},
            explanation_inputs={role_id: c.to_dict() for role_id, c in contexts.items()} if self.explanations else {},
        )
        # Queue only after the commit, so a worker never looks for rows that aren't there yet.
        if self.explanations:
            for role in roles:
                self.explanations.submit(result.request_id, role.role_id)
        return result

    def get(self, request_id: uuid.UUID) -> RecommendationResult | None:
        return activity.load_recommendation(self.session, request_id)

    def _recommend(
        self, model: EmbeddingModel, threshold: float, phrases: list[Phrase]
    ) -> tuple[list[RoleResult], dict[int, RoleContext]]:
        catalog = catalog_repo.load_roadmap_catalog(self.session)

        # 1. Embed the phrases and match them to roadmap concepts (exact threshold scan).
        phrase_vectors = self.embedder_factory(model).embed_query([p.text for p in phrases])
        matches = [
            PhraseConceptMatch(phrases[m.phrase_index], m.concept_id, catalog.concept_roles[m.concept_id], m.similarity)
            for m in vectors.concepts_above_threshold(self.session, model, phrase_vectors, threshold)
        ]

        # 2. Score roles (pure).
        role_scores = scoring.score_roles(matches, catalog.concept_roles, catalog.concepts_per_role)
        if not role_scores:
            return [], {}

        # 3. Concepts each role's courses should target (pure).
        targets: dict[int, list[int]] = {}
        for rs in role_scores:
            known = {m.concept_id for m in matches if m.role_id == rs.role_id and m.phrase.category.is_taken}
            targets[rs.role_id] = selection.concepts_to_learn(catalog.role_concepts_in_order[rs.role_id], known)

        # 4. Candidate courses from the precomputed matches; disliked penalty on candidates only.
        all_targets = sorted({c for ids in targets.values() for c in ids})
        candidates: dict[int, list[CourseCandidate]] = {}
        for concept_id, course_id, similarity in vectors.candidate_courses(self.session, model, all_targets):
            candidates.setdefault(concept_id, []).append(CourseCandidate(concept_id, course_id, similarity))
        disliked_vectors = [v for p, v in zip(phrases, phrase_vectors, strict=True) if p.category is Category.DISLIKED]
        candidate_ids = sorted({c.course_id for cs in candidates.values() for c in cs})
        penalized = vectors.courses_similar_to(self.session, model, disliked_vectors, candidate_ids, threshold)

        # 5. Pick courses per role (pure), then load their details.
        picks = {rs.role_id: selection.pick_courses(targets[rs.role_id], candidates, penalized) for rs in role_scores}
        courses = catalog_repo.courses_by_id(self.session, sorted({p.course_id for ps in picks.values() for p in ps}))

        results, contexts = [], {}
        for rs in role_scores:
            next_ids = targets[rs.role_id][:NEXT_CONCEPTS]
            role = RoleResult(
                rs.role_id,
                catalog.role_names[rs.role_id],
                rs.score,
                next_to_learn=[catalog.node_names[c] for c in next_ids],
                next_concept_ids=next_ids,
            )
            for p in picks[rs.role_id]:
                c = courses[p.course_id]
                role.courses.append(
                    CourseResult(
                        c.id,
                        c.title,
                        c.url,
                        p.similarity,
                        [catalog.node_names[i] for i in p.concept_ids],
                        concept_ids=list(p.concept_ids),
                    )
                )
            contexts[rs.role_id] = self._context(role, matches, courses, catalog)
            results.append(role)
        return results, contexts

    def _context(self, role: RoleResult, matches, courses, catalog) -> RoleContext:
        """The facts an explanation may draw on for this role, and nothing else (ADR-0019)."""
        role_matches = sorted((m for m in matches if m.role_id == role.role_id), key=lambda m: -m.similarity)

        def items(taken: bool) -> list[KnownItem]:
            seen, out = set(), []
            for m in role_matches:
                key = (m.phrase.text, m.concept_id)
                if m.phrase.category.is_taken == taken and key not in seen:
                    seen.add(key)
                    out.append(KnownItem(m.phrase.text, catalog.node_names[m.concept_id], m.phrase.category.value))
            return out[:EXPLAIN_MAX_ITEMS]

        known_concepts = [m.concept_id for m in role_matches if m.phrase.category.is_taken]
        topics = selection.covered_topics(known_concepts, catalog.concept_ancestors, catalog.concepts_per_topic)
        return RoleContext(
            role=role.role,
            score=role.score,
            known=items(taken=True),
            curious=items(taken=False),
            covered_topics=[catalog.node_names[t] for t in topics],
            next_to_learn=role.next_to_learn,
            courses=[
                CourseContext(
                    c.course_id,
                    c.title,
                    courses[c.course_id].headline,
                    (courses[c.course_id].what_you_learn or "")[:400] or None,
                    c.concepts[:5],
                )
                for c in role.courses
            ],
        )
