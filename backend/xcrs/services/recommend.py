"""The recommendation use case: orchestrates I/O adapters and pure domain functions (ADR-0017)."""

import logging
import time
import uuid
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from xcrs.db.models import RecommendationRequest, RecommendedCourse, RecommendedRole
from xcrs.domain import scoring, selection
from xcrs.domain.types import Category, CourseCandidate, Phrase, PhraseConceptMatch
from xcrs.embeddings import embedder_for
from xcrs.explain.base import CourseContext, Explainer, KnownItem, RoleContext
from xcrs.repository import catalog as catalog_repo
from xcrs.repository import vectors

log = logging.getLogger(__name__)

ALGORITHM_VERSION = "1.0.0"  # prototype algorithm, ported (see domain/)
THRESHOLD_SIGMA = 2.5
EXPLAIN_MAX_ITEMS = 12
NEXT_CONCEPTS = 8  # uncovered concepts returned per role, and given to the explanation LLM


@dataclass
class CourseResult:
    course_id: int
    title: str
    url: str
    similarity: float
    concepts: list[str]
    explanation: str | None = None


@dataclass
class RoleResult:
    role_id: int
    role: str
    score: float
    explanation: str | None = None
    prompt_version: str | None = None
    next_to_learn: list[str] = field(default_factory=list)  # decided by the algorithm, not the LLM
    courses: list[CourseResult] = field(default_factory=list)


@dataclass
class RecommendationResult:
    request_id: uuid.UUID
    status: str  # ok | insufficient_input
    model: str
    roles: list[RoleResult]
    latency_ms: int


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
    def __init__(self, session: Session, explainer: Explainer):
        self.session = session
        self.explainer = explainer

    def recommend(self, user_input: dict[Category, list[str]]) -> RecommendationResult:
        started = time.perf_counter()
        model = catalog_repo.active_model(self.session)
        threshold = model.sim_mean + THRESHOLD_SIGMA * model.sim_std
        phrases = to_phrases(user_input)

        roles: list[RoleResult] = []
        if phrases:
            roles = self._recommend(model, threshold, phrases)

        result = RecommendationResult(
            request_id=uuid.uuid4(),
            status="ok" if roles else "insufficient_input",
            model=model.name,
            roles=roles,
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
        self._save(model.id, threshold, user_input, result)
        return result

    def _recommend(self, model, threshold: float, phrases: list[Phrase]) -> list[RoleResult]:
        catalog = catalog_repo.load_roadmap_catalog(self.session)

        # 1. Embed the phrases and match them to roadmap concepts (exact threshold scan).
        phrase_vectors = embedder_for(model).embed_query([p.text for p in phrases])
        matches = [
            PhraseConceptMatch(phrases[m.phrase_index], m.concept_id, catalog.concept_roles[m.concept_id], m.similarity)
            for m in vectors.concepts_above_threshold(self.session, model, phrase_vectors, threshold)
        ]

        # 2. Score roles (pure).
        role_scores = scoring.score_roles(matches, catalog.concept_roles, catalog.concepts_per_role)
        if not role_scores:
            return []

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

        results = []
        for rs in role_scores:
            role = RoleResult(rs.role_id, catalog.role_names[rs.role_id], rs.score)
            role.next_to_learn = [catalog.node_names[c] for c in targets[rs.role_id][:NEXT_CONCEPTS]]
            for p in picks[rs.role_id]:
                c = courses[p.course_id]
                role.courses.append(
                    CourseResult(c.id, c.title, c.url, p.similarity, [catalog.node_names[i] for i in p.concept_ids])
                )
            self._explain(role, matches, courses, catalog)
            results.append(role)
        return results

    def _explain(self, role: RoleResult, matches, courses, catalog) -> None:
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
        context = RoleContext(
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
        explanation = self.explainer.explain(context)
        role.explanation = explanation.role_explanation
        role.prompt_version = explanation.prompt_version
        for c in role.courses:
            c.explanation = explanation.course_explanations.get(c.course_id)

    def _save(self, model_id: int, threshold: float, user_input, result: RecommendationResult) -> None:
        self.session.add(
            RecommendationRequest(
                id=result.request_id,
                model_id=model_id,
                algorithm_version=ALGORITHM_VERSION,
                threshold_used=threshold,
                input={category.value: texts for category, texts in user_input.items()},
                status=result.status,
                latency_ms=result.latency_ms,
            )
        )
        self.session.flush()  # the request row must exist before its roles (foreign key)
        for rank, role in enumerate(result.roles, start=1):
            self.session.add(
                RecommendedRole(
                    request_id=result.request_id,
                    rank=rank,
                    role_id=role.role_id,
                    score=role.score,
                    explanation=role.explanation,
                    prompt_version=role.prompt_version,
                )
            )
        self.session.flush()
        for role in result.roles:
            for rank, c in enumerate(role.courses, start=1):
                self.session.add(
                    RecommendedCourse(
                        request_id=result.request_id,
                        role_id=role.role_id,
                        rank=rank,
                        course_id=c.course_id,
                        similarity=c.similarity,
                        explanation=c.explanation,
                    )
                )
        self.session.commit()
