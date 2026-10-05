"""Engine v2 recommendations (ADR-0029, ADR-0031): chips → skills → role scores, levels and gaps.

A chip is either a catalog skill (picked) or typed text (matched with ADR-0030), with a board category and
an optional 1-4 proficiency. The result is stored with the catalog and algorithm versions (JSONB, as
ADR-0013), and can be read back by id.
"""

import uuid
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from xcrs.db.models import CareerRole, ExplanationV2, FeedbackV2, RecommendationV2
from xcrs.domain.job_titles import market_titles, title_for
from xcrs.domain.role_scoring import Category, Mention, RoleScore, score_roles, suggest_resources
from xcrs.explain.v2 import build_facts
from xcrs.repository import catalog_store
from xcrs.services.skill_matching import SkillMatcher

ALGORITHM_VERSION = "v3.1"
LEVEL_NAMES = {
    "entry": "entry level",
    "mid": "mid level",
    "senior": "senior level",
    "staff": "staff level",
}  # bump when scoring, weights or matching change (results stay comparable)
ROLES_SHOWN = 3
GAPS_SHOWN = 8


@dataclass(frozen=True)
class Chip:
    category: Category
    skill: str | None = None  # a catalog skill id, picked
    text: str | None = None  # typed text, matched to skills
    proficiency: int | None = None


class ExplanationQueueV2(Protocol):
    def submit(self, recommendation_id: uuid.UUID, role: str) -> None: ...


class RecommendationServiceV2:
    def __init__(self, session: Session, matcher: SkillMatcher, explanations: ExplanationQueueV2 | None = None):
        self.session, self.matcher, self.explanations = session, matcher, explanations

    def recommend(
        self, chips: list[Chip], experience: str | None = None, test: bool = False, user_id: uuid.UUID | None = None
    ) -> RecommendationV2:
        """`test` marks results made from the dev test profiles, so activity numbers can leave them out;
        `user_id` is the signed-in owner (ADR-0043), None for anonymous results."""
        snapshot = catalog_store.load_snapshot(self.session)
        typed = [c for c in chips if c.skill is None and c.text]
        results = self.matcher.match([c.text for c in typed]) if typed else []
        matched = {id(c): r for c, r in zip(typed, results, strict=True)}
        mentions: list[Mention] = []
        echo = []
        for c in chips:
            if c.skill is not None:
                skills, method = ([c.skill] if c.skill in snapshot.skill_names else []), "picked"
            else:
                r = matched.get(id(c))
                skills, method = (r.skills, r.method) if r else ([], "none")
            mentions += [Mention(s, c.category, c.proficiency) for s in skills]
            echo.append(
                {
                    "text": c.text or snapshot.skill_names.get(c.skill or "", c.skill),
                    "category": c.category.value,
                    "proficiency": c.proficiency,
                    "method": method,
                    "skills": [{"id": s, "name": snapshot.skill_names[s]} for s in skills],
                }
            )
        # One feeling per skill: when two chips read as the same skill, the later chip wins (the board does the
        # same; this guards other clients).
        mentions = list({m.skill: m for m in mentions}.values())
        status = "ok" if mentions else "insufficient_input"
        roles = score_roles(snapshot, mentions, experience=experience)[:ROLES_SHOWN] if mentions else []
        category = {m.skill: m.category for m in mentions}
        known = {m.skill for m in mentions}
        result = {"matched": echo, "roles": [self._role(snapshot, r, category, known, mentions) for r in roles]}
        row = RecommendationV2(
            catalog_checksum=catalog_store.last_import_checksum(self.session) or "none",
            algorithm_version=ALGORITHM_VERSION,
            status=status,
            input={
                "chips": [
                    {"category": c.category.value, "skill": c.skill, "text": c.text, "proficiency": c.proficiency}
                    for c in chips
                ],
                **({"experience": experience} if experience else {}),
                **({"test": True} if test else {}),
            },
            result=result,
            user_id=user_id,
        )
        self.session.add(row)
        self.session.flush()
        # One explanation job per role (ADR-0037): the facts the LLM may use, stored with the job.
        summaries = dict(
            self.session.execute(
                select(CareerRole.slug, CareerRole.summary).where(
                    CareerRole.slug.in_([r["id"] for r in result["roles"]])
                )
            ).all()
        )
        status = "pending" if self.explanations else "disabled"
        for rank, role in enumerate(result["roles"], 1):
            facts = build_facts(role, result["matched"], summaries.get(role["id"]), LEVEL_NAMES)
            self.session.add(
                ExplanationV2(recommendation_id=row.id, role=role["id"], rank=rank, status=status, input=facts)
            )
        self.session.commit()
        if self.explanations:
            for role in result["roles"]:
                self.explanations.submit(row.id, role["id"])
        return row

    def get(self, recommendation_id: uuid.UUID) -> RecommendationV2 | None:
        return self.session.get(RecommendationV2, recommendation_id)

    def feedback(self, recommendation_id: uuid.UUID, role: str | None, rating: int | None, comment: str | None):
        self.session.add(FeedbackV2(recommendation_id=recommendation_id, role=role, rating=rating, comment=comment))
        self.session.commit()

    @staticmethod
    def _role(snapshot, r: RoleScore, category: dict[str, Category], known: set[str], mentions: list[Mention]) -> dict:
        role = snapshot.roles[r.role]
        relevant = known | {o for q in role.requirements[role.levels[-1]] for o in q.options} | set(role.optional)
        names = snapshot.skill_names

        def level(lv: str | None) -> dict | None:
            if lv is None:
                return None
            return {"id": lv, "title": role.titles.get(lv), "coverage": round(r.level_coverage[lv], 3)}

        return {
            "id": role.id,
            "name": role.name,
            "family": role.family,
            # Titles to search for (ADR-0044): the learner's own first, then the role's market titles.
            "title_for_you": title_for(role, mentions, names, snapshot.languages),
            "job_titles": market_titles(role),
            "score": round(r.score, 4),
            "interest": round(r.interest, 3),
            "coverage": round(r.coverage, 3),
            "level": level(r.level),
            "target_level": level(r.target_level),
            "levels": [level(lv) for lv in role.levels],
            "because": [{"id": s, "name": names[s], "category": category[s].value} for s in r.because[:6]],
            "gaps": [
                {
                    "skills": [{"id": o, "name": names[o]} for o in g.options],
                    "need": g.need,
                    "have": g.have,
                    "stage": g.stage,
                }
                for g in r.gaps[:GAPS_SHOWN]
            ],
            "gaps_total": len(r.gaps),
            "basics": [
                {
                    "skills": [{"id": o, "name": names[o]} for o in g.options],
                    "need": g.need,
                    "have": g.have,
                    "stage": g.stage,
                }
                for g in r.basics
            ],
            "resources": [
                {
                    "id": ref.id,
                    "title": ref.title,
                    "url": ref.url,
                    "provider": ref.provider,
                    "type": ref.type,
                    "level": ref.level,
                    "free": ref.free,
                    "curated": ref.curated,
                    "skills": [{"id": sk, "name": names[sk]} for sk in hits],
                }
                for ref, hits in suggest_resources(snapshot, r.gaps, relevant, known=known)
            ],
        }
