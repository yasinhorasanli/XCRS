"""Engine v2 recommendations (ADR-0029, ADR-0031): chips → skills → role scores, levels and gaps.

A chip is either a catalog skill (picked) or typed text (matched with ADR-0030), with a board category and
an optional 1-4 proficiency. The result is stored with the catalog and algorithm versions (JSONB, as
ADR-0013), and can be read back by id.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from xcrs.db.models import FeedbackV2, RecommendationV2
from xcrs.domain.role_scoring import Category, Mention, RoleScore, score_roles
from xcrs.repository import catalog_store
from xcrs.services.skill_matching import SkillMatcher

ALGORITHM_VERSION = "v2.2"  # bump when scoring, weights or matching change (results stay comparable)
ROLES_SHOWN = 3
GAPS_SHOWN = 8


@dataclass(frozen=True)
class Chip:
    category: Category
    skill: str | None = None  # a catalog skill id, picked
    text: str | None = None  # typed text, matched to skills
    proficiency: int | None = None


class RecommendationServiceV2:
    def __init__(self, session: Session, matcher: SkillMatcher):
        self.session, self.matcher = session, matcher

    def recommend(self, chips: list[Chip]) -> RecommendationV2:
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
        status = "ok" if mentions else "insufficient_input"
        roles = score_roles(snapshot, mentions)[:ROLES_SHOWN] if mentions else []
        category = {m.skill: m.category for m in mentions}
        result = {"matched": echo, "roles": [self._role(snapshot, r, category) for r in roles]}
        row = RecommendationV2(
            catalog_checksum=catalog_store.last_import_checksum(self.session) or "none",
            algorithm_version=ALGORITHM_VERSION,
            status=status,
            input={
                "chips": [
                    {"category": c.category.value, "skill": c.skill, "text": c.text, "proficiency": c.proficiency}
                    for c in chips
                ]
            },
            result=result,
        )
        self.session.add(row)
        self.session.commit()
        return row

    def get(self, recommendation_id: uuid.UUID) -> RecommendationV2 | None:
        return self.session.get(RecommendationV2, recommendation_id)

    def feedback(self, recommendation_id: uuid.UUID, role: str | None, rating: int | None, comment: str | None):
        self.session.add(FeedbackV2(recommendation_id=recommendation_id, role=role, rating=rating, comment=comment))
        self.session.commit()

    @staticmethod
    def _role(snapshot, r: RoleScore, category: dict[str, Category]) -> dict:
        role = snapshot.roles[r.role]
        names = snapshot.skill_names

        def level(lv: str | None) -> dict | None:
            if lv is None:
                return None
            return {"id": lv, "title": role.titles.get(lv), "coverage": round(r.level_coverage[lv], 3)}

        return {
            "id": role.id,
            "name": role.name,
            "family": role.family,
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
        }
