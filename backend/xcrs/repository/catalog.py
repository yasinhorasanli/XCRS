"""Read-only views of the catalog that the domain functions need."""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from xcrs.db.models import Course, EmbeddingModel, RoadmapNode, Role


@dataclass(frozen=True)
class RoadmapCatalog:
    role_names: dict[int, str]
    node_names: dict[int, str]
    concept_roles: dict[int, int]
    concepts_per_role: dict[int, int]  # ordered by role id
    role_concepts_in_order: dict[int, list[int]]  # roadmap (learning) order
    concept_ancestors: dict[int, list[int]]  # topic ids, nearest first
    concepts_per_topic: dict[int, int]


def load_roadmap_catalog(session: Session) -> RoadmapCatalog:
    """The whole roadmap tree (~1K rows today; loaded per request, a few ms)."""
    role_names = dict(session.execute(select(Role.id, Role.name).order_by(Role.id)).all())
    nodes = session.execute(
        select(RoadmapNode.id, RoadmapNode.role_id, RoadmapNode.parent_id, RoadmapNode.type, RoadmapNode.name).order_by(
            RoadmapNode.role_id, RoadmapNode.sequence
        )
    ).all()

    parent = {n.id: n.parent_id for n in nodes}
    concepts = [n for n in nodes if n.type == "concept"]

    def ancestors(node_id: int) -> list[int]:
        chain, p = [], parent[node_id]
        while p is not None:
            chain.append(p)
            p = parent[p]
        return chain

    concept_ancestors = {c.id: ancestors(c.id) for c in concepts}
    concepts_per_topic: dict[int, int] = {}
    for chain in concept_ancestors.values():
        for topic_id in chain:
            concepts_per_topic[topic_id] = concepts_per_topic.get(topic_id, 0) + 1

    role_concepts_in_order: dict[int, list[int]] = {role_id: [] for role_id in role_names}
    for c in concepts:
        role_concepts_in_order[c.role_id].append(c.id)

    return RoadmapCatalog(
        role_names=role_names,
        node_names={n.id: n.name for n in nodes},
        concept_roles={c.id: c.role_id for c in concepts},
        concepts_per_role={r: len(ids) for r, ids in role_concepts_in_order.items() if ids},
        role_concepts_in_order=role_concepts_in_order,
        concept_ancestors=concept_ancestors,
        concepts_per_topic=concepts_per_topic,
    )


def ping(session: Session) -> None:
    session.execute(select(1))


def active_model(session: Session) -> EmbeddingModel:
    model = session.scalars(select(EmbeddingModel).where(EmbeddingModel.status == "active")).one_or_none()
    if model is None or model.sim_mean is None:
        raise RuntimeError("no active embedding model with computed statistics; run `xcrs embed-catalog`")
    return model


def courses_by_id(session: Session, course_ids: list[int]) -> dict[int, Course]:
    return {c.id: c for c in session.scalars(select(Course).where(Course.id.in_(course_ids)))}
