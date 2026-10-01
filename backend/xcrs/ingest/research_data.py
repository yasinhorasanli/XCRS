"""Import of the research dataset (2024) into PostgreSQL: the seed catalog of roles, roadmaps and courses.

Idempotent: re-running updates existing rows instead of duplicating them.

Sources (data/research-2024/, see its README):
  - roadmap_nodes.csv: 1,104 nodes with digit-encoded ids (role 6 → topic 602 → concept 60203).
    Rows are in depth-first roadmap order, which the research prototype relied on as learning order
    (docs/schema.md, note 2), so the row order becomes `sequence`.
  - udemy_courses.csv: 453 courses. Paid prices are in Turkish lira (e.g. "₺299.99").
"""

import csv
import hashlib
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from xcrs.config import REPO_ROOT
from xcrs.db.models import Course, RoadmapNode, Role

DATA_DIR = REPO_ROOT / "data" / "research-2024"

# Research-prototype role ids 1–10 and roadmap.sh file names (data/research-2024/roadmaps/).
ROLES = [
    (1, "ai-data-scientist", "AI Data Scientist"),
    (2, "android", "Android Developer"),
    (3, "backend", "Backend Developer"),
    (4, "blockchain", "Blockchain Developer"),
    (5, "devops", "DevOps Engineer"),
    (6, "frontend", "Frontend Developer"),
    (7, "full-stack", "Full Stack Developer"),
    (8, "game-developer", "Game Developer"),
    (9, "qa", "QA Engineer"),
    (10, "ux-design", "UX Designer"),
]

# The research prototype always excluded this course (SAP Overview) because it was recommended where
# irrelevant (docs/schema.md, note 1). Kept inactive for parity with it.
PROTOTYPE_EXCLUDED_COURSES = {"2602800"}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def legacy_role_id(node_id: int) -> int:
    while node_id >= 100:
        node_id //= 100
    return node_id


def legacy_parent_id(node_id: int) -> int | None:
    parent = node_id // 100
    return parent if parent >= 100 else None


def import_roles(session: Session) -> None:
    stmt = insert(Role).values([{"id": i, "slug": slug, "name": name} for i, slug, name in ROLES])
    session.execute(
        stmt.on_conflict_do_update(index_elements=["id"], set_={"slug": stmt.excluded.slug, "name": stmt.excluded.name})
    )


def import_roadmap_nodes(session: Session) -> int:
    with open(DATA_DIR / "roadmap_nodes.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    sequence_by_role: dict[int, int] = {}
    position_by_parent: dict[int, int] = {}
    new_id_by_legacy: dict[int, int] = {}

    # Parents always precede their children in depth-first order, so one pass resolves parent ids.
    for row in rows:
        legacy_id = int(row["id"])
        role_id = legacy_role_id(legacy_id)
        legacy_parent = legacy_parent_id(legacy_id)
        parent_key = legacy_parent if legacy_parent is not None else -role_id

        sequence_by_role[role_id] = sequence_by_role.get(role_id, 0) + 1
        position_by_parent[parent_key] = position_by_parent.get(parent_key, 0) + 1

        values = {
            "role_id": role_id,
            "parent_id": new_id_by_legacy[legacy_parent] if legacy_parent is not None else None,
            "type": row["type"],
            "name": row["name"],
            "content": row["content"],
            "position": position_by_parent[parent_key],
            "sequence": sequence_by_role[role_id],
            "legacy_id": legacy_id,
            "source": "roadmap.sh",
            "source_version": "research-2024",
            "content_hash": sha256(row["content"]),
        }
        stmt = insert(RoadmapNode).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["legacy_id"], set_={k: stmt.excluded[k] for k in values if k != "legacy_id"}
        ).returning(RoadmapNode.id)
        new_id_by_legacy[legacy_id] = session.execute(stmt).scalar_one()

    return len(rows)


def parse_price(price: str) -> Decimal | None:
    if price.strip().lower() == "free":
        return Decimal(0)
    digits = price.strip().lstrip("₺").replace(",", "")
    return Decimal(digits) if digits else None


def import_courses(session: Session) -> int:
    with open(DATA_DIR / "udemy_courses.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        values = {
            "source": "udemy",
            "source_id": row["id"],
            "title": row["title"],
            "url": "https://www.udemy.com" + row["url"],
            "headline": row["headline"] or None,
            "description": row["description"] or None,
            "what_you_learn": row["what_u_learn"] or None,
            "category": row["category"] or None,
            "language": row["language"] or None,
            "is_paid": row["is_paid"] == "True",
            "price": parse_price(row["price"]),
            "rating": float(row["rating"]) if row["rating"] else None,
            "embed_text": row["concat_text"],
            "content_hash": sha256(row["concat_text"]),
            "is_active": row["id"] not in PROTOTYPE_EXCLUDED_COURSES,
        }
        stmt = insert(Course).values(values)
        session.execute(
            stmt.on_conflict_do_update(
                index_elements=["source", "source_id"],
                set_={k: stmt.excluded[k] for k in values if k not in ("source", "source_id")},
            )
        )
    return len(rows)


def run(session: Session) -> dict[str, int]:
    import_roles(session)
    nodes = import_roadmap_nodes(session)
    courses = import_courses(session)
    session.commit()
    return {
        "roles": len(session.scalars(select(Role)).all()),
        "roadmap_nodes": nodes,
        "courses": courses,
    }
