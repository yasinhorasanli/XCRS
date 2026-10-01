"""ORM models. The source of truth for the schema is docs/schema.md and the Alembic migrations."""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import ClassVar

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    REAL,
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    Numeric,
    SmallInteger,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict] = {datetime: DateTime(timezone=True), float: REAL}


def created_at() -> Mapped[datetime]:
    return mapped_column(server_default=func.now())


# --- Catalog -------------------------------------------------------------------------------------


class Role(Base):
    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    slug: Mapped[str] = mapped_column(Text, unique=True)
    name: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at()


class RoadmapNode(Base):
    __tablename__ = "roadmap_nodes"
    __table_args__ = (
        CheckConstraint("type IN ('topic', 'concept')", name="roadmap_nodes_type_check"),
        Index("roadmap_nodes_role_sequence_idx", "role_id", "sequence"),
        Index("roadmap_nodes_parent_idx", "parent_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    role_id: Mapped[int] = mapped_column(SmallInteger, ForeignKey("roles.id"))
    parent_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("roadmap_nodes.id"))
    type: Mapped[str] = mapped_column(Text)
    name: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    position: Mapped[int]
    sequence: Mapped[int]
    legacy_id: Mapped[int | None] = mapped_column(BigInteger, unique=True)
    source: Mapped[str] = mapped_column(Text)
    source_version: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at()
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    parent: Mapped["RoadmapNode | None"] = relationship(remote_side=[id])


class Course(Base):
    __tablename__ = "courses"
    __table_args__ = (UniqueConstraint("source", "source_id"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    source: Mapped[str] = mapped_column(Text)
    source_id: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    headline: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    what_you_learn: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(Text)
    is_paid: Mapped[bool | None]
    price: Mapped[Decimal | None] = mapped_column(Numeric)
    rating: Mapped[float | None]
    embed_text: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(server_default="true")
    fetched_at: Mapped[datetime | None]
    created_at: Mapped[datetime] = created_at()
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


# --- Vectors -------------------------------------------------------------------------------------


class EmbeddingModel(Base):
    __tablename__ = "embedding_models"
    __table_args__ = (
        CheckConstraint("dimensions > 0", name="embedding_models_dimensions_check"),
        CheckConstraint("status IN ('candidate', 'active', 'retired')", name="embedding_models_status_check"),
    )

    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True)
    runtime: Mapped[str] = mapped_column(Text)
    quantization: Mapped[str] = mapped_column(Text)
    dimensions: Mapped[int]
    query_prefix: Mapped[str | None] = mapped_column(Text)
    document_prefix: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    sim_mean: Mapped[float | None]
    sim_std: Mapped[float | None]
    stats_computed_at: Mapped[datetime | None]
    created_at: Mapped[datetime] = created_at()


class CourseEmbedding(Base):
    __tablename__ = "course_embeddings"

    course_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("courses.id", ondelete="CASCADE"), primary_key=True)
    model_id: Mapped[int] = mapped_column(SmallInteger, ForeignKey("embedding_models.id"), primary_key=True)
    embedding = mapped_column(Vector(), nullable=False)  # untyped: size depends on the model (ADR-0009)
    content_hash: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at()


class NodeEmbedding(Base):
    __tablename__ = "node_embeddings"

    node_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("roadmap_nodes.id", ondelete="CASCADE"), primary_key=True
    )
    model_id: Mapped[int] = mapped_column(SmallInteger, ForeignKey("embedding_models.id"), primary_key=True)
    embedding = mapped_column(Vector(), nullable=False)
    content_hash: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at()


class ConceptCourseMatch(Base):
    __tablename__ = "concept_course_matches"
    __table_args__ = (
        UniqueConstraint("model_id", "concept_id", "rank"),
        CheckConstraint("rank BETWEEN 1 AND 20", name="concept_course_matches_rank_check"),
        Index("concept_course_matches_course_idx", "model_id", "course_id"),
    )

    model_id: Mapped[int] = mapped_column(SmallInteger, ForeignKey("embedding_models.id"), primary_key=True)
    concept_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("roadmap_nodes.id", ondelete="CASCADE"), primary_key=True
    )
    course_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("courses.id", ondelete="CASCADE"), primary_key=True)
    similarity: Mapped[float]
    rank: Mapped[int] = mapped_column(SmallInteger)


# --- Activity ------------------------------------------------------------------------------------


class RecommendationRequest(Base):
    __tablename__ = "recommendation_requests"
    __table_args__ = (
        CheckConstraint("status IN ('ok', 'insufficient_input', 'error')", name="recommendation_requests_status_check"),
        Index("recommendation_requests_created_idx", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, server_default=func.gen_random_uuid())
    created_at: Mapped[datetime] = created_at()
    model_id: Mapped[int] = mapped_column(SmallInteger, ForeignKey("embedding_models.id"))
    algorithm_version: Mapped[str] = mapped_column(Text)
    threshold_used: Mapped[float]
    input: Mapped[dict] = mapped_column(JSONB)  # transitional (ADR-0013)
    status: Mapped[str] = mapped_column(Text)
    latency_ms: Mapped[int | None]
    error: Mapped[str | None] = mapped_column(Text)


class RecommendedRole(Base):
    """A shown role, and its explanation job (ADR-0018): the rows are the queue."""

    __tablename__ = "recommended_roles"
    __table_args__ = (
        UniqueConstraint("request_id", "role_id"),
        CheckConstraint(
            "explanation_status IN ('pending', 'done', 'failed', 'disabled')",
            name="recommended_roles_explanation_status_check",
        ),
        Index("recommended_roles_pending_idx", "request_id", postgresql_where=text("explanation_status = 'pending'")),
    )

    request_id: Mapped[uuid.UUID] = mapped_column(
        UUID, ForeignKey("recommendation_requests.id", ondelete="CASCADE"), primary_key=True
    )
    rank: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    role_id: Mapped[int] = mapped_column(SmallInteger, ForeignKey("roles.id"))
    score: Mapped[float]
    explanation: Mapped[str | None] = mapped_column(Text)
    prompt_version: Mapped[str | None] = mapped_column(Text)  # covers the role's course explanations too
    explanation_status: Mapped[str] = mapped_column(Text, server_default="pending")
    explanation_input: Mapped[dict | None] = mapped_column(JSONB)  # exactly what the LLM gets
    explanation_ms: Mapped[int | None]
    explanation_attempts: Mapped[int] = mapped_column(SmallInteger, server_default="0")
    explained_at: Mapped[datetime | None]
    next_concept_ids: Mapped[list[int]] = mapped_column(ARRAY(BigInteger), server_default="{}")


class RecommendedCourse(Base):
    __tablename__ = "recommended_courses"
    __table_args__ = (
        ForeignKeyConstraint(
            ["request_id", "role_id"],
            ["recommended_roles.request_id", "recommended_roles.role_id"],
            ondelete="CASCADE",
        ),
    )

    request_id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True)
    role_id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    rank: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    course_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("courses.id"))
    similarity: Mapped[float]
    explanation: Mapped[str | None] = mapped_column(Text)
    concept_ids: Mapped[list[int]] = mapped_column(ARRAY(BigInteger), server_default="{}")  # picked for these


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    request_id: Mapped[uuid.UUID] = mapped_column(UUID, ForeignKey("recommendation_requests.id", ondelete="CASCADE"))
    role_id: Mapped[int | None] = mapped_column(SmallInteger, ForeignKey("roles.id"))
    course_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("courses.id"))
    rating: Mapped[int | None] = mapped_column(SmallInteger)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at()


# --- Catalog v2 (schema "catalog", ADR-0028; loaded from catalog/*.yaml by `xcrs catalog import`) -------

CATALOG = "catalog"


class CatalogImport(Base):
    __tablename__ = "imports"
    __table_args__ = ({"schema": CATALOG},)

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    git_commit: Mapped[str | None] = mapped_column(Text)
    git_dirty: Mapped[bool]
    checksum: Mapped[str] = mapped_column(Text)
    stats: Mapped[dict] = mapped_column(JSONB)
    changes: Mapped[dict] = mapped_column(JSONB)
    imported_at: Mapped[datetime] = created_at()


class Level(Base):
    __tablename__ = "levels"
    __table_args__ = ({"schema": CATALOG},)

    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, autoincrement=False)  # position on the ladder
    slug: Mapped[str] = mapped_column(Text, unique=True)
    name: Mapped[str] = mapped_column(Text)
    typical_years: Mapped[str] = mapped_column(Text)
    scope: Mapped[str] = mapped_column(Text)


class Family(Base):
    __tablename__ = "families"
    __table_args__ = ({"schema": CATALOG},)

    id: Mapped[int] = mapped_column(SmallInteger, Identity(always=True), primary_key=True)
    slug: Mapped[str] = mapped_column(Text, unique=True)
    name: Mapped[str] = mapped_column(Text)


class Skill(Base):
    __tablename__ = "skills"
    __table_args__ = ({"schema": CATALOG},)

    id: Mapped[int] = mapped_column(Identity(always=True), primary_key=True)
    slug: Mapped[str] = mapped_column(Text, unique=True)
    name: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    onet: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default="{}")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())


class SkillPrerequisite(Base):
    """One option of one AND-group: rows with the same (skill_id, group_no) are alternatives."""

    __tablename__ = "skill_prerequisites"
    __table_args__ = (Index("skill_prerequisites_option_idx", "option_skill_id"), {"schema": CATALOG})

    skill_id: Mapped[int] = mapped_column(ForeignKey("catalog.skills.id", ondelete="CASCADE"), primary_key=True)
    group_no: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    option_skill_id: Mapped[int] = mapped_column(ForeignKey("catalog.skills.id", ondelete="CASCADE"), primary_key=True)
    min_level: Mapped[int] = mapped_column(SmallInteger)


class CareerRole(Base):
    __tablename__ = "roles"
    __table_args__ = ({"schema": CATALOG},)

    id: Mapped[int] = mapped_column(SmallInteger, Identity(always=True), primary_key=True)
    slug: Mapped[str] = mapped_column(Text, unique=True)
    name: Mapped[str] = mapped_column(Text)
    family_id: Mapped[int] = mapped_column(SmallInteger, ForeignKey("catalog.families.id"))
    summary: Mapped[str] = mapped_column(Text)
    onet_code: Mapped[str] = mapped_column(Text)
    esco_uri: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())


class RoleTitle(Base):
    __tablename__ = "role_titles"
    __table_args__ = (Index("role_titles_role_idx", "role_id"), {"schema": CATALOG})

    id: Mapped[int] = mapped_column(Identity(always=True), primary_key=True)
    role_id: Mapped[int] = mapped_column(SmallInteger, ForeignKey("catalog.roles.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(Text, unique=True)


class RoleTitleSkill(Base):
    __tablename__ = "role_title_skills"
    __table_args__ = ({"schema": CATALOG},)

    title_id: Mapped[int] = mapped_column(ForeignKey("catalog.role_titles.id", ondelete="CASCADE"), primary_key=True)
    item_no: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    option_skill_id: Mapped[int] = mapped_column(ForeignKey("catalog.skills.id", ondelete="CASCADE"), primary_key=True)
    min_level: Mapped[int] = mapped_column(SmallInteger)


class CareerRoleLevel(Base):
    __tablename__ = "role_levels"
    __table_args__ = ({"schema": CATALOG},)

    role_id: Mapped[int] = mapped_column(
        SmallInteger, ForeignKey("catalog.roles.id", ondelete="CASCADE"), primary_key=True
    )
    level_id: Mapped[int] = mapped_column(SmallInteger, ForeignKey("catalog.levels.id"), primary_key=True)
    title: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text)


class RoadmapStage(Base):
    __tablename__ = "roadmap_stages"
    __table_args__ = (
        UniqueConstraint("role_id", "level_id", "position"),
        ForeignKeyConstraint(
            ["role_id", "level_id"],
            ["catalog.role_levels.role_id", "catalog.role_levels.level_id"],
            ondelete="CASCADE",
        ),
        {"schema": CATALOG},
    )

    id: Mapped[int] = mapped_column(Identity(always=True), primary_key=True)
    role_id: Mapped[int] = mapped_column(SmallInteger)
    level_id: Mapped[int] = mapped_column(SmallInteger)
    position: Mapped[int] = mapped_column(SmallInteger)
    name: Mapped[str] = mapped_column(Text)
    optional: Mapped[bool] = mapped_column(server_default="false")


class RoadmapItem(Base):
    """One option of one roadmap requirement: rows with the same (stage_id, item_no) are alternatives."""

    __tablename__ = "roadmap_items"
    __table_args__ = (Index("roadmap_items_skill_idx", "option_skill_id"), {"schema": CATALOG})

    stage_id: Mapped[int] = mapped_column(ForeignKey("catalog.roadmap_stages.id", ondelete="CASCADE"), primary_key=True)
    item_no: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    option_skill_id: Mapped[int] = mapped_column(ForeignKey("catalog.skills.id", ondelete="CASCADE"), primary_key=True)
    min_level: Mapped[int] = mapped_column(SmallInteger)


class CommonPath(Base):
    __tablename__ = "common_paths"
    __table_args__ = (
        UniqueConstraint("from_role_id", "from_level_id", "to_role_id", "to_level_id"),
        ForeignKeyConstraint(
            ["from_role_id", "from_level_id"],
            ["catalog.role_levels.role_id", "catalog.role_levels.level_id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["to_role_id", "to_level_id"],
            ["catalog.role_levels.role_id", "catalog.role_levels.level_id"],
            ondelete="CASCADE",
        ),
        {"schema": CATALOG},
    )

    id: Mapped[int] = mapped_column(Identity(always=True), primary_key=True)
    from_role_id: Mapped[int] = mapped_column(SmallInteger)
    from_level_id: Mapped[int] = mapped_column(SmallInteger)
    to_role_id: Mapped[int] = mapped_column(SmallInteger)
    to_level_id: Mapped[int] = mapped_column(SmallInteger)
    kind: Mapped[str] = mapped_column(Text)
    typical_years: Mapped[str | None] = mapped_column(Text)


class LegacyRoleMap(Base):
    __tablename__ = "legacy_roles"
    __table_args__ = ({"schema": CATALOG},)

    legacy_slug: Mapped[str] = mapped_column(Text, primary_key=True)
    role_id: Mapped[int | None] = mapped_column(SmallInteger, ForeignKey("catalog.roles.id", ondelete="SET NULL"))


class SkillEmbedding(Base):
    """A skill's vector for one model (ADR-0008, ADR-0030), embedded from "name: description"."""

    __tablename__ = "skill_embeddings"
    __table_args__ = ({"schema": CATALOG},)

    skill_id: Mapped[int] = mapped_column(ForeignKey("catalog.skills.id", ondelete="CASCADE"), primary_key=True)
    model_id: Mapped[int] = mapped_column(SmallInteger, ForeignKey("embedding_models.id"), primary_key=True)
    embedding = mapped_column(Vector(), nullable=False)  # untyped: size depends on the model
    content_hash: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class PhraseMatch(Base):
    """What the LLM step decided for a normalized phrase (ADR-0030); derived data, safe to empty."""

    __tablename__ = "phrase_matches"
    __table_args__ = ({"schema": CATALOG},)

    phrase_key: Mapped[str] = mapped_column(Text, primary_key=True)
    catalog_checksum: Mapped[str] = mapped_column(Text, primary_key=True)
    prompt_version: Mapped[str] = mapped_column(Text, primary_key=True)
    llm_model: Mapped[str] = mapped_column(Text, primary_key=True)
    skills: Mapped[list[str]] = mapped_column(ARRAY(Text))  # kept after confirmation
    picked: Mapped[list[str]] = mapped_column(ARRAY(Text))  # what the LLM answered, for auditing
    llm_ms: Mapped[int | None]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
