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
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict] = {datetime: DateTime(timezone=True), float: REAL}


def created_at() -> Mapped[datetime]:
    return mapped_column(server_default=func.now())


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


# --- Accounts (ADR-0043) ----------------------------------------------------------------------------


class User(Base):
    """A signed-in learner. `email` only when a provider verified it (lower-case, unique: links providers)."""

    __tablename__ = "users"
    __table_args__ = (CheckConstraint("email = lower(email)", name="users_email_lower_check"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, server_default=text("gen_random_uuid()"))
    display_name: Mapped[str | None] = mapped_column(Text)
    email: Mapped[str | None] = mapped_column(Text, unique=True)
    created_at: Mapped[datetime] = created_at()
    last_sign_in_at: Mapped[datetime] = mapped_column(server_default=func.now())
    sessions_valid_after: Mapped[datetime | None]  # "sign out everywhere": sessions from before it are cut off


class UserIdentity(Base):
    """One provider account (GitHub, Google, LinkedIn) of a user; `subject` is the provider's stable id."""

    __tablename__ = "user_identities"
    __table_args__ = (
        CheckConstraint("provider IN ('github', 'google', 'linkedin')", name="user_identities_provider_check"),
        Index("user_identities_user_idx", "user_id"),
    )

    provider: Mapped[str] = mapped_column(Text, primary_key=True)
    subject: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID, ForeignKey("users.id", ondelete="CASCADE"))
    email: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at()
    last_sign_in_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Board(Base):
    """A user's saved board: the chips and experience, in the recommendation request's shape."""

    __tablename__ = "boards"

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID, ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    input: Mapped[dict] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())


# --- Engine v2 activity (ADR-0029, ADR-0031): JSONB input and result, as ADR-0013 ---------------------


class RecommendationV2(Base):
    __tablename__ = "recommendations_v2"
    __table_args__ = (
        CheckConstraint("status IN ('ok', 'insufficient_input')", name="recommendations_v2_status_check"),
        Index("recommendations_v2_created_idx", "created_at"),
        Index("recommendations_v2_user_idx", "user_id", "created_at", postgresql_where=text("user_id IS NOT NULL")),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, server_default=text("gen_random_uuid()"))
    created_at: Mapped[datetime] = created_at()
    catalog_checksum: Mapped[str] = mapped_column(Text)
    algorithm_version: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    input: Mapped[dict] = mapped_column(JSONB)
    result: Mapped[dict] = mapped_column(JSONB)
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID, ForeignKey("users.id", ondelete="CASCADE"))  # ADR-0043


class FeedbackV2(Base):
    __tablename__ = "feedback_v2"
    __table_args__ = (Index("feedback_v2_recommendation_idx", "recommendation_id"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    recommendation_id: Mapped[uuid.UUID] = mapped_column(UUID, ForeignKey("recommendations_v2.id", ondelete="CASCADE"))
    role: Mapped[str | None] = mapped_column(Text)
    resource_id: Mapped[int | None] = mapped_column(BigInteger)
    rating: Mapped[int | None] = mapped_column(SmallInteger)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at()


# --- Learning resources (ADR-0026, ADR-0032, ADR-0033) ------------------------------------------------

INGEST = "ingest"


class IngestRun(Base):
    __tablename__ = "runs"
    __table_args__ = ({"schema": INGEST},)

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    source: Mapped[str] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    finished_at: Mapped[datetime | None]
    status: Mapped[str] = mapped_column(Text, server_default="running")
    stats: Mapped[dict] = mapped_column(JSONB, server_default="{}")
    error: Mapped[str | None] = mapped_column(Text)


class RawRecord(Base):
    __tablename__ = "raw_records"
    __table_args__ = (
        UniqueConstraint("source", "external_id", "content_hash"),
        Index("raw_records_latest_idx", "source", "external_id", text("fetched_at DESC")),
        {"schema": INGEST},
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    source: Mapped[str] = mapped_column(Text)
    external_id: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSONB)
    run_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("ingest.runs.id", ondelete="SET NULL"))
    fetched_at: Mapped[datetime] = mapped_column(server_default=func.now())


class LearningResource(Base):
    __tablename__ = "learning_resources"
    __table_args__ = (UniqueConstraint("source", "external_id"), {"schema": CATALOG})

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    source: Mapped[str] = mapped_column(Text)
    external_id: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(Text)
    provider: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text, unique=True)
    title: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str] = mapped_column(Text, server_default="en")
    level: Mapped[str | None] = mapped_column(Text)
    duration_minutes: Mapped[int | None]
    is_free: Mapped[bool]
    price: Mapped[Decimal | None] = mapped_column(Numeric)
    currency: Mapped[str | None] = mapped_column(Text)
    quality: Mapped[dict] = mapped_column(JSONB, server_default="{}")
    fetched_at: Mapped[datetime] = mapped_column(server_default=func.now())
    last_checked_at: Mapped[datetime | None]
    tagged_at: Mapped[datetime | None]  # when `xcrs resources tag` last processed it (migration 0011)
    last_status: Mapped[int | None] = mapped_column(SmallInteger)
    is_active: Mapped[bool] = mapped_column(server_default="true")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ResourceSkill(Base):
    __tablename__ = "resource_skills"
    __table_args__ = (Index("resource_skills_skill_idx", "skill_id", "relation"), {"schema": CATALOG})

    resource_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("catalog.learning_resources.id", ondelete="CASCADE"), primary_key=True
    )
    skill_id: Mapped[int] = mapped_column(ForeignKey("catalog.skills.id", ondelete="CASCADE"), primary_key=True)
    relation: Mapped[str] = mapped_column(Text, primary_key=True, server_default="teaches")
    level: Mapped[int | None] = mapped_column(SmallInteger)
    confidence: Mapped[float | None]
    tagged_by: Mapped[str] = mapped_column(Text)


class ExplanationV2(Base):
    """An engine v2 role's explanation job and result (ADR-0037); the rows are the queue (ADR-0018)."""

    __tablename__ = "explanations_v2"
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'done', 'failed', 'disabled')", name="explanations_v2_status_check"),
        Index("explanations_v2_pending_idx", "created_at", postgresql_where=text("status = 'pending'")),
    )

    recommendation_id: Mapped[uuid.UUID] = mapped_column(
        UUID, ForeignKey("recommendations_v2.id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[str] = mapped_column(Text, primary_key=True)
    rank: Mapped[int] = mapped_column(SmallInteger)
    status: Mapped[str] = mapped_column(Text, server_default="pending")
    input: Mapped[dict] = mapped_column(JSONB)
    explanation: Mapped[str | None] = mapped_column(Text)
    next_step: Mapped[str | None] = mapped_column(Text)
    prompt_version: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(SmallInteger, server_default="0")
    ms: Mapped[int | None]
    created_at: Mapped[datetime] = created_at()
    explained_at: Mapped[datetime | None]
