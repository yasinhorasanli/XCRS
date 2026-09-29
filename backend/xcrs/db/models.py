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
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
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
    __tablename__ = "recommended_roles"
    __table_args__ = (UniqueConstraint("request_id", "role_id"),)

    request_id: Mapped[uuid.UUID] = mapped_column(
        UUID, ForeignKey("recommendation_requests.id", ondelete="CASCADE"), primary_key=True
    )
    rank: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    role_id: Mapped[int] = mapped_column(SmallInteger, ForeignKey("roles.id"))
    score: Mapped[float]
    explanation: Mapped[str | None] = mapped_column(Text)


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


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    request_id: Mapped[uuid.UUID] = mapped_column(UUID, ForeignKey("recommendation_requests.id", ondelete="CASCADE"))
    role_id: Mapped[int | None] = mapped_column(SmallInteger, ForeignKey("roles.id"))
    course_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("courses.id"))
    rating: Mapped[int | None] = mapped_column(SmallInteger)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at()
