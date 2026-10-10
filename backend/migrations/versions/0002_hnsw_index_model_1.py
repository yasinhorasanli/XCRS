"""HNSW index on course embeddings for model 1 (qwen3-embedding:0.6b, 1024 dims).

One migration per embedding model (ADR-0009). The course-side index serves the ingestion
job's "nearest courses to a concept" search. The node side has no index; the request path
uses an exact threshold scan (ADR-0010).

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-28
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS course_emb_m1_hnsw ON course_embeddings
            USING hnsw ((embedding::vector(1024)) vector_cosine_ops)
            WHERE model_id = 1
        """)


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS course_emb_m1_hnsw")
