"""When the LLM last tagged a resource (ADR-0033), so `xcrs resources tag` processes each resource once.

Before this, "untagged" meant "has no LLM tag": a resource whose only tag is the reviewed skill (a YouTube
playlist approved for it, or whose LLM picks matched only that skill) or that matched no skill was processed
again every day. Backfill: resources that already have an LLM tag count as tagged.

Revision ID: 0011
Revises: 0010
Create Date: 2026-10-03
"""

from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE catalog.learning_resources ADD COLUMN tagged_at timestamptz")
    op.execute("""
        UPDATE catalog.learning_resources r SET tagged_at = r.updated_at
        WHERE EXISTS (SELECT 1 FROM catalog.resource_skills rs WHERE rs.resource_id = r.id AND rs.tagged_by = 'llm')
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE catalog.learning_resources DROP COLUMN tagged_at")
