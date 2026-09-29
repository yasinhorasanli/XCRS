"""Record which prompt produced each role's explanations (ADR-0019).

Like algorithm_version (ADR-0013): a past explanation stays traceable after the prompt changes.
NULL means no explanation was produced (LLM disabled or failed).

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-30
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE recommended_roles ADD COLUMN prompt_version text")


def downgrade() -> None:
    op.execute("ALTER TABLE recommended_roles DROP COLUMN prompt_version")
