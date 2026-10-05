"""Title skills per role (ADR-0044): the languages and frameworks that job ads put in the role's title
("Java Backend Engineer"). Skill slugs, in the order roles.yaml lists them (the first wins a tie); imported by
`xcrs catalog import` like the rest of the role.

Revision ID: 0013
Revises: 0012
Create Date: 2026-10-05
"""

from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE catalog.roles ADD COLUMN title_skills text[] NOT NULL DEFAULT '{}'")


def downgrade() -> None:
    op.execute("ALTER TABLE catalog.roles DROP COLUMN title_skills")
