"""Sections of a learning resource (ADR-0046): a long video's chapters (timestamps in its description) or a
playlist's videos, each with the skills it covers, so a gap can link to the part that teaches it.

Sections are YouTube data like their resource: replaced when the resource's sections change, deleted with it
(the 30-day rule, ADR-0033). Their skills are a subset of the resource's own, confirmed by embedding similarity.

Revision ID: 0014
Revises: 0013
Create Date: 2026-10-07
"""

from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE catalog.resource_sections (
            id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            resource_id   bigint NOT NULL REFERENCES catalog.learning_resources (id) ON DELETE CASCADE,
            position      smallint NOT NULL,
            title         text NOT NULL,
            url           text NOT NULL,
            start_seconds integer,
            duration_seconds integer,
            tagged_at     timestamptz,
            UNIQUE (resource_id, position)
        )
    """)
    op.execute("""
        CREATE TABLE catalog.resource_section_skills (
            section_id bigint NOT NULL REFERENCES catalog.resource_sections (id) ON DELETE CASCADE,
            skill_id   integer NOT NULL REFERENCES catalog.skills (id) ON DELETE CASCADE,
            confidence real,
            PRIMARY KEY (section_id, skill_id)
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE catalog.resource_section_skills")
    op.execute("DROP TABLE catalog.resource_sections")
