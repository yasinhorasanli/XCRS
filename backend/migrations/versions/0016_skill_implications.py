"""What knowing a skill shows (ADR-0028 note, 2026-10-12): catalog.skill_implications, from `implies` in
skills.yaml. Java known at working level shows OOP at working level; unlike a prerequisite, OOP is learned
through Java, not before it. One implied skill per row (no choices). Replaced on every catalog import.

Revision ID: 0016
Revises: 0015
Create Date: 2026-10-12
"""

from alembic import op

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE catalog.skill_implications (
            skill_id         integer NOT NULL REFERENCES catalog.skills (id) ON DELETE CASCADE,
            implied_skill_id integer NOT NULL REFERENCES catalog.skills (id) ON DELETE CASCADE,
            max_level        smallint NOT NULL
                             CONSTRAINT skill_implications_max_level_check CHECK (max_level BETWEEN 1 AND 4),
            PRIMARY KEY (skill_id, implied_skill_id),
            CONSTRAINT skill_implications_not_self CHECK (skill_id <> implied_skill_id)
        )
    """)
    op.execute("CREATE INDEX skill_implications_implied_idx ON catalog.skill_implications (implied_skill_id)")


def downgrade() -> None:
    op.execute("DROP TABLE catalog.skill_implications")
