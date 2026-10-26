"""Feedback is one vote per role (security audit 2026-10-26): every POST used to add a row, so one client could
grow feedback_v2 without bound. Duplicates are removed (the newest vote per result and role is kept), and a unique
index makes later votes replace earlier ones. It also serves lookups by result, so the old index goes.

Revision ID: 0017
Revises: 0016
Create Date: 2026-10-26
"""

from alembic import op

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        DELETE FROM feedback_v2 f
         USING feedback_v2 newer
         WHERE newer.recommendation_id = f.recommendation_id
           AND coalesce(newer.role, '') = coalesce(f.role, '')
           AND newer.id > f.id
    """)
    op.execute(
        "CREATE UNIQUE INDEX feedback_v2_one_per_role_idx ON feedback_v2 (recommendation_id, coalesce(role, ''))"
    )
    op.execute("DROP INDEX feedback_v2_recommendation_idx")


def downgrade() -> None:
    # The removed duplicate votes don't come back.
    op.execute("CREATE INDEX feedback_v2_recommendation_idx ON feedback_v2 (recommendation_id)")
    op.execute("DROP INDEX feedback_v2_one_per_role_idx")
