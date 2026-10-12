"""A recommendation can end with no role (ADR-0052): status "no_clear_match" when the board's skills are recognised
but don't single out any role (only dislikes, or only skills most roles use).

Revision ID: 0015
Revises: 0014
Create Date: 2026-10-12
"""

from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE recommendations_v2 DROP CONSTRAINT recommendations_v2_status_check;
        ALTER TABLE recommendations_v2 ADD CONSTRAINT recommendations_v2_status_check
            CHECK (status IN ('ok', 'insufficient_input', 'no_clear_match'));
    """)


def downgrade() -> None:
    # Results without a role go back to the closest old status.
    op.execute("""
        UPDATE recommendations_v2 SET status = 'insufficient_input' WHERE status = 'no_clear_match';
        ALTER TABLE recommendations_v2 DROP CONSTRAINT recommendations_v2_status_check;
        ALTER TABLE recommendations_v2 ADD CONSTRAINT recommendations_v2_status_check
            CHECK (status IN ('ok', 'insufficient_input'));
    """)
