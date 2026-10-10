"""Explanations as background jobs (ADR-0018).

recommended_roles becomes the job queue: explanation_status says where each role's explanation is,
explanation_input holds exactly the facts the LLM gets (so a job can run after a restart, and a
past explanation can be audited), and explanation_ms / explanation_attempts measure and bound it.
The concept id arrays let GET /recommendations/{id} rebuild a result from the database alone.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-30
"""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE recommended_roles
            ADD COLUMN explanation_status text NOT NULL DEFAULT 'pending'
                CONSTRAINT recommended_roles_explanation_status_check
                CHECK (explanation_status IN ('pending', 'done', 'failed', 'disabled')),
            ADD COLUMN explanation_input jsonb,
            ADD COLUMN explanation_ms integer,
            ADD COLUMN explanation_attempts smallint NOT NULL DEFAULT 0,
            ADD COLUMN explained_at timestamptz,
            ADD COLUMN next_concept_ids bigint[] NOT NULL DEFAULT '{}'
    """)
    # Rows from before this migration were explained inline.
    op.execute("""
        UPDATE recommended_roles
        SET explanation_status = CASE WHEN explanation IS NULL THEN 'failed' ELSE 'done' END
    """)
    op.execute("""
        CREATE INDEX recommended_roles_pending_idx ON recommended_roles (request_id)
        WHERE explanation_status = 'pending'
    """)
    op.execute("ALTER TABLE recommended_courses ADD COLUMN concept_ids bigint[] NOT NULL DEFAULT '{}'")


def downgrade() -> None:
    op.execute("ALTER TABLE recommended_courses DROP COLUMN concept_ids")
    op.execute("DROP INDEX recommended_roles_pending_idx")
    op.execute("""
        ALTER TABLE recommended_roles
            DROP COLUMN next_concept_ids,
            DROP COLUMN explained_at,
            DROP COLUMN explanation_attempts,
            DROP COLUMN explanation_ms,
            DROP COLUMN explanation_input,
            DROP COLUMN explanation_status
    """)
