"""Explanations for engine v2 roles (ADR-0018, ADR-0019, ADR-0037): one row per recommended role is the job.

Like recommended_roles for v1: the rows are the queue (a restart re-queues `pending` ones), `input` holds
exactly the facts the LLM gets, and the result is stored with the prompt version and timing.

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-02
"""

from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE explanations_v2 (
            recommendation_id uuid NOT NULL REFERENCES recommendations_v2 (id) ON DELETE CASCADE,
            role              text NOT NULL,
            rank              smallint NOT NULL,
            status            text NOT NULL DEFAULT 'pending' CONSTRAINT explanations_v2_status_check
                              CHECK (status IN ('pending', 'done', 'failed', 'disabled')),
            input             jsonb NOT NULL,
            explanation       text,
            next_step         text,
            prompt_version    text,
            attempts          smallint NOT NULL DEFAULT 0,
            ms                integer,
            created_at        timestamptz NOT NULL DEFAULT now(),
            explained_at      timestamptz,
            PRIMARY KEY (recommendation_id, role)
        )
    """)
    op.execute("CREATE INDEX explanations_v2_pending_idx ON explanations_v2 (created_at) WHERE status = 'pending'")


def downgrade() -> None:
    op.execute("DROP TABLE explanations_v2")
