"""Engine v2 recommendations and feedback (ADR-0029, ADR-0031).

Following ADR-0013 (hybrid now, normalize once the format settles): the input (chips with category and
proficiency) and the result (roles, levels, gaps) are stored as JSONB, with the catalog version and the
algorithm version that produced them. Feedback refers to roles by slug, which stays stable across imports.

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-02
"""

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE recommendations_v2 (
            id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            created_at        timestamptz NOT NULL DEFAULT now(),
            catalog_checksum  text NOT NULL,
            algorithm_version text NOT NULL,
            status            text NOT NULL CONSTRAINT recommendations_v2_status_check
                              CHECK (status IN ('ok', 'insufficient_input')),
            input             jsonb NOT NULL,
            result            jsonb NOT NULL
        )
    """)
    op.execute("CREATE INDEX recommendations_v2_created_idx ON recommendations_v2 (created_at)")
    op.execute("""
        CREATE TABLE feedback_v2 (
            id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            recommendation_id uuid NOT NULL REFERENCES recommendations_v2 (id) ON DELETE CASCADE,
            role              text,
            resource_id       bigint,
            rating            smallint CONSTRAINT feedback_v2_rating_check CHECK (rating IN (-1, 1)),
            comment           text,
            created_at        timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX feedback_v2_recommendation_idx ON feedback_v2 (recommendation_id)")


def downgrade() -> None:
    op.execute("DROP TABLE feedback_v2")
    op.execute("DROP TABLE recommendations_v2")
