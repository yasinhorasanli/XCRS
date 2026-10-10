"""Accounts (ADR-0043): users, their sign-in identities, a saved board, and the owner of a recommendation.

Emails are stored only when the provider says they are verified, lower-cased, and unique: two providers
reporting the same verified email are one account. `sessions_valid_after` cuts off session cookies issued
before it ("sign out everywhere"; NULL = never). Deleting a user deletes everything linked, including their
results (and, through the existing cascades, the feedback and explanations on them). Anonymous results keep
`user_id` NULL.

Revision ID: 0012
Revises: 0011
Create Date: 2026-10-03
"""

from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE users (
            id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            display_name         text,
            email                text CONSTRAINT users_email_key UNIQUE
                                 CONSTRAINT users_email_lower_check CHECK (email = lower(email)),
            created_at           timestamptz NOT NULL DEFAULT now(),
            last_sign_in_at      timestamptz NOT NULL DEFAULT now(),
            sessions_valid_after timestamptz
        )
    """)
    op.execute("""
        CREATE TABLE user_identities (
            provider        text NOT NULL CONSTRAINT user_identities_provider_check
                            CHECK (provider IN ('github', 'google', 'linkedin')),
            subject         text NOT NULL,
            user_id         uuid NOT NULL REFERENCES users (id) ON DELETE CASCADE,
            email           text,
            created_at      timestamptz NOT NULL DEFAULT now(),
            last_sign_in_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (provider, subject)
        )
    """)
    op.execute("CREATE INDEX user_identities_user_idx ON user_identities (user_id)")
    op.execute("""
        CREATE TABLE boards (
            id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            user_id    uuid NOT NULL CONSTRAINT boards_user_key UNIQUE REFERENCES users (id) ON DELETE CASCADE,
            input      jsonb NOT NULL,
            updated_at timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("ALTER TABLE recommendations_v2 ADD COLUMN user_id uuid REFERENCES users (id) ON DELETE CASCADE")
    op.execute(
        "CREATE INDEX recommendations_v2_user_idx ON recommendations_v2 (user_id, created_at) WHERE user_id IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE recommendations_v2 DROP COLUMN user_id")
    op.execute("DROP TABLE boards")
    op.execute("DROP TABLE user_identities")
    op.execute("DROP TABLE users")
