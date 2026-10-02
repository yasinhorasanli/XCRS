"""Learning resources (ADR-0026, ADR-0032, ADR-0033): raw ingested records, normalized resources, skill links.

- ingest.raw_records: what a source returned, as received (JSONB); a new version only when the content
  hash changes. ingest.runs: one row per ingestion run.
- catalog.learning_resources: normalized resources of every type (course, video, playlist, docs,
  tutorial, book), unique per (source, external_id) and per URL.
- catalog.resource_skills: which skills a resource teaches (or requires), at what proficiency, how sure,
  and who said so (curated, llm, reviewed).

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-02
"""

from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA ingest")
    op.execute("""
        CREATE TABLE ingest.runs (
            id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            source      text NOT NULL,
            started_at  timestamptz NOT NULL DEFAULT now(),
            finished_at timestamptz,
            status      text NOT NULL DEFAULT 'running'
                        CONSTRAINT runs_status_check CHECK (status IN ('running', 'ok', 'failed')),
            stats       jsonb NOT NULL DEFAULT '{}',
            error       text
        )
    """)
    op.execute("""
        CREATE TABLE ingest.raw_records (
            id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            source       text NOT NULL,
            external_id  text NOT NULL,
            content_hash text NOT NULL,
            payload      jsonb NOT NULL,
            run_id       bigint REFERENCES ingest.runs (id) ON DELETE SET NULL,
            fetched_at   timestamptz NOT NULL DEFAULT now(),
            UNIQUE (source, external_id, content_hash)
        )
    """)
    op.execute("CREATE INDEX raw_records_latest_idx ON ingest.raw_records (source, external_id, fetched_at DESC)")
    op.execute("""
        CREATE TABLE catalog.learning_resources (
            id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            source           text NOT NULL,
            external_id      text NOT NULL,
            type             text NOT NULL CONSTRAINT learning_resources_type_check
                             CHECK (type IN ('course', 'video', 'playlist', 'docs', 'tutorial', 'book')),
            provider         text NOT NULL,
            url              text NOT NULL UNIQUE,
            title            text NOT NULL,
            description      text,
            language         text NOT NULL DEFAULT 'en',
            level            text CONSTRAINT learning_resources_level_check
                             CHECK (level IN ('beginner', 'intermediate', 'advanced')),
            duration_minutes integer,
            is_free          boolean NOT NULL,
            price            numeric,
            currency         text,
            quality          jsonb NOT NULL DEFAULT '{}',
            fetched_at       timestamptz NOT NULL DEFAULT now(),
            last_checked_at  timestamptz,
            last_status      smallint,
            is_active        boolean NOT NULL DEFAULT true,
            created_at       timestamptz NOT NULL DEFAULT now(),
            updated_at       timestamptz NOT NULL DEFAULT now(),
            UNIQUE (source, external_id)
        )
    """)
    op.execute("""
        CREATE TABLE catalog.resource_skills (
            resource_id bigint NOT NULL REFERENCES catalog.learning_resources (id) ON DELETE CASCADE,
            skill_id    integer NOT NULL REFERENCES catalog.skills (id) ON DELETE CASCADE,
            relation    text NOT NULL DEFAULT 'teaches'
                        CONSTRAINT resource_skills_relation_check CHECK (relation IN ('teaches', 'requires')),
            level       smallint CONSTRAINT resource_skills_level_check CHECK (level BETWEEN 1 AND 4),
            confidence  real,
            tagged_by   text NOT NULL CONSTRAINT resource_skills_tagged_by_check
                        CHECK (tagged_by IN ('curated', 'llm', 'reviewed')),
            PRIMARY KEY (resource_id, skill_id, relation)
        )
    """)
    op.execute("CREATE INDEX resource_skills_skill_idx ON catalog.resource_skills (skill_id, relation)")


def downgrade() -> None:
    op.execute("DROP TABLE catalog.resource_skills")
    op.execute("DROP TABLE catalog.learning_resources")
    op.execute("DROP SCHEMA ingest CASCADE")
