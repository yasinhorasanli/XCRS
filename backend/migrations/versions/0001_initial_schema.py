"""Initial schema: catalog, vectors, activity (docs/schema.md).

Revision ID: 0001
Revises:
Create Date: 2026-09-28
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # --- Catalog ---------------------------------------------------------------------------------
    op.execute("""
        CREATE TABLE roles (
            id          smallint PRIMARY KEY,
            slug        text NOT NULL UNIQUE,
            name        text NOT NULL,
            created_at  timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE TABLE roadmap_nodes (
            id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            role_id         smallint NOT NULL REFERENCES roles (id),
            parent_id       bigint REFERENCES roadmap_nodes (id),
            type            text NOT NULL CONSTRAINT roadmap_nodes_type_check CHECK (type IN ('topic', 'concept')),
            name            text NOT NULL,
            content         text NOT NULL,
            position        integer NOT NULL,
            sequence        integer NOT NULL,
            legacy_id       bigint UNIQUE,
            source          text NOT NULL,
            source_version  text,
            content_hash    text NOT NULL,
            created_at      timestamptz NOT NULL DEFAULT now(),
            updated_at      timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX roadmap_nodes_role_sequence_idx ON roadmap_nodes (role_id, sequence)")
    op.execute("CREATE INDEX roadmap_nodes_parent_idx ON roadmap_nodes (parent_id)")
    op.execute("""
        CREATE TABLE courses (
            id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            source          text NOT NULL,
            source_id       text NOT NULL,
            title           text NOT NULL,
            url             text NOT NULL,
            headline        text,
            description     text,
            what_you_learn  text,
            category        text,
            language        text,
            is_paid         boolean,
            price           numeric,
            rating          real,
            embed_text      text NOT NULL,
            content_hash    text NOT NULL,
            is_active       boolean NOT NULL DEFAULT true,
            fetched_at      timestamptz,
            created_at      timestamptz NOT NULL DEFAULT now(),
            updated_at      timestamptz NOT NULL DEFAULT now(),
            UNIQUE (source, source_id)
        )
    """)

    # --- Vectors ---------------------------------------------------------------------------------
    op.execute("""
        CREATE TABLE embedding_models (
            id                 smallint PRIMARY KEY,
            name               text NOT NULL UNIQUE,
            runtime            text NOT NULL,
            quantization       text NOT NULL,
            dimensions         integer NOT NULL CONSTRAINT embedding_models_dimensions_check CHECK (dimensions > 0),
            query_prefix       text,
            document_prefix    text,
            status             text NOT NULL
                               CONSTRAINT embedding_models_status_check CHECK (status IN ('candidate', 'active', 'retired')),
            sim_mean           real,
            sim_std            real,
            stats_computed_at  timestamptz,
            created_at         timestamptz NOT NULL DEFAULT now()
        )
    """)
    # at most one active model
    op.execute("CREATE UNIQUE INDEX embedding_models_one_active_idx ON embedding_models ((true)) WHERE status = 'active'")

    for entity, table in (("course", "courses"), ("node", "roadmap_nodes")):
        op.execute(f"""
            CREATE TABLE {entity}_embeddings (
                {entity}_id   bigint NOT NULL REFERENCES {table} (id) ON DELETE CASCADE,
                model_id      smallint NOT NULL REFERENCES embedding_models (id),
                embedding     vector NOT NULL,   -- untyped; per-model indexes cast it (ADR-0009)
                content_hash  text NOT NULL,
                created_at    timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY ({entity}_id, model_id)
            )
        """)

    op.execute("""
        CREATE TABLE concept_course_matches (
            model_id    smallint NOT NULL REFERENCES embedding_models (id),
            concept_id  bigint NOT NULL REFERENCES roadmap_nodes (id) ON DELETE CASCADE,
            course_id   bigint NOT NULL REFERENCES courses (id) ON DELETE CASCADE,
            similarity  real NOT NULL,
            rank        smallint NOT NULL CONSTRAINT concept_course_matches_rank_check CHECK (rank BETWEEN 1 AND 20),
            PRIMARY KEY (model_id, concept_id, course_id),
            UNIQUE (model_id, concept_id, rank)
        )
    """)
    op.execute("CREATE INDEX concept_course_matches_course_idx ON concept_course_matches (model_id, course_id)")

    # --- Activity --------------------------------------------------------------------------------
    op.execute("""
        CREATE TABLE recommendation_requests (
            id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            created_at         timestamptz NOT NULL DEFAULT now(),
            model_id           smallint NOT NULL REFERENCES embedding_models (id),
            algorithm_version  text NOT NULL,
            threshold_used     real NOT NULL,
            input              jsonb NOT NULL,   -- transitional (ADR-0013)
            status             text NOT NULL
                               CONSTRAINT recommendation_requests_status_check
                               CHECK (status IN ('ok', 'insufficient_input', 'error')),
            latency_ms         integer,
            error              text
        )
    """)
    op.execute("CREATE INDEX recommendation_requests_created_idx ON recommendation_requests (created_at)")
    op.execute("""
        CREATE TABLE recommended_roles (
            request_id   uuid NOT NULL REFERENCES recommendation_requests (id) ON DELETE CASCADE,
            rank         smallint NOT NULL,
            role_id      smallint NOT NULL REFERENCES roles (id),
            score        real NOT NULL,
            explanation  text,
            PRIMARY KEY (request_id, rank),
            UNIQUE (request_id, role_id)
        )
    """)
    op.execute("""
        CREATE TABLE recommended_courses (
            request_id   uuid NOT NULL,
            role_id      smallint NOT NULL,
            rank         smallint NOT NULL,
            course_id    bigint NOT NULL REFERENCES courses (id),
            similarity   real NOT NULL,
            explanation  text,
            PRIMARY KEY (request_id, role_id, rank),
            FOREIGN KEY (request_id, role_id) REFERENCES recommended_roles (request_id, role_id) ON DELETE CASCADE
        )
    """)
    op.execute("""
        CREATE TABLE feedback (
            id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            request_id  uuid NOT NULL REFERENCES recommendation_requests (id) ON DELETE CASCADE,
            role_id     smallint REFERENCES roles (id),
            course_id   bigint REFERENCES courses (id),
            rating      smallint,
            comment     text,
            created_at  timestamptz NOT NULL DEFAULT now()
        )
    """)


def downgrade() -> None:
    for table in (
        "feedback",
        "recommended_courses",
        "recommended_roles",
        "recommendation_requests",
        "concept_course_matches",
        "node_embeddings",
        "course_embeddings",
        "embedding_models",
        "courses",
        "roadmap_nodes",
        "roles",
    ):
        op.execute(f"DROP TABLE {table}")
    op.execute("DROP EXTENSION IF EXISTS vector")
