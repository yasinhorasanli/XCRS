"""Skill matching (ADR-0030): skill vectors per embedding model, and a cache of matched phrases.

skill_embeddings follows ADR-0008 (one table per entity, keyed by model; the vector column is untyped so
models of any size can coexist). 257 skills need no vector index; add a per-model partial HNSW index
(as in 0002) when the catalog grows past ~10,000 skills.

phrase_matches caches what the LLM step decided for a normalized phrase. The key includes the prompt
version, the LLM and the catalog version (the checksum of the import), so changing any of them asks again.
Fallback results (LLM unavailable) are not cached. The table only holds derived data: it can be emptied.

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-02
"""

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE catalog.skill_embeddings (
            skill_id     integer NOT NULL REFERENCES catalog.skills (id) ON DELETE CASCADE,
            model_id     smallint NOT NULL REFERENCES public.embedding_models (id),
            embedding    vector NOT NULL,
            content_hash text NOT NULL,
            created_at   timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (skill_id, model_id)
        )
    """)
    op.execute("""
        CREATE TABLE catalog.phrase_matches (
            phrase_key       text NOT NULL,
            catalog_checksum text NOT NULL,
            prompt_version   text NOT NULL,
            llm_model        text NOT NULL,
            skills           text[] NOT NULL,
            picked           text[] NOT NULL,
            llm_ms           integer,
            created_at       timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (phrase_key, catalog_checksum, prompt_version, llm_model)
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE catalog.phrase_matches")
    op.execute("DROP TABLE catalog.skill_embeddings")
