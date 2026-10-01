"""Catalog v2 in its own schema (ADR-0028): skills graph, roles, levels, roadmaps, common paths.

The YAML in catalog/ is the source of truth; `xcrs catalog import` loads it here. Entities (levels,
families, skills, roles, role levels) keep stable ids across imports (upserted by slug); their parts
(prerequisites, titles, stages, items, paths) are replaced on every import, since nothing outside the
catalog refers to them. Requirements are stored one row per option: rows sharing (owner, group/item
number) are one AND-requirement whose options are alternatives (OR), each with a minimum proficiency.
Resource tables come with resource ingestion (ADR-0026), once providers are decided. The research
tables in `public` keep serving the current recommender until the new engine switches over.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-01
"""

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA catalog")
    op.execute("""
        CREATE TABLE catalog.imports (
            id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            git_commit  text,
            git_dirty   boolean NOT NULL,
            checksum    text NOT NULL,
            stats       jsonb NOT NULL,
            changes     jsonb NOT NULL,
            imported_at timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE TABLE catalog.levels (
            id            smallint PRIMARY KEY,
            slug          text NOT NULL UNIQUE,
            name          text NOT NULL,
            typical_years text NOT NULL,
            scope         text NOT NULL
        )
    """)
    op.execute("""
        CREATE TABLE catalog.families (
            id   smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            slug text NOT NULL UNIQUE,
            name text NOT NULL
        )
    """)
    op.execute("""
        CREATE TABLE catalog.skills (
            id          integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            slug        text NOT NULL UNIQUE,
            name        text NOT NULL,
            kind        text NOT NULL CONSTRAINT skills_kind_check
                        CHECK (kind IN ('language', 'framework', 'library', 'tool', 'platform', 'concept', 'practice')),
            description text NOT NULL,
            onet        text[] NOT NULL DEFAULT '{}',
            created_at  timestamptz NOT NULL DEFAULT now(),
            updated_at  timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE TABLE catalog.skill_prerequisites (
            skill_id        integer NOT NULL REFERENCES catalog.skills (id) ON DELETE CASCADE,
            group_no        smallint NOT NULL,
            option_skill_id integer NOT NULL REFERENCES catalog.skills (id) ON DELETE CASCADE,
            min_level       smallint NOT NULL
                            CONSTRAINT skill_prerequisites_min_level_check CHECK (min_level BETWEEN 1 AND 4),
            PRIMARY KEY (skill_id, group_no, option_skill_id)
        )
    """)
    op.execute("CREATE INDEX skill_prerequisites_option_idx ON catalog.skill_prerequisites (option_skill_id)")
    op.execute("""
        CREATE TABLE catalog.roles (
            id         smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            slug       text NOT NULL UNIQUE,
            name       text NOT NULL,
            family_id  smallint NOT NULL REFERENCES catalog.families (id),
            summary    text NOT NULL,
            onet_code  text NOT NULL CONSTRAINT roles_onet_code_check CHECK (onet_code ~ '^\\d{2}-\\d{4}\\.\\d{2}$'),
            esco_uri   text,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE TABLE catalog.role_titles (
            id      integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            role_id smallint NOT NULL REFERENCES catalog.roles (id) ON DELETE CASCADE,
            title   text NOT NULL UNIQUE
        )
    """)
    op.execute("CREATE INDEX role_titles_role_idx ON catalog.role_titles (role_id)")
    op.execute("""
        CREATE TABLE catalog.role_title_skills (
            title_id        integer NOT NULL REFERENCES catalog.role_titles (id) ON DELETE CASCADE,
            item_no         smallint NOT NULL,
            option_skill_id integer NOT NULL REFERENCES catalog.skills (id) ON DELETE CASCADE,
            min_level       smallint NOT NULL
                            CONSTRAINT role_title_skills_min_level_check CHECK (min_level BETWEEN 1 AND 4),
            PRIMARY KEY (title_id, item_no, option_skill_id)
        )
    """)
    op.execute("""
        CREATE TABLE catalog.role_levels (
            role_id  smallint NOT NULL REFERENCES catalog.roles (id) ON DELETE CASCADE,
            level_id smallint NOT NULL REFERENCES catalog.levels (id),
            title    text,
            summary  text NOT NULL,
            PRIMARY KEY (role_id, level_id)
        )
    """)
    op.execute("""
        CREATE TABLE catalog.roadmap_stages (
            id       integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            role_id  smallint NOT NULL,
            level_id smallint NOT NULL,
            position smallint NOT NULL,
            name     text NOT NULL,
            optional boolean NOT NULL DEFAULT false,
            UNIQUE (role_id, level_id, position),
            FOREIGN KEY (role_id, level_id) REFERENCES catalog.role_levels (role_id, level_id) ON DELETE CASCADE
        )
    """)
    op.execute("""
        CREATE TABLE catalog.roadmap_items (
            stage_id        integer NOT NULL REFERENCES catalog.roadmap_stages (id) ON DELETE CASCADE,
            item_no         smallint NOT NULL,
            option_skill_id integer NOT NULL REFERENCES catalog.skills (id) ON DELETE CASCADE,
            min_level       smallint NOT NULL
                            CONSTRAINT roadmap_items_min_level_check CHECK (min_level BETWEEN 1 AND 4),
            PRIMARY KEY (stage_id, item_no, option_skill_id)
        )
    """)
    op.execute("CREATE INDEX roadmap_items_skill_idx ON catalog.roadmap_items (option_skill_id)")
    op.execute("""
        CREATE TABLE catalog.common_paths (
            id             integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            from_role_id   smallint NOT NULL,
            from_level_id  smallint NOT NULL,
            to_role_id     smallint NOT NULL,
            to_level_id    smallint NOT NULL,
            kind           text NOT NULL CONSTRAINT common_paths_kind_check
                           CHECK (kind IN ('broaden', 'specialize', 'pivot', 'lead')),
            typical_years  text,
            UNIQUE (from_role_id, from_level_id, to_role_id, to_level_id),
            FOREIGN KEY (from_role_id, from_level_id)
                REFERENCES catalog.role_levels (role_id, level_id) ON DELETE CASCADE,
            FOREIGN KEY (to_role_id, to_level_id)
                REFERENCES catalog.role_levels (role_id, level_id) ON DELETE CASCADE
        )
    """)
    op.execute("""
        CREATE TABLE catalog.legacy_roles (
            legacy_slug text PRIMARY KEY,
            role_id     smallint REFERENCES catalog.roles (id) ON DELETE SET NULL
        )
    """)


def downgrade() -> None:
    op.execute("DROP SCHEMA catalog CASCADE")
