# ruff: noqa: E501  (the downgrade DDL is pg_dump output, kept verbatim)
"""Remove the classic engine's tables (ADR-0039): the 2024 research catalog, its embeddings and activity.

Engine v2 is the only engine (ADR-0037). Before this ran on the dev database, a verified dump was kept in
backups/archive/ and the activity rows (requests, roles, courses, feedback) were exported to JSON there.
`embedding_models` stays: catalog.skill_embeddings uses it.

The downgrade recreates the tables empty, exactly as they were at 0009 (captured with pg_dump), so the
migration chain still runs both ways; the data comes back only from the archived dump.

Revision ID: 0010
Revises: 0009
Create Date: 2026-10-02
"""

from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None

TABLES = [
    "feedback",
    "recommended_courses",
    "recommended_roles",
    "recommendation_requests",
    "concept_course_matches",
    "node_embeddings",
    "course_embeddings",
    "courses",
    "roadmap_nodes",
    "roles",
]


def upgrade() -> None:
    op.execute("DROP TABLE " + ", ".join(TABLES))


def downgrade() -> None:
    op.execute("""
        CREATE TABLE public.concept_course_matches (
            model_id smallint NOT NULL,
            concept_id bigint NOT NULL,
            course_id bigint NOT NULL,
            similarity real NOT NULL,
            rank smallint NOT NULL,
            CONSTRAINT concept_course_matches_rank_check CHECK (((rank >= 1) AND (rank <= 20)))
        );
        CREATE TABLE public.course_embeddings (
            course_id bigint NOT NULL,
            model_id smallint NOT NULL,
            embedding public.vector NOT NULL,
            content_hash text NOT NULL,
            created_at timestamp with time zone DEFAULT now() NOT NULL
        );
        CREATE TABLE public.courses (
            id bigint NOT NULL,
            source text NOT NULL,
            source_id text NOT NULL,
            title text NOT NULL,
            url text NOT NULL,
            headline text,
            description text,
            what_you_learn text,
            category text,
            language text,
            is_paid boolean,
            price numeric,
            rating real,
            embed_text text NOT NULL,
            content_hash text NOT NULL,
            is_active boolean DEFAULT true NOT NULL,
            fetched_at timestamp with time zone,
            created_at timestamp with time zone DEFAULT now() NOT NULL,
            updated_at timestamp with time zone DEFAULT now() NOT NULL
        );
        ALTER TABLE public.courses ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
            SEQUENCE NAME public.courses_id_seq
            START WITH 1
            INCREMENT BY 1
            NO MINVALUE
            NO MAXVALUE
            CACHE 1
        );
        CREATE TABLE public.feedback (
            id bigint NOT NULL,
            request_id uuid NOT NULL,
            role_id smallint,
            course_id bigint,
            rating smallint,
            comment text,
            created_at timestamp with time zone DEFAULT now() NOT NULL
        );
        ALTER TABLE public.feedback ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
            SEQUENCE NAME public.feedback_id_seq
            START WITH 1
            INCREMENT BY 1
            NO MINVALUE
            NO MAXVALUE
            CACHE 1
        );
        CREATE TABLE public.node_embeddings (
            node_id bigint NOT NULL,
            model_id smallint NOT NULL,
            embedding public.vector NOT NULL,
            content_hash text NOT NULL,
            created_at timestamp with time zone DEFAULT now() NOT NULL
        );
        CREATE TABLE public.recommendation_requests (
            id uuid DEFAULT gen_random_uuid() NOT NULL,
            created_at timestamp with time zone DEFAULT now() NOT NULL,
            model_id smallint NOT NULL,
            algorithm_version text NOT NULL,
            threshold_used real NOT NULL,
            input jsonb NOT NULL,
            status text NOT NULL,
            latency_ms integer,
            error text,
            CONSTRAINT recommendation_requests_status_check CHECK ((status = ANY (ARRAY['ok'::text, 'insufficient_input'::text, 'error'::text])))
        );
        CREATE TABLE public.recommended_courses (
            request_id uuid NOT NULL,
            role_id smallint NOT NULL,
            rank smallint NOT NULL,
            course_id bigint NOT NULL,
            similarity real NOT NULL,
            explanation text,
            concept_ids bigint[] DEFAULT '{}'::bigint[] NOT NULL
        );
        CREATE TABLE public.recommended_roles (
            request_id uuid NOT NULL,
            rank smallint NOT NULL,
            role_id smallint NOT NULL,
            score real NOT NULL,
            explanation text,
            prompt_version text,
            explanation_status text DEFAULT 'pending'::text NOT NULL,
            explanation_input jsonb,
            explanation_ms integer,
            explanation_attempts smallint DEFAULT 0 NOT NULL,
            explained_at timestamp with time zone,
            next_concept_ids bigint[] DEFAULT '{}'::bigint[] NOT NULL,
            CONSTRAINT recommended_roles_explanation_status_check CHECK ((explanation_status = ANY (ARRAY['pending'::text, 'done'::text, 'failed'::text, 'disabled'::text])))
        );
        CREATE TABLE public.roadmap_nodes (
            id bigint NOT NULL,
            role_id smallint NOT NULL,
            parent_id bigint,
            type text NOT NULL,
            name text NOT NULL,
            content text NOT NULL,
            "position" integer NOT NULL,
            sequence integer NOT NULL,
            legacy_id bigint,
            source text NOT NULL,
            source_version text,
            content_hash text NOT NULL,
            created_at timestamp with time zone DEFAULT now() NOT NULL,
            updated_at timestamp with time zone DEFAULT now() NOT NULL,
            CONSTRAINT roadmap_nodes_type_check CHECK ((type = ANY (ARRAY['topic'::text, 'concept'::text])))
        );
        ALTER TABLE public.roadmap_nodes ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
            SEQUENCE NAME public.roadmap_nodes_id_seq
            START WITH 1
            INCREMENT BY 1
            NO MINVALUE
            NO MAXVALUE
            CACHE 1
        );
        CREATE TABLE public.roles (
            id smallint NOT NULL,
            slug text NOT NULL,
            name text NOT NULL,
            created_at timestamp with time zone DEFAULT now() NOT NULL
        );
        ALTER TABLE ONLY public.concept_course_matches
            ADD CONSTRAINT concept_course_matches_model_id_concept_id_rank_key UNIQUE (model_id, concept_id, rank);
        ALTER TABLE ONLY public.concept_course_matches
            ADD CONSTRAINT concept_course_matches_pkey PRIMARY KEY (model_id, concept_id, course_id);
        ALTER TABLE ONLY public.course_embeddings
            ADD CONSTRAINT course_embeddings_pkey PRIMARY KEY (course_id, model_id);
        ALTER TABLE ONLY public.courses
            ADD CONSTRAINT courses_pkey PRIMARY KEY (id);
        ALTER TABLE ONLY public.courses
            ADD CONSTRAINT courses_source_source_id_key UNIQUE (source, source_id);
        ALTER TABLE ONLY public.feedback
            ADD CONSTRAINT feedback_pkey PRIMARY KEY (id);
        ALTER TABLE ONLY public.node_embeddings
            ADD CONSTRAINT node_embeddings_pkey PRIMARY KEY (node_id, model_id);
        ALTER TABLE ONLY public.recommendation_requests
            ADD CONSTRAINT recommendation_requests_pkey PRIMARY KEY (id);
        ALTER TABLE ONLY public.recommended_courses
            ADD CONSTRAINT recommended_courses_pkey PRIMARY KEY (request_id, role_id, rank);
        ALTER TABLE ONLY public.recommended_roles
            ADD CONSTRAINT recommended_roles_pkey PRIMARY KEY (request_id, rank);
        ALTER TABLE ONLY public.recommended_roles
            ADD CONSTRAINT recommended_roles_request_id_role_id_key UNIQUE (request_id, role_id);
        ALTER TABLE ONLY public.roadmap_nodes
            ADD CONSTRAINT roadmap_nodes_legacy_id_key UNIQUE (legacy_id);
        ALTER TABLE ONLY public.roadmap_nodes
            ADD CONSTRAINT roadmap_nodes_pkey PRIMARY KEY (id);
        ALTER TABLE ONLY public.roles
            ADD CONSTRAINT roles_pkey PRIMARY KEY (id);
        ALTER TABLE ONLY public.roles
            ADD CONSTRAINT roles_slug_key UNIQUE (slug);
        CREATE INDEX concept_course_matches_course_idx ON public.concept_course_matches USING btree (model_id, course_id);
        CREATE INDEX course_emb_m1_hnsw ON public.course_embeddings USING hnsw (((embedding)::public.vector(1024)) public.vector_cosine_ops) WHERE (model_id = 1);
        CREATE INDEX recommendation_requests_created_idx ON public.recommendation_requests USING btree (created_at);
        CREATE INDEX recommended_roles_pending_idx ON public.recommended_roles USING btree (request_id) WHERE (explanation_status = 'pending'::text);
        CREATE INDEX roadmap_nodes_parent_idx ON public.roadmap_nodes USING btree (parent_id);
        CREATE INDEX roadmap_nodes_role_sequence_idx ON public.roadmap_nodes USING btree (role_id, sequence);
        ALTER TABLE ONLY public.concept_course_matches
            ADD CONSTRAINT concept_course_matches_concept_id_fkey FOREIGN KEY (concept_id) REFERENCES public.roadmap_nodes(id) ON DELETE CASCADE;
        ALTER TABLE ONLY public.concept_course_matches
            ADD CONSTRAINT concept_course_matches_course_id_fkey FOREIGN KEY (course_id) REFERENCES public.courses(id) ON DELETE CASCADE;
        ALTER TABLE ONLY public.concept_course_matches
            ADD CONSTRAINT concept_course_matches_model_id_fkey FOREIGN KEY (model_id) REFERENCES public.embedding_models(id);
        ALTER TABLE ONLY public.course_embeddings
            ADD CONSTRAINT course_embeddings_course_id_fkey FOREIGN KEY (course_id) REFERENCES public.courses(id) ON DELETE CASCADE;
        ALTER TABLE ONLY public.course_embeddings
            ADD CONSTRAINT course_embeddings_model_id_fkey FOREIGN KEY (model_id) REFERENCES public.embedding_models(id);
        ALTER TABLE ONLY public.feedback
            ADD CONSTRAINT feedback_course_id_fkey FOREIGN KEY (course_id) REFERENCES public.courses(id);
        ALTER TABLE ONLY public.feedback
            ADD CONSTRAINT feedback_request_id_fkey FOREIGN KEY (request_id) REFERENCES public.recommendation_requests(id) ON DELETE CASCADE;
        ALTER TABLE ONLY public.feedback
            ADD CONSTRAINT feedback_role_id_fkey FOREIGN KEY (role_id) REFERENCES public.roles(id);
        ALTER TABLE ONLY public.node_embeddings
            ADD CONSTRAINT node_embeddings_model_id_fkey FOREIGN KEY (model_id) REFERENCES public.embedding_models(id);
        ALTER TABLE ONLY public.node_embeddings
            ADD CONSTRAINT node_embeddings_node_id_fkey FOREIGN KEY (node_id) REFERENCES public.roadmap_nodes(id) ON DELETE CASCADE;
        ALTER TABLE ONLY public.recommendation_requests
            ADD CONSTRAINT recommendation_requests_model_id_fkey FOREIGN KEY (model_id) REFERENCES public.embedding_models(id);
        ALTER TABLE ONLY public.recommended_courses
            ADD CONSTRAINT recommended_courses_course_id_fkey FOREIGN KEY (course_id) REFERENCES public.courses(id);
        ALTER TABLE ONLY public.recommended_courses
            ADD CONSTRAINT recommended_courses_request_id_role_id_fkey FOREIGN KEY (request_id, role_id) REFERENCES public.recommended_roles(request_id, role_id) ON DELETE CASCADE;
        ALTER TABLE ONLY public.recommended_roles
            ADD CONSTRAINT recommended_roles_request_id_fkey FOREIGN KEY (request_id) REFERENCES public.recommendation_requests(id) ON DELETE CASCADE;
        ALTER TABLE ONLY public.recommended_roles
            ADD CONSTRAINT recommended_roles_role_id_fkey FOREIGN KEY (role_id) REFERENCES public.roles(id);
        ALTER TABLE ONLY public.roadmap_nodes
            ADD CONSTRAINT roadmap_nodes_parent_id_fkey FOREIGN KEY (parent_id) REFERENCES public.roadmap_nodes(id);
        ALTER TABLE ONLY public.roadmap_nodes
            ADD CONSTRAINT roadmap_nodes_role_id_fkey FOREIGN KEY (role_id) REFERENCES public.roles(id);
    """)
