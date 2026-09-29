# Architecture

A living document: it shows the **current** architecture and the **target** under way. Each change links to the ADR that justified it. Measured characteristics of the starting point are in [baseline.md](baseline.md). The consolidated database design is in [schema.md](schema.md).

## Current (research prototype)

```
Browser ──► Nuxt 3 (SSR + server route /api/recommend)
                 │  5 sequential calls, one per embedding provider
                 ▼
         FastAPI backend (module-level globals)
           ├─ embed user input via provider API
           ├─ cosine similarity against in-memory dense matrices
           ├─ role scoring + course selection (pandas)
           └─ explanations via gpt-4o
                 │ loads at startup
                 ▼
         CSV files (courses, roadmap nodes, 5× embeddings)

Offline: embedding-generation/ script → CSV files
```

## Target (evolving)

Decided so far:

- **Hosting:** containers orchestrated with Docker Compose on a self-hosted bare-metal server; cloud as a later target for the same images. [ADR-0002](adr/0002-self-hosted-first-cloud-last.md)
- **Primary store:** PostgreSQL + pgvector holds courses, roadmap hierarchy with explicit relations, embeddings, and user-facing data; vector search replaces the in-memory matrices. [ADR-0003](adr/0003-postgresql-pgvector-primary-store.md)
- **Embeddings:** self-hosted, produced by `qwen3-embedding:0.6b` on Ollama, behind our own OpenAI-compatible `Embedder` interface. [ADR-0005](adr/0005-local-embedding-models.md), [ADR-0006](adr/0006-own-embedding-interface.md), [ADR-0007](adr/0007-ollama-qwen3-embedding.md)
- **Vector storage:** per-entity embedding tables keyed by model, with a model registry for zero-downtime model migration. Per-model partial HNSW indexes; the top 20 courses per concept are precomputed into `concept_course_matches`. [ADR-0008](adr/0008-embedding-tables-per-entity.md), [ADR-0009](adr/0009-per-model-indexes-and-precomputed-matches.md)
- **Query path:** an exact threshold scan for user phrases × concepts; the disliked-course penalty is checked only against candidate courses, so no request scans the course catalog. [ADR-0010](adr/0010-exact-threshold-search-and-candidate-penalty.md)
- **Data access & schema changes:** SQLAlchemy 2.0 (ORM plus raw SQL) behind a repository, on psycopg 3; schema changes managed with Alembic. [ADR-0011](adr/0011-alembic-schema-migrations.md), [ADR-0012](adr/0012-sqlalchemy-orm-with-raw-sql-repository.md)
- **User activity:** requests store their input as JSONB for now; shown roles and courses, and feedback, are normalized tables. The input will be normalized once its format settles. [ADR-0013](adr/0013-user-activity-hybrid-then-normalized.md)
- **Deployment:** local-first on the MacBook (Ollama native for GPU, the rest in Docker Compose); target: 16 GB VM runs Ollama (private network only), 8 GB VM runs Postgres, the backend and the frontend. [ADR-0014](adr/0014-local-first-then-split-by-role.md)
- **Ingestion store (proposed):** MongoDB for raw scraped data and roadmap drafts. [ADR-0004](adr/0004-mongodb-for-ingestion-layer.md)
- **Concurrency:** synchronous endpoints in FastAPI's thread pool, with all I/O behind adapters so a switch to async doesn't touch the domain. [ADR-0015](adr/0015-sync-endpoints-async-ready.md)
- **API contract:** one versioned endpoint, `POST /api/v1/recommendations`, with structured lists instead of comma-separated text; the model comes from the registry, not the URL. [ADR-0016](adr/0016-versioned-structured-recommendation-api.md)
- **Backend structure:** `api/` → `services/` → pure `domain/` plus adapters (`repository/`, `embeddings/`, `explain/`). [ADR-0017](adr/0017-layered-backend-with-pure-domain.md)
- **Explanations (accepted, not built yet):** the response returns roles and courses immediately; explanations are generated in the background, one LLM call per role, and the UI polls for them. [ADR-0018](adr/0018-decoupled-per-role-explanations.md)
- **Explanation layer:** LangChain chain (prompt template → `ChatOpenAI` on any OpenAI-compatible server → structured output), grounded in the algorithm's actual reasons; the algorithm decides, the LLM explains. A LangChain retriever wraps our own pgvector SQL. [ADR-0019](adr/0019-explanation-layer-on-langchain.md)

```
Browser ──► Frontend (TBD)
                 │
                 ▼
         Backend API (/api/v1, layered)
                 │
                 ▼
         PostgreSQL + pgvector ◄── ingestion pipeline ◄── (MongoDB raw/drafts, proposed)
                                                      ◄── scraper / roadmap generator (future)
```

## New system (running locally on the MacBook, 2026-09-29)

```
Browser ──► Nuxt 3 (prototype UI; server route adapts comma-separated input to /api/v1)
                 │  1 call
                 ▼
         FastAPI  POST /api/v1/recommendations              (backend/xcrs/)
           api/       validation, dependency wiring
           services/  order of steps, saves the request and its results
           domain/    role scoring, course selection (pure Python, no I/O)
           adapters:
             embeddings/ ──► Ollama  qwen3-embedding:0.6b   (native, Apple GPU)
             repository/ ──► PostgreSQL 18 + pgvector 0.8   (Docker)
             explain/    ──► Ollama  qwen3.5:9b             (LangChain chain, one call per role, inline for now)
           retrieval.py  LangChain retriever over repository/ (xcrs search-courses; future chat tool)

Offline: uv run xcrs import-prototype | register-model | embed-catalog
         (catalog vectors, threshold statistics, top-20 concept → course matches)
```

The prototype (`backend/src/`, `embedding-generation/`) stays runnable next to it until the quality comparison passes. First measurements are in [baseline.md](baseline.md#new-system-first-measurements).

Still open: the production explanation model (ADR-0020, after a CPU benchmark), threshold calibration for user phrases, frontend framework, a chat/agent feature and LLM tracing, CI/CD, and the cloud target. See [adr/README.md](adr/README.md#upcoming-decisions).
