# Architecture Decision Records

Significant design decisions for XCRS, one per file. The format is explained in [ADR-0001](0001-record-architecture-decisions.md); new records start from the [template](0000-template.md).

| # | Decision | Status | Date |
|---|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted | 2026-09-27 |
| [0002](0002-self-hosted-first-cloud-last.md) | Self-hosted first, cloud last | Accepted | 2026-09-27 |
| [0003](0003-postgresql-pgvector-primary-store.md) | PostgreSQL + pgvector as the primary data store | Accepted | 2026-09-27 |
| [0004](0004-mongodb-for-ingestion-layer.md) | MongoDB for the ingestion layer | Proposed | 2026-09-27 |
| [0005](0005-local-embedding-models.md) | Self-hosted embedding models instead of hosted APIs | Accepted | 2026-09-27 |
| [0006](0006-own-embedding-interface.md) | Own embedding interface with an OpenAI-compatible adapter (LangChain deferred) | Accepted | 2026-09-27 |
| [0007](0007-ollama-qwen3-embedding.md) | Ollama as the model runtime, `qwen3-embedding:0.6b` as the embedding model | Accepted | 2026-09-28 |
| [0008](0008-embedding-tables-per-entity.md) | Store embeddings in per-entity tables keyed by model | Accepted | 2026-09-28 |
| [0009](0009-per-model-indexes-and-precomputed-matches.md) | Per-model vector indexes and precomputed concept → course matches | Accepted | 2026-09-28 |
| [0010](0010-exact-threshold-search-and-candidate-penalty.md) | Exact threshold search for user phrases; disliked penalty only on candidate courses | Accepted | 2026-09-28 |
| [0011](0011-alembic-schema-migrations.md) | Alembic for database schema migrations | Accepted | 2026-09-28 |
| [0012](0012-sqlalchemy-orm-with-raw-sql-repository.md) | SQLAlchemy 2.0 (ORM + raw SQL) behind a repository, on psycopg 3 | Accepted | 2026-09-28 |
| [0013](0013-user-activity-hybrid-then-normalized.md) | User activity storage: hybrid now, fully normalized once the input format settles | Accepted | 2026-09-28 |
| [0014](0014-local-first-then-split-by-role.md) | Run locally first; target deployment splits the two VMs by role | Accepted | 2026-09-28 |
| [0015](0015-sync-endpoints-async-ready.md) | Synchronous endpoints now, structured to switch to async later | Accepted | 2026-09-29 |
| [0016](0016-versioned-structured-recommendation-api.md) | Versioned recommendation API with structured input | Accepted | 2026-09-29 |
| [0017](0017-layered-backend-with-pure-domain.md) | Layered backend with a pure domain core | Accepted | 2026-09-29 |
| [0018](0018-decoupled-per-role-explanations.md) | Explanations generated after the response, one LLM call per role | Accepted | 2026-09-29 |
| [0019](0019-explanation-layer-on-langchain.md) | Explanation layer on LangChain, with a grounded contract; retriever over our own SQL | Accepted | 2026-09-30 |

## Upcoming decisions

- ADR-0020: Explanation LLM model for production (CPU VM now, GPU later), after a CPU benchmark
- Threshold calibration for user phrases (the open issue in ADR-0010), during the prototype-vs-new comparison
- Frontend framework
- (Later) Data quality: enriching user input, richer generated roadmaps, prerequisite graph and roadmap visualization
- User input collection redesign (clickable suggested phrases instead of comma-separated text); afterwards, normalize the request input (ADR-0013)
- Chat/agent feature (LangGraph) and LLM tracing (LangSmith or self-hosted); see ADR-0019
- CI/CD and the cloud deployment target
