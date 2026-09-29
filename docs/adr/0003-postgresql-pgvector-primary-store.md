# ADR-0003: PostgreSQL + pgvector as the primary data store

- **Status:** Accepted
- **Date:** 2026-09-27
- **Decider:** Muhammed Yasin Horasanli

## Context

The prototype keeps all data in CSV files and memory:

- Courses, roadmap nodes and embeddings for 5 providers are CSVs. Vectors are stored as strings and parsed at startup (`backend/src/util.py:convert_to_float`).
- At startup the backend builds a **dense course × concept similarity matrix per provider** (`backend/src/main.py:main`). Memory grows as courses × concepts × providers: trivial at 453 × 869, but around 20 GB per provider at 100K × 50K.
- The roadmap hierarchy (role → topic → concept) is encoded **inside numeric IDs**, two digits per level (e.g. `60203`), and decoded arithmetically (`util.get_role_id`, `util.get_parent_topics`). That caps each level at 100 children and is effectively a makeshift foreign key.
- The career-role list is hardcoded in three places.
- Planned growth (a new scraper, more course sources, generated roadmaps) makes an in-memory design untenable.

Constraints from [ADR-0002](0002-self-hosted-first-cloud-last.md): zero cost, self-hostable, portable to the cloud later.

## Options considered

### Option A — PostgreSQL + pgvector
- ✅ The data is **relational**: roles contain topics, topics contain concepts, and concepts link to courses. Explicit foreign keys replace the digit-encoded IDs, and role scoring becomes SQL aggregation. *(Note 2026-09-29: role scoring ended up as pure Python in the domain layer ([ADR-0017](0017-layered-backend-with-pure-domain.md)); the database supplies the matches.)*
- ✅ Vectors, metadata and relations live in one database, so a single query can combine similarity search with joins and filters.
- ✅ One container, a mature ecosystem, simple backups (`pg_dump`)
- ✅ Clear cloud path (AWS RDS/Aurora, GCP Cloud SQL/AlloyDB)
- ✅ PostgreSQL is among the most requested database skills
- ❌ pgvector's HNSW index supports at most 2,000 dimensions for `vector` (4,000 for `halfvec`). OpenAI `text-embedding-3-large` produces 3,072.
- ❌ Beyond tens of millions of vectors, dedicated vector engines scale more easily

### Option B — MongoDB Community + self-managed Vector Search (`mongot`)
- ✅ The flexible schema suits scraped data with differing shapes
- ✅ Self-managed Vector Search became generally available in June 2026 and is free in Community Edition, with the same API as Atlas
- ❌ Needs a replica set plus a separate `mongot` process: more moving parts on one server
- ❌ Newer, with fewer operational examples
- ❌ Weaker fit for the tree-shaped, relational roadmap data and the aggregation-heavy scoring

### Option C — Qdrant (dedicated vector database)
- ✅ Fast, simple to self-host, popular in AI stacks
- ❌ Vectors only: a second database would still be needed for courses, roadmaps and user data, meaning two systems to run and keep consistent

### Option D — Pinecone
- ✅ Best-known managed vector database, zero ops
- ❌ Managed-only and paid beyond a small free tier; can't be self-hosted, so it violates ADR-0002

### Option E — MongoDB Atlas
- ✅ Managed, same API as self-managed MongoDB
- ❌ The free tier is too small for 5 providers' vectors; dedicated tiers cost money, which violates ADR-0002

## Decision

Use **PostgreSQL with the pgvector extension** as the single primary store for courses, roadmap nodes (with explicit parent/role relationships), embeddings and user-facing data. Similarity search moves from in-memory dense matrices to pgvector queries with an HNSW index.

## Trade-offs accepted

- **Dimension limit:** for 3,072-dim OpenAI vectors, use `halfvec` (half precision, fits under the 4,000 limit) or request a smaller `dimensions` from the API. The other four providers produce 768–1,536 dims and fit as-is.
- **Less schema flexibility** for heterogeneous scraped data. Handled by keeping raw ingestion data out of the primary store ([ADR-0004](0004-mongodb-for-ingestion-layer.md)), and/or JSONB columns for source-specific fields.
- **Approximate search:** HNSW returns approximate nearest neighbours, not the exact full-matrix results the prototype computes. The prototype's σ-based thresholds, currently computed over the full matrix, need an equivalent (precomputed per model at ingestion time, or replaced by top-k plus a score cutoff). Results must be compared against the prototype before switching.

## Revisit when

- Vector count per provider exceeds ~10M, or p95 query latency on the server becomes unacceptable despite index tuning.
- Most data turns into heterogeneous documents rather than relational entities.
