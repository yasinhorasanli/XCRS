# ADR-0012: SQLAlchemy 2.0 (ORM + raw SQL) behind a repository, on psycopg 3

- **Status:** Accepted
- **Date:** 2026-09-28
- **Decider:** Muhammed Yasin Horasanli

## Context

The backend needs a data-access approach for PostgreSQL + pgvector ([ADR-0003](0003-postgresql-pgvector-primary-store.md)). XCRS has two kinds of queries:

- **Plain data work:** saving requests and feedback, reading courses and nodes, inserting or updating courses and nodes during ingestion.
- **Custom analytical queries:** the exact threshold scan over concept vectors ([ADR-0010](0010-exact-threshold-search-and-candidate-penalty.md)), the role-scoring aggregations *(Note 2026-09-29: role scoring ended up as pure Python in the domain layer ([ADR-0017](0017-layered-backend-with-pure-domain.md)); the database supplies the matches.)*, candidate lookup from `concept_course_matches` with the disliked penalty, and per-model index-matching vector queries ([ADR-0009](0009-per-model-indexes-and-precomputed-matches.md)). These are hand-written SQL regardless of tooling.

Migrations use Alembic ([ADR-0011](0011-alembic-schema-migrations.md)), which can draft migrations from SQLAlchemy table metadata.

## Options considered

### Option A — Raw SQL only (psycopg 3)
- ✅ Full control, no abstraction
- ❌ Boilerplate for every insert and select; result mapping by hand; easy to drift from the schema; low CV value

### Option B — SQLAlchemy Core only (table metadata + SQL expression builder)
- ✅ Composable, schema-aware SQL; Alembic integration
- ❌ Still manual mapping to objects; ceremony for plain data work

### Option C — SQLAlchemy ORM for plain data work, raw SQL for analytical/pgvector queries, all behind a repository module
- ✅ The least boilerplate for plain data work; the custom queries stay explicit SQL
- ✅ Alembic integration; `pgvector-python` provides `Vector` / `HALFVEC` types
- ✅ The most common Python data layer in industry
- ❌ ORM pitfalls: hidden lazy-loading queries (the N+1 problem) and session management

## Decision

**Option C.**

- **SQLAlchemy 2.0**, with the ORM for entities (courses, roadmap nodes, embedding models, user activity).
- Raw SQL via `text()` for the analytical and pgvector queries.
- **All database access sits behind one repository module.** API handlers and domain logic never see SQL or sessions.
- **Driver: psycopg 3.** It supports both sync and async, so the API and the ingestion jobs share one driver. Whether the API is async is decided with the backend structure.

## Trade-offs accepted

- **N+1 risk.** Mitigated by explicit eager loading in the repository, plus tests that count the queries a repository function issues. *(Note 2026-09-29: not written yet; planned with the ADR-0018 read endpoint.)* *(Resolved 2026-09-30: `tests/test_explanation_jobs.py::test_reading_a_result_takes_a_fixed_number_of_queries` asserts that reading a result is 4 queries however many roles and courses it has.)*
- **Two styles of data access** (ORM and raw SQL). Contained by the repository boundary.

## Revisit when

- Most queries end up as raw SQL anyway. Then the ORM layer isn't earning its keep, and Core or psycopg alone would be simpler.
