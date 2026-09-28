# ADR-0011: Alembic for database schema migrations

- **Status:** Accepted
- **Date:** 2026-09-28
- **Decider:** Muhammed Yasin Horasanli

## Context

The PostgreSQL schema ([ADR-0003](0003-postgresql-pgvector-primary-store.md), [ADR-0008](0008-embedding-tables-per-entity.md)–[ADR-0010](0010-exact-threshold-search-and-candidate-penalty.md)) will change repeatedly. Known upcoming changes:
- user activity tables;
- the structured user-input redesign;
- a new per-model vector index for every embedding model (e.g. a `halfvec(2560)` index when a 4B model arrives with the GPU).

Each change must reach at least three databases (the MacBook, a staging VM, a production VM, and later the cloud) in the same order, exactly once, and be reversible.

A missing change can fail silently. A missing per-model index doesn't cause errors; queries just fall back to full scans (ADR-0009).

The backend is Python/FastAPI. Our pgvector DDL (untyped `vector` columns, partial expression indexes, `CREATE INDEX CONCURRENTLY`) is hand-written SQL in any case.

## Options considered

### Option A — Plain `.sql` files + our own runner script
- ✅ Transparent, no dependency
- ❌ Reinvents ordering, version tracking and rollback

### Option B — Alembic
- ✅ The standard migration tool in the Python/SQLAlchemy ecosystem
- ✅ Revisions form an ordered chain; applied versions are tracked in the `alembic_version` table; `upgrade` / `downgrade` built in
- ✅ Migrations can be raw SQL (`op.execute`); an ORM isn't required
- ✅ `autocommit_block()` supports non-transactional DDL such as `CREATE INDEX CONCURRENTLY`
- ❌ Learning curve. Autogenerate can't produce our pgvector expression indexes, so those are hand-written.

### Option C — Standalone tool (dbmate, Flyway, …)
- ✅ Plain SQL with up/down sections
- ❌ A separate, non-Python toolchain; less common in Python backends

## Decision

**Alembic**, with migrations in `backend/migrations/`.

- Migrations are **written and reviewed by hand**. Autogenerate may produce drafts but is never trusted blindly.
- pgvector-specific DDL is raw SQL via `op.execute`.
- Every migration has a working `downgrade()`.
- Deployments run `alembic upgrade head` as a mandatory step.
- **Migrations change structure.** A small, one-time **backfill** that a schema change requires (e.g. converting existing rows to a new column format) may live in the same migration. Recurring or heavy data work (registering a model, embedding the catalog, filling `concept_course_matches`) belongs to the ingestion/admin commands, not to migrations.
- **Before launch** (XCRS has no users during the modernization), breaking changes may be applied directly, the database may be rebuilt from scratch, and early migrations may be squashed into one initial migration. **After launch**, breaking changes (rename, type change, drop) follow the **expand → migrate → contract** pattern, so the running app never sees a schema it can't handle.

Alembic doesn't decide the data-access approach (SQLAlchemy vs raw SQL); that's a separate decision.

## Trade-offs accepted

- The effort of writing migrations and downgrades by hand.
- Discipline required: schema is never changed manually in any environment.

## Revisit when

- Migrations need to be shared with non-Python services that can't run Alembic.
