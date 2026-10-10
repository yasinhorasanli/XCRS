# ADR-0017: Layered backend with a pure domain core

- **Status:** Accepted
- **Date:** 2026-09-29
- **Decider:** Muhammed Yasin Horasanli

## Context

The prototype mixes HTTP handling, module-level global state, pandas scoring, database-like CSV access and OpenAI calls in `backend/src/main.py` and `recom.py`. The recommendation algorithm (weighted concept coverage → sigmoid → top roles; course selection with the disliked penalty) can only be tested by running everything.

The new backend has about three use cases (recommend, feedback, admin/ingestion jobs), one database, and I/O adapters that already exist or are planned: `Embedder` ([ADR-0006](0006-own-embedding-interface.md)), the vector repository ([ADR-0012](0012-sqlalchemy-orm-with-raw-sql-repository.md)), and an explainer. The endpoints are synchronous for now but must be able to switch to async ([ADR-0015](0015-sync-endpoints-async-ready.md)).

## Options considered

### Option A — Flat (like the prototype)
- ✅ Least code
- ❌ The algorithm can't be tested without a database and models; I/O is spread everywhere

### Option B — Layered with a pure domain core
`api/` (HTTP only) → `services/` (the order of steps) → `domain/` (pure functions, no I/O) plus adapters (`repository/`, `embeddings/`, `explain/`)
- ✅ The algorithm is unit-testable in milliseconds, which makes the prototype-parity comparison exact
- ✅ A switch to async touches adapters and signatures only; the domain is unchanged
- ❌ Moderate structure to maintain

### Option C — Full clean architecture / hexagonal / DDD
Formal ports for every dependency, domain entities separate from ORM models with mapping code, one class per use case with input/output objects, a DI container.
- ✅ Maximum decoupling; scales to many use cases and teams
- ❌ For ~3 use cases and one database, much of the code would only map data between layers (ORM ↔ entity ↔ DTO) and every field change touches several classes

## Decision

**Option B.**
- `domain/` imports nothing from FastAPI, SQLAlchemy or httpx.
- Adapters are injected into services via FastAPI `Depends`.
- Formal interfaces (`Protocol`) are added where they pay off (the embedder and explainer already; the repository when a fake is needed in tests).

## Trade-offs accepted

- Domain functions use simple dataclasses and the ORM models are reused where convenient. That's less isolation than C, accepted for less ceremony.

## Revisit when

- Use cases multiply (e.g. user accounts, a roadmap editor, an ingestion pipeline service) or domain rules grow complex enough that C's ports and entities pay for themselves. B can grow into C step by step.
