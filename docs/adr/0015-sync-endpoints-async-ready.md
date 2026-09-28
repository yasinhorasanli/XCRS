# ADR-0015: Synchronous endpoints now, structured to switch to async later

- **Status:** Accepted
- **Date:** 2026-09-29
- **Decider:** Muhammed Yasin Horasanli

## Context

The request path is I/O-bound: an embedding call to Ollama, a few PostgreSQL queries, and LLM calls for explanations (seconds each).

The prototype declared its FastAPI endpoints `async def` but called synchronous SDKs inside them. That blocked the event loop, so concurrent requests were effectively serialized ([baseline](../baseline.md)).

The data layer uses SQLAlchemy 2 on psycopg 3 ([ADR-0012](0012-sqlalchemy-orm-with-raw-sql-repository.md)). Both support sync and async. XCRS has no users during the modernization.

## Options considered

### Option A — Synchronous endpoints (`def`)
- ✅ Simple; the existing sync SQLAlchemy and httpx code works as-is
- ✅ Correct concurrency: FastAPI runs `def` endpoints in a thread pool (~40 threads by default)
- ❌ Each in-flight request holds a thread while waiting on I/O; doesn't scale to hundreds of concurrent slow LLM calls

### Option B — Async end to end (`async def`, async SQLAlchemy, `httpx.AsyncClient`)
- ✅ One event loop can interleave many waiting requests; parallel LLM calls via `asyncio.gather`
- ✅ Natural fit for streaming LLM output later
- ❌ More complex: async sessions, async testing, every I/O call must be awaited (one missed sync call blocks everything, the prototype's bug)

## Decision

**Option A now, designed to switch.**
- All I/O sits behind adapters: repository, embedder, explainer.
- The recommendation logic is pure functions that do no I/O and are therefore identical in sync and async code.
- Switching means changing the adapters and the service/router signatures to `async`, with the domain untouched.

## Trade-offs accepted

- Thread-pool concurrency limits for now.
- Explanation calls for several roles run sequentially unless parallelized with threads.

## Revisit when

- Streaming LLM explanations to the UI is wanted.
- Concurrent requests regularly exhaust the thread pool, or p95 latency is dominated by queued requests.
