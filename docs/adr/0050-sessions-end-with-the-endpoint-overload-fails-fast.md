# ADR-0050: Database sessions end with the endpoint, and overload fails fast instead of queuing

- **Status:** Accepted
- **Date:** 2026-10-11
- **Decider:** Muhammed Yasin Horasanli
- **Incident:** [the API wedged under overload](../incidents/2026-10-11-api-wedged-under-overload.md)

## Context

- The first load test of production (VM-A) found that the API **stops serving after an overload and stays down for 5–10 minutes**, with all 15 pooled database connections "idle in transaction" and every worker thread waiting for a connection. It recovers only after the backlog of already-accepted requests has been worked off. A reproduction on a throwaway stack showed the same.
- Cause (FastAPI 0.141, SQLAlchemy 2.1, all endpoints synchronous):
  - each request's session comes from a `yield` dependency whose connection is returned only after the response is sent;
  - for synchronous endpoints, FastAPI validates the response in a worker thread, so a request that still holds a connection must queue for one of the 40 worker-thread slots again;
  - under overload those slots are all held by requests waiting up to 30 s for a connection.

  Two pools wait on each other.
- No users yet; the launch is planned, and a link shared on Reddit or X can send this kind of spike.

## Options considered

### A: end the session when the endpoint function returns (`Depends(get_session, scope="function")`)
- ✅ One argument in 11 places. The connection is no longer held while the response is sent.
- ✅ Measured on the reproduction (API only): 33 visitors/s served before collapse → 250/s cleanly (p95 22–42 ms).
- ❌ FastAPI's "function" scope still closes *after* response validation, so the cycle can still form, at random around 250/s and reliably above it.

### B: fail fast (a 2 s wait for a connection instead of 30 s; pool 10 + 10 instead of 5 + 10)
- ✅ Settings only. When the cycle forms, it dissolves within seconds: the API is **healthy again ~25 s after the load** instead of 5–10 minutes.
- ❌ Under overload some visitors get errors quickly instead of slow answers. That's acceptable, since slow answers past 30 s were errors anyway.

### C: release the connection inside the worker thread, before the response is validated
- ✅ Removes the cycle by design: async endpoints that run their database work in a thread with a session opened and closed there, or an equivalent wrapper.
- ❌ Touches every endpoint and its dependencies; more than this fix needs before the AWS work continues.

### D: a watchdog that restarts the API when its health check fails
- ✅ A safety net for any future wedge.
- ❌ Hides problems if it's the only fix; it belongs beside a real fix.

## Decision

**A and B now.**
- **A:** every `Depends(get_session)` becomes `Depends(get_session, scope="function")`.
- **B:** new settings `XCRS_DB_POOL_SIZE` (default 10), `XCRS_DB_MAX_OVERFLOW` (10) and `XCRS_DB_POOL_TIMEOUT` (2 s), passed to `create_engine`.

**C and D are planned for later**, together with throughput work (more processes, caching the skill groups, a shared rate-limit store).

## Trade-offs accepted

- **The cycle can still form under heavy overload;** it now costs about 25 s of errors, not minutes of outage.
- **A slow database query longer than 2 s for a free connection fails** instead of waiting. There are none today: queries take milliseconds.
- **Typed-word matching may hold a connection while it waits for the LLM on VM-B.** That's to be checked and fixed with C.

## Revisit when

- The launch is near, or traffic approaches ~100 requests/s: do C and D.
- The load test on VM-A shows the web server (Nuxt) as the next limit: more web processes.
