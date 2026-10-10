# ADR-0018: Explanations generated after the response, one LLM call per role

- **Status:** Accepted
- **Date:** 2026-09-29
- **Decider:** Muhammed Yasin Horasanli

## Context

Each recommended role gets an explanation from a local LLM: why the role fits, and why each course helps. Today the service makes one call per role, sequentially, inside `POST /api/v1/recommendations`, so the response waits for all of them.

- **Measured on the Mac GPU** (`qwen3.5:9b`, thinking off): ~3–13 s per role, **~21–42 s per request** (three roles). The rest of the request (embedding, threshold scan, course selection) is well under a second.
- **Estimated for the production CPU VM** (16 vCPU/16 GB, no GPU yet, [ADR-0014](0014-local-first-then-split-by-role.md)): CPU generation is limited by memory bandwidth. A 9B model at Q4 is ~6 GB of weights, so roughly 3–8 tokens/s. With ~300 output tokens and ~1–1.5k prompt tokens per role, that is **~1–2.5 min per role, several minutes per request**. A smaller model roughly halves this. These are estimates; the model choice waits for a benchmark (ADR-0020).
- The recommendation itself (roles + courses) is useful without explanations, and explanations already degrade to "none" on failure.
- The request, its roles and its courses are already persisted with a `request_id`, and the role and course rows have explanation columns ([ADR-0013](0013-user-activity-hybrid-then-normalized.md)).
- XCRS has no users during the modernization, so losing in-flight work on a restart is acceptable for now.

Two questions: **how explanations reach the client**, and **how many LLM calls produce them**.

## Options considered

### Delivery

#### Option A — Inline, sequential (status quo)
- ✅ Simplest; one request, one response
- ❌ Nothing appears until every explanation is done: ~30 s on the Mac GPU, minutes on the CPU VM
- ❌ A slow LLM makes the whole recommendation slow, although the recommendation itself is ready in under a second

#### Option B — Inline, per-role calls in parallel
- ✅ Small change (thread pool); on a GPU, Ollama batches parallel requests, so the total gets close to the slowest role
- ❌ Little gain on CPU: parallel calls share the same memory bandwidth
- ❌ The response still waits for the slowest role

#### Option C — Decoupled: respond first, explain in the background
- ✅ `POST` returns roles and courses in about a second; explanations fill in role by role
- ✅ Works with any LLM speed, including the CPU VM; one role failing leaves the others untouched
- ✅ The LLM work becomes a queue, which is where a single GPU belongs later
- ❌ More moving parts: a status per role, a background worker, a read endpoint, polling in the UI
- ❌ In-process background work is lost on a restart (mitigated below)

#### Option D — Server-sent events (SSE) streaming in one connection
- ✅ Best perceived speed: text appears token by token
- ❌ Holds a connection open for minutes on CPU; the Nuxt server route must pass the stream through; harder to retry or cache
- ❌ Doesn't remove the need to persist explanations; it's a presentation layer, not a job model

### Call shape

#### One call per role (status quo)
- ✅ Small JSON answer per call, which small models get right more reliably
- ✅ Failure isolation, and a natural unit for role-by-role fill-in
- ❌ The system prompt is processed once per role (a few hundred tokens each)

#### One call for all roles
- ✅ Saves the repeated system-prompt processing
- ❌ Output tokens, the dominant cost, are the same, so almost no time is saved
- ❌ One large JSON answer: one mistake loses every explanation; nothing to show until all are done

## Decision

**Option C with one LLM call per role.**

- `POST /api/v1/recommendations` saves the recommendation, returns it immediately with each role's explanation status `pending`, and queues one explanation job per role.
- An **in-process background worker** runs the jobs, one at a time by default (a CPU LLM gains nothing from concurrency), and writes each result to the role and course rows it belongs to: status `done`, or `failed` with no explanation.
- `GET /api/v1/recommendations/{request_id}` returns the current state; the UI polls it until no role is `pending`.
- The database rows are the source of truth, not the in-memory queue: on startup, the worker re-queues roles still marked `pending`, so a restart delays explanations instead of losing them.
- These are additive changes to the v1 contract ([ADR-0016](0016-versioned-structured-recommendation-api.md)). Endpoints stay synchronous ([ADR-0015](0015-sync-endpoints-async-ready.md)).

Main reason: the recommendation is ready in under a second, but the explanation LLM takes seconds on a GPU and minutes on the CPU VM. Separating them makes the page fast regardless of where the LLM runs, and turns LLM work into a queue that can move to the GPU later.

Tone and length stay as they are (≤ 60 words per role, ≤ 45 per course, grounded prompt) until the prototype-vs-new quality comparison. The production model is decided separately after a CPU benchmark (ADR-0020). The prompt and output contract of the explainer is recorded in ADR-0019.

## Trade-offs accepted

- **More moving parts** than a single response: per-role status, a worker, a read endpoint, UI polling. Mitigated by keeping the worker in-process and using the existing tables.
- **Polling instead of push.** Simple and proxy-friendly; SSE can be added on top later without changing the job model.
- **Single-process only.** Two API processes would each run a worker and could pick up the same role. Fine while one process serves everything (ADR-0014).
- **Not durable mid-job.** A role being generated during a restart is redone from the start (re-queued as `pending`).

## Revisit when

- More than one API process or worker is needed → claim jobs with a PostgreSQL job queue (`SELECT … FOR UPDATE SKIP LOCKED`), still without new infrastructure.
- Token-by-token streaming is wanted in the UI → add SSE on top of the same jobs.
- Explanations on the production hardware reliably finish within a few seconds per request (e.g. a GPU) → consider whether inline delivery would be simpler again.
