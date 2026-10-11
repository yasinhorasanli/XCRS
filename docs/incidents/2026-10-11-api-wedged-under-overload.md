# Incident: the API wedged under overload and stayed down for minutes (2026-10-11)

Blameless write-up. **Fix:** [ADR-0050](../adr/0050-sessions-end-with-the-endpoint-overload-fails-fast.md). **Status:** fixed; production numbers after the fix are at the end.

## Summary

The first load test of production found that a burst of traffic made the API **stop answering, and stay stopped for 5–10 minutes after the traffic ended**: two internal pools (worker threads and database connections) ended up waiting on each other. No users were affected; XCRS has none yet. The fix (sessions end with the endpoint, and a 2-second limit on waiting for a connection) raised the API's overload point roughly eightfold in the reproduction, and turned the multi-minute outage into about 25 seconds of fast errors.

## Impact

- Production (VM-A) was unavailable from 00:07 to 00:14 (UTC+3) on 2026-10-11, during and after the test, until the API was restarted by hand.
- No real users and no data loss. The test's own recommendations stayed in the database; their explanations were marked `disabled`, so none reached the LLM.
- Had it happened after launch, every traffic spike above ~40 visitors a second would have meant ~10 minutes of downtime, with nothing restarting the API automatically.

## How it was found

- **Load test of production VM-A:**
  - k6 on VM-B over the tailnet, constant arrival rates, one level at a time, with a safety stop at p95 > 5 s or > 5% errors;
  - the LLM and rate limits switched off for the duration; a verified database backup first.
- **The "browse" scenario** (home page + skill search + skill groups per visitor): clean up to 20 visitors/s (60 requests/s, p95 124 ms). At 40/s, p95 rose to 30.5 s with 6% errors, and the test stopped itself.
- **Afterwards nothing answered,** not even a health request sent from inside the API container. VM-A was idle (load average 0.22).

## Investigation

1. **Database:**
   - all 15 connections in the pool (SQLAlchemy's default 5 + 10 overflow) were `idle in transaction`, wait event `ClientRead`, for 5½ minutes;
   - their last statements were catalog `SELECT`s.

   So the API had checked out every connection and wasn't using any of them.
2. **Reproduction:** a throwaway stack on a laptop (fresh Postgres, the same API image, the catalog imported), the same overload against the API alone. **Reproduced on the first try:** 33 visitors/s served out of 250/s asked, then the same 15 stuck connections and no health answer.
3. **Thread dump of the stuck process** (py-spy, from a helper container sharing the API's process namespace):
   - 40 of the server's worker threads were inside SQLAlchemy's pool, waiting for a connection;
   - 10 threads were idle, and the event loop was idle;
   - **no thread held a connection.** The 15 connections belonged to requests parked in the event loop.
4. **FastAPI's source** (0.141, read inside the same image) explained the parking:
   - **When the connection comes back:** a `yield` dependency (our `get_session`) is torn down only after the response has been sent.
   - **Why a thread is needed again:** for a synchronous endpoint, response validation runs in the thread pool (`run_in_threadpool(field.validate, …)`).
   - **The cycle:** after its query, a request gives its thread back but keeps its connection, then queues for a thread again, behind the backlog of earlier requests. Each of those takes a thread and waits up to 30 s for a connection that the queued requests hold. Every 30 s, 40 waiters give up and their threads go to more backlog.
5. **Leak or starvation?** Watching the stuck API with no traffic: it recovered by itself after about 9–10 minutes. So it's **starvation that drains** at about (backlog ÷ 40) × 30 s, not a permanent leak.
6. **First fix, `Depends(get_session, scope="function")`:**
   - 250 visitors/s: 0% errors, p95 22–42 ms, healthy at once.
   - 500/s: wedged again for 5–7 minutes.
   - FastAPI's "function" scope still closes *after* response validation (it wraps the whole handler and only excludes sending), so the cycle remained, at higher load.
7. **Fail fast:** a 2 s wait for a connection (instead of 30 s) and a pool of 10 + 10.
   - When the cycle forms, it dissolves quickly: healthy again 24–28 s after the load at 250, 500 and 1,000 visitors/s.
   - Repeated runs at 250/s, warmed up: mostly clean, occasionally a short burst of errors. The cycle forms at random near that rate.
8. **Regression check:** all 134 backend tests pass against a throwaway database.

## Root cause

**Two bounded resources acquired in an order that can close a cycle.** A request holds a database connection while it waits for a worker thread, and other requests hold worker threads while they wait for a database connection.
- **What made it possible:** synchronous endpoints, plus FastAPI validating their responses in the thread pool, plus a session that lives until the response is sent.
- **What made it last for minutes:** SQLAlchemy's 30 s pool timeout, a backlog of hundreds of already-accepted requests, and no automatic restart. Docker's health check marks the container unhealthy but doesn't restart it.

## Fix (ADR-0050)

- The session ends when the endpoint function returns (`scope="function"`, 11 places).
- `XCRS_DB_POOL_SIZE` = 10, `XCRS_DB_MAX_OVERFLOW` = 10, `XCRS_DB_POOL_TIMEOUT` = 2 s.
- **Reproduction, API only, 30 s per level:**

| | Before | After |
|---|---|---|
| 250 visitors/s (500 requests/s) | 33/s served, p99 30 s, then ~10 min down | 250/s served, p95 22–42 ms (occasionally a short error burst) |
| Overload (500–1,000/s) | down for 5–10 min | errors during the overload, healthy again ~25 s after |

## What went well

- The test had a safety stop and a verified backup, and production had no users.
- Production's evidence (pool states, the exact error) was captured before restarting.
- The bug reproduced on a throwaway stack, so the investigation never touched production again.
- Each hypothesis was tested by an experiment that could tell it apart from the others (thread dump, watching without traffic, repeated runs).

## What went wrong

- The first reading of the browse numbers blamed single-process CPU limits; the reproduction showed the starvation was the limit. Corrected during the investigation.
- The first fix looked complete at 250/s and failed at 500/s; testing beyond the first good result caught it.
- The test's orchestration lost two scenarios to a shell quirk (zsh doesn't word-split unquoted variables) and ran them against the stuck API; the runner is now a bash script.

## Follow-ups

- [ ] The structural fix: release the connection inside the worker thread, before the response is validated, so the cycle can't form (ADR-0050, option C).
- [ ] A watchdog that restarts the API when its health check fails (option D).
- [ ] Check whether typed-word matching holds a connection while it waits for the LLM on VM-B.
- [ ] Throughput: more API and web processes, cache the skill groups (they change only with the catalog), a shared rate-limit store.
- [x] Re-run the load test on VM-A after deploying the fix (results below).

## Results on VM-A after the fix

Deployed as 1230b2f; the same test again (numbers in [`baseline.md`](../baseline.md#load-test-of-production-vm-a-2026-10-11-launch-plan-step-3-adr-0050)):

| Scenario | Clean up to | At the first failing level | Healthy again after |
|---|---|---|---|
| browse | 20 visitors/s (60 req/s), p95 144 ms | 40/s: 10.7% errors | **19 s** (before: down until restarted) |
| recommend | 10/s, p95 144 ms | 20/s: 37% errors | **3 s** |
| match (similarity) | 2/s, p95 420 ms | 10/s: 41% errors | **2 s** |

- **The fix delivered what it was for:** after every overload, production recovered by itself within seconds.
- **It didn't raise VM-A's capacity for browsing.** There the next limits are the single web and API processes (157% and 120% CPU, VM-A half idle). For recommendations it's role scoring in the one API process, and for typed phrases it's the embedding model, which takes all 8 cores.
- Those are the throughput follow-ups, and they are now measured.
