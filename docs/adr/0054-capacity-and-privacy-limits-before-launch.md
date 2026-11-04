# ADR-0054: Capacity and privacy limits before the launch

- **Status:** Accepted
- **Date:** 2026-11-04
- **Decider:** Muhammed Yasin Horasanli
- **Source:** security audit run 1 (`NEEDS-VALIDATION.md` leads 2, 4, 5, 18, 20; private), each reproduced with a test before the fix

## Context

- XCRS is about to be shared on LinkedIn and X. A launch post sends a burst of visitors within an hour, mostly anonymous, many typing skills in their own words.
- The hardware stays small: VM-A (8 vCPU, 8 GB) runs the API, the web server and Postgres; VM-B's CPU-only LLM answers **one request at a time** (`OLLAMA_NUM_PARALLEL=1`), shared by skill matching, CV import and the explanations (~12 s each).
- The first security audit found ways in which a few visitors, or one, could slow the site down for everyone, and one way in which the privacy page said more than the code did:
  - **Typed words** (lead 5): the privacy page said the board reaches the server "only when you ask for recommendations", but typed chips are matched as they are added. The LLM's answer is cached per normalized phrase in `catalog.phrase_matches` with no owner and no expiry, so deleting an account can't reach it, and the match endpoint answered `method: "cache"` for a phrase someone had typed before.
  - **Connections** (lead 20): a typed phrase's request held its database connection while it waited for the LLM. Reproduced on a throwaway stack (pool of 2, a stub LLM taking 8 s): two waiting phrases made `/api/v2/health` fail with 500 after the 2 s pool timeout (ADR-0050). In production (pool 10 + 10) about 20 waiting phrases stall every page.
  - **CV parsing** (leads 2, 4): each upload's PDF parse runs in a child process with a 512 MB limit, with nothing stopping several at once, and the API read the child's whole output before checking its size.
  - **Rate-limit map** (lead 18): at 50,000 keys the limiter cleared every bucket, also resetting the bucket of whoever filled the map.
- Constraints: zero budget, no new services (ADR-0035: in-process limits, one API process per VM).

## Options considered

### Typed words
- **A: report a cached answer as `llm`**, like a fresh one. ✅ One line; the response no longer tells anyone that a phrase was typed before. ❌ Answer time still differs (cached: milliseconds; fresh: seconds).
- **B: delete cached answers after 30 days.** ✅ A typed phrase leaves the database within a known time, even though it can't be linked to an account. ❌ Popular phrases cost an LLM call again each month.
- **C: say so on the privacy page.** ✅ The promise matches the code. ❌ None.

### Connections
- **A: end the request's (read-only) transaction before calling the LLM**, so the connection goes back to the pool while it waits. ✅ Small (`MatchStore.release`); the stall above disappears (health 200 in ~10 ms). ❌ One more connection checkout per new phrase.
- **B: at most 2 requests wait for the LLM; other phrases are matched by similarity alone** (not cached, so they get the LLM's answer another time). ✅ Requests stop piling up behind the one-at-a-time LLM; a third phrase answers at once. ❌ Under load some phrases get the weaker similarity match.
- **C: a bigger pool.** ✅ Settings only. ❌ Only moves the limit, and Postgres on VM-A pays for every connection.

### CV parsing
- **A: one PDF parse at a time** (a second upload waits up to 15 s, then gets "try again"), and the child returns at most the text limit + 1 character and 100 hidden pieces of ≤ 300 characters. ✅ Small; memory is bounded at one child. ❌ Two people uploading at the same second: one waits a few seconds, rarely one is asked to retry.
- **B: parse only when the import's turn in the queue comes.** ✅ Cleaner order. ❌ A bigger rewrite of the import flow.
- **C: a memory limit on the container only.** ✅ Protects the VM. ❌ An out-of-memory kill takes the API down for everyone.

### Rate-limit map
- **Drop the least recently used bucket** instead of all of them. Not a real choice; just a fix.

## Decision

Typed words **A + B + C**; connections **A + B** (2 slots, `XCRS_MATCH_LLM_SLOTS`); CV parsing **A**; the rate-limit map drops its least recently used bucket.

The cache expires by age (`created_at`, `XCRS_PHRASE_CACHE_DAYS=30`) rather than by last use: a phrase typed once is gone within 30 days however often others type it, and reading the cache needs no write. Rows past their time are never served, and a background thread in the API deletes them every hour (`PhraseCachePruner`), on VM-A and the AWS copy alike.

## Trade-offs accepted

- Answer time still shows whether a phrase was cached. Hiding it would mean slowing every cached answer down; one unattributed bit per guess, under the per-IP limit, isn't worth that.
- Under load some typed phrases get the similarity match instead of the LLM's. The chip shows what it matched, and the learner can fix it.
- Deferred until there are users, from the same audit: one CV import per user and a quota that survives account deletion (lead 16); explanations taking turns between results and expiring after ~30 min (lead 17); keying the cache by the exact phrase (lead 6).

## Revisit when

- Explanations or matches wait longer than a minute at peak (VM-B's LLM is the bottleneck): two LLM slots on VM-B, a faster model, or the GPU.
- The API runs as more than one process: the slots, the parse lock and the rate limits move to a shared store (ADR-0035).
