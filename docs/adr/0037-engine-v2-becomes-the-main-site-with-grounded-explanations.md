# ADR-0037: Engine v2 becomes the main site, with grounded LLM explanations per role; the classic engine moves to /classic

- **Status:** Accepted
- **Date:** 2026-10-02
- **Decider:** Muhammed Yasin Horasanli

## Context

- Engine v2 (ADR-0029–0033) beats the legacy engine on the labeled profiles: 95% vs 86% first role where both can name it, 98% vs 51% overall (`docs/baseline.md`). It also gives levels, gaps and resources.
- **v2 had no explanations;** the legacy engine has them (ADR-0018, ADR-0019): background jobs, one LLM call per role, grounded in the algorithm's facts.
- **The decider asked** to make `/v2` the main site and add LLM explanations.
- **The legacy engine's data** (its catalog and 31+ requests with feedback) is history; activity data can't be rebuilt.

## Options considered

### The legacy engine
- **Remove it now** (code, pages, tables). ✅ Less code. ❌ Irreversible, and it loses the side-by-side comparison; deserves its own decision with a backup.
- **Move it to `/classic` and keep it working.** ✅ Reversible; links and data intact; removal stays a separate, deliberate step.

### v2 explanations
- **Synchronous, inside the request.** ❌ Seconds per role on CPU (ADR-0018 already rejected this).
- **The ADR-0018 pattern again:** one job row per recommended role, a background worker, polling. ✅ Proven; restart-safe; one LLM call per role.

## Decision

1. **Routes:**
   - The v2 board is `/`, its results `/results/{id}`.
   - The legacy engine moves to `/classic` and `/classic/results/{id}`.
   - Old `/v2` links redirect permanently.
   - The API keeps `/api/v1` (legacy) and `/api/v2` (new).
2. **Explanations for v2 roles:** one row per recommended role in `explanations_v2` (migration 0009) is the job.
   - **Facts:** `input` holds exactly what the LLM may use:
     - the role and its summary;
     - the learner's own words with the skills they were read as and how the learner feels about them;
     - coverage *in words* (little, some, much; no percentages);
     - the estimated or starting level;
     - the first gaps, and the suggested resources.

     Empty fields are left out.
   - **Prompt (`explain-role-v2.1`, LangChain, structured output):** answers `explanation` (2–4 sentences) and `next_step` (1–2 sentences naming a suggested resource). Its rules: attribute everything to what the learner wrote, never turn "didn't enjoy" or "curious" into knowledge, stay modest when the evidence is thin, no scores.
   - **Worker:** `ExplanationWorkerV2`, started with the API, re-queues `pending` jobs on startup and bounds attempts. `GET /api/v2/recommendations/{id}` merges each role's status and text; the results page polls while any is pending.
3. **Removing the legacy engine** (pages, `/api/v1`, its tables, the research catalog) is a later, separate step, after a backup and a decision on keeping its activity history.

## Trade-offs accepted

- Two engines' code stays in the repository for now.
- Explanations take seconds per role (about 4 s on the Mac's GPU with 9B; more on the CPU VM). The page shows the result at once and the explanations as they arrive.
- The LLM can still paraphrase loosely (in the first live run, "REST API design" became "designing interfaces"); the facts are shown beside the text, and the page says it can be imperfect.

## Revisit when

- The legacy engine has had no visits for a while, or the comparison is no longer needed → remove it (backup first).
- Explanations measurably drift from the facts → add a grounding check like the v1 benchmark's (`eval/common.grounding_flags`).
