# ADR-0039: Retire the classic engine: its code, API, tables and research data are removed after an archive

- **Status:** Accepted
- **Date:** 2026-10-02
- **Decider:** Muhammed Yasin Horasanli
- **Note (2026-10-10):** `main` now holds the modernized system ([ADR-0049](0049-one-trunk-main-with-protection.md)); the research version is at the tag `research-prototype` (and `Zenodo-v1.0`), not on `main`.

## Context

- Engine v2 became the main site (ADR-0037); the classic engine stayed at `/classic` and `/api/v1` until a decision on its removal.
- The classic engine carried its own catalog (10 roadmap.sh roles, 1,104 nodes, 453 Udemy courses in `data/research-2024`), its vectors (node and course embeddings, 17,380 precomputed matches), its activity (32 requests, 88 roles, 264 courses, 10 feedback rows) and about a third of the backend code (matching, scoring, selection, suggestions, its explainer and worker).
- Keeping both meant two catalogs to explain, two APIs to rate-limit and deploy, and tests for code that no longer serves anyone. The deployment (ADR-0034) already left the research data out.
- There are no users yet (modernization), and the research version is preserved on `main` and in the Zenodo release.

## Options considered

### Option A: remove it now, after an archive (chosen)
- ✅ One engine, one API, one catalog: less code to secure, deploy and explain.
- ✅ Data that can't be rebuilt is archived first.
- ❌ The side-by-side comparison with the classic engine can't be rerun from this branch (`compare_engines.py` goes; its results stay in `docs/baseline.md`).

### Option B: keep it at `/classic`
- ✅ Old results stay reachable; comparisons can be rerun.
- ❌ Two engines to maintain, and the research data would have to be deployed or the page breaks.

## Decision

1. **Archive first:** a verified dump of the dev database (`backups/archive/xcrs-before-classic-removal-20261002.dump`, kept outside the 14-dump rotation) and a JSON export of the classic activity (requests, roles, courses, feedback, plus role and course names).
2. **Migration 0010** drops the ten classic tables. `embedding_models` stays (catalog skill embeddings use it). The downgrade recreates the tables empty, exactly as at 0009, so the migration chain still runs both ways.
3. **Code removed:** `/api/v1` (recommendations, feedback, knowledge units), the classic domain, explainer and worker, the research importer and catalog embedder, `retrieval.py`, the `import-research-data`, `embed-catalog`, `search` and `search-courses` commands, `data/research-2024`, and the classic pages and components. `/classic/**` redirects to `/` (301).
4. **Kept and ported:** the explainer benchmark now runs on v2 facts and the v2 prompt (needed for the 4B vs 9B decision on the new LLM VM); the health check moves to `/api/v2/health`.

## Trade-offs accepted

- Classic results (`/classic/results/{id}`) no longer open; the rows are only in the archive. With no users, that loses nothing.
- Prototype comparisons need `main` (or the Zenodo release) from now on.
- The ADRs of the classic engine (0008–0010, 0013, 0016, 0018, 0022, 0023) stay as history; their patterns (per-model vectors, rows as the job queue, versioned API, board input) live on in engine v2.

## Revisit when

- Never for the classic engine itself. If the research comparison is needed again, run it from `main` or the Zenodo release against a restore of the archived dump (`scripts/db-restore.sh`).
