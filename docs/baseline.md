# Baseline: the research prototype before modernization

A snapshot of the system as it stood on 2026-09-27 (`main` @ `ee74d82`; the file paths below refer to that branch), taken before any modernization changes. Every claim points to where it comes from, so it can be re-checked later. After each phase, the same metrics get re-measured and compared against this page.

## Data

| Item | Value | Source |
|---|---|---|
| Raw Udemy courses | 973 | `embedding-generation/data/udemy_course_data.csv` |
| Courses after category filter | 453 | `udemy_courses_final.csv` (filter: `embedding-generation/src/util.py:category_matcher`) |
| Roadmap nodes | 1,104 (869 concepts + 235 topics) | `roadmap_nodes_final.csv` |
| Career roles | 10 (hardcoded in 3 places) | `embedding-generation/src/main.py`, `backend/src/main.py:init_data`, frontend |
| Embedding providers | 5 (Google, Voyage, OpenAI, Mistral, Cohere) | `backend/src/models.py` |
| Storage format | CSV; vectors stored as stringified lists | `backend/src/util.py:convert_to_float` |

## Request path

| Characteristic | Value | Source |
|---|---|---|
| Model pipelines per user request | 5, called **sequentially** (each `await`ed in turn), plus 1 `/save_inputs` call | `frontend/server/api/recommend.ts` |
| Embedding API calls per request | 5 (1 per provider, batches of up to 100 user items) | `backend/src/main.py:create_user_embeddings` |
| `gpt-4o` calls per request | up to 20: per provider, 1 for role explanations + 1 per recommended role (≤ 3) for course explanations | `backend/src/recom.py:generate_explanation_for_roles`, `generate_explanation_for_courses` |
| Concurrency | Synchronous SDK calls inside `async def` endpoints block the event loop, so concurrent requests are effectively serialized | `backend/src/main.py:get_recommendations` |
| Per-request overhead | New `RecommendationEngine` per request; it re-reads the OpenAI key file and builds a new client | `backend/src/recom.py:__init__` |
| Observed latency | The UI tells users to expect **1–2 minutes** | `frontend/pages/index.vue` (loading message) |
| LLM output handling | Free-text JSON arrays parsed by hand; a wrong count or invalid JSON silently drops explanations or raises `KeyError` | `backend/src/recom.py` |

## Startup & memory

| Characteristic | Value | Source |
|---|---|---|
| Startup work | Load all CSVs, parse vector strings, compute 5 dense course × concept cosine matrices, compute 2σ / 2.5σ / 3σ thresholds per matrix | `backend/src/main.py:main` |
| Matrix size today | 453 × 869 float64 ≈ 3 MB per provider, ~16 MB for all 5 (small) | derived |
| Matrix growth | O(courses × concepts × providers). At 100K courses × 50K concepts, ≈ 20 GB per provider in float32 | derived |
| Start command | Only `python main.py` works; `uvicorn main:app` never calls `main()`, so the globals are never loaded | `backend/src/main.py` (`if __name__ == "__main__"`) |

## Engineering

| Item | State |
|---|---|
| Tests | None |
| Dependency manifest | None (no `requirements.txt` / `pyproject.toml`); relies on the pre-1.0 `mistralai` API |
| Containerization | None |
| CI/CD | None |
| Secrets | Plain-text files in `embedding-generation/api_keys/` |
| Config | Relative paths, hardcoded production IP in `frontend/server/api/recommend.ts` |
| Persistence of user inputs | JSON files written to `backend/user_inputs/` |
| Logging | A single log file at `backend/log/backend.log` |

## Not yet measured

Measuring these requires the per-provider embedding files and API keys, which aren't in the local checkout. Generating embeddings costs a small amount in API usage.

- [ ] End-to-end latency, p50 and p95
- [ ] Backend startup time
- [ ] Backend resident memory after startup
- [ ] API cost per user request

## New system: first measurements

MacBook Pro M4 Pro, Ollama on the Apple GPU, 2026-09-28/29. **These are not production numbers:** the VMs are CPU-only, and the baselines that count are measured there ([ADR-0007](adr/0007-ollama-qwen3-embedding.md), [ADR-0014](adr/0014-local-first-then-split-by-role.md)). VM numbers are pending.

### Offline (catalog)

| Step | Value |
|---|---|
| Embed the full catalog (1,322 texts, `qwen3-embedding:0.6b`) | 62.8 s |
| Re-run with nothing changed (content hashes) | 0.03 s, 0 texts embedded |
| Threshold statistics + 17,380 top-20 matches | 1.2 s |

### Request path

| Step | Value |
|---|---|
| Embed 10 user phrases | 97 ms |
| Exact threshold scan, phrases × concepts ([ADR-0010](adr/0010-exact-threshold-search-and-candidate-penalty.md)) | ~20 ms (448 ms before parsing each phrase vector once in a materialized CTE) |
| k-NN course search via HNSW | 4 ms |
| Explanation per role (`qwen3.5:9b`, thinking off) | ~3–13 s |
| Explanations per request (3 roles, sequential) | ~21–42 s. Moving them out of the response is [ADR-0018](adr/0018-decoupled-per-role-explanations.md) |
| **Response time with explanations in the background** ([ADR-0018](adr/0018-decoupled-per-role-explanations.md), 2026-09-30) | **0.77 s** (server 685 ms), down from 31.5 s with explanations inline; all 3 roles explained ~29 s later (7.9–12.3 s each) |
| Restart during an explanation job | both roles re-queued on startup and finished; the interrupted one on its 2nd attempt |

### Explanation model, per device (ADR-0020, 2026-09-30)

Same prompt and schema as the app; inputs from 15 synthetic profiles. CPU = Ollama with `num_gpu=0`, 10 threads on the M4 Pro (a best case for the VM).

| Model / device | Median s per role | Decode tok/s | Clean (heuristic grounding checks) |
|---|---|---|---|
| `qwen3.5:9b` / GPU | 8.3 | 38.5 | 80% |
| `qwen3.5:9b` / CPU | 57.9 | 7.1 | 67% |
| `qwen3.5:4b` / GPU | 11.0 | 24.5 | 68% |
| `qwen3.5:4b` / CPU | 33.6 | 10.8 | 73% |

Request-path embedding (10 phrases) while an explanation generates on the **same CPU**: ~100 ms → **~20 s**. That's why embeddings and the explanation LLM run on different VMs.

### Phrase matching (ADR-0022)

50 profile phrases at the 2.5σ threshold: **8 matched nothing** with algorithm 1.0.0 (e.g. "Python", "Spring Boot"); **1** with the fallback (1.1.0).

### Before → after

| Item | Prototype | New system |
|---|---|---|
| Embedding pipelines per request | 5 (one per provider, sequential) | 1 (the active model) |
| External API calls per request | up to 25 (5 embedding + up to 20 `gpt-4o`) | 0 (all local) |
| LLM calls per request | up to 20 | up to 3 (one per recommended role) |
| Data at startup | CSVs parsed into memory, 5 dense matrices | nothing loaded; PostgreSQL + pgvector |
| Tests | none | 37 (+ CI on every push, ADR-0021) |

## Engine v2 vs legacy (2026-10-02)

Measured with `backend/eval/compare_engines.py` (results in `backend/eval/results/compare-engines-*.md`) on the 53 labeled learner profiles of `backend/eval/learner_profiles.yaml`. The legacy engine gets the skills' names as phrases, and its 10 roles are mapped to the new catalog.

| | Legacy engine (research catalog) | Engine v2 (new catalog) |
|---|---|---|
| Roles | 10 (roadmap.sh) | 30, each with 4 levels |
| Top-1 right, profiles the legacy engine can name | 86% | **95%** (98% held-out overall) |
| Top-1 right, all profiles | 51% | **98%** |
| Top-3 contains the expected role, all profiles | 40% | **100%** |
| Scoring latency per profile | 131 ms (embedding, thresholds, course matches) | **2 ms** (skills known; typed text adds matching, ADR-0030) |
| Output | roles + courses + LLM explanations | roles + estimated level + gaps to the next level (resources: next step) |

v2 was calibrated on the same profiles (ADR-0031), so its in-sample numbers flatter it; the cross-validated figure is 98% top-1. Level estimates are weak (about 55% exact from about ten chips) and are shown as estimates.


## First deployment on the VMs (2026-10-02)

Oracle Linux 10 VMs (Xeon E5-2690 v4, AVX2, no GPU), deployed with `deploy/deploy.sh` behind Tailscale (ADR-0040). The LLM is `qwen3.5:4b` on VM-B (24 vCPU), the embedding model on VM-A (8 vCPU).

| Measure | Value |
|---|---|
| Explanation per role, 4B on VM-B (benchmark, 20 cases) | median 12.2 s, p90 13.1 s, 100% clean |
| Explanation per role, 9B on VM-B (same) | median 20.2 s |
| Three explanations for one result, through the app | 38 s |
| Skill matching, three new typed phrases (LLM on VM-B) | 17.1 s (about 5.7 s each) |
| Recommendation once phrases are matched (through Caddy) | 72 ms |
| Database dump (catalog, resources, activity) | 1.4 MB |

## Levels, gaps and resources (2026-10-03, ADR-0041)

On the 15 test profiles of `/dev/profiles` (student to staff, with years of experience) and the 53 calibration profiles:

| | Before (v2.7) | After (v3.0) |
|---|---|---|
| Test: expected role first / exact level / within one | 14 / 4 / – of 15 | 15 / 14 / 15 of 15 |
| Calibration: exact level (no experience given) | 19/53 | 21/53 |
| Gaps of leveled profiles containing Git, debugging, DS, algorithms or fundamentals | most | 0 |
| Most repeated resource across the 15 | Git Tutorials ×6 | System Design Primer ×3 |
