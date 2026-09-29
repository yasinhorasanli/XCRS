# Baseline: the research prototype before modernization

A snapshot of the system as it stood on 2026-09-27 (`main` @ `ee74d82`), taken before any modernization changes. Every claim points to where it comes from, so it can be re-checked later. After each phase, the same metrics get re-measured and compared against this page.

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

### Before → after

| Item | Prototype | New system |
|---|---|---|
| Embedding pipelines per request | 5 (one per provider, sequential) | 1 (the active model) |
| External API calls per request | up to 25 (5 embedding + up to 20 `gpt-4o`) | 0 (all local) |
| LLM calls per request | up to 20 | up to 3 (one per recommended role) |
| Data at startup | CSVs parsed into memory, 5 dense matrices | nothing loaded; PostgreSQL + pgvector |
| Tests | none | 24 |
