# ADR-0020: `qwen3.5:4b` for explanations on the CPU VM, `qwen3.5:9b` on GPUs; embeddings kept off the LLM's machine

- **Status:** Accepted (delegated: made on 2026-09-30 from the benchmark below, while the decider asked for the remaining work to be finished without questions; pending the decider's review)
- **Date:** 2026-09-30
- **Decider:** Muhammed Yasin Horasanli

## Context

Explanations run in a background worker ([ADR-0018](0018-decoupled-per-role-explanations.md)) through a LangChain chain on any OpenAI-compatible server ([ADR-0019](0019-explanation-layer-on-langchain.md)). Production starts on CPU-only VMs (16 vCPU / 16 GB and 8 vCPU / 8 GB); a GPU comes later ([ADR-0014](0014-local-first-then-split-by-role.md)). ADR-0018 estimated 1–2.5 min per role for a 9B model on CPU and left the model choice to a measurement.

**Benchmark** (`backend/eval/bench_explainer.py`; raw results in `backend/eval/results/bench-explainer-20260930-0237-MacBook-Pro.json`). It uses the app's own prompt and output schema on explanation inputs produced by the real algorithm from 15 synthetic profiles (`eval/profiles.json`). CPU runs used Ollama with `num_gpu=0` and 10 threads on the MacBook M4 Pro's performance cores.

| Model / device | Cases | Median s / role | p90 s | Output tokens | Decode tok/s | Prefill tok/s | Clean (all cases) | Clean (same 15 first-role cases) |
|---|---|---|---|---|---|---|---|---|
| `qwen3.5:9b` / GPU | 40 | 8.3 | 9.6 | 240 | 38.5 | 412 | 80% | 80% |
| `qwen3.5:9b` / CPU | 15 | **57.9** | 68.5 | 244 | 7.1 | 40 | 67% | 67% |
| `qwen3.5:4b` / GPU | 40 | 11.0 | 16.4 | 214 | 24.5 | 392 | 68% | 73% |
| `qwen3.5:4b` / CPU | 15 | **33.6** | 47.3 | 201 | 10.8 | 62 | 73% | 73% |

*(Corrected 2026-10-01: the prefill column is unreliable. Ollama reuses the cached system prompt between calls, so its prompt timings don't measure a cold prefill; single runs showed over 5,000 tok/s. No part of the decision used it; the benchmark no longer reports it. Readable summary: `backend/eval/results/bench-explainer-20260930-0237-MacBook-Pro.md`.)*

"Clean" means no automatic grounding flag (invented curiosity, a matched concept attributed to the learner, unknown or missing courses, over-length, mentions of scores). The flags are heuristics that mark explanations to read, not proof. No run produced invalid JSON or failed.

**Request-path embedding latency while an explanation is generating on the same machine** (10 phrases, `qwen3-embedding:0.6b`):

| Explanation LLM on | Embedding idle | Embedding during generation |
|---|---|---|
| GPU (9B / 4B) | ~100 ms | 151 / 302 ms |
| **CPU (9B / 4B)** | ~80–107 ms | **~20,300 ms** |

Findings:
1. **Grounding differences between the models are within run-to-run noise.** The same 9B model scored 80% on GPU and 67% on CPU for the same 15 cases. Both models still sometimes attribute a matched concept to the learner (e.g. "Argo CD" for someone who wrote "Kubernetes").
2. **On CPU, the 4B is 1.7× faster** (33.6 vs 57.9 s per role, about 1.7 vs 2.9 min per three-role request). On GPU, the 9B is faster (8.3 vs 11.0 s).
3. **On a CPU machine, one explanation job makes the request-path embedding ~200× slower.** Without separation, the fast response of ADR-0018 (0.77 s) would become ~20 s whenever a job is running.
4. The Mac's CPU is a best case (high memory bandwidth, fast cores); the VM will likely be slower. The same script runs there: `uv run python eval/bench_explainer.py --devices cpu --threads 16`.
5. The run spanned macOS idle-sleep cycles, which inflate some individual times (p90 / max); medians are robust to that.

## Options considered

### Model on the CPU VM
- **`qwen3.5:9b`:** ✅ the model the prompt was tuned on. ❌ ~1 min per role on the best-case CPU; no measurable grounding advantage in this benchmark; 6.6 GB of RAM.
- **`qwen3.5:4b`:** ✅ 1.7× faster; grounding within noise of the 9B; 3.4 GB of RAM. ❌ A smaller model may fail in ways 15 cases don't show.
- **A hosted API until the GPU arrives:** ❌ zero budget, data leaves the server, and it doesn't serve the local-LLM goal ([ADR-0005](0005-local-embedding-models.md) reasoning).

### Where the embedding model runs
- **Same Ollama as the LLM (as ADR-0014 planned):** ❌ finding 3.
- **Limit the LLM's threads** (e.g. 12 of 16) so embeddings keep some cores: ✅ one machine. ❌ Slows every explanation; contention for memory bandwidth remains.
- **Embeddings on VM-A (with the API and PostgreSQL), the LLM alone on VM-B:** ✅ full isolation; the 0.6B embedding model needs ~1 GB and ran in ~100 ms on CPU. ❌ VM-A (8 GB) gets one more process.

## Decision

1. **CPU VM: `qwen3.5:4b`** for explanations (`XCRS_LLM_MODEL=qwen3.5:4b`). **GPU (the Mac now, the server's GPU later): `qwen3.5:9b`**, the default in `config.py`. Switching is configuration only.
2. **Embeddings run on VM-A, the explanation LLM alone on VM-B**, each with its own Ollama. `XCRS_EMBEDDING_BASE_URL` and `XCRS_LLM_BASE_URL` already point to different hosts. This refines ADR-0014's "VM-B runs Ollama" and makes VM-B the explanation machine.
3. **Output cap:** `XCRS_LLM_MAX_TOKENS=700` (answers use 200–250 tokens). Insurance against a known failure mode of small models in JSON mode (endless whitespace), which would otherwise hold the single worker until the timeout. A cut-off answer degrades to "no explanation".
4. **Re-run the benchmark on VM-B** before launch; the decision stands if the 4B stays under ~1.5 min per role there.

## Trade-offs accepted

- Explanations on the CPU VM take about 1.5–2 minutes per request (more on the VM). Acceptable only because they arrive after the recommendation (ADR-0018).
- The 4B was chosen on speed with grounding "within noise" on 15 synthetic cases; a larger evaluation could separate the models.
- Two Ollama instances to run and keep on the same model tags.
- Mac CPU numbers stand in for the VM until it is measured.

## Revisit when

- The VM benchmark shows more than ~1.5 min per role for the 4B → consider `qwen3.5:2b`, fewer roles explained eagerly, or explanations on demand.
- The GPU arrives → 9B (or larger) on the GPU; re-run the benchmark, including contention.
- Real feedback (thumbs, ADR-0023) shows explanation quality problems that a larger model fixes.
- The grounding heuristics are replaced by a proper evaluation (e.g. an LLM judge with human-checked labels).
