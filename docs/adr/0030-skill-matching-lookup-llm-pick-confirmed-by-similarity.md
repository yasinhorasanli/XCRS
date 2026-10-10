# ADR-0030: Skill matching: name lookup, then the LLM picks from the catalog, confirmed by embedding similarity

- **Status:** Accepted
- **Date:** 2026-10-02
- **Decider:** Muhammed Yasin Horasanli

## Context

- ADR-0029 decided the input (catalog picker plus free text) and that the free-text matching method and its thresholds would be chosen by measurement on an evaluation set.
- **Evaluation set:** `backend/eval/skill_matching.yaml`, 349 phrases with the skills they should match. It covers exact names, aliases, typos, descriptions, tools missing from the catalog, several skills in one phrase, vague phrases, course titles, and 25 phrases that must match nothing. Claude drafted it; the decider reviewed a random 50 and found no errors (PR #11).
- **Method** (`backend/eval/bench_skill_matching.py`):
  - Thresholds are chosen on half the cases (stratified by kind) and measured on the other half, 5 × 2-fold.
  - "Fully right" means every expected skill is matched and nothing wrong is.
  - Models: `qwen3-embedding:0.6b` for embeddings; `qwen3.5:9b` / `qwen3.5:4b` as LLMs via Ollama.
  - Results: `eval/results/skill-matching-20261002-0012-*` (first run, with CPU latency) and `…-0036-*` (all pipelines).
- **Results (held-out):**

| Pipeline | Fully right | Precision | Recall | F1 | Rejects non-skills |
|---|---|---|---|---|---|
| Embed raw phrase | 62.2% | 79.1% | 62.4% | | 96% |
| Embed instruction + phrase | 65.6% | 80.4% | 66.2% | | 88% |
| Embed LLM definition | 60.0–64.9% | ~82% | ~64% | | 96–100% |
| Lookup → embed instruction + phrase | 74.2% | 85.0% | 74.5% | 0.79 | 88% |
| Lookup → embed LLM definition | 69.5–72.6% | ~86% | ~73% | | 96–100% |
| Lookup → LLM picks (prompt 1, 9B) | 84.0% | 68.9% | 91.3% | 0.78 | 100% |
| Lookup → LLM picks (prompt 2, 9B) | 82.5% | 81.8% | 94.9% | 0.88 | 100% |
| Lookup → LLM picks (prompt 2, 4B) | 80.8% | 85.6% | 93.0% | 0.89 | 84% |
| **Lookup → LLM picks (prompt 2, 9B), confirmed by similarity ≥ 0.40** | **84.0%** | **87.1%** | 92.6% | **0.90** | **100%** |
| **Lookup → LLM picks (prompt 2, 4B), confirmed by similarity ≥ 0.37** | 82.1% | **88.1%** | 91.0% | **0.90** | **100%** |

- **Latency per new phrase:**
  - Lookup: 0.3 ms. Embedding: 12 ms.
  - LLM pick: 1.8 s (9B, Mac GPU); 5.4 s median (4B, CPU only, M4 12 threads). The VM will likely be slower.
  - LLM definition: 1.6 s (9B, GPU); 3.0 s (4B, CPU).
- **Findings:**
  - Embedding an LLM-written definition (the decider's suggestion in ADR-0029) did not beat embedding the phrase itself. The LLM helps when it *chooses* skills from the catalog, not when it rewrites the phrase.
  - Prompt 1 over-picked for vague input ("ML" → 28 skills) and returned nothing for tools missing from the catalog (Jira, BigQuery). Prompt 2 ranks its picks, caps them at 5 and maps tools to their skill. Its remaining "errors" are mostly neighbouring skills the set doesn't list as acceptable ("neural networks" also → machine-learning fundamentals).
  - 4B picked skills for 4 of 25 non-skills ("carpentry" → git, linux). Requiring a small embedding similarity removed all four at almost no cost in recall.
  - The best embedding "window" was always 0: for embeddings alone, the rule is simply "the best skill, if similar enough".

## Options considered

- **Lookup → embedding only (no LLM).** ✅ Fast (ms), no LLM dependency. ❌ 74% fully right; weak on tools, several skills in one phrase and vague input.
- **Lookup → LLM definition → embedding.** ✅ Symmetric comparison. ❌ Measured no better than embedding the phrase; an LLM call for nothing.
- **Lookup → LLM picks from the catalog list.** ✅ Best recall (91–95%). ❌ 4B also picks for some non-skills; slower (a ~3k-token prompt).
- **Lookup → LLM picks, confirmed by embedding similarity.** ✅ Best F1 (0.90) and precision (87–88%), rejects every non-skill, with 4B close to 9B. ❌ Two models per new phrase; seconds of CPU latency for phrases the lookup can't resolve.

## Decision

1. **Matching order for free text:**
   1. The **catalog picker** gives skill ids directly.
   2. **Name lookup:** catalog names and their parts, slugs and O\*NET names; typo-tolerant; lists split ("React and TypeScript").
   3. **The LLM picks skills from the catalog list** with prompt 2: ranked, at most 5, tools mapped to their skill, engineering practices counted as skills.
   4. **Each pick is kept only if its embedding similarity to the phrase** (with the retrieval instruction) **is at least 0.40.** The tuned value was 0.40 for 9B and 0.37 for 4B; one constant, re-tuned with the evaluation set when a model changes.
   5. Skills found by the lookup for part of a phrase are kept alongside.
2. **Model:** the explanation LLM per ADR-0020: 4B on the CPU VM, 9B with a GPU.
3. **Cache:** every LLM result is stored per normalized phrase, prompt version, model and **catalog version** (the import checksum), so each phrase costs one LLM call per catalog version, and a catalog change re-asks automatically.
4. **Fallback:** when the LLM is unavailable, times out or answers invalid JSON, matching uses lookup → embedding (the best skill if its similarity is ≥ 0.44). Fallback results are not cached, so the LLM is tried again next time.
5. **Hiding the latency:** clients match each typed chip as soon as it is added to the board (`POST /api/v2/skills/match`); "Recommend" waits only for chips still being matched. Engine v2 endpoints live under `/api/v2` (ADR-0016's versioning); `/api/v1` keeps serving the legacy engine until the switch-over.
6. **Dropped:** the LLM-definition step and the relative σ thresholds for engine v2.
7. **Skill vectors** are stored per model in `catalog.skill_embeddings` (ADR-0008's per-entity pattern), embedded from "name: description" and re-embedded when that text changes. 257 skills need no vector index; add one when the catalog passes about 10,000 skills.

## Trade-offs accepted

- **A new typed phrase takes seconds on the CPU VM** the first time. The cache and background matching hide most of it; a phrase typed at the last moment can delay "Recommend".
- **Matching and explanations share one LLM on the CPU VM;** a matching call can wait behind an explanation. If that shows up in practice, match first (it's interactive), or move matching to the smaller model.
- **Prompt 2 was written after seeing prompt 1's errors on the same 349 phrases,** which risks fitting the prompt to the set. Its changes are generic (ranking, a cap, the tool rule, an example not in the set). Real user phrases will be collected and labeled to re-check it.
- **The strict "fully right" score undercounts:** many remaining errors are reasonable neighbouring skills. Precision, recall and F1 are reported alongside.
- **Two models per new phrase** (LLM plus embedding), both already running for other features.

## Revisit when

- The LLM or embedding model changes → re-run the benchmark and re-tune the 0.40 / 0.44 floors.
- The catalog grows past what fits comfortably in one prompt (about 1,000 skills) → retrieve the top candidates by embedding first and let the LLM pick among those.
- Real user phrases show a different error profile than the evaluation set → add them to the set and re-measure.
- A GPU arrives → 9B for matching, at about 1.8 s per new phrase.
