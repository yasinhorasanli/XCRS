# ADR-0005: Self-hosted embedding models instead of hosted embedding APIs

- **Status:** Accepted
- **Date:** 2026-09-27
- **Decider:** Muhammed Yasin Horasanli

## Context

- The prototype embeds user input live through **5 hosted APIs** (Google, Voyage, OpenAI, Mistral, Cohere), one call per provider per request (`backend/src/main.py:create_user_embeddings`). The catalog was embedded offline with the same 5 providers.
- Budget is zero ([ADR-0002](0002-self-hosted-first-cloud-last.md)).
- Hardware: two CPU-only VMs (8 GB and 16 GB RAM; see ADR-0002, corrected 2026-09-28); a GPU is planned later.
- **Vectors from different models are not comparable.** The query and the catalog must use the same model, and changing the model means re-embedding the whole catalog. With a hosted API, the provider controls when that happens (models get deprecated and retired).
- Workload size:
  - A user request is a handful of short phrases: milliseconds to ~100 ms on CPU.
  - The whole catalog is ~1,300 texts today (minutes on CPU). At 100K+ items it becomes an overnight batch, then incremental.
- The explanation LLM is a **separate** decision; it isn't covered here.

## Options considered

### Option A — Keep hosted embedding APIs
- ✅ Strong quality, no ops
- ❌ Never zero cost; free tiers are rate-limited
- ❌ The provider's deprecation schedule forces re-embedding
- ❌ User input leaves the server

### Option B — Self-hosted embedding model only
- ✅ Zero cost, no rate limits
- ✅ We control model versions, so re-embedding happens only when *we* decide
- ✅ User input stays on the server
- ✅ Leading open embedding models are competitive on public benchmarks
- ❌ We run a model server
- ❌ Quality on *our* data must be verified, not assumed

### Option C — Local by default, plus a hosted "compare mode"
- ✅ Keeps the 5-provider research comparison available
- ❌ Two paths to maintain; costs money whenever compare mode is used
- ❌ The comparison was the research question, which the production app doesn't need

## Decision

**Option B.** Embeddings are produced by a self-hosted model. Application code depends on a **model-agnostic embedding interface**, so hosted providers could be plugged back in without code changes elsewhere. The specific model and runtime are decided separately.

## Trade-offs accepted

- Ops responsibility for the model server.
- A one-time quality check: run the same sample inputs through the prototype (hosted APIs, a few cents) and the new system, then compare the recommendations.
- The 5-provider comparison is dropped from the production app. The research record stays on Zenodo.

## Revisit when

- Local embedding quality measurably underperforms the prototype on the comparison set.
- A budget becomes available and a hosted model offers a clear quality gain.
