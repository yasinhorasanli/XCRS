# ADR-0007: Ollama as the model runtime, `qwen3-embedding:0.6b` as the embedding model

- **Status:** Accepted
- **Date:** 2026-09-28
- **Decider:** Muhammed Yasin Horasanli

## Context

[ADR-0005](0005-local-embedding-models.md) makes embeddings self-hosted; [ADR-0006](0006-own-embedding-interface.md) puts them behind an OpenAI-compatible interface. We now need a concrete runtime and model.

**Hardware:**
- Production: two CPU-only VMs, 8 vCPU / 8 GB RAM and 16 vCPU / 16 GB RAM, each with 40 GB disk. A GPU is planned later.
- Development: a MacBook Pro M4 Pro with 24 GB unified memory. Ollama uses its GPU through Metal.

**Input lengths** (measured on the current data, in words):

| Text | Median | Longest |
|---|---|---|
| Course | 62 | 314 |
| Roadmap concept | 82 | 728 (≈1K tokens) |

A model with at least a 2K-token input limit avoids truncation.

The long-term goal is to run a local LLM for explanations as well. That is a separate decision, but it favours a runtime that can serve both.

## Options considered

### Runtime

| | Ollama | Hugging Face TEI | In-process (sentence-transformers) |
|---|---|---|---|
| OpenAI-compatible API (ADR-0006) | ✅ | ✅ | ❌ |
| Also serves LLMs later | ✅ | ❌ (embeddings only) | ❌ |
| Setup / dev on macOS | Easiest, native Metal | Docker, more tuning | Easy |
| Throughput | Good enough at this scale | Highest | Tied to the backend process |

### Model (Ollama tags, default quantization)

Sizes and benchmark positions below are as reported by public benchmark comparisons, not measured on our data.

| Model | Size | Dims | Context | Notes |
|---|---|---|---|---|
| `nomic-embed-text` | ~0.3 GB | 768 | 8K | Small, popular, older (2024) |
| `embeddinggemma` | ~0.6 GB | 768 (Matryoshka) | 2K | Reported best under 500M parameters |
| **`qwen3-embedding:0.6b`** | ~0.6 GB | 1,024 (Matryoshka) | 32K | Reported best quality for its size |
| `qwen3-embedding:8b` | ~4.7 GB | 4,096 | 32K | Top of the benchmarks, but ~13× the compute: seconds per request and hours to days for catalog runs on CPU. 4,096 dims also exceed pgvector's HNSW limits (2,000 for `vector`, 4,000 for `halfvec`) |

## Decision

- **Runtime:** Ollama, in development (MacBook) and in production (VM).
- **Embedding model:** `qwen3-embedding:0.6b` at 1,024 dimensions. That fits pgvector's HNSW index without workarounds and fits comfortably in the VMs' RAM.

## Trade-offs accepted

- **Lower throughput than TEI.** Irrelevant at the current scale.
- **Benchmark quality, not proven quality.** It must be confirmed by the prototype-vs-new comparison. `embeddinggemma` is the fallback candidate.
- **The exact model tag and quantization must be pinned and stored with every vector.** Embeddings generated on the Mac can be loaded on the server only if the same tag and quantization are used.
- **Dev and prod performance differ.** The Mac runs on its GPU, the VMs on CPU. Latency baselines must be measured on the VM.

## Revisit when

- The GPU arrives. Then consider the `qwen3-embedding:4b`/`8b` models, which require a full re-embed, and possibly vLLM for LLM serving.
- The quality comparison shows the 0.6B model underperforming the prototype.
- Concurrent load on the VM exceeds what Ollama on CPU can serve.
