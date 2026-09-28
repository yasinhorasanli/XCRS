# ADR-0006: Own embedding interface with an OpenAI-compatible adapter (LangChain deferred)

- **Status:** Accepted
- **Date:** 2026-09-27
- **Decider:** Muhammed Yasin Horasanli

## Context

[ADR-0005](0005-local-embedding-models.md) makes embeddings self-hosted, but no local model is running on the server yet. The code that calls the embedding model must work with any provider so that:
- development can start now (e.g. a local model on a laptop),
- the server model can be switched in by configuration later,
- a hosted provider could be plugged back in if ever needed.

In the prototype, provider calls are hardcoded per provider in two places (`backend/src/main.py`, `embedding-generation/src/embedding_generator.py`). It also embeds user queries as documents (Cohere `input_type="search_document"`), although many models expect queries and documents to be prepared differently.

Local model servers (Ollama, Hugging Face TEI, vLLM, llama.cpp) all expose an **OpenAI-compatible HTTP API**, which has become a de facto standard.

## Options considered

### Option A — Keep provider-specific calls
- ❌ Every provider change touches application code

### Option B — LangChain `Embeddings` interface and integration packages
- ✅ Many ready adapters; the same framework could later serve the LLM layer
- ❌ A heavy dependency for a ~30-line need; a history of breaking changes
- ❌ Its vector-store abstraction would hide the custom pgvector SQL that role scoring needs

### Option C — Our own small `Embedder` interface + one OpenAI-compatible adapter
- ✅ Tiny, no framework dependency, easy to test and explain
- ✅ One adapter covers Ollama, TEI, vLLM and OpenAI, with the provider chosen by config (base URL + model name)
- ✅ Separate `embed_query` / `embed_documents` methods fix the query-vs-document handling
- ❌ Additional non-OpenAI-compatible providers need their own adapter

## Decision

**Option C.** The application depends only on an `Embedder` interface (`model_id`, `dimensions`, `embed_documents`, `embed_query`). The first implementation is an OpenAI-compatible HTTP adapter configured with environment variables.

**LangChain is deferred** to the explanation-LLM decision, where it offers more (model swapping, structured output, LangGraph). If it's adopted then, it becomes one more adapter behind this interface, and no rewrite is needed. LangChain's vector-store abstraction won't be used either way; pgvector is queried with SQL.

## Trade-offs accepted

- We maintain a small amount of adapter code ourselves.
- Every stored vector must record the `model_id` that produced it, because vectors from different models are never compared.

## Revisit when

- We need several providers that aren't OpenAI-compatible.
- LangChain is adopted for the LLM layer and a single framework becomes simpler overall.
