# ADR-0019: Explanation layer on LangChain, with a grounded contract; retriever over our own SQL

- **Status:** Accepted
- **Date:** 2026-09-30
- **Decider:** Muhammed Yasin Horasanli

## Context

XCRS explains every recommended role and course with an LLM. The explanation is the product's point: it must say *why*, truthfully.

**The prototype** (`backend/src/recom.py`) sent `gpt-4o` free text and asked for a JSON array of explanations. It matched the array to roles and courses **by position**: a wrong count or invalid JSON silently dropped explanations or raised a `KeyError` ([baseline](../baseline.md)).

**The first version in the new backend** (`backend/xcrs/explain/`, built 2026-09-29 without an ADR) fixed that with a contract, using plain httpx against an OpenAI-compatible endpoint:
- The model receives the algorithm's **actual reasons** as JSON: what the learner wrote and the concept each matched, covered topics, uncovered concepts, and each course with the concepts it was picked for.
- The answer follows a **JSON schema**; course explanations are matched by `course_id`, never by position.
- **Hallucination found and fixed:** a fixed prompt ("refer to what they know and what they are curious about") made the model invent curiosity whenever none was stated (5/5 runs). The system prompt is now **built from the fields present** (0/5), and empty fields are left out.
- Failures degrade to "no explanation"; the recommendation is never lost.

[ADR-0006](0006-own-embedding-interface.md) deferred LangChain "to the explanation-LLM decision". A LangChain review suggested prompt templates, structured output, an explanation chain and a retriever, and warned that the algorithm must decide and the LLM only explain.

Constraints: the embedding and vector design is accepted ([ADR-0008](0008-embedding-tables-per-entity.md)–[ADR-0010](0010-exact-threshold-search-and-candidate-penalty.md)); explanations will run in a background worker ([ADR-0018](0018-decoupled-per-role-explanations.md)); the LLM endpoint must stay swappable (Ollama now, vLLM on a GPU or a hosted API later, [ADR-0014](0014-local-first-then-split-by-role.md)).

## Options considered

### LLM client and orchestration

#### Option A — Keep plain httpx
- ✅ No dependency; already works
- ❌ Hand-written schema, parsing and validation; no standard hook for tracing (LangSmith) or a future chat/agent

#### Option B — LangChain for prompt, structured output and chain
- ✅ `ChatPromptTemplate` separates prompts from logic; a Pydantic model replaces the hand-written schema and parser
- ✅ A standard `Runnable` chain: tracing, batching and a future agent plug in without a rewrite
- ✅ Widely used in industry
- ❌ A dependency with a history of breaking changes (mitigated: only `langchain-core` + `langchain-openai`, pinned)

#### Chat model class (within B)
- **`ChatOpenAI` + `base_url`:** ✅ any OpenAI-compatible server (Ollama, vLLM, hosted) with the same config
- **`ChatOllama`:** ✅ native Ollama features (thinking toggle, `keep_alive`); ❌ ties the explainer to Ollama

### Output fields

- **Keep:** `role_explanation` + one explanation per course (by `course_id`); return the algorithm's uncovered concepts as plain data in the API
- **Add LLM-generated `missing_skills` / `next_steps`:** ❌ the LLM would *decide* what the learner lacks; the algorithm already computes it deterministically

### Retriever

- **LangChain `PGVector` (`langchain_postgres`):** ❌ its own tables (vectors copied into a second schema), no per-model partial indexes (ADR-0009), top-k only (concept matching is a threshold scan by design, ADR-0010), and ADR-0006 excluded LangChain's vector-store abstraction
- **Own `BaseRetriever` over our pgvector SQL:** ✅ standard `retriever.invoke(query)` interface, no schema change; ❌ no consumer on the request path today (course candidates are precomputed, ADR-0009)
- **No retriever yet:** ✅ nothing unused; ❌ the interface arrives only with the chat feature

## Decision

**LangChain for the explanation layer, keeping the grounded contract:**

1. **Prompt:** a `ChatPromptTemplate` (`explain/prompts.py`). The system prompt is still assembled per role from the fields present; the template takes it and the JSON payload as variables, so JSON braces are never parsed as placeholders. Each prompt has a `PROMPT_VERSION`, stored on `recommended_roles.prompt_version` like `algorithm_version`.
2. **Structured output:** `ChatOpenAI(...).with_structured_output(RoleExplanationOut, method="json_schema")` with Pydantic models; course explanations are kept only for recommended `course_id`s.
3. **Chain:** `prompt | structured_llm`, one call per role (ADR-0018). Temperature 0.1, thinking off (`reasoning_effort="none"`), no retries (a retry doubles a slow CPU call), plain chat completions (`use_responses_api=False`).
4. **Model client:** `ChatOpenAI` pointed at `XCRS_LLM_BASE_URL`; `XCRS_LLM_API_KEY` only for hosted endpoints. The `Explainer` protocol stays, so callers don't know LangChain exists.
5. **The algorithm decides, the LLM explains.** The uncovered concepts (`next_to_learn`, in learning order) are returned as data by the API, not generated.
6. **Retriever:** `CourseRetriever(BaseRetriever)` in `xcrs/retrieval.py` wraps the existing k-NN SQL (`repository/vectors.nearest_courses`) and returns LangChain `Document`s. Used by `xcrs search-courses` now and as a tool for a future chat/agent. The recommendation path doesn't use it.

Unchanged from the first version: word limits (≤ 60 per role, ≤ 45 per course), the rules against invention and misattribution, and degrade-on-failure.

**Resolves ADR-0006's deferral** for the explanation layer. LangChain embeddings stay out (our `Embedder` interface remains). An agent/chat feature (LangGraph) and tracing (LangSmith, a hosted service) are separate, later decisions.

## Trade-offs accepted

- A framework dependency for a single structured call today; accepted for a standard chain, tracing hooks and the path to a chat feature. Only `langchain-core` and `langchain-openai` are used, pinned.
- `with_structured_output` relies on the server's JSON-schema mode. Ollama supports it; a server that doesn't would need `method="json_mode"` plus validation.
- The retriever has no request-path consumer yet.
- Model-size limits remain: a 9B model still sometimes blends matched concepts into the prose (e.g. mentioning "Argo CD" for a learner who wrote "Kubernetes"). The model choice is ADR-0020.

## Revisit when

- A chat/agent feature is designed → LangGraph, with the retriever and repository functions as tools.
- Tracing or prompt evaluation is needed → LangSmith (hosted) or a self-hosted alternative.
- LangChain upgrades cost more than the hand-written version they replaced.
- The prompt changes → bump `PROMPT_VERSION`.
