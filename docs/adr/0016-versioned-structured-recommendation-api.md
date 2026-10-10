# ADR-0016: Versioned recommendation API with structured input

- **Status:** Accepted
- **Date:** 2026-09-29
- **Decider:** Muhammed Yasin Horasanli

## Context

The prototype exposes one endpoint per embedding model (`POST /recommendations/{google|voyage|openai|mistral|cohere|mock}`). Each takes four comma-separated strings, split on the server (`backend/src/util.py:split_and_create_dict`). The frontend calls all five one after another.

The new system has one active model ([ADR-0008](0008-embedding-tables-per-entity.md)). A planned input redesign replaces comma-separated text with clickable phrases (chips).

## Options considered

### Option A — Keep the prototype's contract
- ✅ The frontend works unchanged
- ❌ Comma parsing on the server; the model in the URL; no versioning

### Option B — One versioned endpoint with structured lists
- ✅ Clean input that matches the chips redesign; the server no longer guesses at separators
- ✅ The `/v1` prefix allows contract changes without breaking clients
- ❌ The current frontend needs a small adapter

## Decision

**Option B:** `POST /api/v1/recommendations`, with the model chosen by the registry and not the URL:

```json
{ "liked": ["Java", "SQL"], "neutral": [], "disliked": ["PHP"], "curious": ["Docker"] }
```

Until the new UI exists, the existing Nuxt server route splits the comma-separated text and calls this endpoint.

## Trade-offs accepted

- The existing frontend needs an adapter, which is temporary.

## Revisit when

- The input redesign changes what a phrase is (e.g. a concept ID vs free text). That becomes `/v2` or an additive field.
