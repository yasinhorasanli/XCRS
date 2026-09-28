# ADR-0008: Store embeddings in per-entity tables keyed by model

- **Status:** Accepted
- **Date:** 2026-09-28
- **Decider:** Muhammed Yasin Horasanli

## Context

Courses and roadmap concepts each need a vector ([ADR-0003](0003-postgresql-pgvector-primary-store.md)). [ADR-0006](0006-own-embedding-interface.md) requires every vector to record the model that produced it, because vectors from different models aren't comparable. A model upgrade is already planned: `qwen3-embedding:0.6b` now, a larger model once a GPU arrives ([ADR-0007](0007-ollama-qwen3-embedding.md)). The prototype had no way to change models without replacing all its files.

## Options considered

### Option A — A vector column on each entity table (`courses.embedding`)
- ✅ Simplest
- ❌ Only one model at a time; changing models means rewriting the column in place, with a window where the data is inconsistent

### Option B — A separate embeddings table per entity (`course_embeddings(course_id, model_id, embedding)`)
- ✅ Several models can coexist side by side
- ✅ Real foreign keys to `courses` / `roadmap_nodes`
- ✅ Enables zero-downtime model migration
- ❌ One join more than Option A

### Option C — One shared table for everything (`embeddings(entity_type, entity_id, model_id, embedding)`)
- ✅ Several models can coexist
- ❌ No real foreign key (`entity_id` can point to either table), so integrity isn't enforced by the database

## Decision

**Option B.** Embeddings live in `course_embeddings` and `node_embeddings`, keyed by `(entity_id, model_id)`, with foreign keys to their entity and to a model registry (`embedding_models`).

**Zero-downtime model migration** becomes a routine procedure:
1. Register the new model as a *candidate*.
2. Embed the catalog with it, next to the existing vectors.
3. Compare quality.
4. Make it *active*.
5. Retire the old vectors later.

Rollback means switching back.

## Trade-offs accepted

- One extra join compared with a column on the entity table.
- During a migration, storage temporarily holds two sets of vectors.
- pgvector indexes need a fixed dimension, so models with **different dimensions** in the same table need a per-model index strategy. That's decided separately, together with the precomputed-score design.

## Revisit when

Only one model is ever used and the migration flexibility goes unused. (Unlikely, given the planned GPU upgrade.)
