# ADR-0009: Per-model vector indexes and precomputed concept → course matches

- **Status:** Accepted
- **Date:** 2026-09-28
- **Decider:** Muhammed Yasin Horasanli

## Context

[ADR-0008](0008-embedding-tables-per-entity.md) stores several models' vectors in the same per-entity tables. That leaves two design questions.

**1. Mixed dimensions.** pgvector's HNSW index needs a fixed dimension, but future models may produce vectors of different sizes (e.g. 1,024 now, 2,560 later).

**2. Work that doesn't depend on the user.** The prototype (`backend/src/main.py:main`, `backend/src/util.py`) mixes two kinds of similarity:

| Similarity | Depends on the user? | Prototype |
|---|---|---|
| User phrases × concepts | Yes | live cosine against in-memory matrix |
| Disliked items × courses | Yes | live cosine against in-memory matrix |
| **Concept × course** (course selection) | No | full dense matrix built at startup (`top_n_courses_for_concept`) |
| **Threshold statistics** (mean, σ of all concept × course scores) | No | computed at startup (`calculate_threshold`) |

The full concept × course matrix grows as concepts × courses: ~394K pairs today, ~5 billion at 50K concepts × 100K courses.

## Options considered

### Mixed dimensions

**A — One table per model** (`course_embeddings_qwen06`, …)
- ✅ Typed columns, simple indexes, retiring a model is `DROP TABLE`
- ❌ Every new model is a schema change for each entity (2N tables)
- ❌ Table names can't be query parameters, so the code has to build SQL dynamically (error-prone, injection risk)

**B — One table with an untyped `vector` column + one partial expression index per model** (the pattern documented by pgvector)
- ✅ Adding a model means adding one index; queries stay parameterized
- ❌ Queries must repeat the index's exact cast and filter, otherwise Postgres silently falls back to a full scan

**C — Declarative partitioning by `model_id`**
- ✅ Retiring a model is an instant partition drop, with no mass `DELETE`
- ❌ Extra complexity that isn't needed at the current scale

### Precomputed matches

**A — Keep computing concept × course similarity live** (vector search per recommended concept)
- ✅ No precomputation
- ❌ Repeats identical work for every request

**B — Store all concept × course pairs**
- ❌ Grows to billions of rows

**C — Materialized view of the top-k courses per concept**
- ✅ Simple to define
- ❌ `REFRESH` recomputes everything; doesn't scale with an incremental ingestion pipeline

**D — A relational table of the top-k courses per concept, maintained incrementally**
- ✅ Course selection becomes an indexed lookup
- ✅ Scores are stored, so they can be shown as part of explanations
- ❌ The ingestion pipeline must keep it up to date

## Decision

1. **Model registry** `embedding_models(id, name, runtime, quantization, dimensions, status, sim_mean, sim_std)`, with `status ∈ {candidate, active, retired}`. `dimensions` defines each model's vector size.

2. **Embedding tables use an untyped `vector` column, with one partial expression HNSW index per model:**
   ```sql
   CREATE INDEX ON course_embeddings
     USING hnsw ((embedding::vector(1024)) vector_cosine_ops)
     WHERE model_id = 1;
   ```
   Vector-search SQL lives in **one** repository function that generates the matching cast and filter. A test asserts with `EXPLAIN` that the index is actually used. The application checks each vector's size against `embedding_models.dimensions` before inserting.

3. **Precomputed table** `concept_course_matches(model_id, concept_id, course_id, similarity, rank)` holds the **top 20** courses per concept per model:
   - **Why 20:** the prototype halves the scores of courses similar to disliked items before taking the top 3, so there must be headroom for re-ranking. 20 is a starting value to tune.
   - **Maintenance:** the ingestion pipeline updates it incrementally. A new or changed course is compared with all concepts; a new or changed concept gets a fresh k-nearest-neighbour search over courses.

4. **Threshold statistics** (`sim_mean`, `sim_std`) are computed offline per model and stored in the registry: over all pairs today, over a large random sample once the data grows. The request path reads them instead of computing them at startup.

## Trade-offs accepted

- The index-matching discipline for vector queries, enforced by keeping the SQL in a single function plus an `EXPLAIN` test.
- The top-k cut-off: courses outside a concept's top 20 can never be recommended for that concept.
- Threshold statistics from a sample are an approximation of the full distribution.
- More work for the ingestion pipeline.

## Revisit when

- **Partitioning:** retiring models causes noticeable table bloat or slow deletes.
- **k:** recommendation quality shows that 20 is too few (or more than needed).
- **Precomputation:** the recommendation logic stops using per-concept course lists.
