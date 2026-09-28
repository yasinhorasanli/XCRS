# ADR-0010: Exact threshold search for user phrases; disliked penalty only on candidate courses

- **Status:** Accepted
- **Date:** 2026-09-28
- **Decider:** Muhammed Yasin Horasanli

## Context

The request path has two user-dependent similarity steps (see [ADR-0009](0009-per-model-indexes-and-precomputed-matches.md)).

**User phrases × concepts** (`backend/src/main.py:find_similar_concepts_for_courses`)
- The prototype keeps **every** concept whose similarity exceeds a threshold (mean + 2.5σ). It is not a top-k.
- Role scoring then counts those matches per role (`backend/src/recom.py:recommend_role`), so it is a **threshold (range) query**.

**Disliked phrases × courses** (`find_similar_courses_for_disliked_courses`)
- The prototype compares disliked phrases with **every course** and halves the score of those above the threshold (`util.top_n_courses_for_concept`).
- This is the only request-time query that grows with the course count, and the course count is the part expected to explode with the scraper.

**pgvector constraint:** an HNSW index accelerates *k-nearest-neighbour* queries (`ORDER BY distance LIMIT k`). It cannot accelerate "all rows closer than X".

**Growth profile:** concepts come from curated roadmaps and grow slowly (869 today). Courses may reach 100K+.

## Options considered

### User phrases × concepts

**A — Exact scan with a threshold** (compare against every concept vector of the active model)
- ✅ Same meaning as the prototype, which makes validation clean
- ✅ A few ms per request today; all phrases handled in one SQL query
- ❌ Cost grows linearly with concepts (~100 ms–1 s estimated at 50K)

**B — HNSW top-k, then filter by threshold**
- ✅ Fast at any scale
- ❌ Approximate; if more than k concepts pass the threshold, the rest are silently dropped, which changes role scores

### Disliked phrases × courses

**A — Compare against all courses** (prototype behaviour)
- ❌ Grows with the course catalog on every request

**B — Compare only against candidate courses**
- The candidates come from `concept_course_matches` for the recommended concepts, at most ~20 × the number of recommended concepts.
- ✅ Exact, small, and independent of catalog size
- ✅ Same outcome as the prototype for every course that could actually be recommended

## Decision

1. **User phrases × concepts: exact threshold scan.** All phrases of a request go into one SQL query:
   ```sql
   SELECT p.idx AS phrase_idx, n.id AS concept_id,
          1 - (e.embedding <=> p.vec) AS similarity
   FROM   unnest($1::vector[]) WITH ORDINALITY AS p(vec, idx)
   JOIN   node_embeddings e ON e.model_id = $2
   JOIN   roadmap_nodes  n ON n.id = e.node_id AND n.type = 'concept'
   WHERE  1 - (e.embedding <=> p.vec) > $3;
   ```
   An HNSW index on `node_embeddings` is **optional** for now. It's added when the revisit trigger fires.

2. **Disliked penalty: exact similarity against candidate courses only.** No request-time query scans the course catalog.

3. The **HNSW index on `course_embeddings`** stays. The ingestion pipeline needs it for "nearest courses to a new or changed concept" when maintaining `concept_course_matches`.

## Trade-offs accepted

- Concept search cost grows linearly. It's acceptable because concepts grow slowly.
- The penalty logic now depends on the precomputed candidate set. A course that isn't a candidate is never checked, which is fine because it can't be recommended anyway.

## Open issue (for quality validation, not decided here)

The prototype derives its threshold from the **course × concept** score distribution but applies it to **short user phrases × concepts**. Short phrases produce a different distribution. The threshold must be recalibrated for the new model during the prototype-vs-new comparison.

## Revisit when

- Concepts exceed ~50K, or the p95 latency of the concept query becomes noticeable. Then switch to HNSW top-k plus a threshold, using pgvector's iterative index scans.
