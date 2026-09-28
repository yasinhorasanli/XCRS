# ADR-0013: User activity storage — hybrid now, fully normalized once the input format settles

- **Status:** Accepted
- **Date:** 2026-09-28
- **Decider:** Muhammed Yasin Horasanli

## Context

The prototype writes each request's raw input to a JSON file (`backend/src/main.py:save_inputs` → `backend/user_inputs/`). Feedback goes to an external Google Form, linked only by that file name. What was actually *shown* (roles, courses, scores, explanations) is not stored.

Activity data is needed for:
- **Debugging and reproducibility:** "why was this role recommended?"
- **Quality evaluation:** prototype vs new, and old model vs new model, on real inputs.
- **A feedback loop** for tuning thresholds and weights.

The **input format is about to change.** The planned redesign replaces four comma-separated text boxes with clickable suggested phrases (chips) plus free text. Recommendation *results*, by contrast, reference stable entities: roles and courses.

XCRS has no users during the modernization, so schema changes can be applied freely before launch ([ADR-0011](0011-alembic-schema-migrations.md)).

## Options considered

### Option A — One table with request and response as JSONB
- ✅ Simplest; survives any format change
- ❌ Feedback can't reference a specific shown course
- ❌ Analytics need awkward JSON queries

### Option B — Fully normalized now (input phrases, shown roles and courses as rows)
- ✅ Clean analytics and integrity
- ❌ The input tables would be designed before the input redesign and would need reworking right after it

### Option C — Hybrid: input as JSONB; results and feedback normalized
- ✅ Input schema stays flexible during the redesign
- ✅ Results and feedback are relational, with foreign keys to roles and courses
- ❌ Input analytics need JSON queries until normalized

### Option D — MongoDB for activity logs
- ❌ A second database now for no gain; PostgreSQL JSONB covers the flexible part. MongoDB stays reserved for ingestion ([ADR-0004](0004-mongodb-for-ingestion-layer.md)).

## Decision

**Option C now, with a planned move to full normalization.**

Tables:
- `recommendation_requests`: id (uuid), created_at, model_id → embedding_models, algorithm_version, threshold_used, **input (JSONB)**, status (ok | insufficient_input | error), latency_ms
- `recommended_roles`: request_id, rank, role_id → roles, score, explanation
- `recommended_courses`: request_id, role_id, rank, course_id → courses, similarity, explanation
- `feedback`: request_id, role_id or course_id, rating, comment, created_at. The rating format (thumbs vs stars) is decided with the frontend.

`threshold_used` and `algorithm_version` are stored per request, so past results stay reproducible even after the registry's statistics are recomputed.

**Planned evolution.** The JSONB input column is **transitional**. Once the input redesign has settled:
1. add a normalized table (e.g. `request_inputs`: request_id, category, concept_id *or* free_text, position);
2. backfill it from the JSONB in an Alembic migration;
3. drop the JSONB column.

The long-term target is a fully normalized activity schema.

## Trade-offs accepted

- Until normalization, input analytics use JSON queries.
- A later migration is required. That's accepted deliberately in exchange for not designing input tables twice.

## Revisit when

- **The input redesign is implemented and its format hasn't changed for a while:** normalize the input (the planned step above).
- Before launch, a retention period for raw inputs must be set (privacy), even though requests are anonymous.
