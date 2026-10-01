# ADR-0023: Skill-board input with suggestions; results as linked roles and courses; thumbs feedback

- **Status:** Accepted. Delegated: made by Claude on 2026-09-30 while the decider asked for the work to be finished without questions; on 2026-10-01 the decider accepted it without a separate review.
- **Date:** 2026-09-30
- **Decider:** Muhammed Yasin Horasanli

## Context

- **Input (prototype):** four text areas of comma-separated phrases, all required, plus a static "cheatsheet" of ~180 example phrases hardcoded in the page. Users had to know the comma format, type everything, and fill all four boxes even when a category didn't apply.
- **Results (prototype):** one table per model, roles with their courses nested below; a course recommended for two roles appeared twice; explanations arrived with the page after 1–2 minutes.
- **Now:** the API returns roles and courses in under a second and explanations later ([ADR-0018](0018-decoupled-per-role-explanations.md)); requests and shown results are stored, and a `feedback` table exists without an endpoint ([ADR-0013](0013-user-activity-hybrid-then-normalized.md)).
- The frontend framework decision (Next.js rewrite vs Nuxt upgrade) is still open. This ADR is about the UX and the API it needs, not the framework.

## Options considered

### Input
- **A — Keep comma-separated text areas.** ❌ The problems above.
- **B — Chip inputs with autocomplete.** ✅ No format to learn. ❌ Still starts from a blank box; no help deciding what to add.
- **C — A skill board:** four drop zones plus a suggestion panel. Skills are draggable chips; clicking a chip adds it to the selected zone (touch and keyboard users can't drag); typing with autocomplete and pasting "A, B, C" still work. ✅ Recognition instead of recall. ❌ More UI code.

### Where suggestions come from
- **Static list in the page** (as before). ❌ Hardcoded, disconnected from the data.
- **Curated groups + search over every roadmap concept + "For you":** suggestions related to what the learner already added, from the same embedding model and an exact nearest-concept scan (one embedding call, ~0.1 s). ✅ Data-driven, and the list adapts as the board fills. ❌ roadmap.sh node names are file slugs ("ci cd", "csharp"), so they need display labels.

### Results
- **A — List per role** (as before). ❌ Duplicated shared courses; the role ↔ course relation is only implied by nesting.
- **B — Two columns, roles and de-duplicated courses, joined by curved connectors in each role's color;** hovering a role or course highlights its links; a shared course shows one explanation per role. On small screens, no lines: each role lists its courses. ✅ Shows the structure the algorithm produced. ❌ Line positions must follow layout changes (cards grow as explanations arrive).
- **C — A graph library** (Cytoscape, vis-network). ❌ A heavy dependency for at most 3 roles × 9 courses; worse text layout than cards.

### Feedback format
- **Thumbs up/down per role and per course** vs **1–5 stars.** Thumbs: one click, clear signal for tuning; stars: finer, but more friction and noisier.

## Decision

- **Input: C, the skill board** (`pages/index.vue`, `components/board/`). A skill lives in one category at a time (dropping it elsewhere moves it). An "example" button fills a sample profile. The board state survives the trip to the results page ("Edit my answers"), and a shared results link can refill it from the stored input.
- **Suggestions API** (`/api/v1/knowledge-units`): `GET /groups` (curated groups, moved from the old page into `backend/xcrs/data/knowledge_units.json`), `GET ?q=` (curated labels + all roadmap concepts, ranked exact → prefix → word prefix → substring), `POST /related` (nearest concepts to the learner's phrases, minus what they entered, similarity ≥ 0.35). Roadmap slugs get display labels (`domain/labels.py`), in suggestions and in results.
- **Results: B, linked columns** (`pages/results/[id].vue`), a shareable URL per request that polls `GET /api/v1/recommendations/{id}` while explanations are pending, with shimmer placeholders. Responses now include the learner's input (additive, [ADR-0016](0016-versioned-structured-recommendation-api.md)).
- **Feedback: thumbs** (`POST /api/v1/recommendations/{id}/feedback`, rating ±1, optional role / course, validated against what the request showed).
- **Plumbing:** the browser only talks to the Nuxt server, which proxies `/api/v1/**` to the API (`XCRS_API_URL`, read at runtime); the old adapter route and form are gone. The UI stays on Nuxt 3 + Nuxt UI 2 + Tailwind.

## Trade-offs accepted

- Chips still send **text**: a chip that is a known concept isn't mapped to its concept id yet, so it is embedded like free text (ADR-0016's revisit trigger; it would also make "Python" match its five `python` concepts exactly).
- Display labels are a word list, not real labels; our own roadmaps should carry proper labels.
- HTML drag-and-drop doesn't work on touch screens; click-to-add covers them.
- Connector positions are computed in the browser with `ResizeObserver`; there's no server-rendered line layout.
- Building on Nuxt 3 means a framework migration, if chosen, ports these components. They are small and framework-light (no drag-and-drop or graph libraries).

## Revisit when

- The framework decision is made (Next.js rewrite or Nuxt 4 upgrade).
- Feedback volume allows tuning: use thumbs to evaluate thresholds, weights and prompts.
- Mapping chips to concept ids (ADR-0016 revisit) → `/api/v2` or an additive field.
- Results grow beyond 3 roles × 9 courses → reconsider the connector view.
