# ADR-0028: Catalog as code: one skills graph, prerequisites as AND-of-OR with proficiency, reviewed as pull requests

- **Status:** Accepted. Delegated: the decider asked Claude to decide the structure ("make the things make sense", 2026-10-01); reviewed by the decider in PR #9 and merged.
- **Date:** 2026-10-01
- **Decider:** Muhammed Yasin Horasanli
- **Note (2026-10-06):** the import's upsert (`INSERT … ON CONFLICT DO UPDATE`) drew an identity value for every row it tried, existing ones included, so every import used up ids even when nothing changed. The dev database ran out of smallint role ids (32,767): each DB test imports the catalog. The import now updates existing keys and inserts only new ones (`repository/catalog_store.py`, `_upsert`), so ids are drawn only for new entities. The ids, their types and the import's behaviour are otherwise unchanged.

## Context

[ADR-0025](0025-skills-catalog-from-onet-esco-with-llm-learning-paths.md) decided a shared skills catalog with LLM-drafted, human-reviewed learning paths; [ADR-0026](0026-learning-resources-courses-videos-docs.md) one model for courses, videos and docs; [ADR-0027](0027-career-roles-levels-and-transitions.md) the roles. Open questions:

- **What has prerequisites?** Skills need other skills (Kubernetes needs Docker); a course needs what it builds on; "Course 2 of 3" needs course 1. Learners also reach the same goal by different routes (any one back-end language).
- **How deep?** "Knows Docker" isn't enough: Kubernetes needs *working* Docker, a senior needs *advanced* system design.
- **Where do roles and roadmaps live, and how are they reviewed?** Few, high-stakes, slowly changing records that an LLM drafts and a human approves.
- **Data starts from zero.** The 2024 Udemy courses are not carried over; resources will be collected fresh by ingestion (providers and raw storage not decided here).
- **The legacy catalog must stay separable** until the new one proves better.

## Options considered

### Prerequisites
- **Between every kind of thing** (skill→skill, course→course, course→skill …). ❌ N² hand-maintained links that break whenever a course disappears.
- **Only between skills; resources declare what they teach and require.** ✅ One graph; course order is derived (a course that teaches X comes before one that requires X); new resources fit automatically. Explicit resource→resource links only for real series (parts of a course, playlists).
- Within that: **plain AND lists** (can't say "any back-end language") vs **AND-of-OR groups with a minimum proficiency per edge**.

### Skills vs concepts
- **Separate entities.** ❌ The line is blurry ("HTTP", "Docker", "system design"); prerequisites would cross types anyway.
- **One entity with a `kind`** (language, framework, library, tool, platform, concept, practice). ✅ One graph, one matching path.

### Where roles, skills and roadmaps live
- **Database tables edited through an admin UI.** ❌ A UI to build and secure; review history to build.
- **YAML files in the repository, reviewed as pull requests, validated in CI, imported into the database.** ✅ Versioned, diffable, reviewable with existing tools; an LLM (Claude Code) drafts by editing files. ❌ Not editable by non-developers; an import step.

### Separating the legacy catalog
- **A `catalog_version` column in shared tables.** ❌ The shapes differ (roadmap nodes vs skills graph).
- **Rename legacy tables.** ❌ Breaks the running recommender now.
- **New tables in a `catalog` schema; legacy tables stay in `public` until removed.** ✅ No clash, no churn; dropping legacy is one migration.

## Decision

1. **One skills graph.** Skills and concepts are one entity with a `kind`; about 250 skills to start (`catalog/skills.yaml`), each with a description and, where they exist, matching O\*NET technology names as demand evidence.
2. **Prerequisites live only between skills**, as **AND of OR groups**, each with a **minimum proficiency** (1 basic · 2 working · 3 advanced · 4 expert): `kubernetes requires docker:2, computer-networking:2`; `docker requires python|go:2`. They must form a DAG.
3. **Resources** (ADR-0026) declare **teaches** and **requires** links to skills (with proficiency; tagged by the local LLM, reviewable), plus explicit **part-of / next** links only for series. Resource order is derived from the skills graph. **No 2024 Udemy data is imported.**
4. **Roadmaps:** for each role and level, cumulative **stages** of skill requirements (choices allowed, `optional` stages for "good to know"). Rule enforced by the validator: a roadmap asks for a skill only after its required prerequisites, at enough proficiency; optional stages never count as prerequisites; proficiency never drops at a higher level.
5. **Catalog as code:** `catalog/` YAML is the source of truth. Claude Code drafts changes on a branch → pull request → **CI runs `xcrs catalog validate`** (blocks merge on errors) → the decider reviews the diff (checklist in `catalog/README.md`, `xcrs catalog path` and `bridge` to read whole roadmaps and gaps) → merge → `xcrs catalog import` (next step) loads it into the database.
6. **Database (next step):** a `catalog` schema with skills, skill prerequisites (group, option, proficiency), levels, roles, role levels, roadmap items and options, transitions, resources, resource–skill links, resource series, and import records (commit, time, stats). The legacy research tables stay in `public` and keep serving the current recommender until the new engine switches over; then a migration drops them. *(2026-10-01: built as migration `0005` and `xcrs catalog import`, see [schema.md](../schema.md#catalog-v2-schema-catalog-adr-0028). The resource tables are deferred to resource ingestion, since their shape depends on the providers that are still to be chosen.)*
7. **Matching changes accordingly (later):** user phrases match skills (with aliases), roles score by coverage of each level's requirements — which also estimates the learner's level — and resources come from skills the learner is missing.

## Trade-offs accepted

- Only developers can edit the catalog (YAML + pull requests). Fine for one maintainer; an admin UI can come later on top of the same files or tables.
- An explicit import step between merge and database.
- Proficiency levels are coarse (four steps) and judgement calls; review is the control.
- The first draft (238 skills, 23 roadmaps, 1,000+ items) was written from market knowledge and O\*NET evidence before the ESCO link and a systematic coverage check against O\*NET's per-occupation technology lists; those come with the taxonomy import.
- Until the switch-over, two catalogs exist side by side.

## Revisit when

- Another maintainer without Git access needs to edit the catalog → admin UI.
- The graph grows to thousands of skills → generate parts (e.g. from taxonomy imports) instead of hand-editing one file.
- Learners need finer levels than four.
