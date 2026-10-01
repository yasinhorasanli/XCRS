# ADR-0025: A shared skills catalog from O*NET and ESCO, with LLM-drafted learning paths reviewed by a human

- **Status:** Accepted
- **Date:** 2026-10-01
- **Decider:** Muhammed Yasin Horasanli

## Context

The catalog comes from the research dataset ([`data/research-2024`](../../data/research-2024/README.md)): 10 roadmap.sh roadmaps flattened into 1,104 nodes, and 453 Udemy courses. Its quality limits the recommendations:

- **Skills are not shared.** Each roadmap has its own copy of a concept: "python" is five separate `roadmap_nodes` rows, near-tied in similarity, which already forced a workaround ([ADR-0022](0022-threshold-fallback-for-unmatched-phrases.md)).
- **Names are file slugs** ("ci cd", "what is http"), some nodes are section filler ("learn the basics"), and some content is only the name. Display labels and filters ([ADR-0023](0023-skill-board-input-and-linked-results.md)) patch this.
- **The structure is a tree** in roadmap.sh's drawing order: no prerequisites, levels or effort, and roles have no description.
- roadmap.sh's content license must be checked before production use.

Planned growth (more roles, generated roadmaps, more resources) needs a catalog designed for it, and explicit ids ([ADR-0003](0003-postgresql-pgvector-primary-store.md)).

**Taxonomies checked (2026-10-01):**
- **O\*NET** (U.S. Department of Labor): occupations with concrete "technology skills" (e.g. Docker, Kubernetes). **CC BY 4.0**: commercial use allowed; credit O\*NET and USDOL/ETA, link the license, mark changes, and state that USDOL/ETA has not endorsed them.
- **ESCO** (European Commission): occupations and skills with descriptions and relations. Free reuse for any purpose with attribution ("This service uses the ESCO classification of the European Commission").

## Options considered

### Roadmaps
1. **Keep roadmap.sh and enrich it.** ✅ Familiar, curated paths. ❌ License to check; slug names; tree only; per-role duplicates remain.
2. **An open taxonomy** for roles and skills. ✅ Authoritative, licensed for reuse, stable ids, descriptions. ❌ Not learning paths: no order or prerequisites; skills range from tools to broad competences.
3. **LLM-generated roadmaps.** ✅ Ordered paths, prerequisites, levels, rich descriptions. ❌ Can invent or drift; needs review and grounding.

### Skills
- **Per-role concept rows** (today). ❌ Duplicates; no cross-role knowledge.
- **A shared `skills` entity**, with roles linking to skills. ✅ One "Python" for all roles; prerequisites and resources attach once. ❌ A schema and data migration.

### Taxonomy
- **ESCO only**: rich and multilingual, but skills are often broad. **O\*NET only**: concrete technology skills, English only. **Both**: O\*NET's tools for technical granularity, ESCO for broader skills and descriptions; merging needed.

## Decision

**Options 2 + 3, on a shared skills entity, with O\*NET + ESCO as the backbone:**

1. **Roles** come from taxonomy occupations (O\*NET, linked to ESCO where they match), with a description and a reference to the source occupation.
2. **Skills** are one shared table. Each skill keeps references to its source entries (O\*NET technology skill or ESCO skill) and aliases for matching ("JS", "ECMAScript"); skills without a taxonomy entry (newer tools) are allowed and flagged.
3. **Learning paths are drafted by an LLM grounded in the taxonomy:** for each role, which skills matter, in what order, prerequisites between skills (a graph, not a tree), level and a short description. The LLM structures; it does not invent the role's skill set from nothing.
4. **A human reviews every drafted path** before it goes live. Drafts and approved versions are kept with a status and version, so paths can be regenerated and compared.
5. **Attribution** for O\*NET and ESCO is shown in the app and docs, with the O\*NET change notice.

Target entities (designed in the schema ADR that implements this): `skills`, `skill_aliases`, `skill_prerequisites`, `roles`, `role_skills` (importance, level, order), and source references; the current `roadmap_nodes` becomes legacy data until the migration replaces it.

## Trade-offs accepted

- A larger build: taxonomy import, merging two taxonomies, LLM drafting, and a review step.
- Review is manual effort for each role; it is the price of trustworthy paths.
- Two attributions to maintain, and keeping in step with taxonomy releases (O\*NET versions, ESCO versions).
- Taxonomy granularity varies; some tools need manual or LLM-proposed skills outside both taxonomies.

## Revisit when

- Review effort becomes the bottleneck → review sampling or an LLM judge with spot checks.
- A taxonomy release changes ids or structure.
- Users need non-English content (ESCO's multilingual labels become useful).
