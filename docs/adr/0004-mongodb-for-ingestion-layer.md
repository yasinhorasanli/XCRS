# ADR-0004: MongoDB for the ingestion layer (raw scraped data and roadmap drafts)

- **Status:** Superseded by [ADR-0032](0032-raw-ingested-data-in-postgres-jsonb.md) (2026-10-02: raw data in PostgreSQL JSONB; roadmap drafts became reviewed YAML in git, ADR-0028).
- **Date:** 2026-09-27
- **Decider:** Muhammed Yasin Horasanli

## Context

Planned future work:
- A **new course scraper**, possibly covering several platforms besides Udemy, each with its own data shape that changes over time.
- **Roadmaps generated or improved with AI coding agents** (Claude Code / Codex). roadmap.sh roadmaps are nested JSON trees (`embedding-generation/data/roadmaps_in_json/`), and generated drafts need revisions and review before use.

[ADR-0003](0003-postgresql-pgvector-primary-store.md) makes PostgreSQL the primary store for **cleaned, normalized** data. Raw and draft data has a different lifecycle: store it as-is, reprocess it, review it.

## Options considered

### Option A — Store raw and draft data in PostgreSQL (JSONB columns)
- ✅ One database to operate
- ❌ Mixes unstable raw data with the curated, relational core
- ❌ Less natural for deeply nested, frequently changing documents

### Option B — Object storage only (files on disk / S3-compatible storage such as MinIO)
- ✅ Cheapest and simplest for immutable raw dumps
- ❌ No querying, indexing or review-status tracking on drafts

### Option C — MongoDB as a raw-data and draft store feeding PostgreSQL
- ✅ Document model matches the data: raw scraped records stored exactly as received, roadmap drafts as nested trees with version and review status
- ✅ Re-running the cleaning logic doesn't need re-scraping
- ✅ A clear boundary: MongoDB holds untrusted or in-progress data, PostgreSQL holds curated, serving data
- ❌ A second database to operate, back up and monitor

## Decision (proposed)

Use **MongoDB (Community Edition, self-hosted)** as the ingestion-layer store. The pipeline would look like this:

```
scraper / roadmap generator ──► MongoDB (raw + drafts, versioned)
                                      │  clean, normalize, review
                                      ▼
                           PostgreSQL + pgvector (serving data)
```

## Trade-offs accepted (if accepted)

- Running two databases, justified only if the scraper and roadmap generation actually happen. Otherwise Option A or B is enough.

## Revisit when

At the start of the data-growth phase: confirm the scraper sources, data volume and the roadmap-generation workflow, then accept or reject this ADR.
