# ADR-0032: Raw ingested data in PostgreSQL (JSONB, an `ingest` schema), not MongoDB

- **Status:** Accepted. Supersedes the proposed [ADR-0004](0004-mongodb-for-ingestion-layer.md).
- **Date:** 2026-10-02
- **Decider:** Muhammed Yasin Horasanli

## Context

- Learning resources (ADR-0026) arrive from several sources: a curated list, provider APIs and open feeds, later scrapers. Each source has its own shape that changes over time. The raw records should be kept so cleaning and LLM tagging can be re-run without fetching again.
- **ADR-0004 (proposed)** suggested MongoDB for raw data and drafts. Since then, roadmap drafts became reviewed YAML in git (catalog-as-code, ADR-0028), so the remaining need is raw source records only.
- **Constraints:** zero budget. VM-A has 8 GB of RAM and already runs PostgreSQL, the API, the web server and the embedding model (ADR-0014, ADR-0020). One database is backed up today (`scripts/db-backup.sh`).

## Options considered

- **PostgreSQL JSONB in a separate `ingest` schema.** ✅ No new service, RAM or backup path; SQL over JSON with indexes; raw and serving data are joinable while kept apart by schema. ❌ Raw and curated data share one database (and its disk).
- **MongoDB.** ✅ A natural document model; a CV line. ❌ Another service (about 1 GB of RAM on the 8 GB VM), another backup and monitoring path, for a few thousand documents.
- **Files (JSONL per run).** ✅ Simplest, immutable. ❌ No querying, deduplication or run tracking.

## Decision

1. **Raw records live in PostgreSQL, schema `ingest`:**
   - `raw_records`: the source, its external id, the payload as received (JSONB), a content hash and the fetch time.
   - A new version is stored only when the content hash changes, so history is kept without duplicates.
   - `runs` records each ingestion run (source, start and end, status, counts, error).
2. **Normalized resources** live in `catalog.learning_resources`, with skill links in `catalog.resource_skills` (ADR-0026, ADR-0028). They are rebuilt from `ingest.raw_records` by a normalize step, so a cleaning or tagging change never needs a re-fetch.
3. **Backups** cover the `ingest` schema through the existing database backups.
4. **MongoDB is not used** (ADR-0004 is superseded). Revisit if raw data stops fitting the pattern (see below).

## Trade-offs accepted

- Raw payloads take disk on the same 40 GB volume. Fine for thousands of records; pruning old versions is a later job.
- No CV line for MongoDB; the JSONB design (versioned raw records, re-runnable normalization) is the story instead.

## Revisit when

- Raw data grows past a few GB, or needs a different retention than the serving data → move it to files or object storage.
- Scrapers produce deeply nested, schema-less documents that we query in document-store ways.
