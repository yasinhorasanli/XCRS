# ADR-0026: Learning resources: courses, YouTube and documentation in one model; free and paid; English first

- **Status:** Accepted
- **Date:** 2026-10-01
- **Decider:** Muhammed Yasin Horasanli

## Context

- Today the only resources are 453 Udemy courses from 2024 (`courses`), 428 of them without a description. Prices are in Turkish lira without a currency column.
- Skills become a shared catalog ([ADR-0025](0025-skills-catalog-from-onet-esco-with-llm-learning-paths.md)); resources should attach to skills, not to roadmap nodes.
- Udemy's terms forbid scraping; other platforms differ, and their APIs and feeds change over time.
- **YouTube Data API (checked 2026-10-01):** official and free within quota. `search.list` is capped separately at 100 calls a day; other calls share 10,000 units a day at 1 unit each (`videos.list`, `playlists.list`, `playlistItems.list`). Stored API data must be **refreshed or deleted within 30 days**, and pages showing YouTube content must make YouTube the visible source.
- The embedding model, roadmaps and explanation prompts are English.

## Options considered

### Resource types
- **Courses only.** ✅ Structured paths. ❌ Paid-heavy; slow to cover new tools.
- **Courses + YouTube.** ✅ Free options; fresh, skill-sized videos and playlists. ❌ Quality varies; thin text.
- **Courses + YouTube + documentation/tutorials.** ✅ Authoritative, free, current (e.g. official docs). ❌ More sources to maintain; docs aren't "courses", so the UI must show the difference.

### Price
- Free only / paid only / **both, with a free filter**.

### Language
- **English first** / English + Turkish / multilingual.

## Decision

1. **Courses, YouTube videos and playlists, and documentation/tutorials**, in one **`learning_resources`** model with a `type` (course, video, playlist, docs), `provider`, `url`, title, description, language, level, duration, `is_free`, price + currency, published / last-updated dates, quality signals, `fetched_at` / `last_checked_at`, and `is_active`. `courses` migrates into it.
2. **Resources link to skills** (`resource_skills`, tagged by an LLM from the resource's text with a confidence, reviewable), replacing concept → course matches built from roadmap-node text.
3. **Free and paid, with a free filter**, and the aim of at least one free option per role.
4. **English only for now**; `language` is stored so other languages can be added.
5. **Ingestion rules, in order of preference:** official APIs and catalog feeds → structured data the pages publish themselves (schema.org `Course`) → HTML scraping only where the site's terms and robots.txt allow it, rate-limited. Coding agents (Claude Code, Codex) write and maintain the adapters; an LLM normalizes and enriches records (skills, level, prerequisites) as a pipeline step, never as the scraper.
6. **YouTube:** start from curated channels and playlists (cheap lookups), keep searches within the daily cap, refresh stored data at least every 30 days, show YouTube as the source.
7. **Quality signals** per type (course ratings and reviews; video views, likes, channel, recency; docs from official sources) feed ranking; low-quality resources can be deactivated.

Not decided here: which providers come first, and where raw ingested data lives ([ADR-0004](0004-mongodb-for-ingestion-layer.md), still proposed).

## Trade-offs accepted

- Several provider adapters to build and keep working; each source's terms to follow and re-check.
- A 30-day refresh job for YouTube data.
- Mixed resource types need a UI that tells them apart, and ranking that compares unlike things fairly.
- English-only content excludes non-English learners for now.

## Revisit when

- A provider's API or terms change (re-check before each new adapter).
- Non-English demand appears.
- Resource volume makes LLM tagging cost or time noticeable → batch tagging, a smaller model, or embedding-based pre-filtering.
