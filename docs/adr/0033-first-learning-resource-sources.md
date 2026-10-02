# ADR-0033: First learning-resource sources: a curated list as code, freeCodeCamp's open curriculum, and a YouTube adapter switched off until there is an API key

- **Status:** Accepted. The decider chose "curated YAML + adapters" (2026-10-01). The specific sources and rules were chosen by Claude overnight under the decider's instruction to proceed alone, and are to be reviewed.
- **Date:** 2026-10-02
- **Decider:** Muhammed Yasin Horasanli

## Context

- Engine v2 shows each role's **gaps** (ADR-0031), and learners need somewhere to learn them. ADR-0026 decided the model (courses, videos, docs; free and paid, with a free filter; English) and the ingestion rules (official APIs and feeds before scraping; the LLM normalizes and tags, it doesn't scrape).
- **No YouTube Data API key exists yet;** it needs the decider's Google account.
- **Licensing:**
  - Linking to public documentation and courses needs no license, and we store only a title, a URL and our own short description.
  - freeCodeCamp's curriculum repository is BSD-3-Clause, so its curriculum metadata (titles, intros) can be reused with attribution.

## Options considered

- **Curated list only.** ✅ Highest quality, reviewable. ❌ Doesn't grow by itself.
- **Adapters only.** ✅ Grows automatically. ❌ The first adapters cover a fraction of 257 skills, and quality varies.
- **A curated list as code, plus adapters for open sources** (chosen by the decider). ✅ Curated official docs and free courses cover the skills well; adapters prove the pipeline (raw → normalize → LLM tag) and grow the set.

## Decision

1. **`catalog/resources.yaml`** lists curated resources as code, reviewed in pull requests like the catalog (ADR-0028).
   - Each entry: URL, title, provider, type (docs, course, tutorial, video, playlist, book), level, free or not, and the skills it teaches with a proficiency.
   - It starts with official documentation and well-known free courses.
   - `xcrs catalog validate` checks the structure and the skill references; `xcrs catalog import` loads it with `tagged_by = curated`.
   - `xcrs resources check-links` fetches every URL and records its status; CI never touches the network.
2. **freeCodeCamp adapter** (`xcrs resources ingest freecodecamp`):
   - Reads the curriculum's `intro.json` from GitHub and stores it raw (ADR-0032).
   - Normalizes current courses into free `course` resources, skipping legacy, beta, placeholder and spoken-language courses.
   - Tags each course's skills with the matching pipeline of ADR-0030 (the LLM picks from the catalog, confirmed by similarity), `tagged_by = llm`.
3. **YouTube adapter** (`xcrs resources ingest youtube`):
   - Built against the YouTube Data API v3. It uses only `playlists.list` and `playlistItems.list` for channels listed in `catalog/sources/youtube.yaml` (cheap quota; no search calls).
   - Refuses to run without `XCRS_YOUTUBE_API_KEY`.
   - Enforces the 30-day rule of the API terms: YouTube data not refreshed within 30 days is deleted (`xcrs resources expire`).
   - Shows YouTube as the source.
4. **Engine v2 attaches up to three resources per role** to its first gaps. It prefers free resources, curated over LLM-tagged, and a level that fits; the results page lists them under "Start learning". `ALGORITHM_VERSION` goes up.

## Trade-offs accepted

- **The curated list is Claude-drafted.** Every link is checked live, but whether each resource is the *best* one for a skill is a judgement to review.
- **LLM tags** can be wrong; they carry a confidence (the similarity), and curated entries rank first.
- **No videos until a YouTube key exists.**

## Revisit when

- A YouTube API key is added → run the adapter on the curated channel list.
- A paid provider offers an affiliate or catalog API → a priced adapter with the free filter (ADR-0026).
- Links break often (`check-links`) → a scheduled check that deactivates dead resources.
