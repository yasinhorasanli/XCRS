# ADR-0046: YouTube long videos, sections that link a gap to its part, and discovery from trusted channels

- **Status:** Accepted. The decider chose the scope (single videos, chapter and video sections, richer playlist data, channel discovery; not automatic approval) and kept the human approval of every video (2026-10-07). The details below were chosen by Claude while the decider was away, under the instruction to work alone, and are to be reviewed.
- **Date:** 2026-10-07
- **Decider:** Muhammed Yasin Horasanli
- **Approved (2026-10-07):** the decider approved 98 videos for 98 skills: 94 from the screened 99, with 5 doubtful ones (a Power BI-only and a Databricks-only course, a Helm course without chapters, a small observability channel, an attacker's-view web course) replaced or dropped after a second, targeted search, plus an OpenTelemetry course; Helm and observability keep their curated resources and approved playlist.
- **Adds to:** [ADR-0033](0033-first-learning-resource-sources.md) (sources) and [ADR-0038](0038-one-video-slot-in-each-roles-resources.md) (the video slot).

## Context

- The YouTube adapter (ADR-0033) handled **playlists only**: discovery searched `type: playlist`, ingestion read `playlists.list`. Much of the best free material is a single long video ("Docker Crash Course", "CS50P"), which never reached learners.
- A playlist or a 10-hour course was linked as a whole. For a gap that is one topic of a broad resource ("Object-oriented programming" inside a Java course), the learner had to find the part themselves.
- What we stored per playlist was a title and an item count: no duration, no freshness, no numbers to rank with.
- Discovery spent 100 quota units per search (10,000 a day, free). Every catalog skill had been searched once; growing the catalog (CLAUDE.md: plan for data growth) would need more searches every day.
- The YouTube Data API gives cheaply (1 unit per 50 ids or items): playlist contents, video details (duration, views, likes, date, description with chapter timestamps, spoken language), and a channel's uploads. It does **not** give transcripts of other people's videos (owner permission needed; scraping breaks the terms).

## Options considered

### A: single long videos, approved like playlists
- ✅ Smallest change; the same review gate and 30-day refresh.
- ❌ A 10-hour video is still linked as a whole.

### B: A plus sections (chapters of a video, videos of a playlist) tagged with skills
- ✅ A gap can open at the part that teaches it.
- ❌ Thousands of short titles to tag; needs care to avoid wrong deep links.

### C: richer data for the existing playlists (duration, views, likes, freshness)
- ✅ Shown to learners ("about 12 h") and usable for ranking; costs ~300 units a day for 140 playlists.
- ❌ More daily API calls and stored API data (refreshed daily, deleted after 30 days like the rest).

### D: discovery from trusted channels' uploads and playlists
- ✅ 1 unit per 50 uploads instead of 100 per search; scales with the catalog.
- ❌ No query: titles must be matched to skills by name, which is noisier than a search.

### E: automatic approval by rules
- ✅ Scales without the decider.
- ❌ Drops ADR-0033's human gate; popularity is a weak proxy for teaching quality.

## Decision

**A + B + C + D; not E.** Every new playlist or video still needs the decider's approval (`catalog/sources/youtube.yaml`, `playlists:` or `videos:`).

**Ingestion** (`xcrs resources ingest youtube`, daily): for each approved playlist, its videos (up to 500) and their details; for each approved video, its details. Stored per resource: `duration_minutes` and quality numbers (views, likes per view, first and latest upload, captions share); per resource its **sections** (`catalog.resource_sections`, migration 0014): a playlist's available videos (link: the video inside the playlist) or a video's chapters (YouTube's own rule: timestamps at line starts, the first at 0:00, at least three, increasing; link: `&t=…s`). Sections are replaced only when their titles or links change, so their tags survive the daily refresh. A playlist whose numbered episodes run newest first (one of 140: "Session 56" on top) records where episode 1 is. Raw records keep video details without the daily-changing counts.

**Section tags** (`xcrs resources tag`, after the resource's own tags): no LLM. A section may only take skills its resource teaches: the one its title is most similar to, and any other within 0.03 of it, if at least 0.44 similar (ADR-0030's fallback floor). Before embedding, the words all of a playlist's titles share at the start or end and the episode mark are removed ("Full React Tutorial #16 - Using JSON Server" → "Using JSON Server"); chapters about the course itself ("Course structure", "Introduction", "Setup") take no skill.

**Which part to open** (`section_for`, engine v3.2): in gap order, the first section after the first that teaches a gap fewer than half of the resource's sections teach. The resource opens at its start (or episode 1) when a gap is its main subject (the approved skill), a prerequisite of that subject (a Go series for programming fundamentals: that comes from episode 1), or what most of it teaches. The results card shows the duration and "Jump to 3:28:38: Object Oriented Programming", "Watch: …" or "Start with: Session 1 …"; the card itself still opens the whole resource.

**Discovery** (`xcrs resources youtube-discover`, daily): first the watched channels (`watch_channels`: the 33 trusted channels, by id): their uploads since the last scan and their playlists, titles matched to skills by name (never a generic word such as "Flow"; Go, C and R only before "programming", "language", "tutorial" or "course"). Then searches: playlists for skills not yet searched (none left), then long videos (`videoDuration=long`, "<skill> full course") for skills not yet searched for videos. A video is proposed only if it lasts at least 30 minutes, is in English (title and spoken language), has a course word in its title and isn't one numbered episode ("Lecture 8"): the first run's talks and single lectures were the noise. Ranking: trusted channel, then chapters, then the existing quality score. At most two videos per skill.

## Measured

- **Ingestion** (dev DB, 140 playlists): 4,560 sections; average playlist 12 h; ~85 s and ~300 quota units. Captions share is low (the API flags only uploaded captions, not automatic ones), so it is stored but not used.
- **Section tags:** 4,087 of 4,560 playlist sections tagged; all within their playlist's own skills.
- **First discovery** (60 video searches + first full scan of 33 channels, ~7,000 units): 347 video candidates; 236 pass the filters; Claude screened **99 videos for 98 skills** into `untracked/youtube-shortlist.yaml`, pending approval.
- **On the 53 learner profiles** (159 role results, 477 picks), with the 99 shortlisted videos ingested and tagged in a rolled-back transaction: roles showing a video or playlist **135 → 151 (85% → 95%)**; picks 83 videos and 72 playlists (135 playlists before); curated courses 145 → 116, as videos fit some gaps better. Ingesting the 99 videos: 2,552 chapters, 1,751 tagged; 128 LLM tags on the videos themselves.
- **Section links on the same profiles:** 5 of 477 picks, by design rare. 4 chapters, all on a side topic: a Java course at "Object Oriented Programming" (3:28:38) for the OOP gap; CS50P at "Lecture 3 – Exceptions" for debugging; "Databases In-Depth" at "Complexity Comparison of BSTs, Arrays and BTrees" for data structures. Without the main-subject and prerequisite rules, the first version linked 40 picks, most of them wrongly (a "Cloud Fundamentals" playlist at a SaaS video for the cloud-fundamentals gap); before the generic-chapter rule, "Course structure" was linked for data structures. The Selenium playlist listed newest first opens at "Session 1".

## Trade-offs accepted

- More daily API calls (~300 units for refreshes, plus discovery) and more stored YouTube data; both stay inside the free quota and the 30-day rule.
- Section tags come from short titles and embeddings, not from what is said in the video (no transcripts). A side-topic link can be wrong ("Using JSON Server" was tagged state management); the main-subject and prerequisite rules keep links rare, and the card always opens the whole resource too.
- Channel discovery proposes only what trusted channels publish; other good channels still come from search.
- The trusted list is matched by exact channel title, and titles change ("The Net Ninja" became "Net Ninja"); `watch_channels` uses ids, and the renamed titles were added.

## Revisit when

- Learner feedback on resources shows section links followed less than the whole resource, or rated lower.
- The catalog grows past what the channel scan covers (searches would again be the bottleneck): add channels, or approve by rules (option E) for trusted channels only.
- Transcripts become available legitimately (e.g. a channel shares them), which would make section tags far more reliable.
