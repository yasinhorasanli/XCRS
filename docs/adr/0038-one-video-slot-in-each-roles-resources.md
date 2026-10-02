# ADR-0038: Each role's resources keep one slot for a video

- **Status:** Accepted
- **Date:** 2026-10-02
- **Decider:** Muhammed Yasin Horasanli

## Context

- Each recommended role lists up to three resources for its first gaps (ADR-0033). The ranking prefers curated over LLM-tagged resources first, and the curated list (254) covers every skill, so adapter resources were never shown. On the 53 learner profiles, all 477 picks were curated.
- 44 YouTube playlists were approved by the decider and ingested on 2026-10-02 (ADR-0033 note), ranked by views, likes per view, recency and channel size within trusted channels. Learners differ in how they like to learn; many prefer video.
- ADR-0026 asked for courses, videos and documentation in one model; the results page should show that mix.

## Options considered

### Option A: reserve a video slot (chosen)
Two resources from the usual ranking, then the best video or playlist that teaches one of the role's first three gaps.
- ✅ Every role can offer a video next to docs and courses; the approved playlists are actually seen.
- ✅ The usual ranking still decides the other two; only the last slot changes.
- ❌ The video may teach a gap already covered by the first two (another format, not another skill).

### Option B: curated first (as before)
- ✅ No change; highest-reviewed resources only.
- ❌ Adapter resources are never shown; the YouTube work has no effect.

### Option C: one ranking for all sources
- ✅ Simple; best match wins.
- ❌ Needs a comparable quality signal across docs, courses and videos, which we don't have; popularity numbers exist only for YouTube.

## Decision

`suggest_resources` fills the first two slots with the usual ranking (curated, free, gaps covered, on topic, reach, type). If none of them is a video or playlist, the last slot goes to the best video that teaches one of the first three gaps **and nothing beyond the gaps and the learner's own skills**. With no such video, the slot falls back to the usual ranking.

The video must fit the learner, not only the role: a first version accepted any skill of the role and showed a C# playlist to a learner who enjoys Python and Django (the backend roadmap accepts C#, Java, Go or Python). The same rule, "fewest skills beyond the gaps and the learner's", is also a tie-break in the usual ranking, after "on topic". `ALGORITHM_VERSION` v2.5.

Measured on the 53 learner profiles (159 role results): 125 (79%) show a video; 63 different role–video pairs. The looser first version reached 91%, by including playlists for languages the learner never mentioned.

## Trade-offs accepted

- The video can repeat a skill the first two already cover. That's accepted: it's the same gap in a different format.
- Video coverage depends on the approved playlists (44 of 257 skills so far); discovery continues within the daily quota.
- "Curated first" still outranks fit to the learner in the first two slots: a Python learner can get Oracle's "Learn Java" for object-oriented programming although freeCodeCamp's "Introduction to OOP in Python" exists. Changing that order would revise ADR-0033 and is left to the decider.

## Revisit when

- Learner feedback on resources (`feedback_v2.resource_id`) shows videos rated below other types, or the reverse.
- Learners can set a format preference (video, reading, course).
- A quality signal exists for non-video resources, making Option C possible.
