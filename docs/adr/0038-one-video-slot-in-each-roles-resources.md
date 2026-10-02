# ADR-0038: Each role's resources keep one slot for a video, and the learner's languages rank first

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

The video must fit the learner, not only the role: a first version accepted any skill of the role and showed a C# playlist to a learner who enjoys Python and Django (the backend roadmap accepts C#, Java, Go or Python).

**The learner's languages come first in the whole ranking** (decided by the decider on 2026-10-02, after Oracle's "Learn Java" was shown to a Python learner for object-oriented programming): a resource that teaches a programming language (catalog `kind: language`) the learner didn't mention and the gaps don't ask for ranks below every resource that doesn't, even a curated one. This revises the order of ADR-0033 decision 4. The new order: learner's languages, curated, free, gaps covered, on topic, fewest other skills beyond the gaps and the learner's, reach, type. `ALGORITHM_VERSION` v2.6.

Two broader versions were measured and rejected: ranking "fewest extra skills of any kind" first picked "Python Certification" over CS50 for frontend roles (22 language mismatches), and ranking "on topic" first made it 34. Restricting the rule to languages brought it to 0.

Measured on the 53 learner profiles (159 role results, 477 picks): 125 (79%) show a video; 0 picks teach a language the learner didn't mention and the gaps don't need (18 with curated first). Picks: 322 curated, 125 YouTube, 30 freeCodeCamp (0 freeCodeCamp before this ADR). The looser first video rule reached 91% video coverage by including playlists in languages the learner never mentioned.

## Trade-offs accepted

- The video can repeat a skill the first two already cover. That's accepted: it's the same gap in a different format.
- Video coverage depends on the approved playlists (44 of 257 skills so far); discovery continues within the daily quota.
- A language-neutral curated resource still beats one in the learner's language (a Python learner gets the curated "Design Patterns" tutorial for OOP rather than freeCodeCamp's "Introduction to OOP in Python"). Preferring the learner's language positively is a possible next step.
- The rule relies on skill kinds; a resource tagged only with a concept (not its language) escapes it.

## Revisit when

- Learner feedback on resources (`feedback_v2.resource_id`) shows videos rated below other types, or the reverse.
- Learners can set a format preference (video, reading, course).
- A quality signal exists for non-video resources, making Option C possible.
