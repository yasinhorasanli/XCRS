# ADR-0041: Levels from per-level evidence and optional experience; gaps and resources for the next level only

- **Status:** Accepted
- **Date:** 2026-10-03
- **Decider:** Muhammed Yasin Horasanli

## Context

The decider's 15 test profiles (`/dev/profiles`, from a CS student to a staff platform engineer) showed three linked problems with engine v2:

- **Levels.** The estimate was "the longest run of levels, from entry, whose additions are ≥ 55% on the board" (ADR-0031). People list highlights, not basics: a senior engineer writes Kafka and system design, not Git. Entry failed on unlisted basics and stopped the chain. The senior data engineer had 84% of the senior skills and got no level. Meanwhile thin staff levels (4–6 additions in 21 of 30 roadmaps) let senior iOS and embedded profiles pass at staff. Exact level: 4 of 15 test profiles, 19 of 53 calibration profiles.
- **Gaps.** "Skills to learn" were all of the target level's unmet requirements in learning order, so they always started with basics. Git was shown to 9 of 15 profiles, debugging to 7, data structures to 5, even to profiles rated staff.
- **Resources** followed the first gaps: Git Tutorials 6×, MIT Algorithms 5×, Pro Git 4× across the 15. Chrome DevTools went to an embedded engineer ("debugging"), because "curated" outranked "on topic".

## Options considered

### Level estimate
- **Keep the chain, lower the threshold.** ❌ Still stops at the first unlisted basics.
- **Highest level with enough evidence, each level judged on its own** (chosen). ✅ Basics no longer block; reaching a level assumes the ones below. ❌ Thin levels pass easily, so the roadmaps must be even.
- **Ask for experience only.** ✅ Simple, strong signal. ❌ Experience in one field says little about another role (a designer is not a mid frontend engineer).
- **Evidence plus optional experience** (chosen). Experience caps the evidence (a student isn't staff) and lifts it only where the learner shows some of that level's skills.

### Gaps
- **Cumulative unmet requirements** (as before). ❌ Basics first for everyone.
- **What the next level adds** (chosen); unlisted basics of the levels reached are listed separately to check, collapsed.

## Decision

1. **Level** = the highest level with at least 45% of what it adds on the board (`LEVEL_EVIDENCE`), levels judged independently.
2. **Experience**, optional on the board (student, under 2, 2–5, 5–10, 10+ years), gives a band:
   - the highest level of the band caps the estimate;
   - the lowest lifts it, but only with at least 25% evidence at that level (`LIFT_EVIDENCE`).
3. **Gaps** are what the next level adds (at the top level: what's left of it). Skills the learner is curious about stay gaps even below their level. Other unlisted skills of the levels reached are returned as `basics` and shown collapsed: "Assumed at your level… worth a quick check".
4. **Resources** come only from those gaps, and on-topic resources rank above curated ones (after the learner's languages, ADR-0038).
5. **Roadmaps:**
   - every staff level adds 8–13 skills (was 4–6 in 21 roles): leadership, ADRs, engineering processes, product thinking, mentoring, plus deeper versions of the role's core where they don't simply mirror senior;
   - basics at proficiency 1–2 that had entered senior or staff stages only as prerequisites (data structures, SQL, operating systems, networking) moved to each role's first level.

`ALGORITHM_VERSION` v3.0.

## Results

| | Before | After |
|---|---|---|
| Test profiles: expected role first | 14/15 | 15/15 |
| Test profiles: exact level | 4/15 | 14/15 (the student's "none" shows as "start at entry") |
| Test profiles: level within one | – | 15/15 |
| Calibration profiles (53, no experience given): exact level | 19 | 21 |
| Calibration profiles: expected role first | 52/53 | 52/53 |
| Profiles with a level whose gaps include Git, debugging, data structures, algorithms or programming fundamentals | most | 0 |
| Most repeated resource across the 15 | Git Tutorials ×6 | System Design Primer ×3, Scrum Guide ×3 |

## Trade-offs accepted

- Basics are assumed from evidence, not checked; the collapsed list is the safety net.
- 45% / 25% were chosen on 68 labeled profiles (15 test + 53 calibration); more labeled profiles may move them.
- The new staff stages share a common leadership core, so staff levels look more alike across roles than senior levels do. That is true to life, but less specific.
- Calibration profiles without experience are still often rated lower than expected (they list few skills). Experience is optional on purpose.

## Revisit when

- Learner feedback (thumbs on roles) shows levels felt wrong, or a larger labeled set disagrees with the thresholds.
- Accounts exist: progress ("I took this course") and the learner's own corrections become better evidence than self-reported experience.
