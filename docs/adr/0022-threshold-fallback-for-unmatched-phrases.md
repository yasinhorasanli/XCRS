# ADR-0022: Keep the 2.5σ threshold, with a fallback for phrases that match nothing

- **Status:** Accepted (delegated: made on 2026-09-30 while the decider asked for the remaining work to be finished without questions; pending the decider's review)
- **Date:** 2026-09-30
- **Decider:** Muhammed Yasin Horasanli

## Context

Role scoring counts every roadmap concept a user phrase matches above a threshold of mean + 2.5σ ([ADR-0010](0010-exact-threshold-search-and-candidate-penalty.md)). ADR-0010 left an open issue: the mean and σ come from the **course × concept** similarity distribution, but the threshold is applied to **short user phrases × concepts**. With the new model, queries also carry an instruction prefix and documents don't, so the two distributions differ even more.

The comparison harness (`backend/eval/compare_prototype.py`) measured it on 50 phrases from 15 synthetic profiles (`qwen3-embedding:0.6b`; course × concept mean 0.234, σ 0.105):

| σ | Threshold | Median matches per phrase | Max | Phrases matching nothing |
|---|---|---|---|---|
| 1.5 | 0.392 | 10.5 | 60 | 1 |
| 2.0 | 0.444 | 4 | 35 | 3 |
| **2.5 (current)** | 0.497 | 2 | 25 | **8**, including "Python", "Spring Boot", "Express", "User research" |
| 3.0 | 0.549 | 1 | 20 | 19 |

At 2.5σ, **16% of phrases were silently dropped**. The worst case: a learner who liked Python and disliked JavaScript, CSS and HTML got **no recommendation at all**. Short, generic phrases have lower similarity to long concept texts, even when the concept is literally the same word.

The prototype-vs-new comparison, which would calibrate against the prototype's behaviour, can't run yet: the prototype's API keys and per-provider embedding files aren't available.

## Options considered

### Option A — Keep 2.5σ until the prototype comparison runs
- ✅ No change before validation
- ❌ Keeps dropping 16% of inputs, sometimes the whole request

### Option B — Lower the global threshold (e.g. 2.0σ)
- ✅ One constant; fewer lost phrases (3)
- ❌ Doubles the matches of phrases that already match well (median 2 → 4, max 25 → 35), adding noise to every role score

### Option C — Keep 2.5σ; a phrase with no match falls back to its best candidates above 1.5σ
- ✅ Phrases that match well are unchanged; only lost phrases change
- ✅ Every phrase with any plausible candidate contributes (lost phrases 8 → 1)
- ❌ Two constants instead of one; fallback matches are weaker evidence but weighted like normal ones

### Option D — Calibrate on a phrase × concept distribution
- ✅ Fixes the root cause (the distribution mismatch)
- ❌ Needs a representative phrase sample; today that's 50 synthetic phrases, too few to estimate a tail

## Decision

**Option C.** One exact scan down to a floor of mean + 1.5σ, then a pure domain rule (`domain/matching.select_matches`):
- every match above mean + 2.5σ counts, as before;
- a phrase with none keeps its best candidate **and every candidate within 0.05 of it**.

**A margin, not a fixed top-k.** A first version kept the top 2 and was arbitrary: "python" is a concept in five roadmaps with near-tied similarities (0.485, 0.483, 0.462, 0.454), and top-2 picked Backend and Blockchain by a 0.002 difference. With the margin, identical concepts are treated alike.

`ALGORITHM_VERSION` becomes `1.1.0`, stored per request ([ADR-0013](0013-user-activity-hybrid-then-normalized.md)).

## Trade-offs accepted

- The change is validated on synthetic profiles only; the prototype comparison may still move the constants (`FALLBACK_SIGMA`, `FALLBACK_MARGIN`, `THRESHOLD_SIGMA` in `services/recommend.py`).
- Fallback matches weigh as much as normal matches.
- The root cause, calibrating on course × concept pairs, remains. Option D becomes possible once real user phrases are collected.

## Revisit when

- The prototype-vs-new comparison runs (`--prototype-url`): tune the constants against it.
- Enough real user phrases are stored (activity tables) to estimate a phrase × concept distribution (Option D).
- The embedding model changes: re-run `eval/compare_prototype.py`; the table above is model-specific.
