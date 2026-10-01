# ADR-0031: Engine v2 role scoring: a calibrated blend of interest and coverage, levels from each level's additions

- **Status:** Accepted. The decider chose the approach (S2, a calibrated blend) on 2026-10-01. The calibration, the labeled profiles and the level rule were done while the decider slept, under their instruction to proceed autonomously; to be reviewed.
- **Date:** 2026-10-02
- **Decider:** Muhammed Yasin Horasanli

## Context

- ADR-0029 decided that a role's score combines **coverage** (how much of the role's requirements the learner meets) with **interest** (the board categories), calibrated on learner profiles. It also decided that the best-covered level is the learner's estimated level.
- **Input:** a learner puts about ten skills on the board, each with a category (enjoyed, neutral, didn't enjoy, curious) and an optional 1–4 proficiency. A roadmap level asks for about 40 skills.
- **The existing 15 evaluation profiles** had no expected roles, so nothing could be calibrated on them.

## Options considered

- **S1, coverage × interest.** ✅ Simple. ❌ A career changer with no coverage scores zero for the role they're curious about.
- **S2, a·interest + (1−a)·coverage, calibrated.** ✅ Both experts and career changers surface; tuned by measurement like ADR-0030. ❌ Needs labeled profiles.
- **S3, interest only** (the original XCRS idea), with coverage only for the level and gaps. ✅ Simplest. ❌ A role someone fits but feels lukewarm about ranks low.
- **S4, two lists** ("fits you now", "matches your interests"). ✅ Transparent. ❌ More UI, and two answers.

The decider chose **S2**.

## Decision

1. **Coverage** of a role level is the share of its cumulative requirements the learner meets. Each requirement is weighted by distinctiveness (inverse role frequency) × proficiency asked, and partial proficiency counts partly, as `xcrs catalog moves` does.
   - Skills enjoyed, neutral or not enjoyed count as known, at their rating (unrated = 1).
   - Curious skills don't count: they are what the learner wants to learn.
   - **Known skills imply their prerequisites** (the skills graph, ADR-0028): someone who knows Django knows Python at working level. A choice ("python|go:2") implies nothing unless the learner named one of its options.
2. **Interest** of a role is the share of the learner's attention that falls on the role's skills. Every mentioned skill carries category weight × distinctiveness, and a role collects the weights of the mentioned skills it requires, divided by the absolute total of all mentioned.
3. **Score** = a · interest + (1 − a) · coverage, with ranking coverage taken as the mean over the role's levels. **Calibrated values:** a = 0.4; curious 1.25, liked 0.75, neutral 0, disliked −0.5.
4. **Level:** the highest level whose **own additions** (new or raised requirements) are at least 55% met, together with every lower level's. Otherwise "start at the role's first level". **Gaps** are what the next level asks for that the learner doesn't have yet, in roadmap order.
5. **Engine v2 API:**
   - `POST /api/v2/recommendations` takes chips, each a picked catalog skill *or* typed text matched by ADR-0030. It returns the top 3 roles with score, interest, coverage, the estimated level and the target level, the learner's skills that count most, and the gaps.
   - `GET /api/v2/recommendations/{id}` reads one back; `POST …/{id}/feedback` records thumbs.
   - `GET /api/v2/skills?q=` is the picker's search; `GET /api/v2/skills/groups` returns suggested skills per role family.
   - Results are stored as JSONB with the catalog and algorithm versions (`recommendations_v2`, `feedback_v2`; migration 0007; ADR-0013's hybrid approach).

## Evidence

- **Labeled profiles** (`backend/eval/learner_profiles.yaml`, 53 profiles drafted by Claude): one typical learner per role (30), students (5), career changers (4), mixed and disliked-driven (2), and 12 "neighbour" profiles where adjacent roles compete or interest and skills disagree. The neighbour profiles were labeled before any run on them.
- **Calibration** (`backend/eval/bench_role_scoring.py`): grid search, 5 × 2-fold cross-validation stratified by kind; results in `eval/results/role-scoring-*.json`.

| Variant | Held-out top-1 | Top-3 | MRR | Neighbour profiles top-1 |
|---|---|---|---|---|
| **Blend, a = 0.4** | **96%** | **100%** | **0.98** | **100%** |
| Interest only (S3) | 91% | 94% | 0.89 | 75% |
| Coverage only | 94% | 96% | 0.90 | 92% |
| Coverage × interest (S1) | 94% | 100% | 0.94 | |

- **Sensitivity:** `a` between 0.3 and 0.5 performs alike (MRR 0.97–0.98), so 0.4 sits in the middle of the plateau. Interest alone and coverage alone are both clearly worse on the neighbour profiles.
- **An honest note on the first run:** before the neighbour profiles existed, every variant scored about 95%, because the profiles listed each role's signature skills. The first set could not tell the options apart, and that is why the neighbour profiles were added.
- **Level accuracy is weak:** about 53–58% exact with the additions rule (bars 0.30–0.60), against 55% at best with cumulative coverage. The cumulative rule overshot, since meeting all of mid also covers most of staff's cumulative list. Ten chips don't say enough about level.

## Trade-offs accepted

- **The profiles were drafted by Claude**, the same author as the catalog, so the calibration measures agreement with one adviser's judgement. Mitigation: review a random sample, as for ADR-0029, and add real feedback (`feedback_v2`) to the set once there are users.
- **Level estimates are rough.** The UI presents the level as an estimate and encourages the 1–4 ratings, which make it sharper.
- **Typed text costs an LLM call the first time** (ADR-0030); `POST /recommendations` waits for it. The board matches chips in the background so that usually has happened already.
- **`xcrs catalog moves` still estimates starting levels from cumulative coverage** (ADR-0027), which can overshoot the same way. To be aligned with this rule in a later catalog change.

## Revisit when

- Real feedback disagrees with the profiles' labels → add those cases and re-calibrate.
- The catalog changes substantially (roles added or merged) → re-run `eval/bench_role_scoring.py`. The test `test_the_calibrated_weights_keep_their_accuracy_on_the_profiles` fails below 90% top-1.
- Learners rate proficiency most of the time → re-measure the level bar; a higher bar may then pay off.
