# ADR-0052: Show only the roles the board gives evidence for (0 to 3), and show interest as a share of the role

- **Status:** Accepted
- **Date:** 2026-10-12
- **Decider:** Muhammed Yasin Horasanli

## Context

Engine v2 (ADR-0029, ADR-0031) always returned the top three roles, and the results page showed each role's
`interest`: the share of the learner's input that falls on the role's skills. Trying the site on 2026-10-11, the
decider found two cases that looked wrong:

- **"Python", enjoyed, nothing else:** every role that uses Python got "Interest 50%" (Python is asked at working
  level, 2 of 4, and it was the whole board). All 30 roles scored above zero, the top five within 0.360–0.367, and
  Solutions Engineer came first by a 1–3% difference in coverage: noise.
- **"SQL", didn't enjoy, nothing else:** all 30 roles scored exactly 0. The three shown (Frontend Engineer, Game
  Developer, Embedded Software Engineer) were catalog order, and the LLM wrote "This role fits because…" for each.

The same probe found more of the pattern: two dislikes still returned three roles at a score of 0.005; "curious about
machine learning" showed Data Scientist at "Interest 100%"; "Spreadsheets", enjoyed, filled the third place with a
zero-score Frontend Engineer. A real board (five rated frontend skills) was fine.

Two causes: (1) `interest` is normalised by the board, so a thin board makes every matching role look strongly
interesting; it ranks well, but it reads as "how much of this role you like", which it isn't. (2) Nothing decided
whether a role was worth showing at all.

## Options considered

### Which roles to show

#### A1 — An evidence gate: 0 to 3 roles (chosen)
A role is shown only if (a) the board's skills it uses carry enough weight (`evidence`: category weight × how
distinctive the skill is × how much the role relies on it, summed; at least 1.0), (b) it is clearly ahead of the
field (score at least 2× the mean score of all roles), and (c) it isn't a weak tail (at least half the first shown
role's score). When no role passes, the result says why ("Python is used by 22 of 30 roles, so on its own it
doesn't point to one"; "You only told us what you didn't enjoy") and sends the learner back to the board.
- ✅ Fixes every case above; shows one or two roles when only one or two stand out.
- ✅ Calibrated and testable: three values, measured on the labeled profiles and a new set of thin boards.
- ❌ Some learners now see one role where they saw three; a weaker but acceptable third role is no longer shown.

#### A2 — Drop only roles that score zero or less
- ✅ Smallest change; fixes "SQL only" and the zero-score third place.
- ❌ "Python only" still shows three near-tied roles at 50%.

#### A3 — Always three, labelled by confidence
- ✅ The page never looks empty.
- ❌ Still recommends roles the board says nothing about; the explanation has to hedge for them.

### What the interest bar shows

#### B1 — Keep `interest` for ranking, show a role-side share (chosen)
`interest_in_role` = the share of the role's skills (weighted by distinctiveness and reliance) that the learner
enjoys or is curious about. "Python" alone: 2% for Backend Engineer; a typical labeled profile's expected role: 25%
(median; 60% at most).
- ✅ The calibrated ranking (ADR-0031) is untouched; the number means what the label says.
- ❌ Two interest numbers exist (one ranks, one is shown); the code and API say which is which.

#### B2 — Replace the formula, ranking included
- ✅ One number.
- ❌ Needs a new calibration of the weights; a bigger change than the problem.

#### B3 — No percentage, a label (Strong / Some / Weak)
- ✅ Hard to misread.
- ❌ Loses information the bar gives at a glance; thresholds for the labels would be one more calibration.

### The explanation's "next step"

The results page already lists the gaps in learning order and three resources for them, so the LLM's "First step"
repeated the page in other words; the decider asked to remove it from the page. **Chosen:** remove it from the
prompt too (explain-role-v2.2) and stop sending the resources to the model. The quality of the explanations is to be
evaluated separately (launch plan).

## Decision

**A1 + B1, and the prompt without a next step.** `role_scoring.recommendable` applies the gate (`MIN_EVIDENCE` 1.0,
`MIN_LIFT` 2.0, `TAIL` 0.5, at most 3); a recognised board that singles out no role is stored with the new status
`no_clear_match` (migration 0015) and a `no_match` hint (positive skills, and those used by half the roles or more).
Each shown role carries `interest_in_role`, which the results page shows; `interest` keeps ranking. Engine v3.3,
prompt explain-role-v2.2.

**Measured** (`eval/bench_role_gate.py`, 53 labeled profiles + 9 thin boards in `eval/thin_boards.yaml`):

| | Expected role kept | Roles shown (0/1/2/3) | Shown roles an adviser expected or accepted | Thin boards right |
|---|---|---|---|---|
| Always the top three (v3.2) | 53/53 | 0/0/0/53 | 61% | 1/9 |
| Gate (v3.3) | 53/53 | 0/18/18/17 | 83% | 9/9 |

The values sit in a flat region: every evidence floor from 0.8 to 1.5 gives the same result; a lift of 1.5–2.5 keeps
all expected roles and gets the thin boards right, 3.0 starts losing expected roles (51/53); a tail of 0.4 shows
more roles at 79%, 0.6 fewer at 85%. Evidence counts neutral skills at their calibrated half weight: a senior who is
neutral about Java, Spring and Kafka (a labeled profile) still gets Backend Engineer.

**Explanations** (`eval/bench_explainer.py`, qwen3.5:4b on the Mac GPU, the same 53 cases): prompt 671 → 542 tokens,
output 88 → 70 tokens, median 2.2 → 2.1 s per role; no automatic grounding flag (96% → 100% clean; the flag set
changed with the prompt). On VM-B's CPU (about 12 tokens/s) the shorter answer saves roughly 1.5–2 s per role.

## Trade-offs accepted

- A third of the labeled profiles now see one role, a third two. That is the point (83% of what is shown is a role
  an adviser would accept, against 61%), but a learner who wanted options sees fewer. Mitigation: "+ Curious" and
  the board make a second look cheap.
- The "expect" and "accept" labels and the thin boards were drafted by Claude and are a small set; the gate is
  calibrated on them. Mitigation: they are reviewed by sampling, like the other evaluation sets, and every new
  disagreement becomes a board in `thin_boards.yaml`.
- `interest` and `interest_in_role` coexist. Results made before v3.3 keep showing the old number (they have no
  `interest_in_role`).
- The explanations no longer suggest a first step in words; the page's gaps and resources carry that.

## Revisit when

- Thumbs-down feedback (`feedback_v2`) concentrates on the shown roles, or learners often leave a "no clear match"
  result without changing the board.
- The catalog grows well past 30 roles (the lift is relative to the mean of all roles, which changes with the count).
- The explanation-quality evaluation (launch plan) shows the shorter prompt lost something learners valued.
