# ADR-0045: Skills from a CV or a LinkedIn "Save to PDF": a lookup scan plus one local-LLM extraction, reviewed by the learner; signed-in only, nothing kept

- **Status:** Accepted
- **Date:** 2026-10-05
- **Decider:** Muhammed Yasin Horasanli
- **Note (2026-10-05, evening):** the Phase-1 benchmark chose the "pick" shape (decision 2.4); results below. Built while the decider was away, under their instruction to proceed; to be reviewed.

## Context

Filling the skill board by hand is the slowest part of XCRS, and the people with the most to enter (experienced engineers) list the least (ADR-0041). Their skills and dates are already written down in a CV or a LinkedIn profile.

- **LinkedIn's API is closed to us.** Its sign-in (OpenID Connect, ADR-0043) returns name, email, picture and locale only; positions and skills need a partner program (checked 2026-10-03). LinkedIn's own **Save to PDF** (Profile → More → Save to PDF) is the workable route.
- **The LLM is scarce.** One CPU-only VM (VM-B, 24 vCPU) runs `qwen3.5:4b` for explanations and matching (ADR-0020, ADR-0030). An import holds it for minutes and makes everyone else's explanations and matching wait.
- **Privacy** (ADR-0043): no tracking, personal data stays on our servers, store as little as possible. A CV is full of personal data that the board doesn't need.
- **Uploads:** Caddy caps request bodies at 64 KB for the whole site (ADR-0035). The backend has no multipart support yet.
- **Untrusted input.** CV text goes into an LLM prompt. People already hide text in CVs for screening software (white-on-white keywords, "ignore previous instructions").
- **Languages.** Many users will have Turkish CVs, and LinkedIn writes its export in the interface language ("Deneyim", "Nisan 2024 - Mart 2026 (2 yıl)"). The catalog is English.

### Measured before deciding (2026-10-05, Mac M4 Pro; scratch probes, not yet the Phase-1 benchmark)

**PDF text order.** Three synthetic CVs of invented people, printed by Chrome: a two-column LinkedIn-style profile, rendered twice with the columns stored in opposite orders, and a Turkish one-column CV. Then the decider's real LinkedIn export (Apache FOP 2.3, 3 pages; read in place, not copied or stored).

| Extractor | Synthetic two-column (both orders) | Real LinkedIn export | Time (real, 3 pages) |
|---|---|---|---|
| pypdf, plain mode | columns whole (follows the file's order) | **columns whole** (sidebar, then main) | 10 ms |
| pypdf, layout mode | **columns interleaved line by line** | – | – |
| pdfminer.six, default layout analysis | columns whole | **sidebar block spliced into the Summary paragraph** | 22 ms |
| pdfminer.six, `boxes_flow=None` or `line_margin=2` | columns whole | columns whole | – |

All extractors handled Turkish characters. Chrome's PDFs contain ligatures ("Airﬂow"), which the name lookup can't match without NFKC normalization.

**One LLM extraction call** (jobs with dates, plus skills) on a one-page LinkedIn-style CV:

| Model, device | Prompt | Output | Time |
|---|---|---|---|
| 9B, Mac GPU | 600 tokens | 700 tokens | 24.5 s |
| 4B, CPU only (12 threads) | 600 tokens | 700 tokens | 49 s |
| 4B, CPU only, short Turkish CV | 470 tokens | 540 tokens | 21 s |

- **Output tokens dominate the time.** A two-page CV on VM-B is estimated at 1–3 minutes; not measured on VM-B yet.
- **The name lookup resolves most extracted phrases** (17 of 19, 10 of 15). 2–5 per CV are left for the LLM picker, under the per-request cap of 12 (ADR-0035).
- **"Extract phrases" missed practices** (microservices, REST API design, code review) and Kubernetes in "Airflow on Kubernetes".
- **"Pick catalog ids in the same call" found the practices.** With 9B it was clean; with 4B it also invented two skills not in the text and repeated entries.
- **Dates:** both models read "Ocak 2023 – Halen" as 2023-01 to present, and wrote English skill names for the Turkish CV. 9B also listed "English" (a spoken language) as a skill.

## Options considered

### What can be imported
- **PDF only.** ✅ Simplest. ❌ Scanned PDFs have no text layer and there is no OCR; Word users have no way in.
- **PDF plus pasted text** (chosen). ✅ Covers scanned PDFs and Word files (copy-paste) at almost no cost. ❌ A second input.
- **Also DOCX.** ✅ Common for Turkish CVs. ❌ Another parser and attack surface; LinkedIn exports PDF anyway.

### How skills are found
- **(a) LLM only.** ❌ Misses are silent (it missed Kubernetes).
- **(b) Catalog lookup only.** ✅ Instant. ❌ Misses described skills and translations, and finds no dates.
- **(c) Hybrid** (chosen). A deterministic scan for catalog names, plus one LLM call for the job timeline and the remaining skills, everything ending at catalog ids. ✅ The obvious skills cost nothing and can't be missed; the LLM adds dates, practices and translations. ❌ Two paths to merge.

### Waiting
- **Synchronous request.** ❌ 1–3 minutes through Caddy, Funnel and the Nuxt proxy; lost on any timeout.
- **Background job with polling, kept in memory** (chosen). ✅ The proven pattern (ADR-0018). Nothing persisted, and there is one API process (ADR-0014). ❌ A restart loses running imports (the learner retries).

### Where suggestions go
- **Straight onto the board as Neutral.** ✅ One click. ❌ A CV says what you know, not what you enjoyed. It would fill the board with skills the learner may dislike, and a signed-in board saves itself.
- **A review step** (chosen). ✅ The learner decides box and level for each skill. ❌ More UI.

### Levels and years
- **The LLM guesses levels.** ✅ Reads "led", "expert in". ❌ 4B is inconsistent, and the guess can't be explained.
- **A pure rule from dates** (chosen). ✅ Deterministic, testable, explainable. ❌ Years aren't depth; the learner corrects.

### Who can use it
- **Everyone, with per-IP limits.** ✅ Keeps "no account needed" (ADR-0043) and the first-visit effect. ❌ IP limits are weak, and anyone can hold the only CPU LLM for minutes.
- **Signed-in users only** (chosen). ✅ A per-account quota, accounts cost something to fake, and a pre-filled board is worth keeping. ❌ Friction; anonymous visitors see "sign in to import".

### PDF library
- **pypdf** (chosen, BSD, pure Python). ✅ Kept the columns whole on every file, including the real LinkedIn export; fastest. ❌ Follows the order text is stored in the file, so an unusual generator could produce an odd order.
- **pdfminer.six** (MIT). ✅ Orders text by position on the page. ❌ Spliced the sidebar into a paragraph of the real LinkedIn export with its default settings; only tuned settings avoided it, and that tuning is fragile.
- **PyMuPDF.** ❌ AGPL.

## Decision

1. **Input:** a PDF, or pasted text.
   - PDF ≤ 2 MB and ≤ 5 pages; at most 20,000 characters of text either way.
   - Text comes from **pypdf in plain mode**, in a subprocess with a 10-second timeout and a memory limit (malformed or bomb PDFs), then NFKC-normalized.
   - A PDF with under about 50 characters per page gets "this looks scanned; paste the text instead".
   - Caddy allows 3 MB on the upload path only; everything else stays at 64 KB.
2. **Finding skills** (hybrid):
   1. **Scan:** the catalog's names and aliases are looked up in the text with the existing lexical index (ADR-0030), deterministically.
   2. **One LLM call** (LangChain, schema-bound JSON, `CV_PROMPT_VERSION`) returns the jobs (title, employer type, start and end as YYYY-MM), education, and skills.
      - Each skill comes with the jobs it appears in and a **short evidence quote**.
      - Skill names are in English, whatever the CV's language.
      - The output is capped (at most 60 skills, `max_tokens`).
   3. **A skill is kept only if its evidence quote appears in the visible text** (after normalization), and it resolves to a catalog id.
   4. **Which LLM shape is decided in Phase 1 by the benchmark**, as in ADR-0030:
      - **phrases**, which then go through `SkillMatcher` (cached, confirmed by similarity, at most 12 new phrases per import);
      - or **catalog ids picked directly** in the same call, confirmed by similarity to the evidence.
3. **Jobs:** `POST /api/v2/cv-imports` (multipart PDF or text) returns a job id; `GET /api/v2/cv-imports/{id}` returns the status, then the suggestions.
   - Jobs live in memory. They are deleted when their suggestions are read, or after 15 minutes.
   - One import runs at a time and at most 3 wait; beyond that the answer is "busy, try again in a minute".
4. **Signed-in users only**, with a per-account quota (5 a day) on top of the per-IP "heavy" limit (ADR-0035).
   - This ADR records the decider's answer "signed-in only for now".
   - The exact quota is a setting.
5. **Levels and experience by rule** (`xcrs/domain/cv_profile.py`, pure):
   - **Years of a skill:** the union of the periods of the jobs it appears in, so overlapping jobs don't double-count.
   - **Level:** 5+ years → advanced (3); 2–5 → working (2); under 2 → basic (1). One level lower if last used more than 5 years ago. Never expert (4) automatically.
   - **Skills only in a skills list or summary** stay unrated (they count as 1, ADR-0029).
   - **Experience band** (ADR-0041): from the span of professional positions. "Student" when currently studying with no position longer than 6 months.
   - The thresholds are checked on the evaluation set.
6. **The review step** (frontend, `components/cv/`):
   - suggestions grouped by job, then "other skills";
   - each has a checkbox, a box selector (default **Neutral**), level dots and its evidence ("Kafka · 2022–now at Paylane");
   - the suggested experience band is shown too.
   - "Add to board" merges: chips already on the board keep their box. Undo works for 5 seconds.
7. **Prompt injection is detected, ignored and reported:**
   - **Hidden text** is removed before extraction:
     - invisible text rendering;
     - white or near-white fill;
     - font size under 2 pt;
     - text outside the page.
   - **Lines that read as instructions to an AI** are removed before extraction. Detection is pattern-based, in English and Turkish ("ignore previous instructions", "you are an AI", "system prompt", "önceki talimatları yok say"…).
   - **The CV is wrapped as data** in the prompt, which says to ignore instructions inside it.
   - **The review step warns** when anything was removed, and a disclosure shows up to three of the removed snippets: "We ignored part of this file: 2 pieces of hidden text and 1 line that looked like instructions to an AI. Suggestions come only from the visible text."
   - **Evidence is rendered as text, never HTML.**
8. **Nothing from the file is stored:**
   - no file, text or extraction result in the database;
   - logs carry no CV text or file names;
   - what the learner adds to the board is kept like any board (saved for a signed-in user).
   - **The privacy page** describes this before the feature ships.
9. **Evaluation before the feature** (`backend/eval/cv_import/`, `bench_cv_import.py`):
   - 15–20 synthetic CVs of invented people, including:
     - LinkedIn-style two-column exports, some with a Turkish interface;
     - classic CVs, 2–3 of them in Turkish;
     - a scanned one, an injection one with visible and hidden instructions, and a thin one.
   - **Targets:**
     - skill recall ≥ 0.8 and precision ≥ 0.85 on the Mac's 9B;
     - experience band exact on ≥ 80% of CVs;
     - under 90 s for a two-page CV on VM-B's 4B, measured only after telling the decider.
   - A miss is reported with options, not hidden.

## Results (Phase 1, 2026-10-05)

`backend/eval/bench_cv_import.py` on `eval/cv_import/cases.yaml`: 18 synthetic CVs, 16 with text, 285 expected skills. Results are in `eval/results/cv-import-20261005-1755-*` (quality) and `…-1754-*` (the run itself, with latency). Mac M4 Pro: 9B on the GPU, 4B on the CPU only (12 threads).

| Model | Shape | Precision | Recall | F1 | Band exact | LLM time, median (max) |
|---|---|---|---|---|---|---|
| (scan only, no LLM) | – | 0.98 | 0.64 | 0.77 | – | – |
| 9B, GPU | phrases | 0.94 | 0.83 | 0.88 | 16/16 | 23 s (43 s) + matcher calls |
| 9B, GPU | pick | 0.97 | 0.82 | 0.89 | 16/16 | 27 s (56 s) |
| 4B, CPU | phrases | 0.94 | **0.87** | 0.90 | 16/16 | 34 s (86 s) + matcher calls |
| **4B, CPU** | **pick** | **0.97** | 0.84 | 0.90 | 16/16 | 33 s (74 s) |

- **Inputs:**
  - hidden text and the planted instruction line were found in the injection CV;
  - no false alarms on the other 15 CVs, nor on the decider's real LinkedIn export;
  - the scanned PDF was refused with "paste the text".
- **A real 3-page LinkedIn export** (timing only, not stored): 4B on CPU took 81 s with "pick" and 60 s with "phrases". "Phrases" then also needs the matcher: a median of 5 and at most 22 more LLM calls per CV, about 5 s each on CPU.
- **Ablations** (4B, pick):
  - without the scan, recall falls to 0.66;
  - without the similarity confirmation, precision falls to 0.94;
  - the evidence check changes little here, because hidden text never reaches the LLM. A unit test covers planted skills.
- **"Pick" chosen** (`XCRS_CV_SHAPE`, default `pick`):
  - equal F1;
  - higher precision;
  - much better on Turkish CVs. "Phrases" turned "MVVM mimarisi" into Angular, React and Vue, because the matcher sees a phrase without its CV;
  - one LLM call per import, where "phrases" queues up to 12 more on the shared LLM.
- **Labels corrected after the first run:** 10 skills in 5 CVs moved from `expect` to `accept` because the text doesn't name them (Git and Python, for instance, were never mentioned). The first run's numbers on the original labels: 4B pick 0.97 / 0.81; 4B phrases 0.94 / 0.85. Each move is noted in `cases.yaml`.
- **VM-B not measured yet** (the decider's OK first). The Mac's CPU stands in for it; VM-B's server CPU is likely slower per thread.

## Trade-offs accepted

- **An import holds the CPU LLM for 1–3 minutes,** delaying other people's explanations and new-phrase matching. Mitigated by signed-in only, one at a time, a short queue and a quota.
- **Anonymous visitors can't import,** a departure from "everything works without an account". Revisited when the LLM gets cheaper.
- **pypdf trusts the file's text order.** It was right on LinkedIn's generator and Chrome's; an odd generator can mix sections. The LLM tolerates some disorder, and the paste fallback exists.
- **Years are not depth,** and suggested levels are only suggestions. The learner confirms every skill and level before anything reaches the board.
- **Pattern-based injection detection can be evaded** (paraphrase, another language). It's a warning layer, not the defence. The defence is structural: schema-bound output, verbatim evidence, catalog ids only, and results that reach only the uploader.
- **In-memory jobs are lost on a restart,** and a missed final poll loses the result; the learner imports again.
- **The LLM sometimes ties skills from the summary to the current job,** which overstates their years and level. On the Turkish LinkedIn case, four side-project skills got 2.6 years. The learner sees the years and evidence and can change the level. Follow-ups: a prompt revision measured on the set, or a section-aware guard. A guard based on text position fails on LinkedIn, because the headline repeats the title and employer.
- **An instruction sentence that wraps onto a second line** in the PDF is removed only up to the line break; the tail ("as expert level.") reaches the LLM. It is harmless on its own.
- **The synthetic evaluation set was written by Claude,** the same author as the prompt. One real LinkedIn export was checked for layout only; real CVs (with consent) should extend the set later.

## Revisit when

- **A GPU arrives, or imports measure well under a minute on VM-B** → open the import to anonymous visitors with per-IP limits.
- **Users ask for Word files, or many pasted texts start with Word artefacts** → add DOCX.
- **Real CVs show a generator whose text order pypdf mixes up** → add a layout-aware fallback (pdfminer.six with tuned settings) for that case.
- **Imports often have more than 12 phrases the lookup can't resolve** → a separate, higher cap for imports.
- **More than one API process** → move jobs and quotas to PostgreSQL or Redis (ADR-0018's revisit).
