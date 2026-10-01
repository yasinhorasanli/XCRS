# ADR-0029: Engine v2: catalog skills with free-text fallback, optional proficiency, coverage × interest scoring, matching thresholds set by measurement

- **Status:** Accepted
- **Date:** 2026-10-01
- **Decider:** Muhammed Yasin Horasanli

## Context

- The new catalog (ADR-0025–0028) is reviewed and in the database (`catalog` schema, PR #10): 257 skills, 30 roles with levels, roadmaps and common paths. The live recommender still uses the 2024 research data (869 roadmap.sh concepts, 453 Udemy courses), so none of the new catalog reaches users yet.
- **Today's matching** (ADR-0010, ADR-0022): a typed phrase is embedded with an instruction prefix (`qwen3-embedding:0.6b`) and compared with every concept, each embedded from its full markdown article. A concept matches above mean + 2.5σ of that phrase's similarities; a phrase that matches nothing falls back to its near-best concepts above 1.5σ.
  - The two sides look different: a two-word phrase against a long article.
  - The σ rule is relative, so some match always wins, even for input that matches nothing in the catalog.
  - The constants were tuned for 869 long concepts, not 257 short skills.
- The skill board's four categories (enjoyed, neutral, didn't enjoy, curious) say what a learner *likes*, not how well they *know* it, so a learner's level can't be estimated today.
- The catalog already measures distance between roles: requirements covered, weighted by how distinctive each skill is and by proficiency (ADR-0027).

## Options considered

### Order of the remaining work
- **Engine v2 → resources → taxonomy check → operations → AWS.** ✅ The new catalog reaches the UI first, and later work can be measured against it. ❌ Results show roles, levels and skill gaps before resources exist.
- **Resources first, then the engine.** ✅ The engine is built once, with courses. ❌ A long stretch before anything visible changes, and no skill gaps to tell us which resources to collect first.
- **Taxonomy check first.** ✅ Stronger evidence for the catalog. ❌ Improves the evidence, not the product.

### How learners enter skills
- **Catalog picker only.** ✅ Exact, no matching errors. ❌ Anything missing from the catalog can't be expressed.
- **Free text only (today).** ✅ Most flexible. ❌ Fuzzy matching on every input.
- **Picker plus free text.** Catalog skills are suggested first; typed text falls back to matching. ✅ Exact where possible, flexible otherwise. ❌ Two input paths to build.

### Matching free text to skills
- **Instruction-aware embedding only, thresholds re-tuned.** ✅ Fast, no LLM. ❌ Short phrase vs. skill description stays asymmetric.
- **Always add an LLM definition before embedding.** ✅ Symmetric comparison. ❌ An LLM call for input that a name lookup would match.
- **Layered: name/alias lookup → LLM definition (cached) + embedding.** ✅ The cheap exact path first; definitions only where needed, and once per phrase. ❌ More parts; the definition step must prove it helps.

### Proficiency
- **None.** ✅ Simplest. ❌ No level estimate.
- **Optional 1–4 per skill chip** (basic / working / advanced / expert, the catalog's scale). ✅ Enables level estimates and starting-level advice. ❌ One more click per skill; self-ratings are noisy.
- **Years of experience per role.** ✅ Cheap to ask. ❌ Coarse; says nothing about individual skills.

### Role scoring
- **Today's weighted sum, on skills** (curious 1, liked .75, neutral .5, disliked −.25 → sigmoid). ✅ Familiar. ❌ Ignores how much of a role someone covers, and the catalog's proficiencies.
- **Coverage × interest.** ✅ One measure across the product (the same coverage as `xcrs catalog moves`); explainable ("you cover 62% of Data Engineer, and you're curious about its core skills"). ❌ A new formula to calibrate.

## Decision

1. **Order:** engine v2 → learning resources (ADR-0026, with ADR-0004) → taxonomy check (ADR-0025) → operations (backup schedule, VM deployment, rate limiting) → AWS.
2. **Input:** catalog skills (picked, with search over names, aliases and role titles) plus free text.
3. **Matching free text** (the decider's suggestion: give typed input a definition, so it compares like with like):
   1. Picked skills map to their id; no embedding.
   2. Exact or near-exact names (aliases, abbreviations such as "k8s", typos, O\*NET names) are looked up.
   3. Anything else: the local LLM writes a one-line definition in the catalog's description style, cached per normalized phrase; the definition is embedded and compared with skills embedded as *name + description*.
   4. **Thresholds are set by measurement:** an absolute similarity floor plus a margin over the next-best skill (no relative σ rule), chosen on an **evaluation set** of 349 phrases (`backend/eval/skill_matching.yaml`) with the skills they should match. The set includes synonyms, typos, vague phrases and phrases that should match nothing. Claude drafts it; the decider reviews a random sample of 50 in a pull request.
   5. The same set compares raw phrase, phrase + instruction and LLM definition on precision, recall and latency. The definition step stays only if it measurably helps. The measured choice and constants go into ADR-0030.
4. **Proficiency:** an optional 1–4 rating per skill on the board; an unrated skill counts as basic (1) for coverage.
5. **Scoring:** a role's score combines **coverage** (of each level's requirements, by distinctiveness and proficiency, as in ADR-0027) with **interest** from the board categories. The best-covered level is the learner's estimated level, and the missing skills are the gaps that resources will fill. The exact formula is calibrated with the evaluation profiles.

## Trade-offs accepted

- Results show roles, levels and gaps before resources exist. The legacy courses are not mixed into the new results.
- An LLM step in matching adds latency on first sight of a phrase (seconds on the CPU VM); the cache makes repeats free, and the measurement decides whether it stays.
- About 300 of the 349 evaluation cases are reviewed only by sampling; an error rate in the sample above about 10% means reviewing the whole set.
- Self-rated proficiency is noisy; it's optional, and the explanation says how the level was estimated.
- The old σ thresholds and concept matching remain only for the legacy engine until the switch-over.

## Revisit when

- The evaluation shows the definition step doesn't improve precision or recall enough to justify its latency → drop it.
- Matching fails on real user input that the evaluation set didn't cover → extend the set and re-tune.
- Learners skip the proficiency rating most of the time → infer it (e.g. from years of experience) or remove it.
