# ADR-0027: Career roles as specializations on one level ladder, connected by transitions

- **Status:** Accepted. Delegated: the decider asked Claude to decide the career roles and roadmaps (2026-10-01); reviewed by the decider in PR #9 and merged.
- **Changed during review (2026-10-01):** "any role to any other?" (PR #9) → any move is possible and measured; the listed transitions became common paths (decisions 4–5). "Missing roles" (PR #9) → 7 roles added and market titles recorded as aliases, with a measurable rule for which is which (decisions 1 and 7). The decider kept the alias comparison at mid level, and lowered the starting-level bar from 60% to 55% (decision 4).
- **Date:** 2026-10-01
- **Decider:** Muhammed Yasin Horasanli

## Context

- The research catalog has 10 roles from roadmap.sh, one of them a design role (UX Designer). The market has many more distinct software roles, and the decider wants realistic careers: different kinds of roles, levels within a role ("Backend Engineer → senior skills, usually ~5 years → Senior Backend Engineer"), and moves between roles.
- **Evidence (2026-10-01):**
  - Stack Overflow 2025 lists 34 developer types (full-stack 27%, back-end 14.2%, architect 6.1% as a new entry, front-end 4.3%, mobile 3%, embedded 2.8%, engineering manager 2.4%, DevOps 2.3%, data engineer 1.7%, AI/ML engineer 1.4%, data scientist 1.2%, security 1.1%, cloud infrastructure 1%, game 0.9%, QA 0.8%, SRE 0.1% …).
  - 2026 job titles add AI Engineer, MLOps Engineer and Platform Engineer as distinct, fast-growing roles.
  - Common ladders: senior about 5+ years, staff about 8–10+; levels measure scope more than tenure.
  - **O\*NET occupations are broader than market titles:** DevOps, SRE, platform and mobile all fall under 15-1252 Software Developers or 15-1299.08 Computer Systems Engineers/Architects, and O\*NET has no MLOps or AI Engineer occupation. O\*NET does list each occupation's technologies with "hot" and "in demand" flags.
- Skills, prerequisites and roadmaps are a separate decision ([ADR-0028](0028-catalog-as-code-skills-graph-and-prerequisites.md)).

## Options considered

### What a role is
- **O\*NET occupations as roles.** ✅ Authoritative. ❌ Far too broad ("Software Developers" covers back end, mobile and DevOps).
- **Market specializations, each mapped to an O\*NET parent.** ✅ Matches how jobs are advertised and how learners think; O\*NET still provides grounding and demand data. ❌ Our own list to maintain.
- **Roles per seniority** ("Senior Backend Engineer" as its own role). ❌ Duplicates every role four times; hides that it's one path.

### Seniority
- **Per-role titles only.** ❌ No shared meaning across roles.
- **One shared ladder (entry, mid, senior, staff) with per-role content.** ✅ Comparable across roles; each roadmap says what a level adds; titles can still differ (Engineering Manager vs Senior Engineering Manager).

### Moving between roles
- **None (each role isolated).** ❌ Unrealistic; career changes are a main use case.
- **Only explicit, hand-listed transitions.** ✅ Realistic examples. ❌ Implies every other move is impossible, which isn't true: people retrain into anything.
- **Any move, measured; common ones listed as evidence.** Distance computed for every pair from the two roadmaps, plus a curated list of moves people commonly make. ✅ Answers "how far am I from X?" for every role. ❌ The measure needs care: raw counts of missing skills made small roadmaps look near (Backend looked as close to Embedded as to Data Engineer).

## Decision

1. **30 roles in 8 families**, each a market specialization mapped to an O\*NET-SOC parent (ESCO to be linked by the taxonomy import):
   - Product engineering: Frontend, Backend, Full-Stack, Android, iOS, Game Developer, Embedded Software Engineer
   - Quality engineering: QA Automation Engineer (SDET)
   - Infrastructure and reliability: DevOps, Cloud, Site Reliability, AI Reliability, Platform Engineer
   - Data and AI: Data Analyst, Analytics Engineer, Data Engineer, Data Scientist, Machine Learning Engineer, Applied Scientist, MLOps Engineer, AI Platform Engineer, AI Engineer
   - Security: Security Engineer, Penetration Tester
   - Web3: Blockchain Engineer
   - Customer-facing engineering: Forward-Deployed Engineer, Solutions Engineer, Developer Advocate
   - Architecture and leadership: Software Architect, Engineering Manager
2. **One ladder:** entry (0–2 years), mid (2–5), senior (5+), staff/principal (8+), each with a scope statement. Roles start where the market hires: SRE, Platform, MLOps, AI Platform, AI Reliability and Developer Advocate at mid; Software Architect and Engineering Manager at senior (entered from senior engineering roles).
3. **Within a role, moving up a level is implied**; each roadmap level lists what it adds (senior adds system design, incident leadership, mentoring…).
4. **Any role can move to any other.** For every pair the catalog computes:
   - **coverage:** the share of the target's requirements someone already meets, each skill weighted by how *distinctive* it is (rare across roles weighs more: dbt says "data engineer", Git doesn't) and by proficiency, partial proficiency counting partly;
   - **starting level:** the highest target level at least 55% covered, never above the person's current level, since changing roles doesn't promote anyone (a senior backend engineer typically starts data engineering at entry, Software Architect at senior). The bar was 60% in the draft; the decider lowered it to 55% because moves people commonly make at the same level (Backend → Full-Stack, Data Analyst → Analytics Engineer) fell just short of 60%.

   `xcrs catalog moves ROLE@LEVEL` ranks every other role; `xcrs catalog bridge` lists the skills to learn. Results match practice, e.g. from DevOps@senior: Cloud 100%, Platform 91%, SRE 79% (starting at mid); from Data Analyst@mid: Data Scientist 62% (starting at entry).
5. **65 common paths** (e.g. Backend@senior → Software Architect@senior, "lead", typically 8+ years) record moves people really make, typed broaden / specialize / pivot / lead. They are evidence, not a whitelist. The validator flags a non-leadership common path whose target is in the far half of the source's ranked moves (today: Backend → Blockchain, Backend@senior → Platform, Data Engineer → ML Engineer), for a reviewer to confirm.
6. **The research roles become a separate legacy catalog**, mapped to the closest new role for comparison (UX Designer has no counterpart: design is outside this catalog's software-engineering scope). They are removed once the new catalog proves better.
7. **Role or alias, by measurement.** Other market titles for a role are kept as `also_called` (41 titles in all), so search and job-title matching find the role. A title may list `adds`, the skills it asks for on top of the role. **It stays an alias while the role covers at least 80% of it at mid level** (where most hiring happens); below that it is a different job and must be a role with its own roadmap. The validator enforces it. In the first review this measured Product Engineer at 81% of Full-Stack (alias) and AI Platform Engineer and AI Reliability Engineer at 71% and 73% of MLOps and SRE (roles). At senior level those two rise to 87% and 81%, because senior MLOps and SRE engineers pick up LLM serving, so they are distinct mainly at the level people are hired into them.

## Trade-offs accepted

- The role list is ours, not a standard; O\*NET mapping keeps it anchored.
- Years per level are guidance; real promotion depends on scope and impact.
- Dropping UX Designer narrows the product to software engineering.
- 30 roadmaps to keep current; the catalog workflow (ADR-0028) makes changes reviewable.
- The distance is only as good as the roadmaps, and the weights, the 55% starting threshold and the 80% alias threshold are judgement calls; they are constants in `xcrs/catalog/validate.py`, to tune with real feedback.

## Revisit when

- Job-market data shows a new distinct role (or one fading) — e.g. the next Stack Overflow survey.
- Learners ask for non-engineering tech roles (product, design, data governance).
- Feedback shows a level's content doesn't match real expectations.
