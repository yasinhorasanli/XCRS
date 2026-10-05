# ADR-0044: Job titles per role, and one built from the learner's strongest skills

- **Status:** Accepted
- **Date:** 2026-10-05
- **Decider:** Muhammed Yasin Horasanli

## Context

A result names a role ("Backend Engineer") the way the catalog does, but job ads use many titles for the same job: "Backend Developer", "Software Engineer (Backend)", and very often a language or framework in the title ("Java Developer", "Node.js Developer"). A learner who searches job sites only for the catalog name misses most ads, and doesn't see that their own stack is a title of its own.

What the catalog had: `also_called` titles for 7 of 30 roles (ADR-0027), used to keep roles distinct, not shown. What is available locally: O\*NET 31.0's `Job Titles.txt` (54,000 reported titles) lists exactly these titles under our roles' O\*NET codes: *Java Developer*, *Back End Developer*, *Node.js Developer*, *.NET Programmer*.

## Options considered

### A. A fixed list per role
- ✅ Cheap; accurate (curated from O\*NET and market usage); the same for everyone.
- ❌ Generic: it says nothing about the learner's own stack.

### B. The fixed list plus a title built from the learner's skills (chosen)
- ✅ Personal and still deterministic: the catalog says which skills go into titles (`title_skills`), the engine picks the learner's strongest one.
- ✅ Grounded: every word comes from the catalog and the learner's board.
- ❌ One more catalog field to curate, and a migration to store it.

### C. The LLM writes titles per result
- ✅ Flexible.
- ❌ Can invent titles; slower; goes against the grounded-explanations rule (ADR-0037).

## Decision

1. **Market titles:** `also_called` in `roles.yaml` now has 2–4 titles for 25 of 30 roles (74 in all), chosen from O\*NET's reported titles and common job-ad wording. Titles stay unique across roles (validator rule, ADR-0027).
2. **Title skills:** a new `title_skills` list per role (18 roles) names the languages and frameworks job ads put in that role's title, in order of how common they are. The validator requires each to be in the role's roadmap. Stored in `catalog.roles.title_skills` (migration 0013).
3. **The learner's title** (`xcrs/domain/job_titles.py`):
   - A skill counts when the learner enjoyed it (any rating), or is neutral about it at a rating of 2 or more. Curious and didn't-enjoy skills never count: a title is what someone would apply for.
   - The strongest wins: rating first (unrated counts as 2), then enjoyed over neutral, then the `title_skills` order.
   - A language goes in front, a framework or platform in brackets: "Java Backend Engineer", "Backend Engineer (Spring Boot)", "Python Backend Engineer (Django)", "Cloud Engineer (AWS)".
4. **Results** carry `title_for_you` and `job_titles` (the role's name, then the market titles) per role. The page shows them under "Job titles to search for", the learner's own first; each opens a LinkedIn Jobs search for that title (a plain link, no API).
5. `ALGORITHM_VERSION` v3.1: scoring is unchanged; results gain the titles.

## Trade-offs accepted

- The titles are curated by hand; new market titles need a catalog change (reviewed like any other).
- The personal title can read oddly for a few combinations ("TypeScript Frontend Engineer (React)"); it is still a real search phrase.
- LinkedIn Jobs is the only search link; other job sites can be added the same way.
- Stored results from before v3.1 have no titles (the page shows none for them).

## Revisit when

- Job-ad data (a scraper or an API) can measure which titles are actually used per role and region: rank titles by frequency instead of by hand.
- A title skill appears in many learners' boards but in no title: add it to `title_skills`.
