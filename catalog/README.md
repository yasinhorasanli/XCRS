# Catalog: skills, career roles and roadmaps

The reviewed source of truth for what XCRS recommends ([ADR-0025](../docs/adr/0025-skills-catalog-from-onet-esco-with-llm-learning-paths.md), [ADR-0027](../docs/adr/0027-career-roles-levels-and-transitions.md), [ADR-0028](../docs/adr/0028-catalog-as-code-skills-graph-and-prerequisites.md)). Changes arrive as pull requests, CI validates them, and an import loads the merged files into the database.

| File | Contents |
|---|---|
| [`skills.yaml`](skills.yaml) | Every skill and concept, shared by all roles, with prerequisites |
| [`roles.yaml`](roles.yaml) | The level ladder, role families, roles (with O\*NET codes and other market titles), common paths between roles, and the mapping from the legacy research roles |
| [`roadmaps/<role>.yaml`](roadmaps) | For each role and level: what the level adds, in stages |

## Notation

- **Proficiency** 1–4: 1 basic (knows what it is, follows guides) · 2 working (uses it on routine work) · 3 advanced (designs with it, solves hard problems, guides others) · 4 expert (sets direction).
- **`skill:2`**: that skill at working proficiency. **`a|b|c:2`**: any one of them (a choice, e.g. one back-end language).
- **Prerequisites** (`requires` in `skills.yaml`): every entry must hold; an entry with `|` is satisfied by any of its options.
- **Roadmap levels are cumulative.** `senior` lists only what it adds or raises over `mid`. A role may start above `entry` (Site Reliability Engineer starts at `mid`; Software Architect at `senior`), and then its first level lists the whole profile.
- **Stages** order a level's skills. Skills in one stage may depend on each other; a later stage may depend on earlier ones. A stage marked `optional: true` is "good to know": it must meet its own prerequisites but never counts as one.

## Commands (from `backend/`)

```bash
uv run xcrs catalog validate                                            # what CI runs
uv run xcrs catalog path backend-engineer@senior                        # a roadmap, level by level
uv run xcrs catalog moves backend-engineer@mid                          # every other role, nearest first, with a starting level
uv run xcrs catalog bridge backend-engineer@mid data-engineer@mid        # the skills a move asks for
uv run xcrs catalog import                                              # load into the database (after merging)
```

`validate` rejects: unknown skills or roles, prerequisite cycles, a skill asked for before its prerequisites (or at too little proficiency), proficiency that drops at a higher level, common paths inside one role or to levels a role doesn't have, and roles without a roadmap. It warns about skills no roadmap uses and about common paths whose target is far from the source.

**Moving between roles:** any move is possible. `moves` ranks every other role by coverage (the share of its requirements already met, skills weighted by how distinctive they are) and estimates the starting level (highest level ≥ 55% covered). `common_paths` in `roles.yaml` lists the moves people commonly make, as evidence.

**Role or alias:** other market titles go in a role's `also_called`. A title can list `adds`, the skills it asks for on top of the role; `validate` rejects a title the role covers less than 80% of at mid level, because that is a different job and needs its own role and roadmap.

## Learning resources

`resources.yaml` lists curated learning resources (ADR-0033): URL, title, provider, `type` (docs, course, tutorial, video, playlist, book), `level` (beginner, intermediate, advanced), `free`, and `teaches` (skills with the proficiency the resource gets you to). `validate` checks structure and skill references; `uv run xcrs resources check-links` checks the links (needs the network, never in CI). `sources/youtube.yaml` lists YouTube playlists for the YouTube adapter.

### YouTube playlists (ADR-0033)

1. **Get an API key** (free, no billing):
   - In the [Google Cloud console](https://console.cloud.google.com/), create a project.
   - *APIs & Services → Library* → enable **YouTube Data API v3**.
   - *Credentials → Create credentials → API key*, then restrict the key to the YouTube Data API v3.
   - Put it in the repository's `.env` as `XCRS_YOUTUBE_API_KEY=...` (never commit it).
2. `uv run xcrs resources youtube-discover` searches playlists for the skills most roles rely on first, at most 90 searches a run (the free quota allows about 95 a day; later runs continue). Candidate ids go to `sources/youtube-candidates.yaml`; titles go to `untracked/youtube-candidates.md` for review.
3. Approve a candidate by moving its playlist id to `sources/youtube.yaml`. Then run `uv run xcrs resources ingest youtube` and `uv run xcrs resources tag`.
4. YouTube data must be refreshed within 30 days: re-run `ingest youtube` regularly; `xcrs resources expire` deletes what wasn't refreshed.

## Taxonomy evidence

- **ESCO:** each role's `esco:` is the closest ESCO occupation (ESCO © European Union; reuse under the ESCO terms with attribution). It is `null` where ESCO has none (DevRel, MLOps, AI Platform, AI Reliability).
- **O\*NET:** `uv run xcrs catalog coverage` writes `docs/catalog-coverage.md`, comparing each role's roadmap with the technologies O\*NET 31.0 marks hot or in demand for its occupation. It needs the O\*NET text files in `data/taxonomy/raw/` and is evidence for review, not an automatic change.

## Reviewing a change

CI proves the files are consistent; the review is about whether they are **true**. For each changed file, ask:

1. **Roles:** is this a real, distinct kind of job (not a seniority or a rename)? Is the summary what the job actually is? Is a missing title better added as an alias (`also_called`)?
2. **Levels:** would a hiring manager agree with what each level adds? Is anything missing that every job ad asks for, or is anything niche presented as required? Use `path` to read a whole roadmap.
3. **Proficiency:** is `working` vs `advanced` right for that level?
4. **Prerequisites:** is it truly required first, or just related? (Only real prerequisites belong in `requires`.)
5. **Resources:** is it the best free resource for that skill, and does it really get a learner to the proficiency listed?
6. **Common paths:** is it a move people really make? Do `moves` and `bridge` show a believable distance and gap?

## Sources and attribution

Roles map to O\*NET-SOC occupations; skills list matching O\*NET technology examples (`onet:`) as evidence of employer demand. ESCO references are added by the taxonomy import.

- This catalog uses information from the **O\*NET 31.0 Database** by the U.S. Department of Labor, Employment and Training Administration (USDOL/ETA), under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/). The catalog modifies and adds to that information; USDOL/ETA has not approved, endorsed, or tested these modifications.
- ESCO, once linked: this service uses the ESCO classification of the European Commission.
- The research-era roadmaps (roadmap.sh) are not used here; they stay in the legacy catalog until it is removed.
