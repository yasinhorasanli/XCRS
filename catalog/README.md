# Catalog: skills, career roles and roadmaps

The reviewed source of truth for what XCRS recommends ([ADR-0025](../docs/adr/0025-skills-catalog-from-onet-esco-with-llm-learning-paths.md), [ADR-0027](../docs/adr/0027-career-roles-levels-and-transitions.md), [ADR-0028](../docs/adr/0028-catalog-as-code-skills-graph-and-prerequisites.md)). Changes arrive as pull requests, CI validates them, and an import loads the merged files into the database.

| File | Contents |
|---|---|
| [`skills.yaml`](skills.yaml) | Every skill and concept, shared by all roles, with prerequisites |
| [`roles.yaml`](roles.yaml) | The level ladder, role families, roles (with O\*NET codes), transitions between roles, and the mapping from the legacy research roles |
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
```

`validate` rejects: unknown skills or roles, prerequisite cycles, a skill asked for before its prerequisites (or at too little proficiency), proficiency that drops at a higher level, common paths inside one role or to levels a role doesn't have, and roles without a roadmap. It warns about skills no roadmap uses and about common paths whose target is far from the source.

**Moving between roles:** any move is possible. `moves` ranks every other role by coverage (the share of its requirements already met, skills weighted by how distinctive they are) and estimates the starting level (highest level ≥ 60% covered). `common_paths` in `roles.yaml` lists the moves people commonly make, as evidence.

## Reviewing a change

CI proves the files are consistent; the review is about whether they are **true**. For each changed file, ask:

1. **Roles:** is this a real, distinct kind of job (not a seniority or a rename)? Is the summary what the job actually is?
2. **Levels:** would a hiring manager agree with what each level adds? Is anything missing that every job ad asks for, or is anything niche presented as required? Use `path` to read a whole roadmap.
3. **Proficiency:** is `working` vs `advanced` right for that level?
4. **Prerequisites:** is it truly required first, or just related? (Only real prerequisites belong in `requires`.)
5. **Common paths:** is it a move people really make? Do `moves` and `bridge` show a believable distance and gap?

## Sources and attribution

Roles map to O\*NET-SOC occupations; skills list matching O\*NET technology examples (`onet:`) as evidence of employer demand. ESCO references are added by the taxonomy import.

- This catalog uses information from the **O\*NET 31.0 Database** by the U.S. Department of Labor, Employment and Training Administration (USDOL/ETA), under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/). The catalog modifies and adds to that information; USDOL/ETA has not approved, endorsed, or tested these modifications.
- ESCO, once linked: this service uses the ESCO classification of the European Commission.
- The research-era roadmaps (roadmap.sh) are not used here; they stay in the legacy catalog until it is removed.
