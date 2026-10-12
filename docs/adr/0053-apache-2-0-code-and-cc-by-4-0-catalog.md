# ADR-0053: The code is licensed under Apache-2.0 and the catalog under CC BY 4.0

- **Status:** Accepted
- **Date:** 2026-10-12
- **Decider:** Muhammed Yasin Horasanli
- **Replaces:** the GPL-3.0-or-later license the 2024 research prototype was published under (no ADR)

## Context

- The repository was GPL-3.0-or-later, a choice carried over from the 2024 research prototype, not made for the app as it is now.
- XCRS runs as a website (xcrs.app). GPL's duty to share source applies only when the software is distributed, so for a hosted service it acts almost like a permissive license.
- `catalog/` is data, not code (skills, roles, roadmaps, resources), and it adapts O\*NET 31.0, which is CC BY 4.0. A code license is a poor fit for it.
- Git history shows a single author, and the current code is a rewrite: the prototype was deleted on 2026-10-01 and no source files carry over. So the decider can choose the license. The research papers' co-authors are credited for the research; authorship of a paper gives no copyright in the code.
- Dependencies allow any choice: the backend's are permissive except psycopg (LGPL-3.0, used as a library); the frontend's production dependencies are MIT, ISC, BSD, Apache-2.0, BlueOak, MPL-2.0 (lightningcss) and CC0, with node-forge dual-licensed BSD-3-Clause or GPL-2.0.
- The decider's goals: a production-ready XCRS and CV-worthy experience. There is no plan to charge for XCRS today.

## Options considered

### Code

#### A: Keep GPL-3.0-or-later
- ✅ No change.
- ❌ It doesn't cover the way XCRS runs (a hosted service), and it discourages reuse anyway.

#### B: AGPL-3.0-or-later
- ✅ Anyone running a changed copy as a service must offer its source, so there are no closed clones of xcrs.app.
- ❌ Many companies forbid AGPL code, so fewer people read, reuse or contribute. A footer link to the source is required.

#### C: Apache-2.0 (chosen)
- ✅ Anyone can use, change and sell it with no conditions beyond keeping the notices. That is the most reuse and the cleanest CV story ("open source, Apache-2.0, live at xcrs.app").
- ✅ Over MIT: an explicit patent grant from contributors, contributions are under the same license by default (no contributor agreement needed), and no trademark rights are granted, so copies can't present themselves as XCRS.
- ❌ Someone can run or sell a closed, changed copy.

#### D: MIT
- ✅ The shortest and best-known license.
- ❌ It says nothing about patents, contributions or trademarks.

#### E: Source-available (FSL, BSL, PolyForm Noncommercial)
- ✅ Blocks competing commercial use. FSL becomes Apache-2.0 after two years.
- ❌ Not open source; fewer users and contributors, and a weaker CV story. Too much for a project with no business plan.

### Catalog

#### CC BY 4.0 (chosen)
- ✅ A license made for data; the same one O\*NET uses, so its attribution carries over cleanly. Reuse only requires credit.
- ❌ Changed copies may be closed.

#### CC BY-SA 4.0
- ✅ Changed copies must stay open.
- ❌ Harder to reuse, for little gain.

## Decision

The code is licensed under **Apache-2.0** (`LICENSE`, `NOTICE`), and the catalog data under **CC BY 4.0** (`catalog/LICENSE`, credit "XCRS catalog" with a link to the repository). The main reason: a copy of the code isn't the product. The value is the curated catalog, the tuning and running the service, and the bigger risk for XCRS is not being noticed, not being copied.

Third-party content keeps its own terms: O\*NET and ESCO data (attribution in `catalog/README.md` and `NOTICE`), the titles of linked courses and videos, and YouTube data (YouTube Terms of Service). Versions up to the tag `research-prototype` and the Zenodo release stay GPL-3.0-or-later.

## Trade-offs accepted

- Anyone may run or sell a closed copy without giving changes back. Mitigation: the name isn't licensed, and the catalog's upkeep, the tuning and the live service stay with XCRS.
- Released versions can't be taken back. Because there is a single author, future versions can still move to another license (e.g. FSL or AGPL) if that changes.
- Outside contributions arrive under Apache-2.0 (section 5). If the license may need to change later, those contributors have to agree, or their code has to be replaced.

## Revisit when

- XCRS gets a business plan (charging for it), and a competing hosted copy appears or becomes likely.
- Outside contributors join, if a later license change is being considered.
