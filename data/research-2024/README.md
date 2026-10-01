# Research dataset (2024)

The catalog XCRS was built and evaluated on during the research project (see the Zenodo release linked in the root README). `uv run xcrs import-research-data` loads it into PostgreSQL.

| File | Rows | What it is |
|---|---|---|
| `udemy_courses_raw.csv` | 973 | Udemy courses as collected in 2024 |
| `udemy_courses.csv` | 453 | The courses used: categories Development, IT & Software, Office Productivity. `concat_text` (title + headline + category + what you'll learn + description) is the embedded text. Paid prices are in Turkish lira. |
| `roadmaps/*.json` | 10 roadmaps | Career roadmaps from [roadmap.sh](https://github.com/kamranahmedse/developer-roadmap) |
| `roadmap_nodes.csv` | 1,104 (869 concepts, 235 topics) | The roadmaps flattened in depth-first (learning) order. Ids encode the hierarchy two digits per level (role 6 → topic `602` → concept `60203`); the import turns that into explicit `role_id` / `parent_id` and keeps the original as `legacy_id`. |

Counts are rows, not lines: the CSVs contain multi-line fields.

Before production or commercial use, check the terms for the Udemy data and roadmap.sh's content license.
