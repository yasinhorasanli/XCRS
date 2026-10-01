# Engine v2 vs legacy: 2026-10-02T01:27 on MacBook-Pro.local

Labeled learner profiles (`eval/learner_profiles.yaml`). The legacy engine gets the skills' names as phrases;
its roles are mapped to the new catalog (`legacy_roles`). hit@1 = first role expected or acceptable;
hit@3 = the expected role among the first three.

| Profiles | n | Legacy hit@1 | Legacy hit@3 | v2 hit@1 | v2 hit@3 |
|---|---|---|---|---|---|
| expected role exists in the legacy catalog | 21 | 86% | 100% | **95%** | **100%** |
| all | 53 | 51% | 40% | **98%** | **100%** |

Median latency per profile (ms): legacy (embed + threshold + course matches) 131, v2 scoring (skills given) 2

Phrase profiles (`eval/profiles.json`, 15): same first role 53%, top-3 overlap 44%.

| Profile | Legacy top 3 (mapped) | v2 top 3 |
|---|---|---|
| backend-java | devops-engineer, data-scientist, backend-engineer | backend-engineer, devops-engineer, qa-automation-engineer |
| frontend | full-stack-engineer, frontend-engineer, backend-engineer | frontend-engineer, full-stack-engineer, blockchain-engineer |
| data-science | data-scientist, game-developer, blockchain-engineer | data-scientist, machine-learning-engineer, applied-scientist |
| devops | devops-engineer, full-stack-engineer, qa-automation-engineer | cloud-engineer, devops-engineer, penetration-tester |
| android | android-engineer, backend-engineer | android-engineer, ios-engineer, qa-automation-engineer |
| game | game-developer, backend-engineer | game-developer, embedded-software-engineer, applied-scientist |
| blockchain | blockchain-engineer, full-stack-engineer, backend-engineer | blockchain-engineer, qa-automation-engineer, solutions-engineer |
| qa | qa-automation-engineer, frontend-engineer, backend-engineer | qa-automation-engineer, frontend-engineer, full-stack-engineer |
| ux | (ux), qa-automation-engineer, frontend-engineer | frontend-engineer, android-engineer, ios-engineer |
| fullstack | full-stack-engineer, frontend-engineer, backend-engineer | full-stack-engineer, backend-engineer, frontend-engineer |
| edge-single-fact | data-scientist, backend-engineer | data-analyst, analytics-engineer, data-engineer |
| edge-curious-only | data-scientist, devops-engineer | data-scientist, machine-learning-engineer, cloud-engineer |
| edge-misattribution | devops-engineer, backend-engineer | devops-engineer, cloud-engineer, backend-engineer |
| edge-disliked-heavy | data-scientist, blockchain-engineer, devops-engineer | cloud-engineer, devops-engineer, data-engineer |
| edge-abbreviations | devops-engineer, full-stack-engineer, qa-automation-engineer | game-developer, backend-engineer, full-stack-engineer |

v2 first-role misses where the legacy engine could have named the role:

- likes-backend-hates-frontend: expected backend-engineer, v2 ['ai-engineer', 'backend-engineer', 'forward-deployed-engineer'], legacy ['data-scientist', 'backend-engineer', 'devops-engineer']
