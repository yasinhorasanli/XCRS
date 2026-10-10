# CV import benchmark

Run: 2026-10-05 17:55 on MacBook-Pro; prompt `cv-extract-1`; cases: `eval/cv_import/cases.yaml`.

Note: Quality recomputed from the cached LLM answers of the 17:54 run after 10 labels moved from expect to accept (not named in the text) and the scan matched names across line breaks; latency is from the 17:54 run.

## Inputs

| Case | Chars | Hidden | Instruction lines | Error | Checks |
|---|---|---|---|---|---|
| li-en-data-senior | 1606 | 0 | 0 |  | ok |
| li-tr-backend-senior | 1769 | 0 | 0 |  | ok |
| li-en-frontend-mid | 1122 | 0 | 0 |  | ok |
| li-en-ml-staff | 1980 | 0 | 0 |  | ok |
| li-tr-devops-mid | 1190 | 0 | 0 |  | ok |
| classic-en-student | 808 | 0 | 0 |  | ok |
| classic-tr-backend | 844 | 0 | 0 |  | ok |
| classic-tr-mobile | 812 | 0 | 0 |  | ok |
| classic-en-security | 1039 | 0 | 0 |  | ok |
| classic-en-career-changer | 899 | 0 | 0 |  | ok |
| classic-en-embedded | 873 | 0 | 0 |  | ok |
| classic-en-qa | 833 | 0 | 0 |  | ok |
| classic-en-thin | 167 | 0 | 0 |  | ok |
| classic-en-injection | 691 | 2 | 1 |  | ok |
| scanned |  |  |  | scanned | ok |
| li-en-eng-manager | 1494 | 0 | 0 |  | ok |
| classic-en-gamedev | 753 | 0 | 0 |  | ok |
| classic-tr-data-analyst | 605 | 0 | 0 |  | ok |

## Skills and experience band

| Model | Device | Shape | Variant | Precision | Recall | F1 | Band exact |
|---|---|---|---|---|---|---|---|
| - | - | scan only | scan only | 0.98 | 0.64 | 0.77 | – |
| qwen3.5:9b | gpu | phrases | full | 0.94 | 0.83 | 0.88 | 1.0 |
| qwen3.5:9b | gpu | phrases | no scan | 0.94 | 0.77 | 0.84 | 1.0 |
| qwen3.5:9b | gpu | phrases | no evidence check | 0.94 | 0.83 | 0.88 | 1.0 |
| qwen3.5:9b | gpu | pick | full | 0.97 | 0.82 | 0.89 | 1.0 |
| qwen3.5:9b | gpu | pick | no scan | 0.98 | 0.73 | 0.84 | 1.0 |
| qwen3.5:9b | gpu | pick | no evidence check | 0.97 | 0.82 | 0.89 | 1.0 |
| qwen3.5:9b | gpu | pick | no similarity check | 0.97 | 0.82 | 0.89 | 1.0 |
| qwen3.5:4b | cpu | phrases | full | 0.94 | 0.87 | 0.90 | 1.0 |
| qwen3.5:4b | cpu | phrases | no scan | 0.94 | 0.82 | 0.88 | 1.0 |
| qwen3.5:4b | cpu | phrases | no evidence check | 0.94 | 0.87 | 0.90 | 1.0 |
| qwen3.5:4b | cpu | pick | full | 0.97 | 0.84 | 0.90 | 1.0 |
| qwen3.5:4b | cpu | pick | no scan | 0.97 | 0.66 | 0.79 | 1.0 |
| qwen3.5:4b | cpu | pick | no evidence check | 0.96 | 0.84 | 0.89 | 1.0 |
| qwen3.5:4b | cpu | pick | no similarity check | 0.94 | 0.84 | 0.89 | 1.0 |

## LLM latency (one call per CV)

| Model | Device | Shape | Median s | Max s | Median output tokens | Max output tokens |
|---|---|---|---|---|---|---|
| qwen3.5:9b | gpu | phrases | 23.4 | 42.5 | 780 | 1474 |
| qwen3.5:9b | gpu | pick | 26.9 | 55.6 | 729 | 1413 |
| qwen3.5:4b | cpu | phrases | 34.4 | 85.7 | 902 | 2066 |
| qwen3.5:4b | cpu | pick | 33.1 | 74.0 | 730 | 1291 |

A real LinkedIn export (3 pages, 3841 characters; not included):

- qwen3.5:9b gpu phrases: 44.7 s, 1329 prompt tokens, 1437 output tokens
- qwen3.5:9b gpu pick: 40.3 s, 3209 prompt tokens, 1195 output tokens
- qwen3.5:4b cpu phrases: 60.0 s, 1329 prompt tokens, 1297 output tokens
- qwen3.5:4b cpu pick: 80.7 s, 3209 prompt tokens, 1295 output tokens

## Per case (full pipeline)

### qwen3.5:9b gpu phrases

| Case | Found | Right | Recall | Band | Wrong | Missed |
|---|---|---|---|---|---|---|
| li-en-data-senior | 29 | 28 | 25/27 | 5-10 | git | ci-cd, stream-processing |
| li-tr-backend-senior | 25 | 23 | 21/27 | 5-10 | database-design, security-incident-response | ci-cd, event-driven-architecture, incident-management, integration-testing, llm-fundamentals, stakeholder-communication |
| li-en-frontend-mid | 18 | 18 | 17/19 | 2-5 |  | responsive-design, web-performance |
| li-en-ml-staff | 31 | 31 | 29/33 | 10+ |  | fine-tuning, model-serving, public-speaking, technical-leadership |
| li-tr-devops-mid | 16 | 16 | 14/18 | 2-5 |  | cloud-networking, incident-management, infrastructure-as-code, observability |
| classic-en-student | 23 | 21 | 19/19 | student | frontend-testing, integration-testing |  |
| classic-tr-backend | 17 | 17 | 16/19 | 5-10 |  | database-design, integration-testing, microservices |
| classic-tr-mobile | 9 | 9 | 8/15 | 2-5 |  | android-sdk, app-store-release, ci-cd, mobile-architecture, mobile-data-persistence, mobile-testing, rest-api-design |
| classic-en-security | 18 | 17 | 15/17 | 5-10 | observability | penetration-testing, vulnerability-management |
| classic-en-career-changer | 9 | 8 | 8/10 | 0-2 | calculus | data-cleaning, public-speaking |
| classic-en-embedded | 14 | 12 | 10/12 | 10+ | observability, unity | hardware-debugging, unit-testing |
| classic-en-qa | 13 | 13 | 13/16 | 2-5 |  | ci-cd, test-planning, testing-fundamentals |
| classic-en-thin | 3 | 3 | 3/3 | None |  |  |
| classic-en-injection | 14 | 12 | 11/11 | 2-5 | frontend-testing, javascript |  |
| li-en-eng-manager | 22 | 16 | 16/17 | 10+ | caching, community-building, linux, performance-testing, profiling, security-incident-response | engineering-processes |
| classic-en-gamedev | 8 | 7 | 6/12 | 2-5 | infrastructure-as-code | game-design-basics, game-math, game-optimization, game-physics, multiplayer-networking, profiling |
| classic-tr-data-analyst | 7 | 7 | 7/10 | 0-2 |  | data-cleaning, product-analytics, statistics |

### qwen3.5:9b gpu pick

| Case | Found | Right | Recall | Band | Wrong | Missed |
|---|---|---|---|---|---|---|
| li-en-data-senior | 22 | 21 | 21/27 | 5-10 | git | ab-testing, ci-cd, data-warehousing, etl-pipelines, spreadsheets, stream-processing |
| li-tr-backend-senior | 22 | 21 | 21/27 | 5-10 | hiring | ci-cd, event-driven-architecture, incident-management, llm-fundamentals, mentoring, stakeholder-communication |
| li-en-frontend-mid | 15 | 15 | 14/19 | 2-5 |  | frontend-build-tools, frontend-testing, responsive-design, state-management, web-performance |
| li-en-ml-staff | 29 | 29 | 28/33 | 10+ |  | llm-evaluation, llm-serving, model-serving, public-speaking, technical-leadership |
| li-tr-devops-mid | 15 | 15 | 14/18 | 2-5 |  | ci-cd, cloud-networking, infrastructure-as-code, observability |
| classic-en-student | 20 | 20 | 19/19 | student |  |  |
| classic-tr-backend | 19 | 19 | 18/19 | 5-10 |  | database-design |
| classic-tr-mobile | 15 | 15 | 14/15 | 2-5 |  | mobile-testing |
| classic-en-security | 17 | 17 | 15/17 | 5-10 |  | network-security, security-testing-tools |
| classic-en-career-changer | 10 | 8 | 8/10 | 0-2 | calculus, powershell | data-cleaning, public-speaking |
| classic-en-embedded | 14 | 13 | 9/12 | 10+ | unity | communication-protocols, hardware-debugging, rtos |
| classic-en-qa | 13 | 12 | 12/16 | 2-5 | exploratory-data-analysis | ci-cd, e2e-testing, performance-testing, test-planning |
| classic-en-thin | 3 | 3 | 3/3 | None |  |  |
| classic-en-injection | 10 | 10 | 10/11 | 2-5 |  | rest-api-design |
| li-en-eng-manager | 15 | 15 | 14/17 | 10+ |  | engineering-processes, project-planning, stakeholder-communication |
| classic-en-gamedev | 8 | 8 | 8/12 | 2-5 |  | game-design-basics, game-math, oop, profiling |
| classic-tr-data-analyst | 7 | 6 | 6/10 | 0-2 | powershell | data-cleaning, data-visualization, product-analytics, spreadsheets |

### qwen3.5:4b cpu phrases

| Case | Found | Right | Recall | Band | Wrong | Missed |
|---|---|---|---|---|---|---|
| li-en-data-senior | 25 | 24 | 23/27 | 5-10 | git | ab-testing, ci-cd, mentoring, stream-processing |
| li-tr-backend-senior | 24 | 23 | 22/27 | 5-10 | security-incident-response | ci-cd, event-driven-architecture, incident-management, llm-fundamentals, stakeholder-communication |
| li-en-frontend-mid | 16 | 16 | 16/19 | 2-5 |  | code-review, responsive-design, web-performance |
| li-en-ml-staff | 31 | 31 | 29/33 | 10+ |  | mentoring, model-serving, public-speaking, technical-leadership |
| li-tr-devops-mid | 17 | 17 | 15/18 | 2-5 |  | incident-management, infrastructure-as-code, observability |
| classic-en-student | 23 | 21 | 19/19 | student | frontend-testing, integration-testing |  |
| classic-tr-backend | 20 | 20 | 18/19 | 5-10 |  | database-design |
| classic-tr-mobile | 16 | 12 | 10/15 | 2-5 | angular, frontend-architecture, react, vue | android-sdk, app-store-release, mobile-architecture, mobile-data-persistence, mobile-testing |
| classic-en-security | 19 | 18 | 16/17 | 5-10 | observability | vulnerability-management |
| classic-en-career-changer | 9 | 8 | 8/10 | 0-2 | calculus | data-cleaning, public-speaking |
| classic-en-embedded | 17 | 14 | 11/12 | 10+ | manual-testing, observability, unity | communication-protocols |
| classic-en-qa | 17 | 17 | 15/16 | 2-5 |  | ci-cd |
| classic-en-thin | 3 | 3 | 3/3 | None |  |  |
| classic-en-injection | 14 | 12 | 11/11 | 2-5 | frontend-testing, shell-scripting |  |
| li-en-eng-manager | 22 | 19 | 16/17 | 10+ | performance-testing, profiling, security-incident-response | engineering-processes |
| classic-en-gamedev | 11 | 10 | 9/12 | 2-5 | infrastructure-as-code | game-design-basics, game-physics, multiplayer-networking |
| classic-tr-data-analyst | 7 | 7 | 7/10 | 0-2 |  | data-cleaning, product-analytics, statistics |

### qwen3.5:4b cpu pick

| Case | Found | Right | Recall | Band | Wrong | Missed |
|---|---|---|---|---|---|---|
| li-en-data-senior | 25 | 22 | 21/27 | 5-10 | git, system-design, tdd | ci-cd, data-warehousing, etl-pipelines, mentoring, spreadsheets, stream-processing |
| li-tr-backend-senior | 24 | 24 | 23/27 | 5-10 |  | ci-cd, llm-fundamentals, mentoring, stakeholder-communication |
| li-en-frontend-mid | 16 | 16 | 15/19 | 2-5 |  | frontend-build-tools, frontend-testing, responsive-design, state-management |
| li-en-ml-staff | 30 | 30 | 28/33 | 10+ |  | llm-evaluation, mentoring, model-serving, public-speaking, technical-leadership |
| li-tr-devops-mid | 16 | 16 | 15/18 | 2-5 |  | ci-cd, infrastructure-as-code, observability |
| classic-en-student | 19 | 19 | 18/19 | student |  | rest-api-design |
| classic-tr-backend | 17 | 17 | 17/19 | 5-10 |  | code-review, orm |
| classic-tr-mobile | 17 | 15 | 14/15 | 2-5 | nodejs, relational-databases | mobile-architecture |
| classic-en-security | 18 | 17 | 16/17 | 5-10 | cloud-networking | security-compliance |
| classic-en-career-changer | 10 | 8 | 8/10 | 0-2 | calculus, powershell | data-cleaning, public-speaking |
| classic-en-embedded | 14 | 13 | 11/12 | 10+ | unity | hardware-debugging |
| classic-en-qa | 11 | 11 | 11/16 | 2-5 |  | ci-cd, e2e-testing, performance-testing, test-planning, testing-fundamentals |
| classic-en-thin | 3 | 3 | 3/3 | None |  |  |
| classic-en-injection | 10 | 10 | 9/11 | 2-5 |  | rest-api-design, unit-testing |
| li-en-eng-manager | 15 | 15 | 14/17 | 10+ |  | engineering-processes, people-management, stakeholder-communication |
| classic-en-gamedev | 9 | 9 | 8/12 | 2-5 |  | game-design-basics, game-math, game-optimization, multiplayer-networking |
| classic-tr-data-analyst | 8 | 8 | 8/10 | 0-2 |  | data-cleaning, spreadsheets |

