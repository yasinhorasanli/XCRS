# CV import benchmark

Run: 2026-10-12 16:54 on MacBook-Pro; prompt `cv-extract-1 (phrases, pick), cv-extract-3 (compact), cv-extract-4 (found)`; cases: `eval/cv_import/cases.yaml`.

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
| long-li-en-backend-staff | 3052 | 0 | 0 |  | ok |
| long-classic-en-fullstack | 2117 | 0 | 0 |  | ok |
| long-classic-en-ml-research | 2291 | 0 | 0 |  | ok |
| long-li-tr-data-mid | 1269 | 0 | 0 |  | ok |
| long-classic-en-devops-sre | 1782 | 0 | 0 |  | ok |
| long-li-en-mobile-senior | 1710 | 0 | 0 |  | ok |

## Skills and experience band

| Model | Device | Shape | Variant | Precision | Recall | F1 | Band exact |
|---|---|---|---|---|---|---|---|
| - | - | scan only | scan only | 0.98 | 0.67 | 0.80 | – |
| qwen3.5:4b | gpu | compact | full | 0.96 | 0.80 | 0.87 | 1.0 |
| qwen3.5:4b | gpu | compact | no scan | 0.97 | 0.60 | 0.74 | 1.0 |
| qwen3.5:4b | gpu | compact | no evidence check | 0.96 | 0.80 | 0.87 | 1.0 |
| qwen3.5:4b | gpu | compact | no similarity check | 0.95 | 0.80 | 0.87 | 1.0 |
| qwen3.5:4b | gpu | candidates | full | 0.97 | 0.82 | 0.89 | 1.0 |
| qwen3.5:4b | gpu | candidates | no scan | 0.97 | 0.70 | 0.81 | 1.0 |
| qwen3.5:4b | gpu | candidates | no evidence check | 0.97 | 0.82 | 0.89 | 1.0 |
| qwen3.5:4b | gpu | candidates | no similarity check | 0.96 | 0.83 | 0.89 | 1.0 |

## LLM latency (one call per CV)

| Model | Device | Shape | Median s | Max s | Median output tokens | Max output tokens |
|---|---|---|---|---|---|---|
| qwen3.5:4b | gpu | compact | 6.2 | 13.3 | 274 | 577 |
| qwen3.5:4b | gpu | candidates | 7.1 | 12.9 | 312 | 641 |

## Per case (full pipeline)

### qwen3.5:4b gpu compact

| Case | Found | Right | Recall | Band | Wrong | Missed |
|---|---|---|---|---|---|---|
| li-en-data-senior | 21 | 20 | 20/27 | 5-10 | git | ab-testing, ci-cd, data-warehousing, etl-pipelines, mentoring, spreadsheets, stream-processing |
| li-tr-backend-senior | 22 | 21 | 20/27 | 5-10 | hiring | ci-cd, event-driven-architecture, incident-management, llm-fundamentals, mentoring, rest-api-design, stakeholder-communication |
| li-en-frontend-mid | 15 | 15 | 15/19 | 2-5 |  | e2e-testing, frontend-build-tools, frontend-testing, web-performance |
| li-en-ml-staff | 29 | 29 | 28/33 | 10+ |  | llm-evaluation, mentoring, model-serving, public-speaking, technical-leadership |
| li-tr-devops-mid | 17 | 17 | 16/18 | 2-5 |  | ci-cd, infrastructure-as-code |
| classic-en-student | 19 | 19 | 18/19 | student |  | c |
| classic-tr-backend | 12 | 12 | 12/19 | 5-10 |  | code-review, database-design, integration-testing, message-brokers, microservices, orm, rest-api-design |
| classic-tr-mobile | 13 | 13 | 12/15 | 2-5 |  | android-sdk, app-store-release, mobile-architecture |
| classic-en-security | 14 | 14 | 14/17 | 5-10 |  | network-security, security-compliance, security-testing-tools |
| classic-en-career-changer | 10 | 8 | 8/10 | 0-2 | calculus, powershell | data-cleaning, public-speaking |
| classic-en-embedded | 11 | 10 | 8/12 | 10+ | unity | communication-protocols, hardware-debugging, mentoring, rtos |
| classic-en-qa | 16 | 14 | 13/16 | 2-5 | exploratory-data-analysis, jupyter | ci-cd, e2e-testing, testing-fundamentals |
| classic-en-thin | 3 | 3 | 3/3 | None |  |  |
| classic-en-injection | 10 | 10 | 9/11 | 2-5 |  | rest-api-design, unit-testing |
| li-en-eng-manager | 11 | 11 | 11/17 | 10+ |  | engineering-processes, go, incident-management, people-management, project-planning, stakeholder-communication |
| classic-en-gamedev | 7 | 6 | 6/12 | 2-5 | php | game-design-basics, game-math, game-optimization, game-physics, multiplayer-networking, profiling |
| classic-tr-data-analyst | 9 | 8 | 8/10 | 0-2 | powershell | data-visualization, spreadsheets |
| long-li-en-backend-staff | 33 | 32 | 32/33 | 10+ | test-planning | message-brokers |
| long-classic-en-fullstack | 31 | 31 | 29/33 | 5-10 |  | design-systems, frontend-testing, orm, responsive-design |
| long-classic-en-ml-research | 17 | 17 | 14/19 | 5-10 |  | machine-learning-fundamentals, public-speaking, research-methods, statistics, vector-databases |
| long-li-tr-data-mid | 13 | 11 | 11/14 | 5-10 | powershell, regular-expressions | data-cleaning, data-visualization, spreadsheets |
| long-classic-en-devops-sre | 24 | 24 | 24/26 | 10+ |  | computer-networking, sre-practices |
| long-li-en-mobile-senior | 15 | 13 | 11/19 | 5-10 | concurrency, react | accessibility, android-sdk, app-store-release, ci-cd, ios-sdk, mobile-data-persistence, mobile-testing, rest-api-design |

### qwen3.5:4b gpu candidates

| Case | Found | Right | Recall | Band | Wrong | Missed |
|---|---|---|---|---|---|---|
| li-en-data-senior | 22 | 21 | 20/27 | 5-10 | git | ab-testing, ci-cd, data-warehousing, etl-pipelines, mentoring, spreadsheets, stream-processing |
| li-tr-backend-senior | 22 | 21 | 20/27 | 5-10 | hiring | ci-cd, event-driven-architecture, incident-management, llm-fundamentals, mentoring, rest-api-design, stakeholder-communication |
| li-en-frontend-mid | 18 | 18 | 16/19 | 2-5 |  | frontend-build-tools, frontend-testing, web-performance |
| li-en-ml-staff | 33 | 33 | 29/33 | 10+ |  | llm-evaluation, machine-learning-fundamentals, model-serving, technical-leadership |
| li-tr-devops-mid | 17 | 17 | 15/18 | 2-5 |  | ci-cd, infrastructure-as-code, observability |
| classic-en-student | 17 | 17 | 16/19 | student |  | c, rest-api-design, unit-testing |
| classic-tr-backend | 12 | 12 | 12/19 | 5-10 |  | code-review, database-design, integration-testing, message-brokers, microservices, orm, rest-api-design |
| classic-tr-mobile | 13 | 12 | 10/15 | 2-5 | code-review | android-sdk, ci-cd, mobile-data-persistence, mobile-testing, rest-api-design |
| classic-en-security | 16 | 16 | 16/17 | 5-10 |  | network-security |
| classic-en-career-changer | 11 | 10 | 9/10 | 0-2 | calculus | public-speaking |
| classic-en-embedded | 11 | 10 | 9/12 | 10+ | unity | communication-protocols, hardware-debugging, mentoring |
| classic-en-qa | 13 | 12 | 12/16 | 2-5 | code-review | ci-cd, e2e-testing, test-planning, testing-fundamentals |
| classic-en-thin | 3 | 3 | 3/3 | None |  |  |
| classic-en-injection | 10 | 10 | 10/11 | 2-5 |  | rest-api-design |
| li-en-eng-manager | 15 | 15 | 15/17 | 10+ |  | engineering-processes, go |
| classic-en-gamedev | 7 | 7 | 7/12 | 2-5 |  | game-design-basics, game-math, game-optimization, game-physics, oop |
| classic-tr-data-analyst | 8 | 8 | 8/10 | 0-2 |  | data-visualization, product-analytics |
| long-li-en-backend-staff | 34 | 33 | 32/33 | 10+ | test-automation | message-brokers |
| long-classic-en-fullstack | 36 | 33 | 29/33 | 5-10 | gitops, integration-testing, system-design | design-systems, frontend-testing, orm, rest-api-design |
| long-classic-en-ml-research | 17 | 17 | 15/19 | 5-10 |  | machine-learning-fundamentals, public-speaking, research-methods, statistics |
| long-li-tr-data-mid | 13 | 13 | 13/14 | 5-10 |  | spreadsheets |
| long-classic-en-devops-sre | 25 | 25 | 25/26 | 10+ |  | computer-networking |
| long-li-en-mobile-senior | 17 | 14 | 13/19 | 5-10 | concurrency, python, react | accessibility, android-sdk, ios-sdk, mobile-data-persistence, mobile-testing, rest-api-design |

