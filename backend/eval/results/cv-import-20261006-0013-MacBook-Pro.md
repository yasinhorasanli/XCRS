# CV import benchmark

Run: 2026-10-06 00:13 on MacBook-Pro; prompt `cv-extract-1 (phrases, pick), cv-extract-3 (compact), cv-extract-4 (found)`; cases: `eval/cv_import/cases.yaml`.

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
| qwen3.5:9b | gpu | compact | full | 0.97 | 0.79 | 0.87 | 1.0 |
| qwen3.5:9b | gpu | compact | no scan | 0.98 | 0.63 | 0.77 | 1.0 |
| qwen3.5:9b | gpu | compact | no evidence check | 0.96 | 0.79 | 0.87 | 1.0 |
| qwen3.5:9b | gpu | compact | no similarity check | 0.96 | 0.79 | 0.87 | 1.0 |
| qwen3.5:9b | gpu | found | full | 0.91 | 0.74 | 0.81 | 0.824 |
| qwen3.5:9b | gpu | found | no evidence check | 0.81 | 0.74 | 0.78 | 0.824 |
| qwen3.5:9b | gpu | found | no similarity check | 0.86 | 0.74 | 0.80 | 0.824 |
| qwen3.5:4b | cpu | phrases | full | 0.94 | 0.87 | 0.90 | 1.0 |
| qwen3.5:4b | cpu | phrases | no scan | 0.94 | 0.82 | 0.88 | 1.0 |
| qwen3.5:4b | cpu | phrases | no evidence check | 0.94 | 0.87 | 0.90 | 1.0 |
| qwen3.5:4b | cpu | pick | full | 0.97 | 0.84 | 0.90 | 1.0 |
| qwen3.5:4b | cpu | pick | no scan | 0.97 | 0.66 | 0.79 | 1.0 |
| qwen3.5:4b | cpu | pick | no evidence check | 0.96 | 0.84 | 0.89 | 1.0 |
| qwen3.5:4b | cpu | pick | no similarity check | 0.94 | 0.84 | 0.89 | 1.0 |
| qwen3.5:4b | cpu | compact | full | 0.96 | 0.80 | 0.87 | 1.0 |
| qwen3.5:4b | cpu | compact | no scan | 0.95 | 0.62 | 0.75 | 1.0 |
| qwen3.5:4b | cpu | compact | no evidence check | 0.96 | 0.80 | 0.87 | 1.0 |
| qwen3.5:4b | cpu | compact | no similarity check | 0.94 | 0.81 | 0.88 | 1.0 |
| qwen3.5:4b | cpu | found | full | 0.78 | 0.72 | 0.75 | 0.941 |
| qwen3.5:4b | cpu | found | no evidence check | 0.69 | 0.73 | 0.71 | 0.941 |
| qwen3.5:4b | cpu | found | no similarity check | 0.43 | 0.73 | 0.54 | 0.941 |

## LLM latency (one call per CV)

| Model | Device | Shape | Median s | Max s | Median output tokens | Max output tokens |
|---|---|---|---|---|---|---|
| qwen3.5:9b | gpu | phrases | 23.4 | 42.5 | 780 | 1474 |
| qwen3.5:9b | gpu | pick | 26.9 | 55.6 | 729 | 1413 |
| qwen3.5:9b | gpu | compact | 8.7 | 26.0 | 271 | 414 |
| qwen3.5:9b | gpu | found | 11.3 | 83.4 | 319 | 3000 |
| qwen3.5:4b | cpu | phrases | 34.4 | 85.7 | 902 | 2066 |
| qwen3.5:4b | cpu | pick | 33.1 | 74.0 | 730 | 1291 |
| qwen3.5:4b | cpu | compact | 18.7 | 57.1 | 266 | 535 |
| qwen3.5:4b | cpu | found | 24.9 | 195.2 | 333 | 3000 |

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

### qwen3.5:9b gpu compact

| Case | Found | Right | Recall | Band | Wrong | Missed |
|---|---|---|---|---|---|---|
| li-en-data-senior | 22 | 21 | 21/27 | 5-10 | git | ab-testing, ci-cd, data-warehousing, etl-pipelines, spreadsheets, stream-processing |
| li-tr-backend-senior | 21 | 20 | 20/27 | 5-10 | hiring | ci-cd, event-driven-architecture, incident-management, llm-fundamentals, mentoring, rest-api-design, stakeholder-communication |
| li-en-frontend-mid | 16 | 16 | 15/19 | 2-5 |  | frontend-build-tools, frontend-testing, responsive-design, web-performance |
| li-en-ml-staff | 26 | 26 | 26/33 | 10+ |  | ab-testing, llm-evaluation, llm-serving, mentoring, model-serving, public-speaking, technical-leadership |
| li-tr-devops-mid | 15 | 15 | 14/18 | 2-5 |  | ci-cd, cloud-networking, infrastructure-as-code, observability |
| classic-en-student | 21 | 21 | 19/19 | student |  |  |
| classic-tr-backend | 19 | 19 | 18/19 | 5-10 |  | orm |
| classic-tr-mobile | 15 | 15 | 14/15 | 2-5 |  | mobile-testing |
| classic-en-security | 12 | 12 | 11/17 | 5-10 |  | identity-access-management, network-security, security-compliance, security-incident-response, security-testing-tools, web-security |
| classic-en-career-changer | 10 | 8 | 8/10 | 0-2 | calculus, clean-code | data-cleaning, public-speaking |
| classic-en-embedded | 10 | 9 | 8/12 | 10+ | unity | communication-protocols, hardware-debugging, mentoring, rtos |
| classic-en-qa | 17 | 16 | 14/16 | 2-5 | exploratory-data-analysis | ci-cd, e2e-testing |
| classic-en-thin | 3 | 3 | 3/3 | None |  |  |
| classic-en-injection | 9 | 9 | 9/11 | 2-5 |  | rest-api-design, unit-testing |
| li-en-eng-manager | 12 | 12 | 12/17 | 10+ |  | engineering-processes, mentoring, people-management, project-planning, stakeholder-communication |
| classic-en-gamedev | 7 | 7 | 7/12 | 2-5 |  | game-design-basics, game-math, game-optimization, oop, profiling |
| classic-tr-data-analyst | 7 | 6 | 6/10 | 0-2 | powershell | data-visualization, product-analytics, spreadsheets, statistics |

### qwen3.5:9b gpu found

| Case | Found | Right | Recall | Band | Wrong | Missed |
|---|---|---|---|---|---|---|
| li-en-data-senior | 27 | 25 | 23/27 | 5-10 | data-cleaning, git | ci-cd, etl-pipelines, mentoring, spreadsheets |
| li-tr-backend-senior | 18 | 18 | 18/27 | None ✗ |  | ci-cd, code-review, event-driven-architecture, incident-management, integration-testing, llm-fundamentals, mentoring, rest-api-design, stakeholder-communication |
| li-en-frontend-mid | 12 | 12 | 12/19 | None ✗ |  | code-review, e2e-testing, frontend-build-tools, frontend-testing, responsive-design, state-management, web-performance |
| li-en-ml-staff | 38 | 33 | 27/33 | 10+ | agile-scrum, data-modeling, kafka, relational-databases, scala-or-java-for-data | ab-testing, llm-evaluation, llm-serving, public-speaking, technical-leadership, vector-databases |
| li-tr-devops-mid | 13 | 13 | 12/18 | 2-5 |  | ci-cd, cloud-networking, incident-management, infrastructure-as-code, observability, secrets-management |
| classic-en-student | 27 | 24 | 18/19 | student | data-cleaning, react-native, responsive-design | c |
| classic-tr-backend | 12 | 12 | 12/19 | 5-10 |  | code-review, database-design, integration-testing, message-brokers, microservices, orm, rest-api-design |
| classic-tr-mobile | 16 | 15 | 14/15 | 2-5 | javascript | mobile-testing |
| classic-en-security | 19 | 15 | 12/17 | 5-10 | cloud-architecture, cloud-fundamentals, cloud-networking, operating-systems | network-security, penetration-testing, security-compliance, security-incident-response, web-security |
| classic-en-career-changer | 8 | 7 | 7/10 | None ✗ | calculus | data-cleaning, data-visualization, public-speaking |
| classic-en-embedded | 9 | 8 | 7/12 | 10+ | unity | communication-protocols, hardware-debugging, mentoring, rtos, unit-testing |
| classic-en-qa | 16 | 14 | 13/16 | 2-5 | api-security, data-quality | e2e-testing, performance-testing, test-planning |
| classic-en-thin | 3 | 3 | 3/3 | None |  |  |
| classic-en-injection | 13 | 13 | 9/11 | 2-5 |  | rest-api-design, unit-testing |
| li-en-eng-manager | 15 | 12 | 11/17 | 10+ | go-web-services, javascript, kotlin-coroutines | code-review, engineering-processes, go, people-management, project-planning, stakeholder-communication |
| classic-en-gamedev | 7 | 5 | 5/12 | 2-5 | design-systems, gitops | game-design-basics, game-math, game-optimization, game-physics, multiplayer-networking, oop, profiling |
| classic-tr-data-analyst | 9 | 8 | 8/10 | 0-2 | powershell | ab-testing, product-analytics |

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

### qwen3.5:4b cpu compact

| Case | Found | Right | Recall | Band | Wrong | Missed |
|---|---|---|---|---|---|---|
| li-en-data-senior | 22 | 21 | 21/27 | 5-10 | git | ci-cd, data-warehousing, etl-pipelines, mentoring, spreadsheets, stream-processing |
| li-tr-backend-senior | 23 | 22 | 21/27 | 5-10 | hiring | ci-cd, event-driven-architecture, incident-management, mentoring, rest-api-design, stakeholder-communication |
| li-en-frontend-mid | 19 | 19 | 18/19 | 2-5 |  | frontend-build-tools |
| li-en-ml-staff | 29 | 29 | 28/33 | 10+ |  | llm-evaluation, mentoring, model-serving, public-speaking, technical-leadership |
| li-tr-devops-mid | 16 | 15 | 14/18 | 2-5 | vulnerability-management | cloud-networking, infrastructure-as-code, observability, secrets-management |
| classic-en-student | 17 | 17 | 16/19 | student |  | c, rest-api-design, unit-testing |
| classic-tr-backend | 18 | 17 | 17/19 | 5-10 | opentelemetry | database-design, orm |
| classic-tr-mobile | 13 | 12 | 11/15 | 2-5 | e2e-testing | android-sdk, app-store-release, mobile-architecture, mobile-testing |
| classic-en-security | 17 | 16 | 16/17 | 5-10 | php | security-testing-tools |
| classic-en-career-changer | 10 | 9 | 9/10 | 0-2 | calculus | public-speaking |
| classic-en-embedded | 11 | 10 | 9/12 | 10+ | unity | communication-protocols, hardware-debugging, mentoring |
| classic-en-qa | 17 | 16 | 14/16 | 2-5 | exploratory-data-analysis | e2e-testing, testing-fundamentals |
| classic-en-thin | 3 | 3 | 3/3 | None |  |  |
| classic-en-injection | 9 | 9 | 9/11 | 2-5 |  | rest-api-design, unit-testing |
| li-en-eng-manager | 11 | 11 | 11/17 | 10+ |  | engineering-processes, go, incident-management, people-management, project-planning, stakeholder-communication |
| classic-en-gamedev | 7 | 6 | 6/12 | 2-5 | php | game-design-basics, game-math, game-optimization, game-physics, multiplayer-networking, profiling |
| classic-tr-data-analyst | 7 | 6 | 6/10 | 0-2 | powershell | data-visualization, product-analytics, spreadsheets, statistics |

### qwen3.5:4b cpu found

| Case | Found | Right | Recall | Band | Wrong | Missed |
|---|---|---|---|---|---|---|
| li-en-data-senior | 21 | 20 | 20/27 | 5-10 | git | ab-testing, ci-cd, data-warehousing, etl-pipelines, mentoring, spreadsheets, stream-processing |
| li-tr-backend-senior | 18 | 18 | 18/27 | None ✗ |  | ci-cd, code-review, event-driven-architecture, incident-management, integration-testing, llm-fundamentals, mentoring, rest-api-design, stakeholder-communication |
| li-en-frontend-mid | 18 | 15 | 14/19 | 2-5 | design-patterns, gitops, react-native | frontend-build-tools, frontend-testing, responsive-design, state-management, web-performance |
| li-en-ml-staff | 35 | 31 | 25/33 | 10+ | github-actions, kafka, prometheus-grafana, scala-or-java-for-data | ab-testing, fine-tuning, llm-evaluation, llm-serving, mentoring, model-serving, public-speaking, technical-leadership |
| li-tr-devops-mid | 13 | 13 | 12/18 | 2-5 |  | ci-cd, cloud-networking, incident-management, infrastructure-as-code, observability, secrets-management |
| classic-en-student | 31 | 24 | 17/19 | student | data-cleaning, database-design, gitops, jenkins, react-native, responsive-design, scala-or-java-for-data | c, unit-testing |
| classic-tr-backend | 41 | 19 | 17/19 | 5-10 | android-sdk, angular, aspnet-core, browser-devtools, dart, django, fastapi, flutter, go, jetpack-compose, kafka, kotlin, kotlin-coroutines, nextjs, nodejs, nuxt, python, react, react-native, tailwind-css, unreal-engine, vue | code-review, integration-testing |
| classic-tr-mobile | 14 | 12 | 11/15 | 2-5 | gitops, javascript | app-store-release, ci-cd, mobile-architecture, mobile-testing |
| classic-en-security | 21 | 17 | 13/17 | 5-10 | cloud-architecture, cloud-fundamentals, cloud-networking, operating-systems | identity-access-management, network-security, penetration-testing, security-testing-tools |
| classic-en-career-changer | 8 | 7 | 7/10 | 0-2 | calculus | data-cleaning, data-visualization, public-speaking |
| classic-en-embedded | 9 | 8 | 7/12 | 10+ | unity | communication-protocols, hardware-debugging, mentoring, rtos, unit-testing |
| classic-en-qa | 19 | 16 | 14/16 | 2-5 | api-security, code-review, exploratory-data-analysis | e2e-testing, testing-fundamentals |
| classic-en-thin | 21 | 4 | 3/3 | None | angular, aspnet-core, browser-devtools, django, dom, frontend-build-tools, github-actions, jenkins, nextjs, nodejs, nuxt, react-native, selenium, smart-contract-testing, unreal-engine, vue, web-performance |  |
| classic-en-injection | 12 | 12 | 9/11 | 2-5 |  | rest-api-design, unit-testing |
| li-en-eng-manager | 9 | 9 | 8/17 | 10+ |  | code-review, engineering-processes, go, hiring, incident-management, mentoring, people-management, project-planning, stakeholder-communication |
| classic-en-gamedev | 7 | 5 | 5/12 | 2-5 | design-systems, shell-scripting | game-design-basics, game-math, game-optimization, game-physics, multiplayer-networking, oop, profiling |
| classic-tr-data-analyst | 5 | 5 | 5/10 | 0-2 |  | ab-testing, data-cleaning, product-analytics, spreadsheets, statistics |

