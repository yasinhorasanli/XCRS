# Explainer benchmark

- Date: 2026-09-29T23:00:14.472410+00:00; host: MacBook-Pro.local; CPU threads: 10
- Inputs: explanation contexts from the algorithm on `eval/profiles.json`; the app's prompt and schema.
- Clean = no automatic grounding flag (heuristics that mark explanations to read, not proof).
- Prefill speed is not reported: Ollama's prompt cache makes its prompt timings unreliable.

| Model / device | Cases | Errors | Median s/role | p90 s | Prompt tok | Output tok | Decode tok/s | Clean % | Flags | Model load s |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5:9b / gpu | 40 | 0 | 8.3 | 9.6 | 920 | 240 | 38.5 | 80 | invented_curiosity, misattributed | 3.1 |
| qwen3.5:9b / cpu | 15 | 0 | 57.9 | 68.5 | 975 | 244 | 7.1 | 67 | misattributed | 5.8 |
| qwen3.5:4b / gpu | 40 | 0 | 11.0 | 16.4 | 920 | 214 | 24.5 | 68 | invented_curiosity, misattributed | 12.9 |
| qwen3.5:4b / cpu | 15 | 0 | 33.6 | 47.3 | 975 | 201 | 10.8 | 73 | misattributed | 3.9 |

## Request-path embedding while an explanation generates on the same device

| Model / device | Idle ms | During generation ms |
|---|---|---|
| qwen3.5:9b / gpu | 100 | 151 |
| qwen3.5:9b / cpu | 107 | 20334 |
| qwen3.5:4b / gpu | 104 | 302 |
| qwen3.5:4b / cpu | 80 | 20254 |

## Flagged explanations

### qwen3.5:9b / gpu

- **frontend / Frontend Developer**: `misattributed:redux`. This Frontend Developer role fits you because you already know HTML, CSS, and JavaScript basics. Your curiosity about TypeScript, React, and Redux shows a strong desire to build modern web applications using popular frameworks.
- **devops / Full Stack Developer**: `misattributed:infrastructure`. This Full Stack Developer role fits you because you already know Git, Linux, and Bash scripting. Your curiosity about Terraform shows a strong interest in infrastructure, which is essential for building scalable applications.
- **devops / QA Engineer**: `misattributed:data dog, misattributed:grafana`. This QA Engineer role fits you because you already know Git for version control and are curious about monitoring tools like Data Dog and Grafana. This path builds on your existing skills while addressing the testing concepts you haven't covered yet.
- **ux / UX Designer**: `misattributed:balsamiq, misattributed:adobe xd`. This UX Designer role fits you because you already know Figma, Balsamiq, and how to conduct user research. You are also curious about prototyping with Adobe XD and Figma, which aligns perfectly with the design and modeling skills needed for this career.
- **fullstack / Full Stack Developer**: `invented_curiosity`. Your interest in React and Node.js makes you a strong fit for Full Stack Developer. This role combines your frontend skills with backend development, allowing you to build complete web applications from start to finish.
- **edge-curious-only / DevOps Engineer**: `misattributed:aws, misattributed:google cloud`. This DevOps Engineer role fits you because it aligns with your curiosity about Cloud computing, specifically AWS and Google Cloud. The recommended path helps you build the necessary skills to manage these environments effectively.
- **edge-disliked-heavy / DevOps Engineer**: `invented_curiosity`. This DevOps Engineer role fits you because you already know Python, which is a key skill for scripting and automation. Your interest in Python provides a strong foundation for managing infrastructure and building tools.
- **edge-abbreviations / DevOps Engineer**: `misattributed:gitlab ci, misattributed:argo cd, misattributed:jenkins`. This DevOps Engineer role fits you because it aligns with your strong interest in CI/CD tools like Jenkins, GitLab CI, and Argo CD. The recommended path will help you expand your skills beyond these interests to cover essential concepts like Python, JavaScript, and process monitoring required for the role.

### qwen3.5:9b / cpu

- **backend-java / DevOps Engineer**: `misattributed:ecs fargate`. This DevOps Engineer role fits you because you are curious about Kubernetes, Docker, and container orchestration like ECS Fargate. You also want to learn about GitOps tools like Argo CD and Flux CD, which are essential for managing these environments effectively.
- **devops / DevOps Engineer**: `misattributed:datadog, misattributed:datadog, misattributed:prometheus`. This DevOps role fits you because you already know Git for version control and are curious about Terraform, Monitoring tools like Datadog and Prometheus, and Infrastructure as Code concepts. This path builds on your current skills while exploring the automation and observability areas you want to learn.
- **ux / UX Designer**: `misattributed:balsamiq`. This UX Designer role fits you because you already know Figma, Balsamiq, and User Research. You are curious about Prototyping. This path builds on your skills to help users think about actions and apply behavioral science.
- **fullstack / Full Stack Developer**: `misattributed:npm`. Since you already know React, Node.js, and npm, this Full Stack Developer role fits perfectly. You have the core frontend and backend skills needed to build complete web applications.
- **edge-abbreviations / DevOps Engineer**: `misattributed:gitlab ci, misattributed:argo cd, misattributed:teamcity, misattributed:jenkins`. This DevOps Engineer role fits you because you are curious about CI/CD tools like GitLab CI, Argo CD, Jenkins, and TeamCity. This career focuses on automating the build, test, and delivery processes you want to explore.

### qwen3.5:4b / gpu

- **backend-java / DevOps Engineer**: `misattributed:argo cd`. You are curious about Kubernetes, Docker, and CI/CD tools like Argo CD. A DevOps Engineer role fits because it directly utilizes these container orchestration and deployment skills you want to explore.
- **devops / Full Stack Developer**: `misattributed:infrastructure`. You have Linux and Bash skills, which fit Full Stack well. Your curiosity about Terraform aligns with the role's infrastructure needs. This path builds on your existing knowledge while addressing what you want to learn.
- **android / Android Developer**: `misattributed:navigation components`. You know Kotlin and Java, which are perfect for Android development. You are curious about Jetpack Compose and navigation components, making this role a great fit.
- **android / Backend Developer**: `misattributed:monolithic apps`. You know Java, which fits the Backend Developer role. You are curious about mobile app architecture (monolithic apps). This role builds on your Java foundation while addressing your interest in large-scale application structures.
- **game / Backend Developer**: `misattributed:cpp`. You have studied C++, which matches the cpp concept. This role fits because backend development often requires strong programming foundations like C++.
- **blockchain / Backend Developer**: `invented_curiosity`. You have shown interest in JavaScript, which is a core skill for Backend Development. This role leverages your existing knowledge to build server-side applications.
- **qa / Frontend Developer**: `misattributed:playwright`. You know Selenium (playwright) and are curious about Cypress, showing interest in testing. This Frontend Developer role builds on your automation knowledge while introducing React, Next.js, and modern web fundamentals to expand your skillset.
- **ux / QA Engineer**: `misattributed:uat`. You have User research experience matching UAT, and you're curious about Accessibility. This QA Engineer role bridges your background with these interests by covering core testing fundamentals and automation.
- **fullstack / Full Stack Developer**: `misattributed:npm`. You know React, Node.js, and npm, which aligns well with the Full Stack Developer role. This path builds on your frontend and backend skills while introducing essential DevOps practices.
- **fullstack / Backend Developer**: `misattributed:javascript, misattributed:javascript`. You know MongoDB, Node.js, and Express, which aligns with Backend Developer needs. Your curiosity about GraphQL complements your JavaScript skills to build modern APIs.
- **edge-curious-only / DevOps Engineer**: `misattributed:aws, misattributed:google cloud`. You are curious about cloud computing, specifically AWS and Google Cloud. This role fits because it leverages your interest in these platforms while addressing the infrastructure management skills required for a DevOps Engineer.
- **edge-disliked-heavy / Blockchain Developer**: `invented_curiosity`. You liked Python, which supports your interest in Blockchain Development. While you know JavaScript (though disliked), the recommended courses will build your foundational blockchain knowledge from scratch.
- **edge-abbreviations / DevOps Engineer**: `misattributed:argo cd, misattributed:jenkins`. You are curious about CI/CD tools like Jenkins, GitLab, and Argo CD. This role fits because it leverages your interest in automating deployment pipelines while addressing the infrastructure management skills required for DevOps.

### qwen3.5:4b / cpu

- **backend-java / DevOps Engineer**: `misattributed:argo cd`. You are curious about Kubernetes, Docker, and CI/CD tools like Argo CD. This role fits because it builds on your interest in container orchestration and automation while introducing essential gaps like Python, Go, and process monitoring.
- **android / Android Developer**: `misattributed:navigation components`. You know Kotlin and Java, which are perfect for Android development. You are curious about Jetpack Compose and navigation components, making this role a great fit.
- **ux / UX Designer**: `misattributed:adobe xd`. You know Figma, user personas, and research methods like the action funnel. You are curious about prototyping with tools like Adobe XD. This UX Designer role builds on your design and research foundation while introducing behavioral science concepts.
- **edge-abbreviations / DevOps Engineer**: `misattributed:argo cd, misattributed:jenkins`. You are curious about CI/CD tools like Jenkins, GitLab, and Argo CD. This role fits because it requires mastering these automation pipelines to build, test, and deploy software reliably.
