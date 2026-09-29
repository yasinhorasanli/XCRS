# ADR-0021: Continuous integration on GitHub Actions; container images for backend and frontend

- **Status:** Accepted (delegated: made on 2026-09-30 while the decider asked for the remaining work to be finished without questions; pending the decider's review)
- **Date:** 2026-09-30
- **Decider:** Muhammed Yasin Horasanli

## Context

- The backend has 29 tests, a Ruff configuration, and hand-written Alembic migrations whose drift is checked with `alembic check` ([ADR-0011](0011-alembic-schema-migrations.md)). All of it ran only when someone remembered to run it.
- A silent failure is the main risk: a vector query that stops matching its index doesn't error ([ADR-0009](0009-per-model-indexes-and-precomputed-matches.md)), and a pnpm upgrade once silently ignored security overrides.
- [ADR-0002](0002-self-hosted-first-cloud-last.md) says every component runs as a container, but only PostgreSQL had one ([ADR-0014](0014-local-first-then-split-by-role.md), corrected).
- The repository is on GitHub. Budget is zero. Deployment (CD) targets the VMs first and AWS last.

## Options considered

### Option A — GitHub Actions
- ✅ Built into the repository host; free for public repositories; service containers give CI a real PostgreSQL + pgvector
- ✅ The most common CI in job postings; later phases (image push, `terraform plan` via OIDC) use the same tool
- ❌ Runs on GitHub's machines: nothing on the VMs is tested; no GPU or Ollama

### Option B — Self-hosted CI on the VMs (e.g. Woodpecker, Jenkins)
- ✅ Tests where the system runs
- ❌ Another service to operate on 8/16 GB machines owned by someone else; less transferable experience

### Option C — Local scripts / pre-commit hooks only
- ✅ No infrastructure
- ❌ Relies on discipline; nothing checks a pull request

## Decision

**GitHub Actions** (`.github/workflows/ci.yml`) on every push to `main`/`modernization` and every pull request. Three jobs:

1. **backend:** Ruff format + lint; migrations `upgrade head → downgrade base → upgrade head` (every `downgrade()` is exercised); `alembic check`; the test suite against a `pgvector/pgvector:pg18` service container. The embedding model is registered (no Ollama needed) so the index-shape tests run; tests that need embedded vectors skip themselves.
2. **frontend:** `pnpm install --frozen-lockfile` (fails if the lockfile and the security overrides disagree) and a production build.
3. **images:** `docker build` of both images.

**Container images:** `backend/Dockerfile` (multi-stage with uv, non-root, health check; the same image runs `alembic` and `xcrs` admin commands) and `frontend/Dockerfile` (Nuxt server build, non-root). `docker compose --profile app up` runs the whole stack in containers, as on the VMs; day-to-day development still runs the API and UI natively.

**Not included:** deployment (CD), image registry, and tests that need Ollama. The workflow was linted with `actionlint`, and its steps were run locally against a throwaway database; the first real run happens on the next push.

## Trade-offs accepted

- CI can't test LLM output or embedding quality: no Ollama on the runners. Those stay manual (benchmark, quality comparison) until evaluation is automated.
- Vector-dependent tests skip in CI. The index-shape tests, which catch the silent failure, do run.
- Third-party actions are pinned by major version, not by commit hash.

## Revisit when

- Deployment to the VMs starts → add CD (build, push to a registry, deploy), probably still on GitHub Actions.
- The cloud phase starts → OIDC to AWS and `terraform plan` on pull requests.
- An automated LLM evaluation exists → run it on a schedule or on a self-hosted runner with Ollama.
