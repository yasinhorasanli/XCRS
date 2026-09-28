# ADR-0014: Run locally first; target deployment splits the two VMs by role

- **Status:** Accepted
- **Date:** 2026-09-28
- **Decider:** Muhammed Yasin Horasanli

## Context

- **Machines** ([ADR-0002](0002-self-hosted-first-cloud-last.md)):
  - a MacBook Pro M4 Pro with 24 GB unified memory (development);
  - VM-A: 8 vCPU / 8 GB RAM / 40 GB disk;
  - VM-B: 16 vCPU / 16 GB RAM / 40 GB disk.

  The VMs are CPU-only, a GPU is planned later, and the disks can be enlarged.
- **Services:** PostgreSQL + pgvector, Ollama ([ADR-0007](0007-ollama-qwen3-embedding.md)), the backend API, the frontend, and batch jobs (ingestion, embedding).
- **Model load:** the embedding model needs ~1–2 GB of RAM today. A future local LLM for explanations will need far more.
- **No users** during the modernization, so a separate staging environment has little value yet.
- **Ollama has no built-in authentication.**
- **Docker on macOS can't use the Apple GPU,** so Ollama in a container on the Mac would run on CPU only.
- The embedding endpoint is a configurable base URL ([ADR-0006](0006-own-embedding-interface.md)).

## Options considered

### Option A — Split the VMs by role (VM-B: model runtime; VM-A: Postgres, backend, frontend)
- ✅ Inference load is isolated from the database and the API
- ✅ The 16 GB VM is reserved for the future local LLM
- ❌ Needs private networking between the VMs and a locked-down Ollama port
- ❌ No separate staging environment (the Mac fills that role)

### Option B — Staging + production (a full stack on each VM)
- ✅ A realistic test environment
- ❌ Model, database and app compete for resources on one VM
- ❌ Deploys twice, while there are no users to protect

### Option C — Everything on the 16 GB VM
- ✅ Simplest to operate
- ❌ No isolation; leaves the 8 GB VM unused

## Decision

1. **Now: everything runs locally on the Mac.** Ollama runs **natively** (for Metal GPU acceleration). Postgres + pgvector and the backend run in **Docker Compose** and reach Ollama at `http://host.docker.internal:11434`.
2. **Target: Option A.** VM-B runs Ollama; VM-A runs Postgres, the backend and the frontend.
   - **Ollama listens only on the private network between the VMs,** firewalled to VM-A, and is never exposed publicly.
   - Compose files are organized so each VM runs only its own services.
   - The model's location is a single environment variable (the embedding base URL).

## Trade-offs accepted

- The split brings little benefit until the local LLM arrives. It's chosen for where the system is going.
- There's no staging environment; the Mac serves as one. Mac performance (GPU) doesn't represent the VMs (CPU), so baselines are measured on the VMs.
- Cross-VM networking and firewall rules are extra operational work.

## Revisit when

- **The GPU arrives:** check which VM gets it. The model runtime moves there.
- **XCRS gets users:** add a staging environment (Option B becomes more valuable).
- **VM-A runs short on memory** as Postgres and the data grow.
