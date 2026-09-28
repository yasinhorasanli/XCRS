# ADR-0002: Self-hosted first, cloud last

- **Status:** Accepted
- **Date:** 2026-09-27
- **Decider:** Muhammed Yasin Horasanli

## Context

- **Budget is zero.** XCRS has no revenue and no sponsor, and its user count is unknown.
- A **bare-metal server** is available for testing and production. *(Corrected 2026-09-28: originally recorded as "64 GB+ RAM".)* The server owner has allocated **two VMs**: 8 vCPU / 8 GB RAM / 40 GB disk and 16 vCPU / 16 GB RAM / 40 GB disk. CPU-only for now; an NVIDIA GPU is planned later.
- Cloud experience (AWS, and possibly GCP) is still a goal, as a learning target rather than a hosting requirement.
- The current prototype runs as loose processes. The backend hardcodes a public IP in the frontend (`frontend/server/api/recommend.ts`), and there's no containerization.
- The dataset is small today (453 courses, 1,104 roadmap nodes) but is expected to grow once a new scraper and generated roadmaps arrive.

## Options considered

### Option A — Cloud-first (AWS managed services from day one)
- ✅ Cloud experience immediately
- ❌ Ongoing cost: a load balancer, NAT, managed databases and container hosting add up to about $20–100/month, even at idle
- ❌ Free tiers expire or are too small for vector search

### Option B — Multi-cloud / big-data managed services (e.g. GCP BigQuery, Bigtable)
- ✅ Recognizable names
- ❌ Built for terabyte-to-petabyte analytics (BigQuery) or massive low-latency key-value workloads (Bigtable); far larger than a 453 × 869 dataset needs
- ❌ Bigtable's minimum cost alone breaks the budget; cross-cloud IAM and egress add complexity
- ❌ Wrong tool for the problem, so it would be hard to justify

### Option C — Self-hosted first on bare metal, with a portable design; cloud as a later deployment target
- ✅ Zero recurring cost
- ✅ Full control over hardware, which matters for running models locally once the GPU arrives
- ✅ Docker Compose locally and on the server; the same container images can later run on AWS (e.g. ECS) with Terraform
- ❌ We handle our own ops: backups, TLS, uptime and monitoring
- ❌ Cloud experience arrives later instead of immediately

## Decision

**Self-hosted first.** Every component runs as a container, orchestrated with Docker Compose on the bare-metal server. Only portable, open-source building blocks are used. No managed-only services, no cloud-specific APIs in application code. Moving to the cloud (AWS + Terraform + CI/CD) is planned as the **last** modernization phase, as a second deployment target for the same images.

## Trade-offs accepted

- Operational responsibility (backups, TLS, monitoring) sits with us. We mitigate it with simple, scripted backups and container health checks.
- No managed-only databases (e.g. Pinecone). This narrows the options in later decisions ([ADR-0003](0003-postgresql-pgvector-primary-store.md)).
- Cloud skills come later in the project.

## Revisit when

- There's a budget, or traffic grows beyond what one server can handle reliably.
- Uptime requirements exceed what a single self-managed machine can realistically provide.
