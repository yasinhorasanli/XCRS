# ADR-0042: AWS runs a disposable demo copy on one EC2 t4g.small with Docker Compose; both models come from VM-B over Tailscale

- **Status:** Accepted
- **Date:** 2026-10-03
- **Decider:** Muhammed Yasin Horasanli
- **Note (2026-10-07):** building the instance (`infra/demo`), the decider chose Amazon Linux 2023 (SSM agent built in; Compose installed as a pinned, checksummed plugin), access over the tailnet only until CloudFront (no inbound rules at all), and sign-in off on the demo. The instance calls VM-B by its tailnet IP, because containers resolve names through the VPC's DNS, which doesn't know MagicDNS. CPU credits are "standard" (throttle, never bill).

## Context

- ADR-0002 made the cloud the **last** phase, for learning and the CV; the two VMs (ADR-0014, ADR-0034–0036, ADR-0040) are XCRS's real home and are now deployed.
- **Budget is zero.** There is no AWS account yet. Accounts created after 15 July 2025 get the **Free plan**: $100 in credits at sign-up plus up to $100 more for completing introductory activities (such as creating a budget or launching an EC2 instance), valid for 6 months, and AWS can't charge the account beyond them unless it is upgraded. Separately, the **t4g.small free trial** gives every account 750 hours a month of t4g.small until **31 December 2026**.
- The VM-A stack is Caddy, Nuxt (web), FastAPI (API), PostgreSQL + pgvector and the embedding model (`ollama-embed`, `qwen3-embedding:0.6b`, about 1 GB loaded). VM-B runs the LLM (Ollama, `qwen3.5:4b`, ADR-0020) on its private address, and its firewall allows only VM-A.
- Images are built for linux/amd64 only (`.github/workflows/release.yml`).
- There are no users yet, and the AWS copy holds no data that can't be rebuilt: the catalog is YAML, and embeddings are recomputed (ADR-0025, ADR-0030).

## Options considered

### Purpose

#### A: demo/staging copy (chosen)
- ✅ A second, disposable environment that runs the same images as VM-A. It can be destroyed when idle and rebuilt from code, which is the habit Terraform is meant to teach.
- ✅ The VMs are unaffected if something on AWS goes wrong.
- ❌ It holds no real activity data, so it doesn't prove a production failover.

#### B: disaster recovery
- ✅ A more realistic operations story: restorable dumps in S3, a tested runbook, and a switch of DNS.
- ❌ A warm standby costs money or attention; there is no domain to switch yet (ADR-0040).

#### C: move production
- ❌ Breaks the zero budget once the credits end; the VMs would become a fallback without a reason.

### Compute

#### A: EC2 t4g.small + Docker Compose (chosen)
- ✅ Free until 31 Dec 2026 (trial), then about $12/month if always on, and close to $0 if stopped when idle.
- ✅ The same compose stack and `deploy.sh` flow as VM-A, so the AWS work is about AWS rather than a second way to run the app.
- ✅ Graviton (arm64): cheaper per hour than x86, and a portability exercise.
- ❌ Needs arm64 images (free GitHub ARM runners); 2 GB of RAM is tight (see Decision 3).
- ❌ Less "managed" than Fargate + RDS on a CV.

#### B: ECS Fargate + RDS + ALB
- ✅ The textbook managed architecture.
- ❌ About $40–60/month (load balancer, RDS, Fargate tasks, public IPv4), so the $100 sign-up credit would last about two months; every session would end with a destroy.

#### C: EKS
- ❌ About $73/month for the control plane alone. Ruled out.

### LLM and embeddings

#### A: VM-B over Tailscale (chosen)
- ✅ $0, the same models and behaviour as production, and no API keys.
- ❌ The AWS copy depends on VM-B being up, and it shares VM-B's single generation slot (`OLLAMA_NUM_PARALLEL=1`) with production.

#### B: Amazon Bedrock
- ✅ Adds a managed AI service to the story.
- ❌ Pay per token, different models from the VMs, and a second client path in the app.

#### C: no LLM on AWS
- ❌ No explanations and no LLM pick in matching, so the demo would be a lesser copy.

### Fitting into 2 GB of RAM

- **Embeddings from VM-B too (chosen):** pull `qwen3-embedding:0.6b` on VM-B (about 1 GB of its 32 GB). The instance then runs Caddy, web, the API and PostgreSQL, with a 1 GB swap file as headroom. ✅ Stays free. ❌ One more model on VM-B.
- **Keep the embedding model on the instance and add swap:** ❌ Out-of-memory risk while the catalog is embedded.
- **t4g.medium (4 GB):** ❌ Not in the trial: about $24/month always on, drawn from the credits.

### Reaching VM-B's Ollama

- **`tailscale serve` on VM-B plus a tailnet access rule (chosen):** VM-B forwards its tailnet port 11434 to the private Ollama address; the tailnet policy lets only the tagged AWS node (`tag:xcrs-aws`) reach `xcrs-b:11434`. ✅ No firewall change on VM-B. ✅ The AWS node can reach nothing else on the tailnet. ❌ The tailnet's default allow-all policy has to be replaced with explicit rules first.
- **VM-A as a subnet router:** ❌ The AWS copy would then depend on VM-A as well, and all of its traffic would route through VM-A.

## Decision

1. **AWS hosts a disposable demo/staging copy** of XCRS, defined entirely in Terraform under `infra/`, destroyed or stopped when idle, and rebuilt from code. The VMs stay the real home.
2. **One EC2 t4g.small (arm64) running Docker Compose**, started by cloud-init with the same images and flow as VM-A (`deploy/deploy.sh`), in a VPC with a public subnet only (no NAT gateway, no load balancer). The release workflow builds **linux/arm64** images as well.
3. **Both models come from VM-B over Tailscale.** The instance joins the decider's tailnet as `tag:xcrs-aws` using a pre-approved, tagged auth key kept in SSM Parameter Store (SecureString). VM-B also serves `qwen3-embedding:0.6b`; `tailscale serve` on VM-B publishes Ollama to the tailnet, and the tailnet policy allows `tag:xcrs-aws` → `xcrs-b:11434` only. The instance runs no model.
4. **Planned around it, each explained as it is built:** an S3 backend for the Terraform state with native locking; **CloudFront** in front of the instance (free `*.cloudfront.net` HTTPS; the security group allows only CloudFront's managed prefix list); **S3** for backups (also a third copy of the VMs' nightly dumps); IAM roles, **GitHub Actions OIDC** (no stored keys), and **SSM Session Manager** instead of an SSH port; **AWS Budgets** alerts at $5 and $10; a nightly stop by **EventBridge Scheduler**. Any of these that turns out to cost money or to add risk goes back to the decider before it is built.

## Trade-offs accepted

- The AWS copy is down whenever VM-B is down, and its explanations queue behind production's on VM-B's single slot. That is acceptable for a demo; production is unaffected because the VMs don't depend on AWS.
- Compose on one instance is not the managed-services architecture; this ADR gives the reasons, and a short-lived Fargate exercise can come later.
- Every image is built twice (amd64 and arm64), which makes the release workflow slower.
- Running costs after the trial ends (31 Dec 2026), if left on: t4g.small about $12/month, a public IPv4 address about $3.65/month, a 20 GB gp3 disk about $1.60/month. The nightly stop, destroying the copy when idle, and the budget alerts keep this near zero, and the Free plan can't charge beyond its credits.
- The tailnet gains a node in a cloud account; the tagged key and the access rule limit what it can reach.

## Revisit when

- The t4g.small trial ends (31 Dec 2026) or the Free plan's credits run out: decide whether to keep the copy (stop when idle, upgrade the account) or destroy it.
- AWS should hold real data or take over from the VMs: write a disaster-recovery ADR (Purpose B).
- VM-B's load from the demo slows production's explanations.
- A GPU arrives on VM-B, or the app needs more than 2 GB without the models.
