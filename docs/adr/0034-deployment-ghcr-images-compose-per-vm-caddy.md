# ADR-0034: Deployment: images published to GHCR, one Compose file per VM, Caddy in front, a pull-based deploy script

- **Status:** Accepted. Delegated: the decider asked Claude to complete operations overnight (2026-10-02); to be reviewed before the first real deployment.
- **Date:** 2026-10-02
- **Decider:** Muhammed Yasin Horasanli

## Context

- **Topology is decided** (ADR-0014, ADR-0020):
  - VM-A (8 GB) runs PostgreSQL, the API, the web server and the embedding model.
  - VM-B (16 GB) runs only the explanation and matching LLM (`qwen3.5:4b` on CPU).
  - Ollama is reachable only on the private network.
- **Images:** CI already builds both images (ADR-0021) but publishes nothing. The repository is public, so GitHub Container Registry (GHCR) is free.
- **Access:** Claude has no access to the VMs; they belong to someone else and only the decider can log in. Zero budget.
- **Risk:** there are no users yet, so a deploy may cause a short outage. Migrations change data, so a backup must come first (`scripts/db-backup.sh`).

## Options considered

- **Build on the VM from git.** ✅ No registry. ❌ Slow builds on small VMs; what runs isn't what CI tested.
- **CI publishes images to GHCR; the VM pulls (push-to-registry, pull-to-deploy).** ✅ The tested image is what runs; VMs need no build tools or source; rollback means a previous tag. ❌ A registry dependency, and the deploy step stays manual (or needs SSH secrets).
- **CI deploys over SSH (push-based CD).** ✅ Fully automatic. ❌ CI needs SSH keys and network access to someone else's machines; more attack surface. Premature without users.

## Decision

1. **`.github/workflows/release.yml`** builds `xcrs-api` and `xcrs-web` and pushes them to `ghcr.io/<owner>/xcrs-api|xcrs-web` on every push to `modernization` and `main`. Tags: the commit SHA (immutable) and the branch name (moving).
2. **One Compose file per VM** under `deploy/`:
   - **VM-A** (`deploy/vm-a/compose.yaml`):
     - **Caddy** is the only public service (ports 80/443, automatic HTTPS when `XCRS_DOMAIN` is a real domain).
     - The web server, the API, PostgreSQL (internal network only) and an Ollama for embeddings (internal only).
   - **VM-B** (`deploy/vm-b/compose.yaml`): an Ollama for the LLM, published only on the private address that `XCRS_PRIVATE_IP` names. A host firewall rule allows only VM-A (documented in `deploy/README.md`).
   - Images are pinned by `XCRS_IMAGE_TAG` in each VM's `.env`.
3. **`deploy/deploy.sh <tag>`** runs on VM-A:
   1. Verified database backup.
   2. Pull the tagged images.
   3. Run the migrations with the new image.
   4. Restart.
   5. Smoke test through Caddy: health, plus a v2 endpoint that only the new code serves.

   Rollback is `deploy.sh <previous tag>`; migrations stay backward-compatible until ADR-0021's check is extended.
4. **CI-driven deployment is deferred** until there are users and SSH access can be granted safely. The workflow has a manual `workflow_dispatch` input to publish any commit.

## Trade-offs accepted

- **Deploying is a manual command on VM-A.** Acceptable without users; the script makes it one line.
- **GHCR packages are public** for a public repository. The images contain no secrets; configuration and secrets stay in each VM's `.env`.
- **The topology is verified on the Mac, not on the VMs** (Docker Desktop, with Ollama native standing in for VM-B). The first real deploy needs the checklist in `deploy/README.md`.

## Revisit when

- XCRS gets users → push-based CD with environment protection, and a staging environment (ADR-0014).
- The VMs move to another provider, or to AWS (the last phase) → the same images, with a different compose or orchestration.
