# Deploying XCRS to the two VMs

How the self-hosted deployment works (ADR-0014, ADR-0020, ADR-0034–0036), and the checklist for the first one. Verified end to end on the Mac (Docker Desktop, with the Mac's Ollama standing in for VM-B), not yet on the real VMs.

```
            Internet
               │ 80/443
┌──────────────▼──────────── VM-A (8 GB) ──────────────────┐        ┌──── VM-B (16 GB) ────┐
│ caddy ─► web (Nuxt) ─► api (FastAPI) ─► postgres          │ private│ ollama  qwen3.5:4b   │
│                           └──► ollama-embed (0.6B)        │───────►│ (explanations and    │
│ systemd: nightly backup ──────────── rsync over SSH ──────┼───────►│  matching), backups  │
└───────────────────────────────────────────────────────────┘        └──────────────────────┘
```

- **Images:** `.github/workflows/release.yml` publishes `ghcr.io/<owner>/xcrs-api` and `xcrs-web` on every push to `modernization` and `main`, tagged with the commit SHA and the branch name.
- **VM-A:** `deploy/vm-a/compose.yaml`. Only Caddy is public; PostgreSQL, the API and the embedding model stay on the internal Docker network. The API reads the reviewed catalog from this checkout's `catalog/` (mounted read-only).
- **VM-B:** `deploy/vm-b/compose.yaml`. Ollama is published only on the private address; the firewall allows only VM-A.
- **Deploy:** `deploy/deploy.sh <tag>` on VM-A. It takes a verified backup, pulls the images, migrates, imports and embeds the catalog, restarts, and smoke-tests through Caddy. **Rollback** is `deploy/deploy.sh <previous tag>`.

## First deployment checklist

**VM-B (once)**
1. Install Docker. Clone the repository to `/opt/xcrs`.
2. `cd /opt/xcrs/deploy/vm-b && cp .env.example .env`, and set `XCRS_PRIVATE_IP` to VM-B's private address.
3. `docker compose up -d && docker compose exec ollama ollama pull qwen3.5:4b`
4. Firewall: allow port 11434 only from VM-A, for example:
   `ufw allow from <VM-A private IP> to any port 11434 proto tcp && ufw deny 11434 && ufw enable` (keep SSH allowed).
5. Backups target: `useradd -m backup && mkdir -p /srv/xcrs-backups && chown backup /srv/xcrs-backups`, then add VM-A's backup public key to `~backup/.ssh/authorized_keys`.
6. Weekly restore test: `cp deploy/systemd/xcrs-backup-verify.* /etc/systemd/system/ && systemctl enable --now xcrs-backup-verify.timer`

**VM-A (once)**
1. Install Docker. Clone the repository to `/opt/xcrs`.
2. `cd /opt/xcrs/deploy/vm-a && cp .env.example .env`, then set:
   - `POSTGRES_PASSWORD`;
   - `XCRS_LLM_BASE_URL=http://<VM-B private IP>:11434/v1`;
   - `XCRS_DOMAIN` (`:80` until DNS points at VM-A);
   - `XCRS_BACKUP_REMOTE=backup@<VM-B private IP>:/srv/xcrs-backups`.
3. Pick an image tag: a commit SHA from the GHCR packages page (or `modernization`). Then run `/opt/xcrs/deploy/deploy.sh <tag>`.
4. Backups: create an SSH key for root (`ssh-keygen -t ed25519`), give its public key to VM-B (step 5 above), then:
   `cp deploy/systemd/xcrs-backup.* /etc/systemd/system/ && systemctl enable --now xcrs-backup.timer`
5. Check:
   - `systemctl list-timers | grep xcrs`;
   - `curl -s http://<VM-A>/api/v2/health`;
   - the site at `/`.
6. **Restrict the YouTube API key** to VM-A's public IP: Google Cloud console → *Credentials* → the key → *Application restrictions* → *IP addresses*. It was created without that restriction for local use.
7. **Measure on the VMs:** `uv run python eval/bench_explainer.py --devices cpu` on VM-B (ADR-0020), and the matching latency (ADR-0030).

## Everyday operations

| Task | Command (VM-A, in `/opt/xcrs`) |
|---|---|
| Deploy or roll back | `deploy/deploy.sh <tag>` |
| Logs | `docker compose -f deploy/vm-a/compose.yaml logs -f api` |
| Admin command | `docker compose -f deploy/vm-a/compose.yaml run --rm api xcrs <command>` (e.g. `resources ingest freecodecamp`, `resources tag`, `resources check-links`) |
| Backup now | `COMPOSE_FILE=deploy/vm-a/compose.yaml scripts/db-backup.sh && scripts/db-verify-backup.sh` |
| Restore into a new database | `COMPOSE_FILE=deploy/vm-a/compose.yaml scripts/db-restore.sh backups/<dump> xcrs_restored` |

**Limits** (ADR-0035):
- Expensive endpoints allow 10 requests a minute per client IP, in bursts of 5. `XCRS_RATE_*` changes that.
- At most 12 new phrases per request go to the LLM (`XCRS_MATCH_LLM_MAX_NEW`).
- Request bodies are capped at 64 KB in Caddy.

