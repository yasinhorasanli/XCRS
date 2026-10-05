# Deploying XCRS to the two VMs

How the self-hosted deployment works (ADR-0014, ADR-0020, ADR-0034–0036, ADR-0040), and the checklist for the first one. The VMs run **Oracle Linux 10** (SELinux enforcing) on a private network of a server someone else owns; they are managed over **Tailscale**, and the site is published with **Tailscale Funnel**.

```
 Visitors ──HTTPS──► Tailscale Funnel (https://xcrs-a.<tailnet>.ts.net)
                         │ tailscaled on VM-A → 127.0.0.1:8080
┌──────────── VM-A (8 vCPU, 8 GB) ─────────▼───────────────┐          ┌──── VM-B (24 vCPU, 32 GB) ────┐
│ caddy ─► web (Nuxt) ─► api (FastAPI) ─► postgres          │ private  │ ollama (host network,         │
│                           └──► ollama-embed (0.6B)        │ ───────► │  192.168.99.171:11434)        │
│ systemd: nightly backup ─────────── rsync over SSH ───────┼────────► │ qwen3.5 4b/9b; backups        │
└───────────────────────────────────────────────────────────┘          └───────────────────────────────┘
 Admin: the Mac, VM-A and VM-B in one tailnet (ssh xcrs-a-ts / xcrs-b-ts), or the owner's VPN
```

- **Images:** `.github/workflows/release.yml` publishes `ghcr.io/<owner>/xcrs-api` and `xcrs-web` on every push to `modernization` and `main`, tagged with the commit SHA and the branch name.
- **VM-A:** `deploy/vm-a/compose.yaml`. Caddy listens on `127.0.0.1:8080` only; Funnel publishes it. PostgreSQL, the API and the embedding model stay on the internal Docker network. The API reads the reviewed catalog from this checkout's `catalog/` (mounted read-only).
- **VM-B:** `deploy/vm-b/compose.yaml`. Ollama has no authentication: it uses host networking on the private address, and firewalld allows port 11434 from VM-A only (a published Docker port would bypass firewalld).
- **Deploy:** `deploy/deploy.sh <tag>` on VM-A: verified backup, pull, migrate, import and embed the catalog, restart, smoke test through Caddy. **Rollback** is `deploy/deploy.sh <previous tag>`.
- **Your local notes:** `deploy/inventory.local.yaml` (gitignored; template `inventory.example.yaml`) holds addresses and users; passwords stay in the macOS Keychain. `scripts/vm-facts.sh` prints a VM's facts read-only: `ssh xcrs-b 'bash -s' < scripts/vm-facts.sh`.

## First deployment checklist (Oracle Linux 10)

**Both VMs (once)**
1. SSH key login: `ssh-copy-id -i ~/.ssh/id_ed25519_xcrs.pub <user>@<vm>`; `Host` entries in `~/.ssh/config`.
2. Docker CE from Docker's repository (Oracle Linux 10 still uses dnf 4's `--add-repo`):
   ```
   sudo dnf config-manager --add-repo https://download.docker.com/linux/rhel/docker-ce.repo
   sudo dnf -y --setopt=install_weak_deps=False install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin git
   echo '{"log-driver":"json-file","log-opts":{"max-size":"10m","max-file":"3"}}' | sudo tee /etc/docker/daemon.json
   sudo systemctl enable --now docker && sudo usermod -aG docker $USER
   ```
3. Tailscale: `curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up --hostname xcrs-a` (or `xcrs-b`); approve the link; disable key expiry for the machine in the admin console.
4. Clone: `sudo mkdir -p /opt/xcrs && sudo chown $USER /opt/xcrs && git clone https://github.com/yasinhorasanli/XCRS.git /opt/xcrs`.

**VM-B (once)**
1. Firewall (keep the owner's services open), with an automatic undo while testing:
   ```
   sudo systemd-run --unit=fw-rollback --on-active=300 systemctl stop firewalld
   sudo firewall-offline-cmd --add-service=ssh --add-service=cockpit --add-port=9100/tcp --add-port=41641/udp
   sudo firewall-offline-cmd --add-rich-rule='rule family="ipv4" source address="<VM-A private IP>/32" port port="11434" protocol="tcp" accept'
   sudo systemctl enable --now firewalld && sudo systemctl restart docker
   # test a new SSH session, then: sudo systemctl stop fw-rollback.timer
   ```
2. `cd /opt/xcrs/deploy/vm-b && cp .env.example .env` (set `XCRS_PRIVATE_IP`), `docker compose up -d`, then `docker compose exec ollama ollama pull qwen3.5:4b` (and `qwen3.5:9b` for the benchmark).
3. Backups target: `sudo useradd -m backup && sudo mkdir -p /srv/xcrs-backups && sudo chown backup /srv/xcrs-backups`; add VM-A root's backup public key to `~backup/.ssh/authorized_keys`.
4. Weekly restore test: `sudo cp deploy/systemd/xcrs-backup-verify.* /etc/systemd/system/ && sudo systemctl enable --now xcrs-backup-verify.timer`.

**VM-A (once)**
1. `cd /opt/xcrs/deploy/vm-a && cp .env.example .env && chmod 600 .env`, then set `POSTGRES_PASSWORD`, `XCRS_LLM_BASE_URL=http://<VM-B private IP>:11434/v1`, `XCRS_LLM_MODEL`, `XCRS_YOUTUBE_API_KEY`, `XCRS_BACKUP_REMOTE=backup@<VM-B private IP>:/srv/xcrs-backups`.
2. Pick an image tag (a commit SHA from the GHCR packages page, or `modernization`) and run `/opt/xcrs/deploy/deploy.sh <tag>`.
3. Resources from the adapters: `docker compose -f deploy/vm-a/compose.yaml run --rm api xcrs resources ingest freecodecamp`, `… ingest youtube`, `… tag`.
4. Backups: `sudo ssh-keygen -t ed25519 -f /root/.ssh/id_ed25519 -N ""`, give its public key to VM-B (above), then `sudo cp deploy/systemd/xcrs-backup.* /etc/systemd/system/ && sudo systemctl enable --now xcrs-backup.timer`.
5. Publish: `sudo tailscale funnel --bg 8080`. The first time, Tailscale prints a link to allow Funnel (and HTTPS certificates) for the tailnet; open it while signed in as the tailnet admin. Check `tailscale funnel status`, then open `https://xcrs-a.<tailnet>.ts.net`.
6. **Restrict the YouTube API key** to the VMs' egress address: Google Cloud console → *Credentials* → the key → *Application restrictions* → *IP addresses* (the shared NAT IPv4 and the VMs' IPv6 addresses; see the inventory). Do this once discovery runs on VM-A, or the Mac's calls stop working.
7. **YouTube job** (daily at 11:30, after the quota resets: refresh and ingest the approved playlists, expire data older than 30 days, tag new resources, discover candidates): `sudo install -d -o 10001 -g 10001 /srv/xcrs-discovery`, copy `catalog/sources/youtube-candidates.yaml` there (it carries the skills already searched), then `sudo cp deploy/systemd/xcrs-youtube-discover.* /etc/systemd/system/ && sudo systemctl enable --now xcrs-youtube-discover.timer`. On the Mac, `scripts/youtube-candidates-pull.sh` copies the results into the repo for review.
8. **Measure on the VMs:** the explainer benchmark against VM-B (ADR-0020), and the matching latency (ADR-0030).
9. **Accounts (ADR-0043, optional):** in `.env` set `XCRS_PUBLIC_URL` (the Funnel address), `XCRS_INTERNAL_SECRET` and `NUXT_SESSION_PASSWORD` (`openssl rand -hex 32` each), then register the OAuth apps with these callback URLs and set their ids and secrets:
   - **GitHub:** Settings → Developer settings → OAuth Apps → New; callback `<XCRS_PUBLIC_URL>/auth/github`.
   - **Google:** Cloud console → Google Auth Platform → Clients → Web application; redirect URI `<XCRS_PUBLIC_URL>/auth/google`; scopes openid, email, profile. Google may ask to verify the domain for the consent screen; if it won't accept the `ts.net` address, Google waits for the bought domain.
   - **LinkedIn:** Developer portal → Create app → Products → *Sign In with LinkedIn using OpenID Connect*; Auth → redirect URL `<XCRS_PUBLIC_URL>/auth/linkedin`.

   Restart with `deploy/deploy.sh <tag>`; `/sign-in` then shows a button per configured provider. Changing `NUXT_SESSION_PASSWORD` signs everyone out.

## Everyday operations

| Task | Command (VM-A, in `/opt/xcrs`) |
|---|---|
| Deploy or roll back | `deploy/deploy.sh <tag>` |
| Logs | `docker compose -f deploy/vm-a/compose.yaml logs -f api` |
| Admin command | `docker compose -f deploy/vm-a/compose.yaml run --rm api xcrs <command>` (e.g. `resources ingest freecodecamp`, `resources tag`, `resources check-links`) |
| Backup now | `COMPOSE_FILE=deploy/vm-a/compose.yaml scripts/db-backup.sh && scripts/db-verify-backup.sh` |
| Funnel on / off / status | `sudo tailscale funnel --bg 8080` / `sudo tailscale funnel --https=443 off` / `tailscale funnel status` |
| YouTube job: run now / results | `sudo systemctl start xcrs-youtube-discover` / `scripts/youtube-candidates-pull.sh` (on the Mac) |
| Test profiles (tailnet only) | `http://xcrs-a.<tailnet>.ts.net/dev/profiles` with `XCRS_DEV_TOOLS=true` in `.env`; public requests get 404 from Caddy |
| VM facts (read-only) | `ssh xcrs-a 'bash -s' < scripts/vm-facts.sh` |
| Restore into a new database | `COMPOSE_FILE=deploy/vm-a/compose.yaml scripts/db-restore.sh backups/<dump> xcrs_restored` |

**Limits** (ADR-0035):
- Expensive endpoints allow 10 requests a minute per client IP, in bursts of 5. `XCRS_RATE_*` changes that.
- At most 12 new phrases per request go to the LLM (`XCRS_MATCH_LLM_MAX_NEW`).
- Request bodies are capped at 64 KB in Caddy, except `/api/v2/cv-imports` (CV uploads, ADR-0045): 3 MB. The API itself refuses PDFs over 2 MB or 5 pages.
- CV imports (ADR-0045): signed-in users only, 5 a day each (`XCRS_CV_PER_DAY`), one running and at most 3 waiting (`XCRS_CV_MAX_WAITING`); `XCRS_CV_IMPORT_ENABLED=false` turns them off. **After deploying, reload Caddy** (`docker compose -p xcrs-vm-a exec caddy caddy reload --config /etc/caddy/Caddyfile`) so the upload limit applies.
- Behind Funnel, the visitor's address comes from `X-Forwarded-For`: Caddy trusts private ranges, and the API takes the rightmost entry that isn't a trusted proxy.

