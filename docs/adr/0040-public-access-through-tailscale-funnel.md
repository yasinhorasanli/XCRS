# ADR-0040: Public access through Tailscale Funnel for now, a Cloudflare Tunnel once there is a domain; the VMs are managed over Tailscale

- **Status:** Accepted
- **Date:** 2026-10-02
- **Decider:** Muhammed Yasin Horasanli
- **Note (2026-10-03):** dev tools (the `/dev/profiles` test page and `/api/v2/dev`) are reachable on localhost and the tailnet only: Caddy serves `/dev` only to requests that carry the tailnet user (`Tailscale-User-Login`; public Funnel requests carry `Tailscale-Funnel-Request: ?1` instead) and answers 404 otherwise; the API also needs `XCRS_DEV_TOOLS=true`. A first version blocked on the Funnel header but never loaded: Caddy's single-file mount kept the pre-`git pull` file until a restart, which `deploy.sh` now does.

## Context

- The two VMs (ADR-0014) sit on a private network (192.168.99.0/24) of a server someone else owns. They have no public address of their own: outgoing traffic leaves through a shared NAT address, and inbound ports would have to be forwarded by the owner.
- ADR-0034 assumed Caddy on public ports 80/443 with Let's Encrypt certificates for a domain. There is no domain yet (one will be bought later), and the budget is zero.
- The VMs run Oracle Linux 10 with SELinux enforcing; firewalld was off. Docker publishes container ports past firewalld's zones.
- Tailscale was already on VM-A (the owner's tailnet); a Funnel demo there showed public HTTPS at `https://<host>.<tailnet>.ts.net` with no open ports.

## Options considered

### Option A: Tailscale Funnel (chosen for now)
- ✅ Free; public HTTPS with an automatic certificate; no inbound ports on the server or the owner's network.
- ✅ The same tool gives private SSH and admin access to the VMs (one tailnet for the Mac and both VMs).
- ❌ Only `*.ts.net` names (`https://xcrs-a.tail3afc6e.ts.net`); no custom domain.
- ❌ Bandwidth limits and Tailscale in the request path; fine for our traffic, not for a large launch.

### Option B: Cloudflare Tunnel
- ✅ Free; works with our own domain; CDN, caching and DDoS protection in front.
- ❌ Needs a domain with its DNS on Cloudflare; another account and agent.

### Option C: public ports 80/443 to Caddy
- ✅ The classic setup ADR-0034 was written for; Caddy handles certificates.
- ❌ Needs the owner to forward ports; the server is directly exposed; a domain is still needed for HTTPS.

## Decision

1. **Now: Tailscale Funnel on VM-A.** Caddy listens on `127.0.0.1:8080` only; `tailscale funnel --bg 8080` publishes it. Caddy keeps the body limit and security headers, and trusts forwarded headers from private ranges; the API takes the rightmost `X-Forwarded-For` entry that isn't a trusted proxy, so rate limits (ADR-0035) see the visitor rather than Docker's gateway.
2. **When a domain is bought: a Cloudflare Tunnel** to the same `127.0.0.1:8080`, with a new note or ADR. Nothing else in the stack changes.
3. **Management over Tailscale.** The decider's own tailnet (`tail3afc6e.ts.net`) holds the Mac and both VMs; VM-A moved out of the owner's tailnet. SSH uses keys (passphrase in the macOS Keychain); key expiry is disabled for the servers.
4. **Oracle Linux specifics.** Docker CE from Docker's repository. VM-B's firewalld allows SSH, the owner's Cockpit (9090) and node_exporter (9100), Tailscale (41641/udp), and port 11434 from VM-A only. Ollama on VM-B uses host networking on the private address, because a published Docker port would bypass that rule.

## Trade-offs accepted

- The public URL is a `ts.net` name until the domain exists; links shared now will change.
- Funnel depends on Tailscale's service and its limits; moving to a tunnel is a configuration change, not a redesign.
- VM-A's firewall stays off for now: its only listeners are SSH, the owner's services, and Caddy on localhost.

## Revisit when

- A domain is bought: switch to Option B.
- Traffic approaches Funnel's limits, or the site needs a CDN.
- The owner offers public addresses or port forwarding: Option C becomes possible.
