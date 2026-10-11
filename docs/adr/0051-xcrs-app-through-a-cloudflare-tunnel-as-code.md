# ADR-0051: xcrs.app reaches production through a Cloudflare Tunnel managed with Terraform; the AWS copy is aws.xcrs.app

- **Status:** Accepted
- **Date:** 2026-10-11
- **Decider:** Muhammed Yasin Horasanli
- **Carries out:** [ADR-0040](0040-public-access-through-tailscale-funnel.md) ("a Cloudflare Tunnel once there is a domain")

## Context

- The decider bought `xcrs.app` at Cloudflare Registrar on 2026-10-11 (DNSSEC on; mail-safety records set by hand). The app keeps the name XCRS.
- Production (VM-A) was public through Tailscale Funnel at a `ts.net` name. The VMs have no public address and no open ports (ADR-0040).
- Production's sign-in (GitHub and Google, ADR-0043) is configured for the `ts.net` address; LinkedIn waits for a company page.
- **Production runs with the dev tools on.** Caddy shows them only to requests carrying Tailscale's identity header (ADR-0040 note). Any proxy that forwards visitors' headers lets them forge that header; CloudFront needed a guard for that (ADR-0047).
- The AWS demo copy (ADR-0042, ADR-0047) was reachable only at `*.cloudfront.net`.

## Options considered

### How Cloudflare is configured

#### A: Terraform with the Cloudflare provider (chosen)
- ✅ DNS, the tunnel and its routes are reviewed in pull requests, have a history, and can be rebuilt.
- ✅ The demo's certificate and the DNS record that proves it come in one apply (both in `infra/demo`).
- ✅ DNS is public by design (anyone can query it), so the records reveal nothing new in a public repository.
- ❌ One more credential: an API token. It's limited to the `xcrs.app` zone (DNS edit, zone read) and the account's tunnels (edit), kept in the macOS Keychain, never in the repository. **No expiry date, by the decider's choice:** it's rotated only if it leaks.

#### B: the Cloudflare dashboard by hand
- ✅ Fastest, no token.
- ❌ No history or review; the certificate check means copying values between two dashboards.

### How `cloudflared` runs on VM-A
- **A: a container in VM-A's compose file, behind a `tunnel` profile (chosen).** Deployed, restarted and logged like the rest. The tunnel's token sits in VM-A's mode-600 `.env`.
- **B: a system service from Cloudflare's package.** Separate from the app's deploy flow.

### Tailscale Funnel afterwards
- **A: off (chosen).** One public entrance. VM-A keeps a tailnet-only HTTPS address through `tailscale serve`, so the decider's devices and the dev tools still work privately.
- **B: kept as a backup public address.** A second public path to secure.

### The AWS copy's address
- **`aws.xcrs.app` (chosen):** says what it is, the same app on AWS. Search engines are told not to index it (`X-Robots-Tag: noindex` from CloudFront), and a banner can point to xcrs.app later.
- **`demo.xcrs.app`:** reads like a lesser trial version and confuses real visitors.
- **`xcrs.horasanli.dev`:** frames it as a portfolio project, but ties the app to the personal domain.
- **`*.cloudfront.net` only:** no setup, but the name changes if the distribution is rebuilt.

### Sign-in on the AWS copy
- **Off (chosen):** the copy stays anonymous; sign-in would need its own OAuth apps and secrets.

## Decision

**Production: `xcrs.app` → Cloudflare's edge → a tunnel → Caddy on VM-A.**
- **`infra/cloudflare`** holds the tunnel `xcrs-vm-a`, its routes (`xcrs.app` and `www.xcrs.app` → `http://caddy:80`, anything else → 404), and proxied CNAME records for both names.
- **On VM-A:** `cloudflared` runs as a compose service with the `tunnel` profile.
- **Caddy** redirects `www` to the apex. It also answers 404 for `/dev` and `/api/v2/dev` when a request carries Cloudflare's `Cf-Ray` header, which visitors can't remove; this is the counterpart of the CloudFront guard.
- **Sign-in:** the public URL becomes `https://xcrs.app`, and the OAuth apps get its callback URLs.
- **Funnel** is switched off once the site works; VM-A keeps `tailscale serve` for the tailnet.

**The AWS copy: `aws.xcrs.app`.**
- **`infra/demo`** requests an ACM certificate in `us-east-1`, validated through a record in the `xcrs.app` zone.
- **The distribution** gets the alias and a `noindex` response header.
- **DNS:** a DNS-only CNAME (Cloudflare's proxy off, so two CDNs aren't chained).

**Ownership:** each stack owns the DNS records of what it serves. Records made by hand (mail safety) stay outside Terraform.

## Trade-offs accepted

- **Production depends on Cloudflare** (edge and tunnel) and keeps working only while `cloudflared` runs; Docker restarts it if it stops.
- **Cloudflare sees the traffic in clear text at its edge,** as any CDN does.
- **A token without expiry:** a leak would stay usable until noticed. It's limited to one zone (DNS) and tunnels, and stored only in the Keychain.
- **Cloudflare caches by file type** (e.g. `.js` and `.css`) and respects Nuxt's cache headers for `/_nuxt`. Pages and `/api` aren't cached, because they're neither static files nor marked cacheable.

## Revisit when

- CI runs Terraform (AWS step 7): the token moves to GitHub's encrypted secrets, scoped the same way.
- Traffic or abuse calls for Cloudflare rules (rate limiting, WAF) at the edge.
- LinkedIn sign-in arrives (after the company page), or the AWS copy should allow sign-in.
