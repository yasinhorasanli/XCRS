# ADR-0047: CloudFront in front of the AWS demo copy, through a VPC origin, on pay-as-you-go pricing, caching only Nuxt's build files

- **Status:** Accepted
- **Date:** 2026-10-10
- **Decider:** Muhammed Yasin Horasanli
- **Builds on:** [ADR-0042](0042-aws-demo-copy-on-one-ec2-instance.md) (the AWS demo copy)

## Context

- The AWS demo copy (ADR-0042) runs on one EC2 t4g.small in a public subnet. Its security group has **no inbound rules**; it is reached over Tailscale only. Step 5 of the AWS phase gives it a public HTTPS address.
- The goal is learning and CV material at **zero cost**. No domain yet, so the free `*.cloudfront.net` name and certificate are enough.
- The instance needs its public IPv4 address for outbound traffic (GHCR has no IPv6), and a NAT gateway (~$35/month) is out of budget, so it stays in the public subnet.
- The app's rate limits (ADR-0035) key on the visitor's IP: they read `X-Forwarded-For` only from trusted, by default private, addresses.
- Since late 2025 CloudFront has two pricing models: pay-as-you-go (always-free tier: 1 TB and 10 M requests a month) and flat-rate plans (Free: $0, 1 M requests, 100 GB, WAF with 5 rules, **no overage charges**). Checked 2026-10-10: the Free plan has no private VPC origins (Business, $200/month, and up), allows only AWS-managed cache and origin-request policies, has no access logs, and requires a WAF web ACL. Its Terraform resource (`aws_pricingplanmanager_subscription`) was merged for provider 6.69.0, which was not yet released.

## Options considered

### How CloudFront reaches the instance

#### A: VPC origin (chosen)
- ✅ CloudFront reaches the instance's private address through a CloudFront-managed network interface inside our VPC; no public path to the instance at all. Free.
- ✅ The security group admits only CloudFront's service security group, so only **our** distributions get in, with no shared secret.
- ✅ Requests arrive from a private address, so the app's rate limits see the real visitor without configuration.
- ❌ A VPC origin takes ~15 minutes to deploy and is tied to the instance; replacing the instance replaces it.
- ❌ Not available on the flat-rate Free plan.

#### B: public origin + CloudFront's managed prefix list
- ✅ The classic pattern; works on the Free plan.
- ❌ The prefix list is shared by all CloudFront customers: anyone's distribution could reach the instance unless a secret origin header is checked (unverified on the Free plan).
- ❌ Unencrypted HTTP to the origin's public address; CloudFront's ~50 address ranges must become trusted proxies for the rate limits.

### Pricing

#### A: pay-as-you-go, always-free tier (chosen)
- ✅ $0 at demo traffic; everything in Terraform today; works with VPC origins.
- ❌ **No cost cap**: a request flood beyond the free tier would be billed; budgets only notify. No WAF (pay-as-you-go WAF is ~$5+/month).

#### B: flat-rate Free plan
- ✅ Guaranteed $0 even under attack; WAF with 5 rules included.
- ❌ Forces option B above, no logs, subscription by hand until the provider release.

### Caching

#### A: cache `/_nuxt/*` only (chosen)
- ✅ Nuxt's build files carry a content hash in their names, so they never go stale; faster loads and fewer requests to the 2 GB instance.
- ❌ One more cache behavior to reason about.

#### B: cache nothing
- ✅ Simplest; ❌ every request reaches the instance.

## Decision

1. **CloudFront distribution with a VPC origin** pointing at the instance's private address, port 80, HTTP inside the VPC. The instance's security group gets one inbound rule: TCP 80 from `CloudFront-VPCOrigins-Service-SG`. Caddy also listens on the instance's private address (port 80); the tailnet path (`tailscale serve` → `127.0.0.1:8080`) stays.
2. **Pay-as-you-go** pricing with the always-free tier; price class 100 (North America and Europe edges); the default `*.cloudfront.net` certificate; HTTP redirected to HTTPS; HTTP/2 and HTTP/3; IPv6 on.
3. **Caching:** `/_nuxt/*` with the managed `CachingOptimized` policy; everything else (pages, `/api/*`) with the managed `CachingDisabled` policy and the managed `AllViewerExceptHostHeader` origin-request policy (all methods allowed).
4. **Dev tools stay tailnet-only:** CloudFront forwards viewers' headers, so a visitor could send `Tailscale-User-Login` themselves. Caddy now also answers 404 for `/dev` and `/api/v2/dev` when the request carries `X-Amz-Cf-Id`, a header CloudFront sets on every origin request and viewers can't remove. The demo's API also runs with `XCRS_DEV_TOOLS=false`.

## Trade-offs accepted

- **No hard cost cap.** Mitigated by the `zero-spend` and `monthly-10` budgets, the app's rate limits, and a backlog item: a budget action that stops the instance when the month's cost reaches $20.
- No WAF; Shield Standard and the app's own limits are the protection.
- A replaced instance means a replaced VPC origin (~15 minutes) on the next apply.
- CloudFront → instance is plain HTTP, inside the VPC only.

## Revisit when

- Traffic or attacks make a cost cap or WAF worth it: the flat-rate plans (Business tier keeps VPC origins), or pay-as-you-go WAF.
- A domain is bought: an ACM certificate and an alias on the distribution (and on the VMs, ADR-0040).
- The provider ships the pricing-plan resource and the Free plan gains private origins.
