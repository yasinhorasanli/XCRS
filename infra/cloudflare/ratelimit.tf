# A flood guard at Cloudflare's edge (ADR-0051 note, 2026-11-12): an IP that sends more than 100 requests to /api/
# within 10 seconds is answered 429 by Cloudflare for the next 10 seconds, so the flood never reaches VM-A. It is
# coarse on purpose: the API's own per-IP limits (xcrs/api/limits.py, ADR-0035) stay the real ones. 100 leaves
# room for many visitors behind one shared address (a class on campus Wi-Fi), each results page polling every
# 2.5 s (4 requests per 10 s).
#
# Free plan (developers.cloudflare.com/waf/rate-limiting-rules, availability): one rule, the path as the only field,
# counted per IP, 10-second period and timeout only, rate limits also cached files (not an issue: /api isn't cached).
# Only proxied names see it: xcrs.app and www. aws.xcrs.app is DNS-only and goes straight to CloudFront.
# Needs the token's "Zone WAF: Edit" permission (infra/README.md).
resource "cloudflare_ruleset" "rate_limit" {
  zone_id     = data.cloudflare_zone.xcrs.id
  name        = "xcrs rate limits"
  description = "Managed by Terraform (infra/cloudflare)"
  kind        = "zone"
  phase       = "http_ratelimit"

  rules = [{
    ref         = "api_flood_per_ip"
    description = "More than 100 /api/ requests in 10 s from one IP: 429 for 10 s"
    expression  = "starts_with(http.request.uri.path, \"/api/\")"
    action      = "block"
    ratelimit = {
      characteristics     = ["cf.colo.id", "ip.src"]
      period              = 10
      requests_per_period = 100
      mitigation_timeout  = 10
    }
  }]
}
