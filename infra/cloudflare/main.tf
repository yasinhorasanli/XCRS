# Production's public entrance (ADR-0040, ADR-0051): xcrs.app → Cloudflare's edge → a tunnel that cloudflared on
# VM-A keeps open from the inside (no open ports) → Caddy. www.xcrs.app takes the same path and Caddy redirects it
# to the apex. aws.xcrs.app (the AWS demo copy) is owned by infra/demo.

data "cloudflare_zone" "xcrs" {
  filter = { name = var.zone }
}

resource "cloudflare_zero_trust_tunnel_cloudflared" "vm_a" {
  account_id = var.account_id
  name       = "xcrs-vm-a"
  config_src = "cloudflare" # routes are managed here (below), not in a file on VM-A
}

resource "cloudflare_zero_trust_tunnel_cloudflared_config" "vm_a" {
  account_id = var.account_id
  tunnel_id  = cloudflare_zero_trust_tunnel_cloudflared.vm_a.id
  config = {
    ingress = [
      { hostname = var.zone, service = var.origin },
      { hostname = "www.${var.zone}", service = var.origin },
      { service = "http_status:404" }, # any other name that reaches the tunnel
    ]
  }
}

# The token cloudflared presents (VM-A's .env: XCRS_TUNNEL_TOKEN). It lands in the encrypted state, never in git.
data "cloudflare_zero_trust_tunnel_cloudflared_token" "vm_a" {
  account_id = var.account_id
  tunnel_id  = cloudflare_zero_trust_tunnel_cloudflared.vm_a.id
}

# Both names point at the tunnel through Cloudflare's proxy (orange cloud): visitors get Cloudflare's TLS,
# caching and DDoS protection; the apex works as a CNAME thanks to Cloudflare's CNAME flattening.
resource "cloudflare_dns_record" "apex" {
  zone_id = data.cloudflare_zone.xcrs.id
  name    = var.zone
  type    = "CNAME"
  content = "${cloudflare_zero_trust_tunnel_cloudflared.vm_a.id}.cfargotunnel.com"
  proxied = true
  ttl     = 1 # "automatic", required for proxied records
  comment = "production on VM-A through the tunnel (infra/cloudflare, ADR-0051)"
}

resource "cloudflare_dns_record" "www" {
  zone_id = data.cloudflare_zone.xcrs.id
  name    = "www.${var.zone}"
  type    = "CNAME"
  content = "${cloudflare_zero_trust_tunnel_cloudflared.vm_a.id}.cfargotunnel.com"
  proxied = true
  ttl     = 1
  comment = "redirected to the apex by Caddy (infra/cloudflare, ADR-0051)"
}
