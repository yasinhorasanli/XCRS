output "tunnel_id" {
  value = cloudflare_zero_trust_tunnel_cloudflared.vm_a.id
}

output "tunnel_token" {
  description = "For VM-A's .env (XCRS_TUNNEL_TOKEN); read with `terraform output -raw tunnel_token`, never print it."
  value       = data.cloudflare_zero_trust_tunnel_cloudflared_token.vm_a.token
  sensitive   = true
}

output "site" {
  value = "https://${var.zone}"
}
