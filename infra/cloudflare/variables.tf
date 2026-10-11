variable "account_id" {
  description = "Cloudflare account id (not secret, but kept out of the code: set it in terraform.tfvars, gitignored)."
  type        = string
}

variable "zone" {
  type    = string
  default = "xcrs.app"
}

variable "origin" {
  description = "Where the tunnel delivers requests on VM-A: Caddy, reached by cloudflared inside the compose network."
  type        = string
  default     = "http://caddy:80"
}
