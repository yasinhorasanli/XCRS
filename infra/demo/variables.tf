variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "availability_zone" {
  description = "t4g.small is offered in all three Frankfurt zones."
  type        = string
  default     = "eu-central-1a"
}

variable "release" {
  description = "What the instance runs: a branch or commit SHA of the repo, also used as the image tag (both platforms)."
  type        = string
  default     = "modernization"
}

variable "instance_type" {
  description = "t4g.small: Graviton (arm64), 2 vCPU, 2 GB, in AWS's free trial until 31 Dec 2026 (ADR-0042)."
  type        = string
  default     = "t4g.small"
}

variable "root_volume_gb" {
  type    = number
  default = 20
}

variable "tailscale_auth_key_parameter" {
  description = "SSM SecureString holding the Tailscale auth key; created by hand so the key never enters Terraform state."
  type        = string
  default     = "/xcrs/demo/tailscale-auth-key"
}

variable "tailnet_hostname" {
  type    = string
  default = "xcrs-aws"
}

variable "vm_b_ollama_url" {
  description = "VM-B's Ollama over Tailscale (`tailscale serve` on VM-B). An IP, not a MagicDNS name: containers resolve through the VPC's DNS."
  type        = string
  default     = "http://100.76.33.94:11434/v1"
}

variable "compose_version" {
  description = "Docker Compose plugin (not packaged for Amazon Linux 2023); the same version as the VMs."
  type        = string
  default     = "v5.5.1"
}

variable "compose_sha256" {
  description = "sha256 of docker-compose-linux-aarch64 for compose_version (from the release's .sha256 file)."
  type        = string
  default     = "732e3a84c1a0f67256ce80bc2598a24546b10ca05f9faa97efceb1171ece2ef7"
}

variable "stop_at_usd" {
  description = "Monthly actual spend (credits excluded) at which AWS Budgets stops the instance (ADR-0048)."
  type        = number
  default     = 20
}

variable "alert_email" {
  description = "Where budget alerts and the stop notice go. Set it in terraform.tfvars (gitignored), not in code."
  type        = string
}
