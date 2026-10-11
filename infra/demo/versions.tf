terraform {
  required_version = ">= 1.11"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.67"
    }
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 5.26"
    }
  }
}

# Credentials come from the environment (AWS_PROFILE=xcrs-tf locally, OIDC in CI), never from this code.
provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "xcrs"
      Stack     = "demo"
      ManagedBy = "terraform"
    }
  }
}

# CLOUDFLARE_API_TOKEN from the environment, for the demo's DNS records in the xcrs.app zone (infra/README.md).
provider "cloudflare" {}
