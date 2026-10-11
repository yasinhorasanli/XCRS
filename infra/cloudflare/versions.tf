terraform {
  required_version = ">= 1.11"

  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 5.26"
    }
  }
}

# Credentials come from the environment: CLOUDFLARE_API_TOKEN, a token limited to the xcrs.app zone (DNS edit,
# zone read) and the account's tunnels (edit), kept in the macOS Keychain (infra/README.md). Never in this code.
provider "cloudflare" {}
