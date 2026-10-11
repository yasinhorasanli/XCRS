terraform {
  backend "s3" {
    bucket       = "xcrs-tfstate-d6b0fd" # infra/bootstrap (the AWS profile is still needed for the state)
    key          = "cloudflare/terraform.tfstate"
    region       = "eu-central-1"
    use_lockfile = true
    encrypt      = true
  }
}
