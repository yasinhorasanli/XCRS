terraform {
  backend "s3" {
    bucket       = "xcrs-tfstate-d6b0fd" # infra/bootstrap
    key          = "demo/terraform.tfstate"
    region       = "eu-central-1"
    use_lockfile = true
    encrypt      = true
  }
}
