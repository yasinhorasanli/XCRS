# Added after the first apply: this stack's own state lives in the bucket it creates. To rebuild the bucket
# from nothing, move this file aside, apply with local state, put it back, then `terraform init -migrate-state`.
terraform {
  backend "s3" {
    bucket       = "xcrs-tfstate-d6b0fd"
    key          = "bootstrap/terraform.tfstate"
    region       = "eu-central-1"
    use_lockfile = true # native S3 locking: a .tflock object next to the state, no DynamoDB table
    encrypt      = true
  }
}
