# infra/: the AWS demo copy (Terraform)

AWS runs a disposable demo/staging copy of XCRS; the two VMs stay its real home ([ADR-0042](../docs/adr/0042-aws-demo-copy-on-one-ec2-instance.md)). Everything in the AWS account is created from this folder, and everything except the state bucket can be destroyed when idle.

| Stack | What it holds | Cost |
|---|---|---|
| `bootstrap/` | The S3 bucket `xcrs-tfstate-d6b0fd` (eu-central-1) that holds the Terraform state of every stack, its own included | ~$0 (a few KB) |

## Account and credentials

- Account `xcrs` on AWS's Free plan, region **eu-central-1** (Frankfurt, near the VMs). The root user is used only for account-level tasks (two MFA devices); daily work is the IAM user `yasin-admin` (group `admins`, MFA). There is no AWS Organization or IAM Identity Center: joining an Organization ends the Free plan.
- Budgets `zero-spend` and `monthly-10` ($10; alerts at 50% and 100% of actual spend and 100% of forecast). Both exclude the charge types Credit and Refund, so they see usage that credits pay for.
- **No long-lived access keys.** The CLI signs in with `aws login --profile xcrs` (short-lived credentials, refreshed for up to 12 hours). Terraform's Go SDK can't read those sessions yet, so a second profile lends them over:

  ```ini
  # ~/.aws/config
  [profile xcrs]
  login_session = arn:aws:iam::<account id>:user/yasin-admin
  region = eu-central-1

  [profile xcrs-tf]
  credential_process = aws configure export-credentials --profile xcrs --format process
  region = eu-central-1
  ```

  The code never names a profile; the environment chooses (`AWS_PROFILE=xcrs-tf` locally, GitHub OIDC in CI later).

## Running a stack

```bash
aws login --profile xcrs                 # when the session has expired
export AWS_PROFILE=xcrs-tf
cd infra/<stack>
terraform init
terraform plan -out=<stack>.tfplan       # read it before applying
terraform apply <stack>.tfplan
```

- **State** lives in `s3://xcrs-tfstate-d6b0fd/<stack>/terraform.tfstate`: versioned, encrypted, HTTPS only, never public. Locking uses S3's native lock file (`use_lockfile`), so there is no DynamoDB table. Never edit the state by hand; old versions can be restored from the bucket's version history.
- `.terraform.lock.hcl` is committed and pins provider checksums for macOS arm64 and Linux amd64/arm64. After changing a provider version, run `terraform providers lock -platform=darwin_arm64 -platform=linux_amd64 -platform=linux_arm64`.
- `.terraform/`, local state files and saved plans are gitignored.

## Rebuilding the state bucket from nothing

The bootstrap stack stores its state in the bucket it creates. If the bucket is ever gone: move `bootstrap/backend.tf` aside, `terraform init && terraform apply` (local state), put `backend.tf` back, `terraform init -migrate-state`, then delete the local `terraform.tfstate*` files. The bucket has `prevent_destroy`, so `terraform destroy` refuses to remove it.
