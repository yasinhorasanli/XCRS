# infra/: the AWS demo copy (Terraform)

AWS runs a disposable demo/staging copy of XCRS; the two VMs stay its real home ([ADR-0042](../docs/adr/0042-aws-demo-copy-on-one-ec2-instance.md)). Everything in the AWS account is created from this folder, and everything except the state bucket can be destroyed when idle.

| Stack | What it holds | Cost |
|---|---|---|
| `bootstrap/` | The S3 bucket `xcrs-tfstate-d6b0fd` (eu-central-1) that holds the Terraform state of every stack, its own included | ~$0 (a few KB) |
| `demo/` | VPC (one public subnet, no NAT), a security group whose only inbound rule admits CloudFront's VPC origin, the instance role, and one EC2 t4g.small (Amazon Linux 2023, arm64, 20 GB gp3) that cloud-init turns into the demo copy (`deploy/aws/compose.yaml`), and a CloudFront distribution in front (ADR-0047) | ~$5.55/month during the t4g trial (until 31 Dec 2026: public IPv4 + disk, paid from Free-plan credits); ~$19.57/month always-on after it ($14.02 instance, $3.65 IPv4, $1.90 disk); stops itself at $20 (ADR-0048) |

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

## The demo stack (`demo/`)

**What happens on `terraform apply`:** the instance boots and cloud-init (`demo/cloud-init.sh.tftpl`) does the rest in about 10 minutes:
1. adds 1 GB of swap;
2. installs Docker and the Compose plugin;
3. joins the tailnet as `xcrs-aws` (`tag:xcrs-aws`) with the auth key from SSM Parameter Store;
4. publishes Caddy to the tailnet with `tailscale serve`;
5. clones the repo at `release` (default `main`; a branch or SHA whose images exist for arm64);
6. writes `deploy/aws/.env` (a fresh Postgres password) and runs `XCRS_DEPLOY_DIR=deploy/aws deploy/deploy.sh`.

The catalog is imported and embedded on every first boot. Both models are on VM-B.

**Before the first apply (once):**
1. **VM-B** serves the embedding model too, and publishes Ollama to the tailnet. Ollama itself still listens only on the private address:
   ```bash
   cd /opt/xcrs/deploy/vm-b && docker compose exec ollama ollama pull qwen3-embedding:0.6b
   sudo tailscale serve --bg --tcp 11434 tcp://<VM-B private IP>:11434
   ```
   `serve` forwards to the private address (checked 2026-10-10, Tailscale 1.102), so Ollama, firewalld and VM-A's path stay as they were; `tailscale serve --tcp=11434 off` undoes it. The tailnet policy decides who may connect.
2. **Tailnet policy** (Tailscale admin console → Access controls). Add a tag and rules so the AWS machine can reach only VM-B's Ollama, while your own devices keep reaching everything. Keep any existing `nodeAttrs` (Funnel) and `ssh` sections:
   ```jsonc
   "tagOwners": { "tag:xcrs-aws": ["autogroup:admin"] },
   "hosts":     { "xcrs-b": "100.76.33.94" },
   "grants": [
     // your devices (the Mac, VM-A, VM-B): everything, as before
     { "src": ["autogroup:member"], "dst": ["*"], "ip": ["*"] },
     // the AWS demo copy: VM-B's Ollama only
     { "src": ["tag:xcrs-aws"], "dst": ["xcrs-b"], "ip": ["tcp:11434"] }
   ]
   ```
   This replaces the default allow-all rule (`"src": ["*"], "dst": ["*"]`).
   The policy also carries a `tests` block that Tailscale runs on every save: `tag:xcrs-aws` reaches `xcrs-b:11434` but not SSH on either VM or VM-A's site.
3. **Auth key** (Settings → Keys → Generate auth key): reusable, ephemeral (the machine leaves the tailnet when destroyed), pre-approved, tag `tag:xcrs-aws`, 90 days. Store it as an encrypted parameter, typed in rather than pasted on the command line:
   ```bash
   read -rs TS_KEY && aws ssm put-parameter --profile xcrs --name /xcrs/demo/tailscale-auth-key \
     --type SecureString --value "$TS_KEY" --overwrite && unset TS_KEY
   ```
   The key expires after 90 days; repeat this step before the next apply after that.
4. **Session Manager plugin** for shells on the instance: `brew install --cask session-manager-plugin`.
5. **Alert email:** `cp terraform.tfvars.example terraform.tfvars` in `infra/demo` and put your address in it (gitignored).

**Public access (ADR-0047):** `terraform output site` is the CloudFront address (`https://d….cloudfront.net`).
- CloudFront reaches the instance through a **VPC origin**: a CloudFront-managed network interface in our VPC that connects to Caddy on the instance's private address, port 80. The instance's one inbound rule admits only CloudFront's service security group.
- Pages and `/api` are never cached; `/_nuxt/*` (content-hashed build files) is.
- `/dev` answers 404 through CloudFront: Caddy refuses requests marked with `X-Amz-Cf-Id`.
- A first apply, or a replaced instance, takes ~15 minutes extra while the VPC origin deploys.
- A request may run at most 60 seconds at the edge, which matters for typed phrases that need the LLM on VM-B.

**Cost guard (ADR-0048):** a budget `xcrs-demo-stop-at-20` emails at 80% and, when the month's **actual** spend (credits excluded) reaches **$20**, AWS Budgets stops the instance on its own through SSM. A normal month is ~$5.55 in the trial and ~$19.57 always-on after it, so it fires only on unusual spending, hours after the fact (budget data refreshes a few times a day). It doesn't stop CloudFront from counting requests. After a stop:
```bash
aws ec2 start-instances --profile xcrs --instance-ids "$(terraform output -raw instance_id)"
```
Docker restarts the containers, Tailscale reconnects, and the private address (CloudFront's origin) stays the same. Tested 2026-10-10 with the action's own SSM document (`AWS-StopEC2Instance`): stopped in 31 s, public IPv4 released (no charge while stopped), CloudFront answered 504 meanwhile; after `start-instances` the site was healthy through CloudFront 25 s later, with LLM matching over the tailnet working.

**Everyday:**
- `terraform apply` → `terraform output shell` gives a shell (no SSH) → `sudo tail -f /var/log/cloud-init-output.log` shows the first boot → the site opens at `https://xcrs-aws.<tailnet>.ts.net` from a tailnet device.
- `terraform destroy` when idle. Everything goes except the state bucket, and the next apply rebuilds the same thing.
- A newer Amazon Linux image: `terraform apply -replace=aws_instance.demo`.
- Any change to the first-boot script replaces the instance (`user_data_replace_on_change`).
- After a replacement the new machine can join the tailnet as `xcrs-aws-1`, while the old ephemeral entry is still being removed. Rename it in the Tailscale admin console (Machines → the machine → Edit machine name → `xcrs-aws`); `tailscale set --hostname` on the instance doesn't change the machine name. Then re-register `serve` under the new name, or HTTPS on the tailnet fails with a TLS error: in an SSM shell, `sudo tailscale serve reset && sudo tailscale serve --bg 8080`.

## Backlog

- **Nightly stop** (EventBridge Scheduler) and **S3 backups**: steps 6 and 8 of the AWS plan (ADR-0042).
- **GitHub Actions OIDC** (plan on PR, apply on approval): step 7.

## Rebuilding the state bucket from nothing

The bootstrap stack stores its state in the bucket it creates. If the bucket is ever gone: move `bootstrap/backend.tf` aside, `terraform init && terraform apply` (local state), put `backend.tf` back, `terraform init -migrate-state`, then delete the local `terraform.tfstate*` files. The bucket has `prevent_destroy`, so `terraform destroy` refuses to remove it.
