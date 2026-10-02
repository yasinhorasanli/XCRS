# ADR-0036: Backups: nightly verified dumps on VM-A, copied to VM-B, with a weekly restore test

- **Status:** Accepted. Delegated (overnight, 2026-10-02); to be reviewed.
- **Date:** 2026-10-02
- **Decider:** Muhammed Yasin Horasanli

## Context

- `scripts/db-backup.sh`, `db-verify-backup.sh` and `db-restore.sh` exist (2026-10-01) but run by hand, and the dumps stay on the machine that made them.
- **What can be rebuilt:** the catalog (from git), embeddings (from the models) and raw resource records (re-fetched). Activity data (requests, results, explanations, feedback) **cannot** be rebuilt.
- **Machines:** two VMs with 40 GB disks, one provider, zero budget. A dump is about 8 MB today.

## Options considered

- **Keep dumps on VM-A only.** ❌ Losing VM-A loses the backups too.
- **Copy to VM-B.** ✅ Free; a different machine; VM-B can also run the restore test. ❌ The same provider and location, so not a disaster-proof copy.
- **Copy to cloud object storage** (S3, B2). ✅ Off-site. ❌ Costs money (small, but the budget is zero) and needs credentials. Natural once the AWS phase starts.

## Decision

1. **Nightly on VM-A** (systemd timer, 03:30): `scripts/db-backup.sh` (custom-format dump plus manifest, keeps 14), then `scripts/db-offsite-copy.sh`, which copies new dumps to VM-B over SSH (rsync, a dedicated key, a backups directory only) and keeps 30 there.
2. **Weekly on VM-B** (systemd timer, Sundays): `scripts/db-verify-backup.sh` on the newest copy. It restores into a throwaway container and compares row counts with the manifest. A failure leaves a marker file and a non-zero unit status (`systemctl --failed`).
3. **Before every deploy** `deploy/deploy.sh` takes and verifies a backup (ADR-0034), as does any data-changing step (the project's backup rule).
4. **Off-provider copies** (S3 in the AWS phase, or the decider's own disk via `scripts/db-offsite-copy.sh` with another target) come later.

## Trade-offs accepted

- Both copies are with one provider; a provider-wide loss loses the data. Mitigated later by an off-provider target, which is one environment variable.
- A nightly schedule means losing up to a day of activity. Fine without users; WAL archiving or point-in-time recovery comes with real traffic.

## Revisit when

- Real users → a shorter interval or continuous WAL archiving, and an off-provider copy.
- The AWS phase → copy to S3 with lifecycle rules.
