# ADR-0036: Backups: nightly verified dumps on VM-A, copied to VM-B, with a weekly restore test

- **Status:** Accepted. Delegated (overnight, 2026-10-02); to be reviewed.
- **Reviewed (2026-10-02):** the decider confirmed the delegated choices.
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

## Note (2026-10-26, security audit run 1): retention by age, and copies VM-A can't remove

Two findings of the audit (`#13`, `#14`) changed decision 1:

- **Retention by age, the same on both VMs** (the decider chose this over 14 dumps on VM-B or a 30-day privacy promise): `scripts/db-prune-backups.sh` deletes dumps older than 13 days by the UTC time in their names and always keeps the newest 3. It runs at least daily (after each backup on VM-A, hourly on VM-B), so no dump reaches the 14 days the privacy page promises. Counting dumps had let VM-B keep 30 (about a month), and missed nights or extra deploy dumps change what a count means.
- **VM-B prunes its own copies.** `db-offsite-copy.sh` only sends new files over rsync; it ran its retention as a shell command on VM-B, which needed a key that could run anything there. The key is now a dedicated one (`/root/.ssh/xcrs-backup`), and VM-B limits it with `restrict,from="<VM-A>",command="rrsync -wo -no-del /srv/xcrs-backups"`: no shell, writes only, no deletes.
- **Copies are locked.** `xcrs-backup-prune.timer` (hourly) first hands every complete copy to root, read-only (`scripts/db-lock-backups.sh`), in a root-owned sticky directory, so someone with root on VM-A can't replace or delete them either.

Trade-offs accepted: if no new dump arrives for two weeks, the newest 3 stay longer than 14 days (better than having no backup; the privacy page says so since this change); a new copy can still be replaced from VM-A for up to an hour, until the timer locks it; if VM-B is down, its copies age past the limit until it runs again.

## Note (2026-10-27, security audit run 2): file names and types from VM-A, and failed backups

- **The lock only touches plain files.** The names and file types in `/srv/xcrs-backups` come from VM-A, and `chown`/`chmod` follow symbolic links, so a link sent from VM-A pointed root's hourly lock at any file on VM-B (reproduced in an Oracle Linux 10 container: two root-only files became world-readable). `db-lock-backups.sh` now skips links and hard-linked files, uses `chown -h`, checks again once the entry is root's (the sticky directory then stops swaps), and refuses to run with `fs.protected_hardlinks` off. VM-B's key line adds rrsync's `-munge`, so a link VM-A sends points nowhere.
- **Names dated in the future are pruned.** They held the newest-3 places, so every real copy could age out while they stayed.
- **VM-A prunes even when the backup fails**, and removes the unfinished `.dump.partial`. The prune used to be the last step of a successful backup, so failed nights kept every old dump (VM-A has no prune timer).
- **The restore test leaves nothing.** The pgvector image keeps its data in an anonymous volume, which `docker rm -f` left behind with the whole restored database in it, once per deploy and weekly on VM-B; `db-verify-backup.sh` now removes it (`-v`). Volumes left by earlier runs are removed by hand (deploy/README.md).
