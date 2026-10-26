#!/usr/bin/env bash
# Copy backups to another machine (ADR-0036): rsync over SSH to $XCRS_BACKUP_REMOTE. Only new files are sent;
# nothing is deleted or run there. VM-B limits this key to writing into its backup directory (rrsync, see
# deploy/README.md) and prunes and locks the copies itself (xcrs-backup-prune.timer), so this machine, which faces
# the internet, can't remove or replace the copies that exist to outlive it.
#
#   XCRS_BACKUP_REMOTE=backup@10.0.0.2: XCRS_BACKUP_SSH='ssh -i /root/.ssh/xcrs-backup' scripts/db-offsite-copy.sh
#
# With rrsync the remote path is relative to the directory the key is limited to, so `backup@host:` is that directory.
set -euo pipefail

cd "$(dirname "$0")/.."
REMOTE="${XCRS_BACKUP_REMOTE:?set XCRS_BACKUP_REMOTE=user@host:path}"
BACKUP_DIR="${XCRS_BACKUP_DIR:-backups}"
SSH="${XCRS_BACKUP_SSH:-ssh -i /root/.ssh/xcrs-backup} -o BatchMode=yes -o IdentitiesOnly=yes"

rsync -a --ignore-existing -e "$SSH" "$BACKUP_DIR"/xcrs-*.dump "$BACKUP_DIR"/xcrs-*.manifest "$REMOTE"
echo "copied to $REMOTE"
