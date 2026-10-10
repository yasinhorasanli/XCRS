#!/usr/bin/env bash
# Copy backups to another machine (ADR-0036): rsync over SSH to $XCRS_BACKUP_REMOTE (user@host:/path), then
# keep the newest $REMOTE_KEEP there. Use a dedicated key that can only write to the backup directory.
#
#   XCRS_BACKUP_REMOTE=backup@10.0.0.2:/srv/xcrs-backups scripts/db-offsite-copy.sh
set -euo pipefail

cd "$(dirname "$0")/.."
REMOTE="${XCRS_BACKUP_REMOTE:?set XCRS_BACKUP_REMOTE=user@host:/path}"
REMOTE_KEEP="${REMOTE_KEEP:-30}"
BACKUP_DIR="${XCRS_BACKUP_DIR:-backups}"
SSH="${XCRS_BACKUP_SSH:-ssh -o BatchMode=yes}"

rsync -a --ignore-existing -e "$SSH" "$BACKUP_DIR"/xcrs-*.dump "$BACKUP_DIR"/xcrs-*.manifest "$REMOTE/"
host="${REMOTE%%:*}"; path="${REMOTE#*:}"
$SSH "$host" "cd '$path' && ls -1t xcrs-*.dump | tail -n +$((REMOTE_KEEP + 1)) | while read -r f; do rm -f \"\$f\" \"\${f%.dump}.manifest\"; done"
echo "copied to $REMOTE (keeping $REMOTE_KEEP)"
