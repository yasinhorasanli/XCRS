#!/usr/bin/env bash
# VM-B (ADR-0036, note of 2026-10-26): hand each complete copy (a dump whose manifest has arrived; rsync sends the
# manifest after its dump) to root, read-only. The directory is root:backup 1770 (sticky), so VM-A's write-only key
# can add files but can't replace or delete these.
#
#   scripts/db-lock-backups.sh /srv/xcrs-backups
set -euo pipefail

DIR="${1:-${XCRS_BACKUP_DIR:?give the backup directory}}"
for manifest in "$DIR"/xcrs-*.manifest; do
  dump="${manifest%.manifest}.dump"
  [ -f "$manifest" ] && [ -f "$dump" ] || continue
  chown root:root "$manifest" "$dump"
  chmod 0444 "$manifest" "$dump"
done
