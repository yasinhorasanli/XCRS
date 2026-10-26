#!/usr/bin/env bash
# VM-B (ADR-0036, note of 2026-10-26): hand each complete copy (a dump whose manifest has arrived; rsync sends the
# manifest after its dump) to root, read-only. The directory is root:backup 1770 (sticky), so VM-A's write-only key
# can add files but can't replace or delete these.
#
# The names and file types come from VM-A, so only plain files are touched: a link would point chown and chmod at
# any file on VM-B (security audit run 2, 2026-10-27). `chown -h` comes first and never follows a link; once an
# entry is root's, the sticky directory stops the backup user from swapping it, so the second check holds. Hard
# links to files that aren't the backup user's are blocked by fs.protected_hardlinks, which must stay on.
#
#   scripts/db-lock-backups.sh /srv/xcrs-backups
set -euo pipefail

DIR="${1:-${XCRS_BACKUP_DIR:?give the backup directory}}"
[ "$(cat /proc/sys/fs/protected_hardlinks)" = 1 ] || { echo "fs.protected_hardlinks is off: not locking"; exit 1; }

plain() { [ -f "$1" ] && [ ! -L "$1" ] && [ "$(stat -c %h -- "$1")" = 1 ]; }

for manifest in "$DIR"/xcrs-*.manifest; do
  dump="${manifest%.manifest}.dump"
  plain "$manifest" && plain "$dump" || continue
  chown -h root:root -- "$manifest" "$dump"
  for f in "$manifest" "$dump"; do
    if plain "$f"; then chmod 0444 -- "$f"; else rm -f -- "$f"; echo "removed $f (not a plain file)"; fi
  done
done
