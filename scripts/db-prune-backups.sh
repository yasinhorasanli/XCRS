#!/usr/bin/env bash
# Backup retention (ADR-0036, note of 2026-10-26), the same on both VMs: delete dumps (and their manifests) older
# than KEEP_DAYS days, by the UTC time in their names, but always keep the newest KEEP_MIN, so failed backups never
# leave none. It runs at least daily (VM-A after each backup, VM-B from xcrs-backup-prune.timer), so with 13 days no
# dump reaches the 14 days the privacy page promises. Names that aren't xcrs-<UTC time>.dump are left alone. A name
# dated more than a day ahead is deleted first: on VM-B the names come from VM-A, and future names would otherwise
# hold the newest-KEEP_MIN places while every real copy ages out (security audit run 2, 2026-10-27).
#
#   scripts/db-prune-backups.sh                     # backups/ (or $XCRS_BACKUP_DIR)
#   KEEP_DAYS=13 KEEP_MIN=3 scripts/db-prune-backups.sh /srv/xcrs-backups
set -euo pipefail

DIR="${1:-${XCRS_BACKUP_DIR:-backups}}"
KEEP_DAYS="${KEEP_DAYS:-13}"
KEEP_MIN="${KEEP_MIN:-3}"
# GNU date (the VMs) or BSD date (macOS)
cutoff="$(date -u -d "-$KEEP_DAYS days" +%Y%m%d-%H%M%S 2>/dev/null || date -u -v-"$KEEP_DAYS"d +%Y%m%d-%H%M%S)"
ahead="$(date -u -d "+1 day" +%Y%m%d-%H%M%S 2>/dev/null || date -u -v+1d +%Y%m%d-%H%M%S)"

dumps() { find "$DIR" -maxdepth 1 -name 'xcrs-*.dump' | grep -E '/xcrs-[0-9]{8}-[0-9]{6}\.dump$' || true; }

stamp_of() { local s="${1##*/xcrs-}"; echo "${s%.dump}"; }

dumps | while read -r dump; do
  if [[ "$(stamp_of "$dump")" > "$ahead" ]]; then
    rm -f "$dump" "${dump%.dump}.manifest"
    echo "pruned $dump (dated in the future)"
  fi
done

dumps | sort -r | tail -n +"$((KEEP_MIN + 1))" | while read -r dump; do
  if [[ "$(stamp_of "$dump")" < "$cutoff" ]]; then
    rm -f "$dump" "${dump%.dump}.manifest"
    echo "pruned $dump"
  fi
done
