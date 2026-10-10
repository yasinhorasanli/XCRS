#!/usr/bin/env bash
# Back up the XCRS PostgreSQL database (docker compose service "postgres").
#
#   scripts/db-backup.sh            # -> backups/xcrs-<UTC time>.dump + .manifest
#   KEEP=30 scripts/db-backup.sh    # keep the newest 30 dumps (default 14)
#
# The dump is pg_dump's custom format (compressed, restorable table by table with pg_restore).
# The manifest records exact row counts, the Alembic revision, the git commit and a SHA-256, so a
# restore can be checked (scripts/db-verify-backup.sh). Backups stay on this machine: copy them
# somewhere else too (another disk or host), or a lost machine loses them with the database.
set -euo pipefail

# sha256sum on Linux, shasum on macOS
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi; }

cd "$(dirname "$0")/.."
BACKUP_DIR="${XCRS_BACKUP_DIR:-backups}"
KEEP="${KEEP:-14}"
DB="${POSTGRES_DB:-xcrs}"
USER_NAME="${POSTGRES_USER:-xcrs}"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
DUMP="$BACKUP_DIR/xcrs-$STAMP.dump"
MANIFEST="$BACKUP_DIR/xcrs-$STAMP.manifest"

mkdir -p "$BACKUP_DIR"
psql_q() { docker compose exec -T postgres psql -U "$USER_NAME" -d "$DB" -tAX -c "$1"; }

docker compose exec -T postgres pg_dump -U "$USER_NAME" -d "$DB" --format=custom --compress=9 > "$DUMP.partial"
mv "$DUMP.partial" "$DUMP"   # a dump only gets its final name once it is complete

{
  echo "created_utc: $STAMP"
  echo "database: $DB"
  echo "postgres: $(psql_q 'SHOW server_version')"
  echo "alembic_revision: $(psql_q 'SELECT version_num FROM alembic_version' 2>/dev/null || echo none)"
  echo "git_commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "sha256: $(sha256 "$DUMP" | cut -d' ' -f1)"
  echo "bytes: $(wc -c < "$DUMP" | tr -d ' ')"
  echo "row_counts:"
  for table in $(psql_q "SELECT table_schema || '.' || table_name FROM information_schema.tables
                         WHERE table_type = 'BASE TABLE' AND table_schema NOT IN ('pg_catalog', 'information_schema')
                         ORDER BY 1"); do
    echo "  $table: $(psql_q "SELECT count(*) FROM $table")"
  done
} > "$MANIFEST"

# Retention: delete the oldest dumps (and their manifests) beyond KEEP.
ls -1t "$BACKUP_DIR"/xcrs-*.dump 2>/dev/null | tail -n +"$((KEEP + 1))" | while read -r old; do
  rm -f "$old" "${old%.dump}.manifest"
done

echo "Backup: $DUMP ($(du -h "$DUMP" | cut -f1)), manifest: $MANIFEST"
