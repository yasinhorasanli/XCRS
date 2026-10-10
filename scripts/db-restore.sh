#!/usr/bin/env bash
# Restore a backup into the compose PostgreSQL.
#
#   scripts/db-restore.sh backups/xcrs-....dump xcrs_restored     # into a new database (safe)
#   scripts/db-restore.sh backups/xcrs-....dump xcrs --replace    # replace the live database
#
# Replacing drops the live database first; take a fresh backup before doing that.
set -euo pipefail

cd "$(dirname "$0")/.."
DUMP="${1:?usage: db-restore.sh DUMP TARGET_DB [--replace]}"
TARGET="${2:?usage: db-restore.sh DUMP TARGET_DB [--replace]}"
USER_NAME="${POSTGRES_USER:-xcrs}"
LIVE="${POSTGRES_DB:-xcrs}"

if [ "$TARGET" = "$LIVE" ] && [ "${3:-}" != "--replace" ]; then
  echo "Refusing to overwrite the live database '$LIVE'. Restore into another name, or pass --replace."
  exit 1
fi
psql_admin() { docker compose exec -T postgres psql -U "$USER_NAME" -d postgres -tAX -c "$1"; }
psql_admin "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '$TARGET' AND pid <> pg_backend_pid()" >/dev/null
psql_admin "DROP DATABASE IF EXISTS \"$TARGET\""
psql_admin "CREATE DATABASE \"$TARGET\" OWNER \"$USER_NAME\""
docker compose exec -T postgres pg_restore -U "$USER_NAME" -d "$TARGET" --no-owner --exit-on-error < "$DUMP"
echo "Restored $DUMP into database '$TARGET'"
