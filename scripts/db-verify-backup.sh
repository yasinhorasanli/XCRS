#!/usr/bin/env bash
# Prove a backup restores: load it into a throwaway pgvector container and compare row counts with its
# manifest. Never touches the real database.
#
#   scripts/db-verify-backup.sh                     # the newest backup
#   scripts/db-verify-backup.sh backups/xcrs-....dump
set -euo pipefail

# sha256sum on Linux, shasum on macOS
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi; }

cd "$(dirname "$0")/.."
DUMP="${1:-$(ls -1t "${XCRS_BACKUP_DIR:-backups}"/xcrs-*.dump | head -1)}"
MANIFEST="${DUMP%.dump}.manifest"
NAME="xcrs-verify-$$"
IMAGE="pgvector/pgvector:pg18"

[ -f "$MANIFEST" ] || { echo "no manifest for $DUMP"; exit 1; }
expected_sha="$(grep '^sha256:' "$MANIFEST" | cut -d' ' -f2)"
actual_sha="$(sha256 "$DUMP" | cut -d' ' -f1)"
[ "$expected_sha" = "$actual_sha" ] || { echo "CHECKSUM MISMATCH: $DUMP is damaged"; exit 1; }

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
docker run -d --name "$NAME" -e POSTGRES_USER=xcrs -e POSTGRES_PASSWORD=verify -e POSTGRES_DB=xcrs "$IMAGE" >/dev/null
until docker exec "$NAME" pg_isready -U xcrs -d xcrs >/dev/null 2>&1; do sleep 1; done
sleep 1
docker exec -i "$NAME" pg_restore -U xcrs -d xcrs --no-owner --exit-on-error < "$DUMP"

failures=0
while IFS= read -r line; do
  table="$(echo "$line" | sed -E 's/^ +([^:]+): .*/\1/')"
  expected="$(echo "$line" | sed -E 's/.*: //')"
  actual="$(docker exec "$NAME" psql -U xcrs -d xcrs -tAX -c "SELECT count(*) FROM $table")"
  if [ "$expected" != "$actual" ]; then
    echo "MISMATCH $table: manifest $expected, restored $actual"; failures=$((failures + 1))
  fi
done < <(sed -n '/^row_counts:/,$p' "$MANIFEST" | tail -n +2)

tables="$(sed -n '/^row_counts:/,$p' "$MANIFEST" | tail -n +2 | wc -l | tr -d ' ')"
if [ "$failures" -eq 0 ]; then
  echo "OK: $DUMP restores; checksum and row counts of $tables tables match"
else
  echo "FAILED: $failures of $tables tables differ"; exit 1
fi
