#!/usr/bin/env bash
# Deploy a tagged release on VM-A (ADR-0034): verified backup, pull, migrate, import the catalog, restart,
# smoke test through Caddy. Rollback = deploy.sh <previous tag>.
#
#   deploy/deploy.sh <image tag>        # a commit SHA (immutable) or a branch tag such as "modernization"
#
# Needs deploy/vm-a/.env (copy .env.example). Safe to re-run.
set -euo pipefail

TAG="${1:?usage: deploy/deploy.sh <image tag>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT/deploy/vm-a/.env"
[ -f "$ENV_FILE" ] || { echo "missing $ENV_FILE (copy deploy/vm-a/.env.example)"; exit 1; }
export COMPOSE_FILE="$ROOT/deploy/vm-a/compose.yaml"   # also used by scripts/db-*.sh
export XCRS_IMAGE_TAG="$TAG"
cd "$ROOT"
set -a; . "$ENV_FILE"; set +a
export XCRS_IMAGE_TAG="$TAG"
step() { printf '\n== %s\n' "$*"; }

step "Database up"
docker compose up -d postgres
until docker compose exec -T postgres pg_isready -U "${POSTGRES_USER:-xcrs}" -d "${POSTGRES_DB:-xcrs}" >/dev/null 2>&1; do sleep 1; done

if docker compose exec -T postgres psql -U "${POSTGRES_USER:-xcrs}" -d "${POSTGRES_DB:-xcrs}" -tAc \
     "SELECT 1 FROM information_schema.tables WHERE table_name = 'alembic_version'" | grep -q 1; then
  step "Backup before changing anything (ADR-0036)"
  scripts/db-backup.sh
  scripts/db-verify-backup.sh
fi

if [ "${XCRS_SKIP_PULL:-0}" = "1" ]; then
  step "Using local images for $TAG (XCRS_SKIP_PULL=1, local verification)"
else
  step "Pull $TAG"
  docker compose pull api web
fi

step "Models for embeddings (VM-A); the LLM lives on VM-B"
docker compose up -d ollama-embed
docker compose exec -T ollama-embed ollama pull qwen3-embedding:0.6b >/dev/null

step "Migrate, register the model, import and embed the catalog"
run() { docker compose run --rm --no-deps api "$@"; }
run alembic upgrade head
run xcrs register-model qwen3-embedding:0.6b --id 1 --status active >/dev/null || true   # no-op when registered
run xcrs catalog import
run xcrs catalog embed

step "Restart"
docker compose up -d
sed -i.bak "s/^XCRS_IMAGE_TAG=.*/XCRS_IMAGE_TAG=$TAG/" "$ENV_FILE" && rm -f "$ENV_FILE.bak"

step "Smoke test through Caddy"
if [ "${XCRS_DOMAIN:-:80}" = ":80" ]; then BASE="http://localhost:${XCRS_HTTP_PORT:-80}"; else BASE="https://$XCRS_DOMAIN"; fi
for i in $(seq 1 60); do curl -fsS "$BASE/api/v2/health" >/dev/null 2>&1 && break; sleep 2; done
curl -fsS "$BASE/api/v2/health"; echo
# A v2 endpoint only this code serves: proves the new image answers, not just that something is healthy.
curl -fsS "$BASE/api/v2/skills?q=kubernetes" | grep -q '"id":"kubernetes"' && echo "v2 catalog search: ok"
curl -fsS -o /dev/null -w "home page: %{http_code}\n" "$BASE/v2"
echo "deployed $TAG"
