#!/usr/bin/env bash
# Copy the daily discovery results from VM-A into this checkout for review (ADR-0033):
# ids → catalog/sources/youtube-candidates.yaml (committed), titles → untracked/youtube-candidates.md (local only).
#   scripts/youtube-candidates-pull.sh [ssh host, default xcrs-a]
set -euo pipefail
cd "$(dirname "$0")/.."
host="${1:-xcrs-a}"
scp -q "$host:/srv/xcrs-discovery/youtube-candidates.yaml" catalog/sources/youtube-candidates.yaml
mkdir -p untracked && scp -q "$host:/srv/xcrs-discovery/youtube-candidates.md" untracked/youtube-candidates.md
ssh "$host" 'journalctl -u xcrs-youtube-discover.service -n 3 --no-pager -o cat | grep searches || true'
git diff --stat catalog/sources/youtube-candidates.yaml
