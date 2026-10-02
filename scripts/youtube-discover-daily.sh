#!/usr/bin/env bash
# Daily YouTube discovery (ADR-0033): one quota day of searches, continuing where the last run stopped.
# Run by a launchd agent on the dev Mac (see deploy/README.md, "YouTube discovery"); safe to run by hand.
# Candidates go to catalog/sources/youtube-candidates.yaml and untracked/youtube-candidates.md for review;
# nothing is ingested until the decider approves ids into catalog/sources/youtube.yaml.
set -euo pipefail
cd "$(dirname "$0")/../backend"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
log_dir=../untracked/logs
mkdir -p "$log_dir"
{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S')"
  uv run xcrs resources youtube-discover --max-searches "${MAX_SEARCHES:-85}" 2>&1 | grep -v "^INFO" || true
} >> "$log_dir/youtube-discover.log"
