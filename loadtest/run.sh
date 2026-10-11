#!/usr/bin/env bash
# Step the load up for one scenario until k6's safety thresholds fail, one summary per rate.
#
#   loadtest/run.sh BASE_URL SCENARIO [RATES...]      e.g. loadtest/run.sh http://100.88.82.63:8081 browse 2 5 10 20 40
#
# Runs k6 in Docker (no install). Summaries land in loadtest/results/<scenario>-<rate>.json (gitignored).
set -euo pipefail
BASE="${1:?base url}"; SCENARIO="${2:?browse|recommend|match}"; shift 2
if [ "$#" -gt 0 ]; then RATES=("$@"); else RATES=(2 5 10 20 40 80 160); fi
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/results"
for rate in "${RATES[@]}"; do
  echo "== $SCENARIO at $rate/s"
  set +e
  docker run --rm --network host -u "$(id -u):$(id -g)" -v "$HERE:/lt" grafana/k6:2.3.0 run --quiet \
    -e BASE_URL="$BASE" -e SCENARIO="$SCENARIO" -e RATE="$rate" \
    --summary-export "/lt/results/$SCENARIO-$rate.json" /lt/xcrs.js > "$HERE/results/$SCENARIO-$rate.log" 2>&1
  code=$?
  set -e
  python3 - "$HERE/results/$SCENARIO-$rate.json" "$rate" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1])); m = d["metrics"]
dur = m["http_req_duration"]; it = m.get("iterations", {})
print(f"  rate {sys.argv[2]}/s: {it.get('rate', 0):.1f} it/s done, {m['http_reqs']['rate']:.1f} req/s, "
      f"med {dur['med']:.0f} ms, p95 {dur['p(95)']:.0f} ms, p99 {dur['p(99)']:.0f} ms, "
      f"failed {m['http_req_failed']['value']*100:.1f}%, dropped {m.get('dropped_iterations', {}).get('count', 0)}")
EOF
  if [ "$code" -ne 0 ]; then echo "  stopped: thresholds crossed (k6 exit $code)"; break; fi
  sleep 15   # let the target settle between levels
done
