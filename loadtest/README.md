# loadtest/: load tests with k6

How many visitors XCRS serves before it slows down or fails, and what gives out first. First run and what it found: [the incident write-up](../docs/incidents/2026-10-11-api-wedged-under-overload.md); numbers in [`docs/baseline.md`](../docs/baseline.md).

| Scenario | One iteration = one visitor doing |
|---|---|
| `browse` | the home page (server-rendered), a skill search, the skill groups |
| `api` | a skill search and the skill groups, against the API alone (for a local container) |
| `recommend` | a board of 4–8 catalog skills sent for recommendations (scoring + one database write) |
| `match` | two typed phrases matched by lookup and similarity (the embedding model, no LLM) |

## Running it

```bash
loadtest/run.sh http://TARGET:PORT browse 2 5 10 20 40      # one 45 s run per rate (visitors per second)
```

- `run.sh` runs k6 in Docker, steps the rate up, and stops at the first level that breaks the safety rule (p95 > 5 s or > 5% errors). One summary per level goes to `loadtest/results/` (gitignored).
- Use bash, not zsh: zsh doesn't split unquoted variables into words, which garbles loops over arguments.

## Preparing the target

- **Rate limits:** they key on the client's IP, so one load generator is throttled after a few requests. Switch them off for the test (`XCRS_RATE_LIMITS_ENABLED=false`).
- **The LLM:** switch it off too (`XCRS_LLM_ENABLED=false`), unless the LLM itself is under test. With it off, new recommendations get explanations marked `disabled`, so nothing piles up on VM-B when it's switched back on.
- **On VM-A**, a temporary Compose override does both, and also publishes Caddy on the tailnet address so VM-B can reach it. Remove it afterwards by re-running Compose without the file:

  ```yaml
  # /tmp/xcrs-loadtest.override.yaml (not in the repo)
  services:
    api:
      environment:
        XCRS_LLM_ENABLED: "false"
        XCRS_RATE_LIMITS_ENABLED: "false"
    caddy:
      ports:
        - "<VM-A tailnet IP>:8081:80"
  ```

  ```bash
  docker compose -f deploy/vm-a/compose.yaml -f /tmp/xcrs-loadtest.override.yaml up -d api caddy   # test mode
  docker compose -f deploy/vm-a/compose.yaml up -d api caddy                                        # back to normal
  ```
- **Back up first** (`scripts/db-backup.sh` + `scripts/db-verify-backup.sh`): the test writes recommendations into the database.
- Run the generator from another machine (VM-B), so it doesn't compete with the target for CPU.
