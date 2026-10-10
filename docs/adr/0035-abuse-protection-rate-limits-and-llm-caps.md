# ADR-0035: Abuse protection: per-client rate limits on expensive endpoints, and a cap on new LLM matches per request

- **Status:** Accepted. Delegated (overnight, 2026-10-02); to be reviewed.
- **Reviewed (2026-10-02):** the decider confirmed the delegated choices.
- **Date:** 2026-10-02
- **Decider:** Muhammed Yasin Horasanli

## Context

- **The expensive endpoints** call models running on CPU:
  - recommendations v1 and v2: embeddings, then LLM explanations and matching;
  - skill matching: LLM picks, about 5 s per new phrase on the CPU VM (ADR-0030);
  - related suggestions: embeddings.

  One client can tie up the CPU VM, and the matching LLM is shared with explanations.
- **A request can carry many phrases:** v2 takes up to 60 chips, each possibly new text, so one request could ask for minutes of LLM time.
- **One API process** per VM (ADR-0018's in-process worker), behind Caddy (ADR-0034). No Redis, zero budget. No users or accounts yet.

## Options considered

- **Rate limit in Caddy.** ✅ Before the app. ❌ Standard Caddy has no rate-limit module (it needs a custom build), and it can't tell expensive from cheap requests by cost.
- **In-app per-client token buckets on the expensive endpoints.** ✅ No new service; limits follow the cost of each endpoint; returns `429` with `Retry-After`. ❌ The state lives in one process (fine for one process, needs Redis to scale out). The client IP comes from `X-Forwarded-For`, which must only be trusted from the proxy.
- **Accounts and API keys.** ✅ Precise. ❌ No users yet; friction for a public demo.

## Decision

1. **In-app token buckets per client IP** (`xcrs/api/limits.py`), configured with `XCRS_RATE_*`:
   - **"heavy"** (LLM- or embedding-backed: `POST /api/v1/recommendations`, `POST /api/v2/recommendations`, `POST /api/v2/skills/match`): 10 a minute, bursts of 5.
   - **"light"** (embedding-only suggestions, `POST /api/v1/knowledge-units/related`): 60 a minute.
   - Reads (GET) are not limited.
2. **The client IP** is the last address in `X-Forwarded-For`, added by Caddy, and only when the request comes from a trusted proxy (`XCRS_TRUSTED_PROXIES`, by default Docker's private ranges and localhost). Otherwise it is the peer address.
3. **At most `XCRS_MATCH_LLM_MAX_NEW` (12) new phrases per request go to the LLM.** The rest use the embedding fallback (ADR-0030), so one request can't queue minutes of LLM work. Cached phrases don't count.
4. **Caddy limits request bodies to 64 KB.**

## Trade-offs accepted

- **Limits are per process and reset on restart;** acceptable for one API process. Several processes or VMs need a shared store (Redis), and the limiter's interface keeps that a one-file change.
- **IP-based limits** treat users behind one NAT as one client, and don't stop a distributed attacker. A WAF or accounts come later if abuse appears.
- **Phrases past the cap match with lower quality** (embedding fallback, 74%).

## Revisit when

- More than one API process, or real users → Redis-backed limits, maybe per-account quotas.
- `429`s show up for legitimate use in the logs → raise the limits.
