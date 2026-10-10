# Explainer benchmark

- Date: 2026-10-02T10:42:19.680577+00:00; host: MacBook-Pro.local; CPU threads: 14; prompt: explain-role-v2.1
- Inputs: engine v2 facts for `eval/learner_profiles.yaml`; the app's v2 prompt and schema.
- Clean = no automatic grounding flag (heuristics that mark explanations to read, not proof).
- Prefill speed is not reported: Ollama's prompt cache makes its prompt timings unreliable.

| Model / device | Cases | Errors | Median s/role | p90 s | Prompt tok | Output tok | Decode tok/s | Clean % | Flags | Model load s |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5:4b / gpu | 53 | 0 | 2.2 | 2.4 | 671 | 88 | 60.1 | 96 | invented_curiosity | 2.2 |

## Request-path embedding while an explanation generates on the same device

| Model / device | Idle ms | During generation ms |
|---|---|---|
| qwen3.5:4b / gpu | 74 | 112 |

## Flagged explanations

### qwen3.5:4b / gpu

- **sre-vs-platform / site-reliability-engineer**: `invented_curiosity`. This role fits because you enjoyed SRE practices, incident management, Prometheus, Grafana, and observability. While you are neutral about Kubernetes and Terraform, your strong interest in monitoring and reliability aligns well with the core responsibilities of keeping production systems stable.
- **security-vs-pentest / security-engineer**: `invented_curiosity`. This role fits you because your enjoyment of cloud security, SIEM monitoring, incident response, and vulnerability management aligns directly with the core duties of protecting applications and infrastructure. The recommendation assumes a mid-level starting point based on your existing interest in these specific security domains.
