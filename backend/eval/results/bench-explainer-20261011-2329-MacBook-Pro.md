# Explainer benchmark

- Date: 2026-10-11T23:29:09.391116+00:00; host: MacBook-Pro.local; CPU threads: 14; prompt: explain-role-v2.2
- Inputs: engine v2 facts for `eval/learner_profiles.yaml`; the app's v2 prompt and schema.
- Clean = no automatic grounding flag (heuristics that mark explanations to read, not proof).
- Prefill speed is not reported: Ollama's prompt cache makes its prompt timings unreliable.

| Model / device | Cases | Errors | Median s/role | p90 s | Prompt tok | Output tok | Decode tok/s | Clean % | Flags | Model load s |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5:4b / gpu | 53 | 0 | 2.1 | 2.5 | 542 | 70 | 58.6 | 100 | - | 2.4 |

## Request-path embedding while an explanation generates on the same device

| Model / device | Idle ms | During generation ms |
|---|---|---|

## Flagged explanations
