# Explainer benchmark

- Date: 2026-10-02T13:52:47.043136+00:00; host: xcrs-b (VM-B: 24 vCPU Xeon E5-2690 v4, 30 GiB; Ollama reached through an SSH tunnel from the Mac); CPU threads: 24; prompt: explain-role-v2.1
- Inputs: engine v2 facts for `eval/learner_profiles.yaml`; the app's v2 prompt and schema.
- Clean = no automatic grounding flag (heuristics that mark explanations to read, not proof).
- Prefill speed is not reported: Ollama's prompt cache makes its prompt timings unreliable.

| Model / device | Cases | Errors | Median s/role | p90 s | Prompt tok | Output tok | Decode tok/s | Clean % | Flags | Model load s |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5:4b / cpu | 20 | 0 | 12.2 | 13.1 | 671 | 92 | 11.7 | 100 | - | 0.0 |
| qwen3.5:9b / cpu | 20 | 0 | 20.2 | 89.7 (one case queued behind an app request) | 671 | 104 | 8.3 | 100 | - | 0.0 |


## Flagged explanations
