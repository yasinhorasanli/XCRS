# Skill matching: 2026-10-02T00:12 on MacBook-Pro.local

349 cases, 257 skills, `qwen3-embedding:0.6b`, LLM `qwen3.5:9b`. Held-out = thresholds chosen on half the cases (stratified by kind), measured on the other half; 5 x 2-fold. Accuracy = a case is fully right (every expected skill, nothing wrong).

| Pipeline | Held-out accuracy | Precision | Recall | Rejects non-skills | Floor | Window |
|---|---|---|---|---|---|---|
| lookup → LLM picks from the list | **84.0%** ± 1.6% | 68.9% | 91.3% | 100% | 0.0 | 0.0 |
| LLM picks from the list | **83.7%** ± 1.6% | 68.9% | 91.0% | 100% | 0.0 | 0.0 |
| lookup → embed instruction + phrase | **74.2%** ± 2.6% | 85.0% | 74.5% | 88% | 0.44 | 0.0 |
| lookup → embed instruction + definition (gate) | **72.6%** ± 3.2% | 88.0% | 73.1% | 100% | 0.42 | 0.0 |
| lookup → embed definition (no gate) | **72.2%** ± 2.9% | 85.6% | 74.5% | 100% | 0.63 | 0.0 |
| lookup → embed definition (gate) | **69.5%** ± 3.0% | 85.8% | 71.1% | 96% | 0.55 | 0.0 |
| embed instruction + phrase | **65.6%** ± 2.6% | 80.4% | 66.2% | 88% | 0.44 | 0.0 |
| embed instruction + definition (gate) | **64.9%** ± 2.7% | 84.6% | 64.1% | 100% | 0.42 | 0.0 |
| embed phrase | **62.2%** ± 2.3% | 79.1% | 62.4% | 96% | 0.61 | 0.0 |
| embed LLM definition (gate) | **60.0%** ± 2.6% | 79.1% | 63.7% | 96% | 0.45 | 0.0 |

Accuracy by kind (thresholds tuned on all cases):

| Pipeline | alias | broad | course | exact | multi | paraphrase | tool | typo | unrelated |
|---|---|---|---|---|---|---|---|---|---|
| lookup → LLM picks from the list | 89% | 44% | 70% | 100% | 96% | 85% | 72% | 96% | 100% |
| LLM picks from the list | 89% | 44% | 70% | 100% | 96% | 85% | 70% | 96% | 100% |
| lookup → embed instruction + phrase | 80% | 44% | 55% | 100% | 72% | 78% | 61% | 88% | 88% |
| lookup → embed instruction + definition (gate) | 78% | 48% | 45% | 100% | 68% | 58% | 69% | 92% | 100% |
| lookup → embed definition (no gate) | 82% | 28% | 55% | 100% | 68% | 71% | 56% | 96% | 100% |
| lookup → embed definition (gate) | 84% | 32% | 60% | 100% | 64% | 51% | 58% | 96% | 96% |
| embed instruction + phrase | 76% | 44% | 50% | 90% | 8% | 78% | 61% | 68% | 88% |
| embed instruction + definition (gate) | 76% | 44% | 40% | 85% | 8% | 58% | 69% | 80% | 100% |
| embed phrase | 64% | 52% | 55% | 94% | 8% | 78% | 41% | 64% | 96% |
| embed LLM definition (gate) | 80% | 28% | 55% | 85% | 8% | 51% | 56% | 88% | 96% |

Latency:

- lookup_ms_per_phrase: 0.31
- embed_one_phrase_ms: median 12, p90 14, n 25
- define_qwen3.5:9b_gpu_ms: median 1601, p90 1815, n 349
- pick_qwen3.5:9b_gpu_ms: median 1779, p90 2179, n 349
- define_qwen3.5:4b_cpu_ms: median 3047, p90 3899, n 25
- pick_qwen3.5:4b_cpu_ms: median 5445, p90 5955, n 8
