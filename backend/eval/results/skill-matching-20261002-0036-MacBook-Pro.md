# Skill matching: 2026-10-02T00:36 on MacBook-Pro.local

349 cases, 257 skills, `qwen3-embedding:0.6b`, LLM `qwen3.5:9b`. Held-out = thresholds chosen on half the cases (stratified by kind), measured on the other half; 5 x 2-fold. Accuracy = a case is fully right (every expected skill, nothing wrong).

| Pipeline | Held-out accuracy | Precision | Recall | Rejects non-skills | Floor | Window |
|---|---|---|---|---|---|---|
| lookup → LLM picks (pick-1, qwen3.5:9b) | **84.0%** ± 1.6% | 68.9% | 91.3% | 100% | 0.0 | 0.0 |
| lookup → LLM picks (pick-2, qwen3.5:9b), confirmed by similarity | **84.0%** ± 1.9% | 87.1% | 92.6% | 100% | 0.4 | 0.0 |
| LLM picks (pick-1, qwen3.5:9b) | **83.7%** ± 1.6% | 68.9% | 91.0% | 100% | 0.0 | 0.0 |
| lookup → LLM picks (pick-2, qwen3.5:9b) | **82.5%** ± 2.0% | 81.8% | 94.9% | 100% | 0.0 | 0.0 |
| lookup → LLM picks (pick-2, qwen3.5:4b), confirmed by similarity | **82.1%** ± 2.6% | 88.1% | 91.0% | 100% | 0.37 | 0.0 |
| LLM picks (pick-2, qwen3.5:9b) | **81.4%** ± 2.5% | 81.9% | 93.8% | 100% | 0.0 | 0.0 |
| lookup → LLM picks (pick-2, qwen3.5:4b) | **80.8%** ± 2.7% | 85.6% | 93.0% | 84% | 0.0 | 0.0 |
| LLM picks (pick-2, qwen3.5:4b) | **80.2%** ± 2.7% | 85.1% | 93.0% | 84% | 0.0 | 0.0 |
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
| lookup → LLM picks (pick-1, qwen3.5:9b) | 89% | 44% | 70% | 100% | 96% | 85% | 72% | 96% | 100% |
| lookup → LLM picks (pick-2, qwen3.5:9b), confirmed by similarity | 89% | 52% | 70% | 100% | 96% | 78% | 80% | 96% | 100% |
| LLM picks (pick-1, qwen3.5:9b) | 89% | 44% | 70% | 100% | 96% | 85% | 70% | 96% | 100% |
| lookup → LLM picks (pick-2, qwen3.5:9b) | 89% | 36% | 55% | 100% | 96% | 75% | 83% | 96% | 100% |
| lookup → LLM picks (pick-2, qwen3.5:4b), confirmed by similarity | 93% | 64% | 80% | 100% | 96% | 72% | 67% | 96% | 100% |
| LLM picks (pick-2, qwen3.5:9b) | 89% | 36% | 55% | 98% | 92% | 75% | 81% | 92% | 100% |
| lookup → LLM picks (pick-2, qwen3.5:4b) | 91% | 52% | 75% | 100% | 96% | 71% | 70% | 96% | 84% |
| LLM picks (pick-2, qwen3.5:4b) | 91% | 52% | 75% | 96% | 96% | 71% | 70% | 96% | 84% |
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
- embed_one_phrase_ms: median 10, p90 14, n 25
- define_qwen3.5:9b_gpu_ms: median 1601, p90 1815, n 349
- pick (pick-1, qwen3.5:9b) gpu_ms: median 1779, p90 2179, n 349
- pick (pick-2, qwen3.5:9b) gpu_ms: median 1823, p90 2242, n 349
- pick (pick-2, qwen3.5:4b) gpu_ms: median 1054, p90 1236, n 349
