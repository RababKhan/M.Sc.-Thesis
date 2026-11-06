# Paper 3 — 1-epoch proxy validation report

Pre-registered in `paper3_proxy_protocol.md` (amendment 2). Validation pairs frozen before measurement (`paper3_proxy_validation_policies.json`). VAL-RL only; the test set is never read. A = VAL-RL accuracy after recovery; q = A / A(identity) at the same recovery length.

| | SimpleCNN | ResNet-8 |
|---|---|---|
| Pairs | 156 | 31 |
| **Spearman ρ(A₁, A₅)** (primary) | 0.844 | 0.980 |
| Pearson r | 0.973 | 0.990 |
| MAE of q | 0.0088 | 0.0679 |
| Bias A₁ − A₅ (pp) | -1.29 | -5.68 |
| Bias q₁ − q₅ | -0.0073 | -0.0679 |
| Share of pairs with A₁ < A₅ | 1.000 | 1.000 |
| Top-10 overlap (by A₅ vs A₁) | 3 | 10 |
| Pairs with q₅ ≥ 0.98 | 112 | 2 |
| τ₁ (Youden rule) | 0.9715 | — |
| Youden J | 0.864 | — |
| **Proxy decision (ρ ≥ 0.9)** | **NO-GO** | **GO** |

**SimpleCNN.** Confusion of q₁ ≥ τ₁ against q₅ ≥ 0.98: {'tp': 112, 'fp': 6, 'fn': 0, 'tn': 38}. At q₁ ≥ 0.98: {'tp': 95, 'fp': 4, 'fn': 17, 'tn': 40}. Spearman by 5-epoch source: {'pilot': {'n': 126, 'spearman': 0.771063555463545}, 'validation': {'n': 30, 'spearman': 0.967174814043581}}. Identity: A₁ = 79.88, A₅ = 80.60.

**ResNet-8.** Confusion of q₁ ≥ τ₁ against q₅ ≥ 0.98: τ₁ not set (fallback: PI). At q₁ ≥ 0.98: {'tp': 2, 'fp': 0, 'fn': 0, 'tn': 29}. Spearman by 5-epoch source: {'validation': {'n': 31, 'spearman': 0.9804396047590239}}. Identity: A₁ = 77.96, A₅ = 78.40.
