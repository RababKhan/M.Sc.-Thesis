# Paper 3 — ResNet-8 proxy-threshold calibration report

Protocol `resnet8_threshold_calibration_protocol.md` and the 48 frozen policy IDs were committed in
`5b12780` before any calibration run. The rules were applied mechanically
(`resnet8_threshold_calibration_decision.json`). VAL-RL only; the test set was never read.

## Calibration

| Item | Value |
|---|---|
| Calibration policies | 48: 8 per q₁ bin, none reallocated (bins held 224, 80, 74, 55, 34 and 18 policies) |
| 5-epoch identity A₅ | 78.40 |
| Truly feasible (q₅ ≥ 0.98) | **14** (stop threshold 5: passed) |
| Spearman ρ(q₁, q₅) on the 48 | 0.975 |
| δ = q₅ − q₁ | mean +0.026, median +0.025, range −0.0005 to +0.068; positive in 47/48 |
| t (100% recall) | 0.9651 (the lowest q₁ of a truly feasible policy) |
| t (≥ 20 in the landscape) | 0.9790 (the 20th-largest landscape q₁) |
| t = min of both | 0.9651 |
| **τ_proxy = t − 0.005** | **0.9601** |
| At τ_proxy on the 48 | recall 1.00, precision 0.875, TP 14, FP 2, FN 0, TN 32 |

δ by q₁ bin: the proxy underestimates more for weaker policies.

| q₁ bin | n | Mean δ | Truly feasible | Best table speed-up |
|---|---|---|---|---|
| [0.85, 0.90) | 8 | 0.050 | 0 | 1.219 |
| [0.90, 0.92) | 8 | 0.034 | 0 | 1.165 |
| [0.92, 0.94) | 8 | 0.028 | 0 | 1.135 |
| [0.94, 0.96) | 8 | 0.024 | 0 | 1.092 |
| [0.96, 0.98) | 8 | 0.015 | 6 | 1.089 |
| [0.98, ∞) | 8 | 0.005 | 8 | 1.064 |

## Latency room at τ_proxy (full frozen 1-epoch landscape)

| Item | Value |
|---|---|
| Proxy-feasible policies | 52 (≥ 20: passed) |
| Latency | 0.7016 – 0.7702 ms |
| **Best table speed-up** | **1.098×** (≥ 1.10× required: **failed**); 0 policies at ≥ 1.10× |
| Parameter reduction | 0 – 28.9% |
| MAC reduction | 0 – 18.4% |
| Pareto policies (A₁, latency) | 13 |

The best speed-up among the 14 truly feasible calibration policies is 1.089× ([0, 3, 1, 1]: q₅ 0.985,
17% MAC reduction).

## Decision

- **Threshold calibration: GO.** τ_proxy = 0.9601.
- **Latency-room rule: NOT MET** (best proxy-feasible speed-up 1.098× < 1.10×).
- **PPO: NO-GO. STOP**, as pre-registered.

Neither of the following conditional steps was taken, because both required GO:
- the final structured sensitivity vector (`paper3_resnet8_sensitivity.csv`);
- the PPO pre-registration and pre-PPO freeze.

On this CPU (i7-1260P, batch 1, 1 thread), at a 2% validation-accuracy tolerance after the fixed
recovery, physically compacted structured pruning of CIFAR-10 ResNet-8 leaves less than about 10%
latency room. Speed-ups of 1.13–1.22× exist only among policies whose 5-epoch retention is below 0.98.
