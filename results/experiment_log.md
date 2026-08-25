# Experiment log

Every result file in `results/`, what produced it, and whether it is leakage-free.
Raw numbers only — no interpretation.

Last updated: 2026-08-25

---

## Data protocol (all current experiments)

| Split | Size | Source | Used for |
|---|---|---|---|
| train | 45,000 | CIFAR-10 train, seeded permutation | all weight updates |
| validation | 5,000 | CIFAR-10 train, held out | epoch selection, layer sensitivity, PPO reward |
| RL probe | 1,000 | first 1,000 of the validation split | PPO reward during training (fast subset) |
| test | 10,000 | CIFAR-10 test | final reported numbers only |

The PPO training environment is constructed with `test_loader=None`, so the test
set cannot influence training or model selection.

Sparsity is reported as a **percentage** (58.2), not a fraction (0.582). The
reward function consumes that value directly.

---

## Current results

| File | Produced by | Status |
|---|---|---|
| `final_results.csv` | `notebooks/Main code.ipynb`, cells 26–55 (executed 2026-08-25) | leakage-free |
| `ppo_multiseed.csv` | `evaluation/matched_sparsity_analysis.py`, extracted from Main notebook cells 51 and 63 | leakage-free |
| `matched_sparsity_comparison.csv` | `evaluation/matched_sparsity_analysis.py` | leakage-free |
| `matched_sparsity_methods.csv` | `evaluation/matched_sparsity_analysis.py` | leakage-free |
| `per_layer_allocation.csv` | `evaluation/matched_sparsity_analysis.py` | leakage-free |
| `predictions/test_predictions.csv` | `evaluation/matched_sparsity_analysis.py` | leakage-free |

### Superseded — do not cite

| File | Why |
|---|---|
| `final_results_finetune.csv` | RL reward, sensitivity probe and in-episode fine-tuning all used test images; epoch selection on test accuracy |
| `final_results_fair_finetune.csv` | same |
| `../checkpoints/manual_and_rl_df_current.csv` | pre-dates the split fix |
| README "Final Results" table | same leakage as the two CSVs above |

---

## Baseline

`checkpoints/cnn_baseline_FIXED.pth` — SimpleCNN, 10 epochs, Adam lr=1e-3, seed 42,
trained on the 45,000-image train split.

- validation accuracy 77.32%
- test accuracy 77.03%

Every number below is computed against this checkpoint. Retraining the baseline
invalidates all of them, including the McNemar statistics.

Prunable weights: 619,872 total.

| Layer | Params | Share |
|---|---|---|
| `features.0` | 864 | 0.14% |
| `features.3` | 18,432 | 2.97% |
| `features.6` | 73,728 | 11.89% |
| `classifier.1` | 524,288 | **84.58%** |
| `classifier.3` | 2,560 | 0.41% (outside the action space) |

Total sparsity is dominated by `classifier.1`; report per-layer ratios alongside it.

---

## PPO runs (4 seeds)

Source: `notebooks/Main code.ipynb` cell 51 (seed 42) and cell 63 (seeds 1–3).

| Seed | Policy | Per-layer % | Sparsity % | Test acc % |
|---|---|---|---|---|
| 42 | `[1,1,5,5]` | 10/10/60/60 | 58.20 | 76.49 |
| 1 | `[0,0,5,5]` | 0/0/60/60 | 57.88 | 76.65 |
| 2 | `[0,5,5,5]` | 0/60/60/60 | 59.67 | 75.16 |
| 3 | `[0,5,5,5]` | 0/60/60/60 | 59.67 | 75.16 |

Mean 75.87 ± 0.82% (sample sd, n=4) at 58.9% mean sparsity.

Seeds 2 and 3 converged to the same policy, so their deterministic test accuracy
is identical. Reported as-is, not deduplicated.

---

## Matched-sparsity comparisons

Each baseline is calibrated to the *same total sparsity as that seed's policy*,
over the same four layers, so only the allocation differs. McNemar exact
two-sided test on the same 10,000 test images.

| Seed | Sparsity % | PPO % | Global magnitude % | Δ pp | χ² | p | Significant |
|---|---|---|---|---|---|---|---|
| 42 | 58.20 | 76.49 | 76.35 | +0.14 | 0.278 | 0.598 | no |
| 1 | 57.88 | 76.65 | 76.54 | +0.11 | 0.163 | 0.686 | no |
| 2 | 59.67 | 75.16 | 76.36 | −1.20 | 16.660 | 4.35e-05 | **yes (PPO worse)** |
| 3 | 59.67 | 75.16 | 76.36 | −1.20 | 16.660 | 4.35e-05 | **yes (PPO worse)** |

| Seed | Sparsity % | PPO % | Uniform per-layer % | Δ pp | χ² | p | Significant |
|---|---|---|---|---|---|---|---|
| 42 | 58.20 | 76.49 | 41.20 | +35.29 | 2742.19 | <1e-300 | yes |
| 1 | 57.88 | 76.65 | 40.69 | +35.96 | 2812.02 | <1e-300 | yes |
| 2 | 59.67 | 75.16 | 40.68 | +34.48 | 2602.24 | <1e-300 | yes |
| 3 | 59.67 | 75.16 | 40.68 | +34.48 | 2602.24 | <1e-300 | yes |

Per-sample predictions for the seed-42 models are in
`predictions/test_predictions.csv`, so these tests can be recomputed without
re-running any model.

---

## Reference baselines (own sparsity levels, not matched)

| Method | Sparsity % | Test acc % |
|---|---|---|
| Uniform per-layer 20% | 19.92 | 75.69 |
| Uniform per-layer 40% | 39.83 | 67.75 |
| Uniform per-layer 60% | 59.75 | 40.48 |
| Global magnitude 10% | 10.00 | 76.95 |
| Global magnitude 20% | 20.00 | 77.04 |
| Global magnitude 40% | 40.00 | 76.88 |
| Global magnitude 60% | 60.00 | 76.51 |
| Global magnitude 80% | 80.00 | 73.14 |

Uniform per-layer and global magnitude are **different methods** and are kept as
separate baselines. The global-magnitude sweep also prunes `classifier.3`, which
the action space excludes; the matched-sparsity global-magnitude runs above do not.

---

## Per-layer allocation at seed 42's sparsity (58.20%)

| Layer | Share of weights | PPO | Global magnitude | Uniform |
|---|---|---|---|---|
| `features.0` | 0.14% | 10.0% | 9.8% | 58.4% |
| `features.3` | 2.97% | 10.0% | 28.7% | 58.4% |
| `features.6` | 11.89% | 60.0% | 32.0% | 58.4% |
| `classifier.1` | 84.58% | 60.0% | 63.3% | 58.4% |

---

## Reward function (verified, unchanged)

```
R = 1.5 * (accuracy / baseline_accuracy) + 0.04 * sparsity_percent
      + 0.05 * (number of distinct actions)
```

Verified by reconstructing the recorded seed-42 reward exactly
(3.9173135875256566). Marginal rates near baseline: +1 pp accuracy = +0.0195
reward, +1 pp sparsity = +0.0400 reward.

The `0.05 × distinct actions` term differs between rows of a comparison table
(a uniform policy scores 1 distinct action, `[1,1,5,5]` scores 2), so the Reward
column is not comparable across methods and is not used as a quality metric.

---

## Pending

- [ ] Clean fine-tuning notebook (`notebooks/Final fine tuned.ipynb`) — running
- [ ] Sensitivity ablation: with / zeroed / shuffled, 4 seeds each
- [ ] Reward sparsity-coefficient sweep: 0.01 / 0.02 / 0.04 / 0.08
- [ ] Storage and latency measurement (`evaluation/storage_and_latency.py`, needs an idle machine)
- [ ] Consolidated final results table, once the above land

---

## Environment

Python 3.13.13 · torch 2.7.1+cpu · torchvision 0.22.1 · gymnasium 1.2.3 ·
stable-baselines3 2.8.0 · numpy 2.2.2 · pandas 2.2.3 · CPU only.

## Reproduction

```
python evaluation/matched_sparsity_analysis.py     # matched baselines + McNemar
python evaluation/storage_and_latency.py           # storage + latency (idle machine)
```

`matched_sparsity_analysis.py` asserts that it reproduces the notebook's recorded
seed-42 sparsity and accuracy exactly before computing anything new.
