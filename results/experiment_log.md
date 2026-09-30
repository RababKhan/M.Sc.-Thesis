# Experiment log

Every result file in `results/`, what produced it, and whether it is leakage-free.
Raw numbers only — no interpretation.

Last updated: 2026-09-29

---

## Data protocol (all current experiments)

| Split | Size | Source | Used for |
|---|---|---|---|
| train | 45,000 | CIFAR-10 train, seeded permutation | all weight updates |
| validation | 5,000 | CIFAR-10 train, held out | epoch selection, layer sensitivity, PPO reward |
| RL probe | 1,000 | first 1,000 of the validation split | PPO reward during training (fast subset) |
| test | 10,000 | CIFAR-10 test | final reported numbers only |

The PPO training environment is constructed with `test_loader=None`, so the test
set cannot influence training or model selection. In the scripted experiments
(`evaluation/ppo_experiments.py`) the test set is additionally locked while
`agent.learn()` runs: requesting it raises an error.

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
| `final_results_fair_finetune_holdout.csv` | `notebooks/Final fine tuned.ipynb`, full top-to-bottom run (executed 2026-09-26) | leakage-free; the clean fine-tuning results |
| `finetune_matched_global_magnitude.csv` | `evaluation/finetune_matched_baselines.py` (2026-09-27) | leakage-free |
| `runs/*.json`, `training_curves/*.csv` | `evaluation/ppo_experiments.py` (2026-09-26/27), one file per PPO run | leakage-free |
| `sensitivity_ablation.csv`, `_summary.csv`, `_stats.csv` | `evaluation/aggregate_experiments.py` from `runs/` | leakage-free |
| `reward_coefficient_sweep.csv`, `_summary.csv` | `evaluation/aggregate_experiments.py` from `runs/` | leakage-free |
| `latency_results.csv`, `latency_samples.csv` | `evaluation/latency_benchmark.py` (2026-09-27) | — |
| `accuracy_sparsity_tradeoff.csv` (fig. A) | `evaluation/aggregate_experiments.py` from `matched_sparsity_methods.csv` | leakage-free |
| `ppo_policy_heatmap.csv` (fig. B) | `evaluation/aggregate_experiments.py` from `ppo_multiseed.csv` | leakage-free |
| `sensitivity_ablation_plot.csv` (fig. C) | `evaluation/aggregate_experiments.py` | leakage-free |
| `reward_sweep_plot.csv` (fig. D) | `evaluation/aggregate_experiments.py` | leakage-free |
| `ppo_training_curves.csv`, `_aggregate.csv` (fig. E) | `evaluation/aggregate_experiments.py` from `training_curves/` | leakage-free |
| `sensitivity_feature_diagnostic.md` (+ `sensitivity_diagnostic_data.json`, `sensitivity_noise_folds.csv`, `sensitivity_*_original.csv`) | `evaluation/sensitivity_diagnostic.py` (2026-09-27) | validation only |
| `sensitivity_probe_curves.csv`, `refined_sensitivity_values.csv`, `refined_sensitivity_folds.csv` | `evaluation/refined_sensitivity.py probe` (registered 2026-09-27 01:24) | validation only |
| `refined_runs/*.json`, `refined_training_curves/*.csv` | `evaluation/refined_sensitivity.py run` (100 runs, 2026-09-27) | leakage-free |
| `refined_sensitivity_ablation.csv`, `_summary.csv`, `_policy_stats.csv`, `_statistics.csv`, `_mcnemar.csv`, `_state_usage.csv`, `_perturbation.csv`, `refined_fig_*.csv` | `evaluation/refined_aggregate.py` | leakage-free |
| `reward_landscape.csv`, `learned_policy_reward_rank.csv` | `evaluation/reward_landscape.py` (`--rank-learned` for the second) | validation only |
| `constrained_preregistration.md` | written before the constrained experiment (2026-09-27 15:27) | — |
| `constrained_sensitivity_curves.csv`, `sensitivity_action_masks.csv`, `constrained_sensitivity_state.csv`, `constrained_heuristic_policies.json` | `evaluation/constrained_masks.py` | validation only |
| `policy_landscape_validation.csv`, `sensitivity_predictive_validity.csv`, `sensitivity_predictive_validity.md` | `evaluation/policy_landscape.py`, `evaluation/predictive_validity.py` | validation only |
| `constrained_runs/*.json`, `constrained_training_curves/*.csv` | `evaluation/constrained_ppo.py run` (180 runs) | leakage-free |
| `constrained_ppo_all_runs.csv`, `_summary.csv`, `_policy_stats.csv`, `_statistics.csv`, `constrained_matched_sparsity.csv`, `constrained_policy_landscape.csv`, `constrained_predictions/`, `constrained_fig_A…E_*.csv` | `evaluation/constrained_aggregate.py` | leakage-free |
| `soft_prior_preregistration.md`, `soft_prior_implementation_checks.md` | pre-registration (commit `4a7a5a5`); `evaluation/soft_prior.py check` | — |
| `soft_prior_runs/*.json`, `soft_prior_training_curves_raw/*.csv` | `evaluation/soft_prior.py run` (240 runs) | leakage-free |
| `soft_prior_all_runs.csv`, `_summary.csv`, `_statistics.csv`, `_policy_stats.csv`, `_training_curves.csv`, `_sample_efficiency.csv`, `_matched_sparsity.csv`, `_policy_landscape.csv`, `_classification.json`, `soft_prior_predictions/`, `soft_prior_fig_A…E_*.csv` | `evaluation/soft_prior_aggregate.py` | leakage-free |
| `exploration_levelA_preregistration.md`, `exploration_levelA_implementation_checks.md` | pre-registration (commit `c3a9484`); `evaluation/exploration_confirm.py check` | — |
| `exploration_levelA_runs/*.json`, `exploration_levelA_training_curves_raw/*.csv` | `evaluation/exploration_confirm.py run` (120 runs, seeds 100–129) | leakage-free |
| `exploration_levelA_all_runs.csv`, `_trajectory.csv`, `_auc.csv`, `_safety.csv`, `_sample_efficiency.csv`, `_final_models.csv`, `_statistics.csv`, `_summary.csv`, `_classification.json`, `exploration_levelA_predictions/`, `exploration_levelA_fig_A…D_*.csv` | `evaluation/exploration_confirm_aggregate.py` | leakage-free |
| `archive_confirmatory_preregistration.md`, `archive_validation_split.npz`, `archive_sensitivity_vector.json` | pre-registration commit `1bda844`; `evaluation/archive_confirm.py prepare` | V_RL / V_SELECT only |
| `archive_implementation_checks.md` | `evaluation/archive_confirm.py check` | — |
| `archive_runs/<condition>/seed_*_{training,archive,selected,run}.*`, `archive_predictions/<condition>/seed_*.csv` | `evaluation/archive_confirm.py run` (160 runs, seeds 200–239) | leakage-free |
| `archive_all_runs.csv`, `_unique_policies.csv`, `_selection_scores.csv`, `_selected_models.csv`, `_matched_sparsity.csv`, `_statistics.csv`, `_mcnemar_summary.csv`, `_conversion_analysis.csv`, `_discovery_time.csv`, `_quality.csv`, `_global_magnitude.csv`, `_summary.csv`, `_classification.json`, `archive_fig_A…E_*.csv` | `evaluation/archive_aggregate.py` | leakage-free |

### Superseded — do not cite

| File | Why |
|---|---|
| `final_results_finetune.csv` (removed from `results/`; in git history, commit `2f37f37`) | RL reward, sensitivity probe and in-episode fine-tuning all used test images; epoch selection on test accuracy |
| `final_results_fair_finetune.csv` (removed from `results/`; in git history, commit `2f37f37`) | same |
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

## Storage and inference cost

Source: `evaluation/storage_and_latency.py`, run on an idle machine, seed-42
policy `[1,1,5,5]` at 58.20% sparsity (259,133 of 619,872 weights non-zero,
41.8% density).

### Storage

| Representation | Size | vs its dense counterpart |
|---|---|---|
| dense `.pth` | 2427.1 KB | — |
| pruned `.pth` | 2427.1 KB | **0.0%** |
| dense `.pth`, gzipped | 2253.4 KB | — |
| pruned `.pth`, gzipped | 1142.2 KB | **−49.3%** |
| CSR (value + int32 index), theoretical | 2024.5 KB | −16.4% vs dense float32 |
| bitmask + values, theoretical | 1087.9 KB | −55.1% vs dense float32 |

The raw checkpoint is byte-for-byte identical in size: unstructured pruning
stores the zeros. The reduction is realised only by a representation that
exploits them, and naive CSR gives most of it back because an int32 index costs
as much as the float value it points at. Compressed-vs-compressed (−49.3%) is
the like-for-like figure; comparing a gzipped pruned checkpoint against an
uncompressed dense one overstates the saving as −52.9%.

### Inference latency

> Earlier, lighter protocol, on the machine that ran Main (hardware not recorded).
> Superseded for latency by "Latency (controlled protocol)" below, which adds
> `inference_mode`, a matched global-magnitude model, 600 timed passes,
> interleaved order and confidence intervals. The storage figures above stand.

Batch 128, 4 threads, 20 warmup iterations discarded, 100 timed.

| Model | Median | Mean | Min | SD |
|---|---|---|---|---|
| dense | 75.89 ms | 78.77 | 68.34 | 23.53 |
| pruned (58.2% zeros) | 75.27 ms | 77.97 | 68.64 | 8.27 |

Difference in medians: −0.62 ms (−0.8%), far smaller than the run-to-run spread
(SD 23.53 ms on the dense model). **No measurable latency difference.** Dense
kernels perform the same multiply-accumulates whether or not the operands are
zero.

### Does a sparse kernel help at this density?

`classifier.1` (524,288 weights, 40.0% density), batch 128:

| Kernel | Median |
|---|---|
| dense matmul | 1.331 ms |
| `torch.sparse.mm` (CSR) | 3.093 ms (**2.32× slower**) |

At this density the sparse kernel is slower than the dense one, so the sparsity
does not convert into compute savings even with sparse support available.

---

## Fine-tuning arm (1 PPO seed)

Source: `notebooks/Final fine tuned.ipynb`, executed top to bottom on 2026-09-26
(34 code cells, in order, no errors, 2075 s). Loads `cnn_baseline_FIXED.pth`
and asserts val 77.32% / test 77.03% before anything else; both reproduced.
Layer sensitivities reproduce Main's exactly (probe accuracy 76.1; drops
0.6 / 0.2 / 0.1 / 0.0).

Every pruned model is fine-tuned for 3 epochs (Adam, lr 1e-4) on the 45,000-image
train split, epoch selected on the 5,000-image validation split, then measured
once on test. PPO trained for 300 timesteps (seed 42) with 1-epoch in-episode
fine-tuning on a 5,000-image train subset and reward on the 1,000-image RL probe.

| Method | Actions | Val % | Test % | Sparsity % |
|---|---|---|---|---|
| Baseline | - | 77.32 | 77.03 | 0.00 |
| Baseline + fine-tune | - | 80.02 | 79.91 | 0.00 |
| FT uniform 10% | `[1,1,1,1]` | 80.08 | 79.58 | 9.96 |
| FT uniform 20% | `[2,2,2,2]` | 80.12 | 79.75 | 19.92 |
| FT uniform 40% | `[4,4,4,4]` | 79.44 | 79.60 | 39.83 |
| FT uniform 60% | `[5,5,5,5]` | 77.78 | 78.62 | 59.75 |
| PPO + fine-tune | `[4,4,4,4]` | 79.68 | 79.65 | 39.83 |

The learned policy `[4,4,4,4]` is identical to FT uniform 40%. The two rows
apply the same pruning and differ only in fine-tuning randomness, so their
0.05 pp test gap is run-to-run noise, not a method difference. This run shows
no layer-wise adaptation. The notebook's own guard (cell 50) prints the same
warning.

This notebook defines its own reward function (cell 31); its Reward column is
not comparable with Main's and is omitted here.

`checkpoints/baseline_finetuned.pth` and `checkpoints/ppo_finetune_agent_holdout.zip`
were written by this run.

### Matched global magnitude + fine-tune (added 2026-09-27)

Source: `evaluation/finetune_matched_baselines.py` → `finetune_matched_global_magnitude.csv`.
Global magnitude over the four agent-controlled layers, calibrated to the PPO +
fine-tune sparsity (39.83%), then fine-tuned with the notebook's protocol above
(masks kept during fine-tuning; sparsity unchanged afterwards, asserted).
Single run; 206 s.

| Method | Per-layer sparsity (f.0 / f.3 / f.6 / c.1) | Val % | Test % | Sparsity % |
|---|---|---|---|---|
| FT global magnitude @ 39.83% | 6.0 / 17.6 / 19.8 / 43.7 | 80.22 | 80.09 | 39.83 |
| PPO + fine-tune (from the table above) | 40 / 40 / 40 / 40 | 79.68 | 79.65 | 39.83 |

Validation accuracy by epoch (pre-fine-tune, 1, 2, 3): 77.24, 79.56, 79.68, 80.22.
Every row in the fine-tuning arm is one run with fine-tuning randomness, and
the notebook did not save per-sample predictions, so no significance test is
available for this comparison.

---

## PPO experiment harness (sensitivity ablation and reward sweep)

Script: `evaluation/ppo_experiments.py` with `evaluation/rl_env.py` (a cell-by-cell
copy of Main's environment, state, action space and reward, with the sensitivity
vector and λ_s as parameters). Aggregation: `evaluation/aggregate_experiments.py`.

| Item | Value |
|---|---|
| Baseline | `checkpoints/cnn_baseline_FIXED.pth`, sha256 `ca28f8345c23365ee7b92686e780ae0a9befa76c9d6bb9b3bb0b9e42272e111c` (val 77.32%, test 77.03%, probe 76.10%, asserted) |
| Split | seed 42; 45,000 / 5,000 / 1,000 probe / 10,000 test; val indices sha256 `9d3648af…f02fd7` |
| Measured sensitivity | features.0 0.6, features.3 0.2, features.6 0.1, classifier.1 0.0 (identical to Main) |
| PPO | MlpPolicy, lr 3e-4, n_steps 64, batch 32, n_epochs 10, γ 0.99, ent_coef 0.01 |
| Budget | 2,000 timesteps requested → 2,048 trained (32 rollouts × 64), 512 episodes, every run |
| Seeds | 42, 1, 2, 3 for every cell; none dropped |
| Reward during training | probe accuracy / baseline **full-validation** accuracy (77.32), exactly as in Main |
| Reported policy | final deterministic policy after training (as Main); the highest-reward training episode is recorded separately and never test-evaluated |
| Test set | measured once per run, after training; locked during `agent.learn()` |
| Code | git `c220caf` plus uncommitted scripts; sha256 `ppo_experiments.py` `1649aab5…`, `rl_env.py` `3c92b735…`, `common.py` `0afa8e51…` (full hashes in each `runs/*.json`) |
| Hardware / software | Intel Core i5-6400 @ 2.70 GHz, 4 threads, CPU only; Windows 11 (build 22621); Python 3.11.9, torch 2.7.1+cpu, SB3 2.8.0, gymnasium 1.2.3, numpy 2.2.2 |
| Runs | 28, from 2026-09-26 22:02 to 2026-09-27 00:13 (+06:00); 2.24 h total |

**Regression check.** The four "correct sensitivity, λ_s = 0.04" runs are the
same configuration as Main's recorded runs. The harness requires them to match
before it continues; all four reproduced Main's policy, sparsity and test
accuracy exactly, and seed 42's reported reward matched bit for bit
(3.9173135875256566). These four runs serve as the "correct" condition of the
ablation and the λ_s = 0.04 cell of the sweep; they were not run twice.

### Sensitivity ablation — ORIGINAL (λ_s = 0.04, original feature)

Preserved unchanged. The refined analysis below is a separate experiment and
does not replace it.

Conditions change only the value in the state's sensitivity slot (dimension 7
unchanged). Shuffled uses a seed-deterministic derangement, so every layer gets
another layer's value; the mapping for each seed is in `sensitivity_ablation.csv`
(seeds 42 and 3 happened to draw the same derangement).

| Condition | Seed 42 | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|---|
| correct | `[1,1,5,5]` 76.49 | `[0,0,5,5]` 76.65 | `[0,5,5,5]` 75.16 | `[0,5,5,5]` 75.16 |
| zeroed | `[1,1,5,5]` 76.49 | `[0,0,5,5]` 76.65 | `[0,0,5,5]` 76.65 | `[0,0,5,5]` 76.65 |
| shuffled | `[1,1,5,5]` 76.49 | `[0,0,5,5]` 76.65 | `[0,1,5,5]` 76.56 | `[0,0,5,5]` 76.65 |

(policy, test accuracy %)

| Condition | Test acc % (mean ± SD) | Val acc % | Sparsity % (mean ± SD) | Retention % | Distinct policies |
|---|---|---|---|---|---|
| correct | 75.87 ± 0.82 | 75.53 | 58.85 ± 0.95 | 98.49 | 3 |
| zeroed | 76.61 ± 0.08 | 76.53 | 57.96 ± 0.16 | 99.45 | 2 |
| shuffled | 76.59 ± 0.08 | 76.50 | 58.04 ± 0.18 | 99.43 | 3 |

Paired comparisons (`sensitivity_ablation_stats.csv`), correct minus other, per seed:

| Comparison | Identical policy | Δ test acc (pp) per seed 42/1/2/3 | Mean Δ acc | Mean Δ sparsity | Sign-flip p (acc) |
|---|---|---|---|---|---|
| correct vs zeroed | 2 of 4 seeds | 0 / 0 / −1.49 / −1.49 | −0.745 | +0.89 | 0.50 |
| correct vs shuffled | 2 of 4 seeds | 0 / 0 / −1.40 / −1.49 | −0.723 | +0.82 | 0.50 |

The sign-flip test is exact over the 16 sign patterns; with n = 4 its smallest
attainable two-sided p is 0.125. Per-seed McNemar tests where the policies
differ (seeds 2 and 3) give p = 4.6e-9 (vs zeroed) and 4.0e-8 / 4.6e-9 (vs
shuffled). These compare the two specific pruned models, which also differ in
sparsity by 1.5–1.8 pp, not the effect of the feature in general.

### Reward sparsity-coefficient sweep (correct sensitivity)

λ_s multiplies sparsity in **percent** (0–100), as in Main. No normalised-sparsity
variant was run.

| λ_s | Test acc % (mean ± SD) | Val acc % | Sparsity % (mean ± SD) | Retention % | Policies (frequency) |
|---|---|---|---|---|---|
| 0.00 | 77.05 ± 0.04 | 77.35 | 8.76 ± 17.51 | 100.03 | `[0,0,0,0]` ×3, `[0,0,1,4]` ×1 |
| 0.01 | 76.61 ± 0.08 | 76.53 | 57.96 ± 0.16 | 99.45 | `[0,0,5,5]` ×3, `[1,1,5,5]` ×1 |
| 0.02 | 76.59 ± 0.08 | 76.50 | 58.04 ± 0.18 | 99.43 | `[0,0,5,5]` ×2, `[1,1,5,5]` ×1, `[0,1,5,5]` ×1 |
| 0.04 | 75.87 ± 0.82 | 75.53 | 58.85 ± 0.95 | 98.49 | `[0,5,5,5]` ×2, `[1,1,5,5]` ×1, `[0,0,5,5]` ×1 |
| 0.08 | 74.87 ± 0.20 | 74.46 | 59.68 ± 0.01 | 97.19 | `[1,5,5,5]` ×3, `[0,5,5,5]` ×1 |

The largest action is 60%, so the highest reachable total sparsity over the four
layers is 59.75%. Per-run reward components are in `reward_coefficient_sweep.csv`.

### Highest-reward training episode vs final policy

In all 28 runs the final deterministic policy differs from the highest-reward
configuration sampled during training. At λ_s = 0.04 and 0.08 that
configuration was `[0,4,5,5]` in every run (seed 42, λ_s 0.04: reward 3.970 vs
3.885 for the final policy, both on the validation probe); at 0.01 and 0.02 it
was `[0,2,4,5]`. Recorded in `best_episode_actions`; not test-evaluated.

---

## Refined sensitivity analysis

A separate experiment; the original ablation above is unchanged. Diagnosis
first (`sensitivity_feature_diagnostic.md`), then a multi-ratio estimator,
then a paired ablation with 10 seeds.

### Design

| Item | Value |
|---|---|
| Scripts | `evaluation/refined_sensitivity.py` (probe, verify, run), `evaluation/refined_aggregate.py`, `evaluation/policy_probe.py`, `evaluation/reward_landscape.py` |
| Estimator | each layer alone L1-pruned at 10/20/40/60% from a fresh baseline copy; full 5,000-image validation split; accuracy and cross-entropy loss |
| Definition A (accuracy) | mean over ratios of (acc₀ − acc_r)/acc₀ |
| Definition B (loss) | mean over ratios of (loss_r − loss₀)/loss₀ |
| Normalisation | min-max across the four layers to [0, 1] (all equal → 0); the normalised value enters the state |
| Registered | `refined_sensitivity_values.csv`, 2026-09-27 01:24 (+06:00), before any refined PPO run; its sha256 is recorded in every refined run and was unchanged at the end |
| Conditions | correct (normalised vector); zeroed (0.0, same state dimension); shuffled (seed-deterministic derangement in which every layer also receives a different value; mapping recorded per run) |
| λ_s | 0.01, 0.02 (accuracy coef 1.5, diversity 0.05 unchanged) |
| Seeds | 42, 1, 2, 3, 4, 5, 6, 7, 8, 9 for every cell; none dropped |
| Runs | 100 = 2 λ_s × 10 seeds × (2 definitions × {correct, shuffled} + 1 zeroed). The zeroed state does not depend on the definition, so one zeroed run per (λ_s, seed) serves both |
| Everything else | identical to the original harness: baseline sha256 `ca28f834…`, split, layers, action space, PPO settings, 2,048 timesteps, reward baseline 77.32 on the probe, final deterministic policy, test once after training |
| Execution | 4 parallel workers × 1 torch thread, i5-6400. Before use, 1- and 2-thread execution were each verified to reproduce original runs (seeds 42/1/2/3 and 42/1) exactly, **including all 512 training episodes**. An unexpected power loss (Kernel-Power 41, no bugcheck) some time after 02:02 interrupted the batch after 4 runs; no file was damaged (null-byte scan, JSON parse, 512-row curves) and the resumable runner restarted at 08:24 without re-running completed runs. Last run finished 2026-09-27 14:00 |
| Agent files | saved without a `.zip` extension because SB3 treats `.01_seed-…` in the name as a suffix; they load normally |

### Refined sensitivity values

| Layer | Val acc at 10 / 20 / 40 / 60% | Accuracy raw → norm | Loss raw → norm | Original (pp) |
|---|---|---|---|---|
| features.0 | 77.28 / 76.58 / 68.94 / 42.72 | 0.1415 → 1.000 | 0.7839 → 1.000 | 0.6 |
| features.3 | 77.36 / 77.40 / 77.00 / 75.76 | 0.0057 → 0.051 | 0.0243 → 0.032 | 0.2 |
| features.6 | 77.26 / 77.32 / 77.20 / 76.68 | 0.0027 → 0.030 | 0.0094 → 0.013 | 0.1 |
| classifier.1 | 77.34 / 77.40 / 77.44 / 77.60 | −0.0016 → 0.000 | −0.0011 → 0.000 | 0.0 |

Fold stability (5 disjoint 1,000-image validation folds): loss definition,
1 distinct ranking; accuracy definition, 2; original, 4.

### Results (test accuracy %, 10 seeds)

| Def. | λ_s | Condition | Mean ± SD | Worst | Best | Sparsity % mean ± SD | Excess over matched GM (pp) | features.3 at 60% | Unique policies |
|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.01 | correct | 76.615 ± 0.074 | 76.46 | 76.65 | 57.92 ± 0.10 | +0.093 | 0 | 3 |
| accuracy | 0.01 | zeroed | 76.618 ± 0.067 | 76.49 | 76.65 | 57.95 ± 0.13 | +0.116 | 0 | 2 |
| accuracy | 0.01 | shuffled | 76.582 ± 0.345 | 75.66 | 76.91 | 57.00 ± 2.33 | +0.056 | 0 | 5 |
| accuracy | 0.02 | correct | 75.951 ± 1.186 | 73.22 | 76.65 | 58.43 ± 0.87 | −0.539 | 3 | 5 |
| accuracy | 0.02 | zeroed | 76.626 ± 0.053 | 76.50 | 76.65 | 57.98 ± 0.21 | +0.128 | 0 | 3 |
| accuracy | 0.02 | shuffled | 76.594 ± 0.148 | 76.18 | 76.65 | 58.03 ± 0.38 | +0.085 | 0 | 3 |
| loss | 0.01 | correct | 76.615 ± 0.074 | 76.46 | 76.65 | 57.92 ± 0.10 | +0.093 | 0 | 3 |
| loss | 0.01 | zeroed | 76.618 ± 0.067 | 76.49 | 76.65 | 57.95 ± 0.13 | +0.116 | 0 | 2 |
| loss | 0.01 | shuffled | 76.750 ± 0.131 | 76.65 | 76.96 | 55.27 ± 3.45 | +0.159 | 0 | 3 |
| loss | 0.02 | correct | 76.257 ± 0.702 | 74.77 | 76.65 | 58.39 ± 0.77 | −0.215 | 2 | 5 |
| loss | 0.02 | zeroed | 76.626 ± 0.053 | 76.50 | 76.65 | 57.98 ± 0.21 | +0.128 | 0 | 3 |
| loss | 0.02 | shuffled | 76.464 ± 0.462 | 75.16 | 76.65 | 58.21 ± 0.59 | −0.007 | 1 | 5 |

"Excess over matched GM" = run's test accuracy minus that of global magnitude
pruning over the same four layers at the run's exact total sparsity. Zeroed
rows are the same runs in both definitions. Accuracy/0.01/correct and
loss/0.01/correct have identical aggregates by coincidence: their per-seed
policies differ in two seeds (seeds 6 and 7 swap `[1,0,5,5]`).
features.0 was pruned ≤ 10% in 98 of 100 runs (exceptions, both seed 8: accuracy/0.01/shuffled `[2,0,5,5]` and accuracy/0.02/correct `[2,5,5,5]`).

### Paired statistics (correct minus other; `refined_sensitivity_statistics.csv`)

Primary tests (test accuracy, exact sign-flip, Holm over 8):

| Def. | λ_s | Comparison | Identical policies | Mean Δ | Median Δ | Bootstrap 95% CI | Sign-flip p | Holm p | d_z |
|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.01 | vs zeroed | 8/10 | −0.003 | 0 | [−0.057, 0.048] | 1.000 | 1.000 | −0.04 |
| accuracy | 0.01 | vs shuffled | 5/10 | +0.033 | 0 | [−0.136, 0.241] | 0.813 | 1.000 | 0.10 |
| accuracy | 0.02 | vs zeroed | 5/10 | −0.675 | 0 | [−1.404, −0.031] | 0.125 | 0.875 | −0.57 |
| accuracy | 0.02 | vs shuffled | 5/10 | −0.643 | 0 | [−1.425, −0.010] | 0.188 | 1.000 | −0.53 |
| loss | 0.01 | vs zeroed | 8/10 | −0.003 | 0 | [−0.057, 0.048] | 1.000 | 1.000 | −0.04 |
| loss | 0.01 | vs shuffled | 4/10 | −0.135 | −0.175 | [−0.207, −0.062] | 0.031 | 0.250 | −1.11 |
| loss | 0.02 | vs zeroed | 5/10 | −0.369 | 0 | [−0.829, −0.006] | 0.188 | 1.000 | −0.53 |
| loss | 0.02 | vs shuffled | 5/10 | −0.207 | 0 | [−0.593, 0.040] | 0.438 | 1.000 | −0.36 |

No primary comparison is significant after correction, and none has a
positive mean difference beyond +0.033 pp. Secondary, uncorrected results in
the same direction: accuracy definition pooled over λ_s, excess over matched GM,
correct vs zeroed −0.345 pp (sign-flip p = 0.016); loss definition pooled,
correct vs shuffled test accuracy −0.171 pp (p = 0.015) with +1.41 pp more
sparsity. Spread across seeds: at λ_s = 0.02 correct has SD 1.19 (accuracy
def.) and 0.70 (loss def.) vs zeroed 0.05; paired permutation p = 0.125 and 0.25.
Harmful policy (features.3 at 60%): correct 3/10 and 2/10 at λ_s = 0.02, zeroed
0/10 in every cell (exact McNemar over seeds p = 0.25 and 0.50).

### Policy use of the sensitivity input (`refined_sensitivity_state_usage.csv`)

Changing only the sensitivity input changes the chosen action in 14.2% of
cases for correct-condition agents (12.5% shuffled). Its permutation importance
is comparable to the parameter-count inputs and below layer index. The agents
use the input; it does not make them better.

### Reward landscape (`reward_landscape.csv`, `learned_policy_reward_rank.csv`)

All 1,296 policies scored with the training reward on the validation probe.

| λ_s | Reward-optimal | Median rank of learned policies (of 1,296) | Median rank of best sampled episode |
|---|---|---|---|
| 0.00 | `[0,1,3,4]` / `[0,2,4,5]` (tie) | 817 | 3 |
| 0.01 | `[0,2,4,5]` | 88 (identical for correct, zeroed, shuffled) | 1 |
| 0.02 | `[0,2,4,5]` | 58 (identical for correct, zeroed, shuffled) | 1 |
| 0.04 | `[0,4,5,5]` | 34.5 | 1 |
| 0.08 | `[0,4,5,5]` | 6 | 1 |

At λ_s = 0.02 the rewards of `[0,0,5,5]`, `[1,1,5,5]`, `[0,5,5,5]` and
`[1,5,5,5]` lie within 0.0022 of each other.

---

## Sensitivity-Aware Constrained PPO

A new experiment: layer sensitivity as a **constraint on the action space**
(real categorical action masking) instead of a state feature. The original and
refined sensitivity ablations above are unchanged.

### Provenance

| Item | Value |
|---|---|
| Pre-registration | `constrained_preregistration.md`, written and hashed (sha256 `386de2f0…`) at 2026-09-27 15:27, before curves, masks, validity analysis or any run; hash recorded in every run and unchanged at the end |
| Scripts | `evaluation/constrained_masks.py` (curves, masks, state, heuristic), `evaluation/policy_landscape.py` (1,296-policy validation landscape), `evaluation/predictive_validity.py`, `evaluation/constrained_ppo.py` (runner, sha256 `80e5fe5a…`), `evaluation/constrained_aggregate.py` |
| Code | git `c220caf` + uncommitted scripts (hashes in each `constrained_runs/*.json`) |
| Baseline | `cnn_baseline_FIXED.pth`, sha256 `ca28f834…`; val 77.32%, test 77.03% (asserted) |
| Split / folds | seed-42 split as before; validation folds = `val[0:1000]` … `val[4000:5000]` in the fixed order (fold 0 = RL probe) |
| Masks | `sensitivity_action_masks.csv` (sha256 `425deeae…`); state values `constrained_sensitivity_state.csv` (`9a1fd2b7…`) |
| Algorithm | `sb3_contrib.MaskablePPO` 2.8.0 for **every** PPO condition (unconstrained ones get all-valid masks). MlpPolicy, lr 3e-4, n_steps 64, batch 32, n_epochs 10, γ 0.99, GAE λ 0.95, clip 0.2, ent 0.01, vf 0.5, max_grad_norm 0.5; 2,048 timesteps (512 episodes) |
| Reward | 1.5 · acc_probe/77.32 + λ_s · S(%) + 0.05 · distinct actions; λ_s 0.01 (primary), 0.02 (robustness) |
| Seeds | 42, 1, …, 9 for every condition. The pre-registered rule chose 10 over 20: at a measured 274 s/run, 20 seeds project to ≥ 6.9 h at 4-way parallelism (> 6 h limit) |
| Runs | 180 = 2 λ_s × 10 seeds × 9 (C0 shared; C1–C4 per definition); C5 heuristic evaluated separately. 0 masked actions taken in 368,640 training steps (asserted per run) |
| Execution | 4 workers × 1 thread, i5-6400, Python 3.11.9, torch 2.7.1+cpu; 2026-09-27 16:32–18:12 |
| Test set | each run's frozen policy applied to a fresh baseline copy, evaluated once; per-example predictions saved (`constrained_predictions/seed_*.csv`, `heuristic.csv`) |

**Verification before the runs.** Terminal evaluations are memoised by policy.
A first memoised run did **not** reproduce the recorded original seed-42 run:
creating a PyTorch `DataLoader` iterator draws one int64 from the global torch
generator (`torch/utils/data/dataloader.py`, `_BaseDataLoaderIter.__init__`),
which PPO also samples from, so skipping evaluations shifted PPO's random
stream. After the cached path was made to create the iterator as well, (1)
PPO with the memoised environment and (2) **MaskablePPO with all-valid masks**
each reproduced the recorded original run exactly, all 512 training episodes
included. MaskablePPO with no restriction is therefore the same algorithm as
the PPO used in every earlier experiment.

### Masks (generated programmatically; 1.0 pp margin, one-sided 95% t bound over 5 folds)

| Layer | Accuracy rule (primary): UCB at 10/20/30/40/60% (pp) | Allowed | Loss rule: allowed |
|---|---|---|---|
| features.0 | 0.45 / **0.998** / 3.65 / 10.47 / 36.72 | 0–20% | 0–10% |
| features.3 | 0.31 / 0.22 / 0.59 / 0.65 / 2.52 | 0–40% | 0–40% |
| features.6 | 0.28 / 0.20 / 0.41 / 0.52 / 1.63 | 0–40% | 0–40% |
| classifier.1 | 0.12 / −0.00 / 0.18 / 0.21 / −0.05 | 0–60% | 0–60% |

Loss rule margin: 4.41% relative cross-entropy increase (= 1.0 pp of the
baseline's 22.68% validation error, applied to the other error measure).
Heuristic C5: accuracy `[2,4,4,5]`, loss `[1,4,4,5]`.

### Predictive validity (before PPO; `sensitivity_predictive_validity.md`)

Sensitivity predicts in-context pruning damage (in-context tolerated ratios
10/40/60/60%, same order as sensitivity; Pareto policies prune features.0 at
10% and classifier.1 at 60% on average). Confounded with layer size (Spearman
−1.0). The accuracy mask is too permissive for features.0 and both masks are
too strict for features.6 relative to in-context tolerance. The correct masks
admit the global reward optimum and 7 of 19 validation-Pareto policies;
shuffled masks admit none and cap sparsity at 37–42%.

### Results (test accuracy %, 10 seeds)

| Def. | λ_s | Condition | Mean ± SD | Median | Worst | Sparsity % | Excess over matched GM (pp) | Distinct policies | features.3 at 60% |
|---|---|---|---|---|---|---|---|---|---|
| — | 0.01 | C0 unconstrained | 76.618 ± 0.067 | 76.65 | 76.49 | 57.95 ± 0.13 | +0.116 | 2 | 0 |
| accuracy | 0.01 | C1 state-only | 76.674 ± 0.111 | 76.65 | 76.56 | 57.56 ± 1.14 | +0.150 | 3 | 0 |
| accuracy | 0.01 | **C2 correct mask** | 76.548 ± 0.057 | 76.53 | 76.53 | 56.55 ± 0.47 | −0.026 | 2 | 0 |
| accuracy | 0.01 | C3 shuffled mask | 76.419 ± 0.466 | 76.50 | 75.62 | 38.18 ± 1.75 | −0.544 | 6 | 2 |
| accuracy | 0.01 | C4 mask, zero state | 76.548 ± 0.057 | 76.53 | 76.53 | 56.55 ± 0.47 | −0.026 | 2 | 0 |
| accuracy | — | C5 heuristic `[2,4,4,5]` | 75.05 | | | 56.72 | −1.530 | | 0 |
| loss | 0.01 | C1 | 76.609 ± 0.069 | 76.65 | 76.49 | 57.98 ± 0.15 | +0.129 | 3 | 0 |
| loss | 0.01 | C2 | 76.603 ± 0.128 | 76.53 | 76.53 | 56.10 ± 1.04 | −0.003 | 3 | 0 |
| loss | 0.01 | C3 | 76.376 ± 0.551 | 76.49 | 75.63 | 37.13 ± 2.24 | −0.583 | 4 | 3 |
| loss | 0.01 | C4 | 76.582 ± 0.126 | 76.53 | 76.50 | 56.10 ± 1.04 | −0.021 | 4 | 0 |
| loss | — | C5 heuristic `[1,4,4,5]` | 76.43 | | | 56.71 | −0.140 | | 0 |
| — | 0.02 | C0 | 76.626 ± 0.053 | 76.65 | 76.50 | 57.98 ± 0.21 | +0.128 | 3 | 0 |
| accuracy | 0.02 | C1 | 75.978 ± 1.142 | 76.59 | 73.22 | 58.51 ± 0.85 | −0.503 | 5 | 3 |
| accuracy | 0.02 | C2 | 76.556 ± 0.087 | 76.53 | 76.43 | 56.40 ± 0.63 | −0.044 | 3 | 0 |
| accuracy | 0.02 | C3 | 76.105 ± 0.531 | 76.00 | 75.62 | 38.41 ± 1.56 | −0.842 | 5 | 5 |
| accuracy | 0.02 | C4 | 76.556 ± 0.087 | 76.53 | 76.43 | 56.40 ± 0.63 | −0.044 | 3 | 0 |
| loss | 0.02 | C1 | 76.391 ± 0.491 | 76.65 | 75.16 | 58.30 ± 0.69 | −0.113 | 4 | 1 |
| loss | 0.02 | C2 | 76.538 ± 0.068 | 76.53 | 76.43 | 56.55 ± 0.47 | −0.038 | 3 | 0 |
| loss | 0.02 | C3 | 75.797 ± 0.803 | 75.63 | 73.87 | 37.49 ± 1.95 | −1.173 | 5 | 6 |
| loss | 0.02 | C4 | 76.520 ± 0.032 | 76.53 | 76.43 | 56.70 ± 0.00 | −0.032 | 2 | 0 |

C2's modal policy is `[0,4,4,5]` (accuracy def.: 9/10 at λ_s 0.01); C0's is `[0,0,5,5]` (8/10).

### Primary family (accuracy definition, λ_s = 0.01, test accuracy, exact sign-flip, Holm over 3)

| Comparison | Identical policies | Mean Δ | Median Δ | Bootstrap 95% CI | p | Holm p | d_z |
|---|---|---|---|---|---|---|---|
| C2 − C0 | 0/10 | −0.070 | −0.12 | [−0.120, 0.014] | 0.094 | 0.281 | −0.62 |
| C2 − C3 | 0/10 | +0.129 | +0.03 | [−0.114, 0.418] | 0.418 | 0.836 | +0.29 |
| C2 − C4 | 10/10 | 0 | 0 | [0, 0] | 1.000 | 1.000 | — |

Criterion 3 (excess over matched global magnitude): C2 − C0 = −0.142 pp
(p = 0.002, 10/10 seeds negative); C2 − C3 = +0.518 pp (p = 0.002).
Sparsity: C2 − C0 = −1.40 pp (p = 0.002).

Validation accuracy tells a different story from test: C2 − C0 = +0.488 pp on
validation (p = 0.002) but −0.070 pp on test. For the modal policies,
`[0,4,4,5]` vs `[0,0,5,5]`: validation 77.06 vs 76.70, test 76.53 vs 76.65.

Robustness (not in the Holm family): at λ_s = 0.02, C2 − C3 test accuracy
+0.451 (accuracy def., p = 0.039) and +0.741 (loss def., p = 0.014); C2 − C0
−0.070 (p = 0.094) and −0.088 (loss, p = 0.008). C2 − C5 (heuristic): +1.50 pp
(accuracy), +0.17 pp (loss), p ≤ 0.004 at both λ_s.

### Convergence (validation reward landscape)

Median global reward rank of final policies (of 1,296): C0 88, C2 36 (λ_s 0.01);
C0 58, C2 25 (λ_s 0.02). C2 sampled its mask's reward-optimal policy
(`[0,2,4,5]`, the global optimum) in 10/10 runs and ended there in 0/10.

### Pre-registered classification: **Level C (no support)**

Criterion 1 fails (C2 is not better than unconstrained PPO; at matched
sparsity it is worse, −0.14 pp, p = 0.002). Criterion 2 fails on test accuracy
(Holm p = 0.84). Level B does not apply: relative to C0, correct masks do not
reduce harmful decisions (0 vs 0 features.3-at-60% runs) or variance
(SD 0.057 vs 0.067, p = 0.20).

---

## Soft Sensitivity-Prior PPO (final pre-registered sensitivity experiment)

Layer sensitivity as a **soft prior on the policy logits**:
adjusted_logit(ℓ, a) = raw_logit(ℓ, a) − β · S_ℓ · (ratio/60). No action is
masked. All earlier sensitivity experiments are unchanged.

### Provenance

| Item | Value |
|---|---|
| Pre-registration | `soft_prior_preregistration.md`, sha256 `22919379…`, written 2026-09-27 19:09:17 +06:00 and **committed as `4a7a5a5`** before any implementation or run; the file is unchanged relative to that commit and its hash is recorded in every run |
| Scripts | `evaluation/soft_prior.py` (SoftPriorPolicy, checks, runner; sha256 `85425e13…`), `evaluation/soft_prior_aggregate.py`; reuses the verified memoised environment in `evaluation/constrained_ppo.py` (unchanged, `80e5fe5a…`) |
| Code | git `4a7a5a5` + uncommitted scripts (hashes per run) |
| Baseline | `cnn_baseline_FIXED.pth`, sha256 `ca28f834…`; val 77.32%, test 77.03% (asserted) |
| Sensitivity | `refined_sensitivity_values.csv` (sha256 `3efd306f…`, registered 01:24): loss (primary) [1.000, 0.0323, 0.0134, 0.000], mean 0.2614; accuracy (secondary) [1.000, 0.0511, 0.0298, 0.000], mean 0.2702 |
| β | 0.50 primary, 1.00 robustness; no other values |
| Policy / PPO | SB3 2.8.0 PPO with `SoftPriorPolicy` = MlpPolicy + logit prior; [64, 64] tanh, lr 3e-4, n_steps 64, batch 32, n_epochs 10, γ 0.99, GAE λ 0.95, clip 0.2, ent 0.01, vf 0.5, max_grad_norm 0.5; 2,048 timesteps (512 episodes) |
| Reward | 1.5 · acc_probe/77.32 + 0.01 · S(%) + 0.05 · distinct actions (λ_s = 0.01) |
| Seeds | 42, 1, …, 19 (20) for all 12 cells. Pre-registered rule: 20 seeds unless > 6 h projected; a cold-cache measured run took ≤ 230 s, so the worst case was 3.8 h |
| Runs | 240 = 20 seeds × 12 (P0–P5 loss β 0.5; P1–P3 loss β 1.0; P1–P3 accuracy β 0.5); 2026-09-27 19:23–21:16, 4 workers × 1 thread, i5-6400, Python 3.11.9, torch 2.7.1+cpu |
| Implementation checks | `soft_prior_implementation_checks.md`: all pass. β = 0 reproduces standard PPO (all 512 episodes vs MlpPolicy, and vs the recorded original run); S = 0 gives bit-identical logits; every action keeps probability > 0; shuffled prior keeps the multiset with every layer's value changed; constant prior has the identical mean penalty; test set locked during training |
| Test set | frozen final policy → fresh baseline copy → test once; per-example predictions in `soft_prior_predictions/seed_*.csv` |

### Results (test accuracy %, 20 seeds)

| Cell | Mean ± SD | Worst | Sparsity % | Excess over matched GM (pp) | Unique policies | Most common (share) | f.3 = 60% | f.6 = 60% |
|---|---|---|---|---|---|---|---|---|
| P0 no prior | 76.572 ± 0.177 | 75.88 | 57.00 ± 4.33 | +0.061 | 3 | `[0,0,5,5]` (0.70) | 0 | 19 |
| **P1 correct, β 0.5** | 76.623 ± 0.114 | 76.49 | 56.70 ± 4.57 | +0.137 | 6 | `[0,0,5,5]` (0.50) | 0 | 18 |
| P2 shuffled, β 0.5 | 76.679 ± 0.351 | 75.35 | 50.82 ± 9.68 | +0.037 | 8 | `[0,0,5,5]` (0.45) | 0 | 12 |
| P3 constant, β 0.5 | 76.606 ± 0.328 | 75.35 | 54.53 ± 7.09 | +0.060 | 10 | `[0,0,5,5]` (0.40) | 0 | 15 |
| P4 state only | 76.524 ± 0.424 | 74.77 | 57.98 ± 0.72 | +0.041 | 7 | `[0,0,5,5]` (0.55) | 1 | 19 |
| P5 prior + state | 76.614 ± 0.148 | 76.18 | 57.57 ± 1.74 | +0.110 | 8 | `[0,0,5,5]` (0.55) | 0 | 18 |
| P1 correct, β 1.0 | 76.664 ± 0.098 | 76.56 | 57.39 ± 1.76 | +0.148 | 4 | `[0,0,5,5]` (0.75) | 0 | 18 |
| P2 shuffled, β 1.0 | 76.637 ± 0.383 | 75.62 | 48.35 ± 9.99 | −0.041 | 10 | `[0,0,5,5]` (0.30) | 0 | 9 |
| P3 constant, β 1.0 | 76.660 ± 0.155 | 76.49 | 53.55 ± 8.43 | +0.104 | 7 | `[0,0,5,5]` (0.45) | 0 | 15 |
| P1 accuracy def., β 0.5 | 76.644 ± 0.118 | 76.49 | 56.77 ± 4.42 | +0.126 | 5 | `[0,0,5,5]` (0.65) | 0 | 18 |
| P2 accuracy def., β 0.5 | 76.594 ± 0.410 | 75.35 | 52.00 ± 8.77 | −0.032 | 9 | `[0,0,5,5]` (0.40) | 0 | 13 |
| P3 accuracy def., β 0.5 | 76.518 ± 0.426 | 75.35 | 55.25 ± 6.60 | −0.012 | 8 | `[0,0,5,5]` (0.45) | 0 | 17 |

### Primary family (loss, β = 0.5; metric = excess over matched GM; sign-flip; Holm over 3)

| Comparison | Identical policies | Mean Δ | Median Δ | Bootstrap 95% CI | p | Holm p | d_z | Seeds +/− |
|---|---|---|---|---|---|---|---|---|
| P1 − P0 | 13/20 | +0.076 | 0 | [−0.001, 0.202] | 0.141 | 0.281 | 0.29 | 6 / 1 |
| P1 − P2 | 10/20 | +0.100 | 0 | [0.006, 0.238] | 0.053 | 0.158 | 0.35 | 9 / 1 |
| P1 − P3 | 9/20 | +0.077 | 0 | [−0.013, 0.210] | 0.210 | 0.281 | 0.28 | 7 / 4 |

None of the three criteria holds. Raw test accuracy: P1 − P0 +0.051 (p = 0.27),
P1 − P2 −0.057, P1 − P3 +0.017. Sparsity: P1 − P0 −0.30 pp; P2 and P3 prune
5.9 and 2.2 pp less than P1.

Robustness, not in the Holm family: β = 1.0, P1 − P0 +0.087 (p = 0.031;
raw +0.093, p = 0.016, 7 seeds +, 0 −), P1 − P2 +0.189 (p = 0.003),
P1 − P3 +0.044 (p = 0.10). Accuracy definition, β = 0.5: P1 − P0 +0.066
(p = 0.38), P1 − P2 +0.159 (p = 0.074), P1 − P3 +0.139 (p = 0.20).
P1 − P4 +0.096 (p = 0.33); P1 − P5 +0.027 (p = 0.38).

### Secondary family (P1 vs P0, loss, β = 0.5; Holm over 7)

| Outcome | P1 − P0 | p | Holm p |
|---|---|---|---|
| a. SD of test accuracy | −0.063 (0.114 vs 0.177) | 0.83 | 1.00 |
| b. harmful final policies | −1 (0 vs 1) | 1.00 | 1.00 |
| c. final global reward rank | −3.45 | 0.81 | 1.00 |
| d. time to validation target (≥ 76.0% val, ≥ 55% sparsity) | P1 earlier in 2 more seeds | 0.69 | 1.00 |
| **e. trajectory AUC, validation accuracy** | **+0.44 pp** | **< 0.0001** | **< 0.0001** |
| **f. trajectory AUC, reward rank** | **−39.6** | **0.0005** | **0.003** |
| **g. trajectory AUC, utility** | **+0.030** | **0.0005** | **0.003** |

On e–g, P1 is also better in mean than both P2 and P3. All runs reached the
validation target (P2: 19 of 20).

**Mechanism (descriptive).** The share of training episodes sampling
features.0 ≥ 40% is 20.7% (P0), 17.9% (P1), 21.7% (P2), 20.5% (P3), 16.7%
(P4) and 14.1% (P5). Excluding those samples, mean sampled validation accuracy
is 75.80% (P0) vs 75.87% (P1). The trajectory advantage is therefore mainly
fewer catastrophic features.0 samples during training. State-only sensitivity
(P4, AUC 73.05) and prior + state (P5, 73.54) reach a similar or larger
trajectory advantage than P1 (72.87).

### Pre-registered classification: **Level B (limited/partial effect)**

Computed mechanically (`soft_prior_classification.json`): no primary
criterion holds. Secondary outcomes e, f, g favour P1 over P0 after Holm, and
P1 ≥ P2 and P3 in mean on each. The effect is on the quality of the
policies *sampled during training* (safer exploration), not on the final
pruned models.

---

## Confirmatory study: sensitivity-guided exploration (fresh seeds)

Prospective confirmation of the soft-prior study's exploration finding.
All earlier experiments are unchanged.

### Provenance

| Item | Value |
|---|---|
| Pre-registration | `exploration_levelA_preregistration.md`, sha256 `af6881f7f090765ba070bfc31038493757671e0c73bba02aff7908d00313f777`, written 2026-09-28 09:11:47 +06:00, **committed `c3a9484`** before any check or run; unchanged relative to that commit; hash recorded in every run |
| Scripts | `evaluation/exploration_confirm.py` (runner, metrics, statistics, checks; sha256 `04cbee90…`), `evaluation/exploration_confirm_aggregate.py`; training through the unmodified `soft_prior.py` (`85425e13…`) and `constrained_ppo.py` (`80e5fe5a…`) |
| Baseline / split | `cnn_baseline_FIXED.pth` (`ca28f834…`), seed-42 split 45,000 / 5,000 / 10,000 |
| Sensitivity, prior | S = [1.0, 0.032278303448596, 0.0133578022446062, 0.0] (loss, `refined_sensitivity_values.csv`); adjusted_logit = raw_logit − 0.5 · S_ℓ · ratio/60 |
| Reward / PPO | λ_s 0.01; identical PPO settings to the soft-prior study; 2,048 timesteps, 512 episodes |
| Seeds | 100–129 (never used before; checked against every earlier run) |
| Conditions | E0 no prior, E1 correct, E2 shuffled (per-seed permutations listed in the pre-registration), E3 constant mean(S) = 0.26140902642330055 |
| Checkpoints / endpoints | end of every episode (timesteps 4…2048); the sampled policy's full-validation accuracy from `policy_landscape_validation.csv` (`a90f33c0…`); trapezoidal AUC normalised by duration |
| Implementation checks | `exploration_levelA_implementation_checks.md`: all 10 pass (β = 0 = standard PPO and = recorded P0; E1 = recorded P1, all 512 episodes; multiset, mean, non-zero probabilities, test lock, identical schedule, deterministic metrics, seed-level exact statistics validated against brute force) |
| Execution | 120 runs, 2026-09-28 09:42–11:10, 4 workers × 1 thread, i5-6400; no failures, no retries; no interim statistics (progress logs showed run counts only) |

### Results (30 seeds; mean ± SD)

| Condition | Val-accuracy AUC (primary) | Rank-score AUC | Utility AUC | Destructive f.0 freq. | Worst trajectory val acc | Final val acc | Final test acc | Final sparsity |
|---|---|---|---|---|---|---|---|---|
| E0 no prior | 72.437 ± 0.576 | 0.7098 ± 0.0558 | 1.8055 ± 0.0517 | 0.1984 ± 0.0398 | 40.669 ± 0.225 | 76.486 ± 0.572 | 76.536 ± 0.345 | 56.26 ± 4.99 |
| **E1 correct** | **72.944 ± 0.474** | **0.7329 ± 0.0509** | **1.8265 ± 0.0477** | **0.1732 ± 0.0321** | 40.607 ± 0.166 | 76.547 ± 0.366 | 76.587 ± 0.205 | 57.42 ± 1.81 |
| E2 shuffled | 72.498 ± 0.541 | 0.6942 ± 0.0635 | 1.7905 ± 0.0563 | 0.2020 ± 0.0387 | 40.761 ± 0.241 | 76.647 ± 0.491 | 76.666 ± 0.184 | 53.00 ± 9.26 |
| E3 constant | 72.582 ± 0.607 | 0.7017 ± 0.0569 | 1.7977 ± 0.0529 | 0.1969 ± 0.0394 | 40.751 ± 0.350 | 76.601 ± 0.492 | 76.605 ± 0.258 | 54.63 ± 6.97 |

### Primary endpoint: validation-accuracy AUC (exact sign-flip over 2³⁰ patterns; Holm over 3)

| Comparison | Mean Δ | Median Δ | SD Δ | 95% t CI | p | Holm p | d_z | Seeds +/− |
|---|---|---|---|---|---|---|---|---|
| E1 − E0 | +0.507 | +0.465 | 0.357 | [0.374, 0.640] | 1.2e-7 | 3.5e-7 | 1.42 | 29 / 1 |
| E1 − E2 | +0.446 | +0.489 | 0.423 | [0.288, 0.604] | 1.3e-5 | 1.6e-5 | 1.05 | 26 / 4 |
| E1 − E3 | +0.362 | +0.377 | 0.380 | [0.220, 0.504] | 8.1e-6 | 1.6e-5 | 0.95 | 25 / 5 |

### Secondary endpoints (E1 − comparator; Holm within each endpoint)

| Endpoint | vs E0 | vs E2 | vs E3 |
|---|---|---|---|
| S1 rank-score AUC | +0.023 [0.015, 0.031], Holm p 2.8e-6, d 1.08 | +0.039, 7.8e-8, d 1.20 | +0.031, 8.8e-7, d 1.03 |
| S2 utility AUC | +0.021 [0.014, 0.028], 1.4e-6, d 1.11 | +0.036, 2.8e-8, d 1.31 | +0.029, 2.2e-7, d 1.09 |
| S3 destructive f.0 frequency | −0.025 [−0.033, −0.018], 2.8e-7, d −1.22 | −0.029, 1.9e-6, d −1.17 | −0.024, 2.0e-7, d −1.19 |
| S4 worst trajectory val acc | −0.062 [−0.145, 0.021], 0.14 | −0.154, 0.001 | −0.143, 0.016 |
| S5 final policy val acc | +0.061, 1.00 | −0.099, 1.00 | −0.054, 1.00 |

S4 goes against E1 relative to E2 and E3 (by 0.14–0.15 pp near a floor of
about 40.6% set by catastrophic features.0 samples, which every condition
still draws). S5 shows no difference in final policy quality.

Descriptive: all 30 E0, E1 and E3 seeds and 29 E2 seeds reached the
validation target; paired time-to-target comparisons are not significant
(E1 earlier/later vs E0 5/2, p = 0.45). Final test models: McNemar vs
same-seed E0 significant in 1 of 30 E1 seeds; vs matched global magnitude,
1 of 30. Validation-Pareto final policies: E0 16, E1 20, E2 14, E3 17.

Safety (mean per run, of 512 episodes): destructive features.0 samples E0
101.6, E1 88.7, E2 103.4, E3 100.8. E1 samples features.0 = 0% in 120.4
episodes vs 99.1 for E0. Post-hoc description (not pre-registered): excluding
destructive samples, mean sampled validation accuracy is 75.68 (E0) vs 75.78
(E1).

### Pre-registered classification: **Level A — confirmed exploration benefit**

Computed mechanically (`exploration_levelA_classification.json`): all three
primary comparisons have a positive mean, Holm p < 0.05 and a 95% t CI
excluding 0; all three of S1, S2, S3 favour E1 over E0 with Holm p < 0.05
(two were required). The claim is about exploration during training. It does
not extend to final pruned-model accuracy (S5 and test accuracy show no
difference).

---

## Archive-confirmatory experiment: sensitivity-guided PPO with best-policy archive

Tests whether the confirmed exploration advantage converts into higher
**final** test accuracy via a pre-registered best-policy archive. All earlier
experiments are unchanged.

### Provenance

| Item | Value |
|---|---|
| Pre-registration | `archive_confirmatory_preregistration.md`, sha256 `bcbeb6d9047056b4d59e776210dffd0ba0b09b856ad231a1a5b40a4af3ed0ad0`, written 2026-09-28 11:40:06 +06:00, **committed `1bda844`** together with the split and sensitivity files it locks; all three unchanged at the end |
| Validation partition | `archive_validation_split.npz` (sha256 `55b7ff40…`): stratified, split seed 20260928; V_RL 3,000 (sensitivity, reward, training), V_SELECT 2,000 (post-training archive evaluation only) |
| Sensitivity | `archive_sensitivity_vector.json` (sha256 `de30b20c…`): refined loss algorithm on V_RL only; S = [1.0, 0.0316, 0.00935, 0.0] |
| Reward | 1.5 · acc_{V_RL}/76.9667 + 0.01 · S(%) + 0.05 · distinct actions |
| PPO / prior | as in the soft-prior and exploration studies; β 0.50; 2,048 timesteps |
| Seeds | 200–239 (40; verified unused) × A0–A3 = 160 runs; 2026-09-28 12:20–16:05; 4 workers × 1 thread, i5-6400; no failures |
| Code | git `1bda844` + scripts `evaluation/archive_confirm.py` (sha256 `37aefdb7…`), `archive_checks.py`, `archive_aggregate.py` |
| Implementation checks | `archive_implementation_checks.md`: all 15 pass (A0 = standard PPO over all 512 episodes; V_SELECT and TEST locked during training; TEST locked until the selection is frozen; archive completeness; no repeated evaluation; deterministic selection with synthetic tie cases; frozen-before-test verified by hash and timestamps) |
| Protocol | the selected and terminal policies are written and hashed before the test loader opens; each is evaluated once on test; predictions in `archive_predictions/` |

### Results (40 seeds; archive-selected model)

| | A0 no prior | **A1 correct** | A2 shuffled | A3 constant |
|---|---|---|---|---|
| Band reached | 38 | **40** | 34 | 37 |
| Band candidates per run (mean) | 9.0 | 11.0 | 7.0 | 9.0 |
| Best band V_SELECT accuracy (mean) | 76.34 | 77.59 | 76.14 | 76.49 |
| Selected test accuracy, mean ± SD | 74.42 ± 7.69 | **76.56 ± 0.02** | 73.00 ± 9.38 | 75.43 ± 5.32 |
| Worst seed | 40.68 | 76.56 | 40.68 | 43.53 |
| Selected sparsity, mean | 58.16 | 58.18 | 58.20 | 58.14 |
| Selected `[0,1,5,5]` | 30 | 38 | 25 | 33 |
| Runs with selected test < 75% | 5 | 0 | 9 | 3 |
| Terminal test accuracy (mean) | 76.50 | 76.65 | 76.39 | 76.42 |
| Terminal sparsity (mean) | 54.69 | 57.02 | 51.87 | 55.52 |

### Primary analysis (matched set = both reached the band; exact sign-flip; Holm over 3)

| Comparison | N | Mean Δ | Median Δ | 95% t CI | p | Holm p | d_z | Wins / ties / losses |
|---|---|---|---|---|---|---|---|---|
| A1 − A0 | 38 | +1.243 | 0 | [−0.582, 3.067] | 0.227 | 0.5625 | 0.22 | 4 / 30 / 4 |
| A1 − A2 | 34 | +1.392 | 0 | [−0.653, 3.437] | 0.205 | 0.5625 | 0.24 | 5 / 24 / 5 |
| A1 − A3 | 37 | +1.090 | 0 | [−0.757, 2.936] | 0.188 | 0.5625 | 0.20 | 4 / 32 / 1 |

Matched-sparsity validity holds for all three (mean |Δ| 0.045, 0.059,
0.033 pp; 97–100% of pairs within 0.5 pp); ≥ 30 matched seeds in each.
Criteria C1–C3 fail; C4 and C5 hold; no comparison is explained by pruning
less; B1–B4 all false.

Sensitivity analysis on all 40 seeds (including runs without band
candidates, not primary): A1 − A0 +2.148 (p = 0.057), A1 − A2 +3.559
(p = 0.003), A1 − A3 +1.138 (p = 0.023). Selected-policy V_SELECT accuracy,
matched set: A1 − A0 +1.242 (p = 0.039), A1 − A2 +1.443 (p = 0.014),
A1 − A3 +1.093 (p = 0.19).

McNemar per seed (selected models): significantly favouring A1 in 5 (vs A0),
9 (vs A2) and 5 (vs A3) seeds; significantly favouring the control in 0.
Global magnitude at the exact selected sparsity: A1 +0.24 pp on average,
no seed significant either way; A0 / A2 / A3 −1.93 / −3.35 / −0.92 pp, with
5 / 9 / 5 seeds significantly worse than global magnitude.

Secondary: archive gain (selected − terminal test accuracy) A0 −2.08,
A1 −0.09, A2 −3.39, A3 −1.00 (median −0.09 for all). A1 vs A2 gain p = 0.011,
others not significant. First episode of the selected policy (median): A0 352,
A1 320, A2 384, A3 346; no significant difference.

**Mechanism (descriptive).** Every control selection below 75% is either a
band policy with destructive features.0 pruning chosen because it was the
run's only band candidate (`[4,1,5,5]`, `[5,0,5,5]`, `[4,0,5,5]`), or a
fallback when no band policy was sampled (`[5,5,4,5]` 40.7%, `[3,4,5,5]`,
`[4,4,5,5]`, …). The rule maximises V_SELECT accuracy only *within* the band,
so it deploys a destructive policy when that is all the archive holds. A1's
archives always contained a safe band candidate.

### Pre-registered classification: **LEVEL C-FA — no confirmed final-accuracy benefit**

Computed mechanically (`archive_classification.json`): C1–C3 fail (Holm
p = 0.56 for each comparison; most matched seeds tie). The a-priori
structural note held: the band has 14 policies, and 30 of 38 matched A1/A0
pairs selected the identical model. The secondary evidence that A1 avoids
catastrophic deployments (0 vs 3–9 runs below 75%; SD 0.02 vs 5–9 pp) does
not change the classification.

---

## Phase 2 (CPU-only): consolidated library, three architectures, CIFAR-100, dense baselines

### Protocols

Written and committed in `366d21a`, before any Phase-2 benchmark, split or training run:

- `rocksolid_cpu_master_protocol.md`. It supersedes the hardware-dependent parts of
  `rocksolid_master_protocol.md`, which is unchanged.
- `cpu_runtime_budget.md`. Part A holds the rules; the Part B measurements were appended in
  `2055dac`.
- `cpu_dense_training_protocol.md`.
- `cpu_efficiency_protocol.md`.

The project is designed for commodity CPUs. No GPU run or GPU latency is part of it.

### Consolidated library (`src/`) and its verification

`src/` re-implements the shared pieces of the historical scripts once, for any registered
architecture:

- data, splits and guards;
- models;
- pruning units and one-shot baselines;
- prepared structured pruning;
- sensitivity;
- the PPO environment and soft prior;
- evaluation and efficiency;
- statistics.

The historical scripts are unchanged. `experiments/phase2/verify_library.py` →
`results/reproducibility/library_verification.json`: **all pass**.

| Check | Result |
|---|---|
| Splits, validation/probe/test tensors, V_RL/V_SELECT | bit-identical to `rl_env.Splits` / `archive_validation_split.npz`; val hash `9d3648af…` |
| Baseline | val 77.32 / probe 76.10 / test 77.03; 620,362 parameters |
| Sensitivity | accuracy drop identical; archive V_RL loss vector identical (4 threads) |
| Pruning | `[1,1,5,5]` masks identical, sparsity 58.19572427856073; 103 landscape policies (1 thread); 6 matched and 8 reference baselines with exact test accuracies |
| Complete PPO runs | seeds 42 (uncached), 1, 2, 3 (memoised) and soft-prior E1 seed 100: all 512 episodes, and the final policy, sparsity and test accuracy, bit-identical |
| Statistics | sign-flip, Wilcoxon and Holm identical to the historical code; exact t-quantiles |

The verification found two numerical facts. The first is now a rule in the CPU master
protocol §11.

- **Thread count changes float reduction order.** Loss values move at the 1e-8 level, and an
  argmax can flip on an exact logit tie.
  - Example: policy `[3,4,2,2]`, validation image 3771 scores 73.24% at 1 thread and 73.26% at 2
    or 4 threads. The historical and the new code behave identically.
  - Consequence: evaluations that are compared with each other use one fixed thread count. That
    count is 1, as in all historical landscape and PPO evaluations.
- **pandas misreads floats.** Its default CSV float parser misreads about 20% of 17-digit floats in
  the last bit. Recorded curves are therefore compared with `float_precision="round_trip"`.

### Architectures and pruning units

Each architecture has four units (6^4 = 1,296 policies), and the output classifier is excluded.
Details are in `architecture_generalization/cpu_prunable_units.csv`.

| Architecture | Units (weights) | Largest unit share |
|---|---|---|
| SimpleCNN | features.0 864 · features.3 18,432 · features.6 73,728 · classifier.1 524,288 | 84.6% |
| LeNet-5 (CIFAR; 5×5 convs 6/16, FC 400-120-84) | conv1 450 · conv2 2,400 · fc1 48,000 · fc2 10,080 | 77.7% |
| ResNet-8 (He et al. n = 1; 16/32/64) | stem 432 · stage1 4,608 · stage2 14,336 · stage3 57,344 | 74.1% |

The pre-registered fallback, SmallVGG, was not needed: every setting stayed within the time budget.

### One-epoch CPU benchmark (4 threads)

Results are in `architecture_generalization/cpu_epoch_benchmark.csv`. The benchmark recorded
training loss only, no accuracy.

| Setting | s / epoch | Projected run | Actual runs (3 seeds) |
|---|---|---|---|
| C10 LeNet-5 | 8.1 | 0.07 h | 4.2–4.9 min |
| C10 ResNet-8 | 77.1 | 0.68 h | 38.9–40.1 min |
| C100 SimpleCNN | 43.0 | 0.39 h | 27.2–29.9 min |
| C100 LeNet-5 | 6.8 | 0.06 h | 4.8–5.2 min |
| C100 ResNet-8 | 76.8 | 0.68 h | 38.5–38.6 min |

- **Stability exception.** It did not trigger: every epoch-1 training loss was below ln C. The
  learning rate is therefore 1e-3 everywhere.
- **Cost.** Dense training took 5.8 machine-hours in total, against a projection of 5.7 h. The
  budget is ≤ 2 h per run and ≤ 30 h for Phase 2.

### CIFAR-100 split

The split is in `reproducibility/cifar100_split_indices.npz` (sha256 `d7351925…`, `c156f62`),
with manifest `cifar100_split_manifest.json`.

- **Train / validation:** seed-42 permutation, 45,000 / 5,000. The index lists coincide with
  CIFAR-10's.
- **V_RL / V_SELECT:** stratified, 3,000 / 2,000, seed 20260928.
- **Images per class:** validation 37–65, V_RL 22–39, V_SELECT 15–26.
- **Normalisation:** the constants recomputed from the 50,000 training images round to the constants
  used.
- **Test set:** read only by frozen dense models.

### Dense baselines

The protocol:
- Adam at lr 1e-3, cosine schedule over 30 epochs, batch 128, crop + flip augmentation;
- the checkpoint is the best full-validation epoch;
- the test set is read once, after the checkpoint was written and hashed;
- seeds 0–2, with seed 0 the pruning reference by rule.

Files:
- per run: `architecture_generalization/cpu_baseline_runs.csv`;
- per setting: `cpu_baselines.csv`;
- run records: `architecture_generalization/dense_runs/`;
- checkpoints: `checkpoints/phase2_dense/`.

| Setting | Val mean ± SD | Test mean ± SD | Reference seed 0: val / test |
|---|---|---|---|
| C10 SimpleCNN | — | — | 77.32 / 77.03 (`cnn_baseline_FIXED.pth`, historical recipe) |
| C10 LeNet-5 | 65.63 ± 1.94 | 66.21 ± 1.51 | 64.32 / 64.96 |
| C10 ResNet-8 | 78.62 ± 0.23 | 78.69 ± 0.51 | 78.72 / 78.69 |
| C100 SimpleCNN | 48.78 ± 0.95 | 49.84 ± 0.65 | 49.08 / 50.50 |
| C100 LeNet-5 | 32.15 ± 0.42 | 31.42 ± 0.75 | 32.58 / 32.28 |
| C100 ResNet-8 | 43.07 ± 0.54 | 43.58 ± 0.53 | 43.48 / 43.99 |

The selected epochs are 24–30. The selected epoch's validation accuracy is at most 0.26 pp above
the last epoch's.

### Efficiency of the dense references (4 threads)

Results are in `architecture_generalization/cpu_efficiency_baselines.csv`, with the timings in
`cpu_efficiency_latency_samples.csv`. CPU load was 8.0% before the measurement and 4.1% after.
Each model and batch size got 500 timed passes.

| Setting | Params | MACs / image | Raw / gzip bytes | Latency b1 / b32 / b128 (ms, median) | Peak RAM b128 (MB) |
|---|---|---|---|---|---|
| C10 SimpleCNN | 620,362 | 10.85 M | 2,485,317 / 2,307,464 | 0.74 / 12.75 / 52.15 | 232 |
| C10 LeNet-5 | 62,006 | 0.65 M | 252,037 / 232,527 | 0.62 / 1.65 / 5.61 | 194 |
| C10 ResNet-8 | 78,042 | 12.50 M | 332,219 / 297,340 | 1.40 / 12.29 / 64.82 | 220 |
| C100 SimpleCNN | 643,492 | 10.87 M | 2,577,861 / 2,398,700 | 0.77 / 12.81 / 52.45 | 233 |
| C100 LeNet-5 | 69,656 | 0.66 M | 282,629 / 260,967 | 0.60 / 1.69 / 5.59 | 189 |
| C100 ResNet-8 | 83,892 | 12.51 M | 355,643 / 319,202 | 1.39 / 12.08 / 64.55 | 219 |

- **FLOPs.** torch FlopCounterMode FLOPs equal 2 × MACs in every case.
- **ptflops.** ptflops 0.7.5 MACs, which include BN and activations, are in the CSV.
- **Consistency.** The SimpleCNN numbers agree with the earlier controlled measurement:
  0.737 ms at batch 1 and 51.25 ms at batch 128.

### Implementation checks

`architecture_generalization/cpu_phase2_checks.md`: **69 passed, 0 failed, 0 skipped**. The checks
cover:
- architectures, and forward passes on both datasets;
- parameter formulas and unit mapping;
- the zero-pruning policy;
- checkpoint reload, including all 15 dense runs;
- deterministic splits and the data guards;
- the five one-shot baselines;
- structured channel removal;
- training resume and determinism;
- the PPO environment on all architectures;
- efficiency counters and sensitivity.

### Prepared but not run

Phase 2 stops here. The following are prepared but not run:
- multi-seed PPO on the new settings;
- the one-shot baseline comparisons;
- structured-pruning experiments;
- fine-tuning;
- robustness.

---

## Phase 3 (CPU-only): architecture and dataset generalisation of PPO and sensitivity-guided exploration

Pre-registration: `results/phase3_preregistration.md`, sha256 `064c6c05…`, committed in `2dbcc77`
before any landscape or PPO run.

- **Frozen inputs** (`phase3_frozen_inputs.json`, same commit): the six V_RL loss-sensitivity
  vectors (1 thread), the destructive-action rules, and the per-seed shuffled and constant priors.
- **Landscapes and implementation checks** (`a2cf5e5`): 57/57 checks passed on the
  non-experimental seed 42.
- **Runs:** 6 settings × 4 conditions (P0 plain; P1 correct prior β 0.5; P2 shuffled; P3 constant)
  × 20 fresh seeds 300–319, so **480 runs**. Every run:
  - uses the identical PPO budget (2,048 timesteps = 512 episodes) at 1 thread;
  - takes its reward from V_RL (λ_s 0.01) with an exact V_RL landscape cache;
  - reads the test set once per frozen final policy.
- **Integrity checks:** 22/22 (`phase3_integrity_checks.md`).

### Infrastructure events (`phase3_runs/infrastructure_log.txt`)

1. **A mains power cut** struck after 75 completed runs.
   - The integrity scan found every record verified, with no partial or corrupt file.
   - The four in-flight runs had written nothing and were repeated with the same seed and
     configuration.
2. **A worker crash** (`PermissionError`, a transient Windows lock) hit the shared
   test-prediction cache file when two workers produced the same final policy.
   - The affected run had written nothing and was repeated.
   - Fix `5633700`, infrastructure only:
     - every run computes its own test predictions;
     - the shared cache is written once, and an existing file must be identical;
     - writes are fsynced before the atomic rename.

### Frozen sensitivity (V_RL, normalised, unit order)

| Setting | S | Most sensitive (destructive = it at ≥ 40%) |
|---|---|---|
| C10 SimpleCNN | 1.000, 0.032, 0.009, 0.000 | features.0 |
| C10 LeNet-5 | 1.000, 0.422, 0.046, 0.000 | conv1 |
| C10 ResNet-8 | 1.000, 0.138, 0.331, 0.000 | stem |
| C100 SimpleCNN | 1.000, 0.103, 0.014, 0.000 | features.0 |
| C100 LeNet-5 | 1.000, 0.587, 0.000, 0.086 | conv1 |
| C100 ResNet-8 | 0.987, 0.310, 1.000, 0.000 | stage2 |

### Test accuracy at matched sparsity

Means over 20 seeds. Each baseline has exactly the zero count of each P0 run. `phase3_summary.csv`
holds the SDs and the rows matched to P1–P3.

| Setting | Dense | P0 | P1 | P0 sparsity | Uniform | Global | LAMP | ERK | Random |
|---|---|---|---|---|---|---|---|---|---|
| C10 SimpleCNN | 77.03 | 76.44 | 76.51 | 55.1% | 45.08 | 76.51 | 76.99 | 75.70 | 16.84 |
| C10 LeNet-5 | 64.96 | 61.67 | 62.17 | 51.5% | 45.38 | 58.94 | 63.31 | 59.74 | 12.99 |
| C10 ResNet-8 | 78.69 | 76.95 | 76.87 | 28.3% | 62.79 | 69.57 | 76.80 | 77.01 | 12.59 |
| C100 SimpleCNN | 50.50 | 49.66 | 49.48 | 53.3% | 19.96 | 49.67 | 50.30 | 45.31 | 2.37 |
| C100 LeNet-5 | 32.28 | 31.18 | 30.81 | 34.6% | 27.51 | 31.32 | 31.86 | 30.88 | 3.29 |
| C100 ResNet-8 | 43.99 | 43.01 | 43.13 | 9.5% | 42.78 | 42.74 | 43.76 | 43.81 | 4.25 |

### Pre-registered cross-setting classification

Mechanical, from `phase3_cross_setting_summary.csv` and `phase3_classification.json`. "Supported"
means the hypothesised direction holds, Holm p < 0.05 within the family, and the 95% t-CI excludes 0.

| Statement | Supported / opposite (of 6) | Class |
|---|---|---|
| 1 PPO > uniform | 5 / 0 (C100 ResNet-8: 19 of 20 ties, PPO chose uniform `[1,1,1,1]`) | **STRONG** |
| 2 PPO > global magnitude | 3 / 0 (C10 LeNet-5 +2.74, C10 ResNet-8 +7.38, C100 ResNet-8 +0.28 pp) | **MODERATE** |
| 3 PPO > LAMP | 0 / 5 (LAMP better by 0.55–1.64 pp; C10 ResNet-8 inconclusive) | **NOT SUPPORTED, REVERSED** |
| 4 PPO > ERK | 3 / 1 (ERK better in C100 ResNet-8) | **LIMITED** |
| 5 PPO > random | 6 / 0 | **STRONG** |
| 6 correct prior improves exploration (V_RL accuracy AUC) | 3 / 0 (C10 SimpleCNN, C100 SimpleCNN, C100 ResNet-8) | **MODERATE** |
| 7 exploration gain is layer-specific (P1 > P2 and P1 > P3) | 1 / 0 (C10 SimpleCNN only) | **LIMITED** |
| 8 correct prior improves final accuracy (excess over matched GM) | 0 / 0 | **NOT SUPPORTED** |
| secondary: rank-score AUC, utility AUC, destructive frequency (P1 − P0) | 6 / 0 each | STRONG each |
| historical Level-A/B/C rule, recomputed per setting | A in C10 and C100 SimpleCNN; B in the other four | — |

### Other observations

- **Final policies.**
  - No final policy in any condition is destructive.
  - PPO learns architecture-specific allocations: heavy pruning of the large FC/last-stage unit,
    light pruning of the first unit.
  - In C100 ResNet-8, PPO converges to `[1,1,1,1]` in 19 of 20 P0 runs, which is only 9.5% sparsity.
    With λ_s = 0.01 the reward favours accuracy over sparsity for this small, low-accuracy model.
- **Exploration.** The prior's advantage is concentrated early in training (roughly the first
  100–150 episodes) and the trajectories converge afterwards.
  - On LeNet-5 the shuffled and constant priors help as much as the correct prior.
- **McNemar (supportive).**
  - P1 vs P0 same seed: significant pairs are rare and split in direction.
  - P0 vs matched GM: PPO is significantly better (McNemar p < 0.05) in 18 of 20 (C10 LeNet-5)
    and 19 of 20 (C10 ResNet-8) seed pairs.
- **CPU cost** (measured):
  - landscapes: 2.1 h wall on 4 workers;
  - PPO: 5.8 process-hours, which is 1.4 machine-hours;
  - matched baselines: 2.0 h;
  - checks: 0.6 h;
  - total ≈ 6.5 machine-hours.

### Files

All in `results/architecture_generalization/` unless noted.

- **Run tables:** `phase3_all_runs.csv`, `phase3_summary.csv`, `phase3_policy_stats.csv`.
- **Exploration:** `phase3_training_curves.csv`, `phase3_trajectory.csv.gz`,
  `phase3_exploration_auc.csv`.
- **Statistics and classification:** `phase3_statistics.csv`, `phase3_cross_setting_summary.csv`,
  `phase3_classification.json`.
- **McNemar:** `phase3_mcnemar.csv`, `phase3_mcnemar_summary.csv`.
- **Inputs and rules:** `phase3_sensitivity_vectors.csv`, `phase3_destructive_action_rules.json`,
  `phase3_frozen_inputs.json`.
- **Checks:** `phase3_implementation_checks.md`, `phase3_integrity_checks.md`.
- **Landscapes:** `phase3_policy_landscapes.csv`, `phase3_landscapes/`.
- **Runs and predictions:** `phase3_runs/` (records, curves, infrastructure log),
  `phase3_predictions/`.
- **Figures:** `phase3_fig_A…F_*.csv` (data), `phase3_figures/*.png`.
- **Baselines:** `results/stronger_baselines/phase3_all_runs.csv`, `phase3_summary.csv`,
  `phase3_grid.csv`, `phase3_baseline_evaluations.csv`.
- **Agents:** `checkpoints/phase3_agents/`.

---

## Phase 4 (CPU-only, branch `phase4-cpu`): reward redesign, sensitivity-specific prior, elite-archive search

Pre-registration `results/phase4/phase4_preregistration.md` (sha256 `914b0418…`, `f52e87d`), committed
before any Phase-4 computation.

- **Runs:** 1,200 PPO runs, seeds 400–419, 1 thread each.
- **Checks:** implementation 21/21; integrity 16/16.
- **Selections (validation only, mechanical):** Stage A kept R0 (no redesigned reward met both
  criteria); Stage B chose P2 (centred sensitivity-action prior); Stage C chose S2A (elite BC +
  archive selection).
- **Test:** read only after the final-policy freeze (`bf5f3fb`).

**Results.** Criteria B, C and E were met; A and D were not. Full summary with qualifications:
`results/phase4/phase4_summary.md`.
- **Sensitivity-specific prior:** STRONG (5/6) on reward-rank AUC.
- **Archive search:** selected median rank 1–10 vs 65–173, top-20 reach 100%; mostly a selection
  effect, since PPO already visits the top policies.
- **Final pipeline vs control:** better in 3/6, worse in both ResNet-8 settings.
- **LAMP:** significantly better than the final pipeline in 6/6; non-inferior within 0.5 pp in 3/6.

**Infrastructure.** A file-lock crash and a power interruption during the test evaluation were both
resumed with nothing lost (`results/phase4/runs/infrastructure_log.txt`).

---

## Latency (controlled protocol)

Source: `evaluation/latency_benchmark.py` → `latency_results.csv` (summary) and
`latency_samples.csv` (all 3,600 timed passes). Run 2026-09-27 on the i5-6400,
4 threads, CPU, Python 3.11.9, torch 2.7.1+cpu, with no other experiment running
(system CPU load 5.8% before, 9.4% after).

Models: dense baseline; PPO seed 42 `[1,1,5,5]` (58.20%); global magnitude over
the same four layers at 58.20%. Masks baked in with `prune.remove`. `eval()` +
`inference_mode()`, fixed input `(B, 3, 32, 32)`, 100 warm-up passes per model,
then 10 rounds × 60 timed passes with the model order shuffled every round
(600 per model). CIs: bootstrap (2,000 resamples) for medians and median
differences.

| Batch | Model | Median ms [95% CI] | Mean ms | SD ms | Median diff vs dense, ms [95% CI] |
|---|---|---|---|---|---|
| 1 | dense | 0.737 [0.733, 0.741] | 0.789 | 0.180 | — |
| 1 | PPO seed 42 | 0.721 [0.716, 0.723] | 0.771 | 0.167 | −0.016 [−0.022, −0.012] |
| 1 | global magnitude, matched | 0.737 [0.734, 0.740] | 0.793 | 0.179 | 0.000 [−0.005, +0.005] |
| 128 | dense | 51.251 [50.786, 51.578] | 52.674 | 6.011 | — |
| 128 | PPO seed 42 | 51.248 [50.957, 51.608] | 52.971 | 6.662 | −0.003 [−0.442, +0.597] |
| 128 | global magnitude, matched | 50.928 [50.450, 51.210] | 51.985 | 5.539 | −0.323 [−0.910, +0.228] |

At batch 1 the PPO model's 0.016 ms (2.2%) difference has a CI excluding zero,
while the global-magnitude model at identical total sparsity shows none.

---

## Figure data

| Figure | File | Content |
|---|---|---|
| A | `accuracy_sparsity_tradeoff.csv` | dense baseline; PPO seeds 42/1/2/3; matched global magnitude and matched uniform for each seed; unmatched reference sweeps tagged separately (`family` column). Seed 3's matched rows are marked as sharing seed 2's policy. Asserts every seed's rows are at matched sparsity. |
| B | `ppo_policy_heatmap.csv` | rows = seeds, columns = nominal prune % per layer and realised per-layer sparsity |
| C | `sensitivity_ablation_plot.csv` | condition × seed: test/val accuracy, sparsity, retention, policy |
| D | `reward_sweep_plot.csv` | λ_s × seed: test/val accuracy, sparsity, retention, policy |
| E | `ppo_training_curves.csv` | every training episode of all 28 runs: raw reward, 25-episode rolling mean, reward terms, probe accuracy, sparsity, actions |
| E | `ppo_training_curves_aggregate.csv` | per (condition, λ_s, episode): mean and SD across the 4 seeds, raw and smoothed |

Training curves exist only for these scripted runs. Main's original runs did not
log episodes; the four regression runs reproduce those runs' final policies,
sparsities and accuracies exactly, but their curves are from the re-run.

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

- [x] Storage and latency measurement — see above
- [x] Clean fine-tuning notebook — see "Fine-tuning arm" above. Single PPO seed;
      more seeds are needed before drawing any conclusion about the fine-tuned arm.
      `checkpoints/baseline_clean.pth` is no longer produced or used by any code;
      its working copy came from an abandoned Aug 25 run with a separately trained
      15-epoch baseline.
- [x] Sensitivity ablation: correct / zeroed / shuffled, 4 seeds each
- [x] Reward sparsity-coefficient sweep: 0.00 / 0.01 / 0.02 / 0.04 / 0.08, 4 seeds each
- [x] Controlled latency measurement with a matched global-magnitude model
- [x] Matched global magnitude + fine-tune baseline for the fine-tuning arm
- [x] Sensitivity feature diagnostic and refined sensitivity analysis (100 runs, 10 seeds)
- [x] Sensitivity-Aware Constrained PPO (pre-registered; 180 runs, 10 seeds; Level C)
- [x] Soft Sensitivity-Prior PPO (pre-registered, committed `4a7a5a5`; 240 runs, 20 seeds; Level B). Final sensitivity experiment: per the pre-registration no further β values, transformations, masks, rewards or definitions are tried
- [x] Confirmatory exploration study on fresh seeds 100–129 (pre-registered, committed `c3a9484`; 120 runs; Level A for the exploration claim). No further sensitivity experiment without a new hypothesis
- [x] Archive-confirmatory experiment on fresh seeds 200–239 (pre-registered, committed `1bda844`; 160 runs; Level C-FA). Per the pre-registration: no further redesign of sensitivity for final accuracy without a new hypothesis
- [ ] More PPO seeds in the fine-tuning arm (currently 1)
- [ ] Consolidated final results table (left to the paper write-up)
- [x] Phase 2 (CPU-only): consolidated `src/` library (verified), LeNet-5 / ResNet-8, CIFAR-100 split, 15 dense baselines, efficiency baselines, implementation checks
- [x] Phase 3 (CPU-only): 480 pre-registered PPO runs over 6 settings, exact landscapes, matched baselines; mechanical cross-setting classification
- [x] Phase 4 (CPU-only): reward redesign, sensitivity-specific prior, elite-archive search; 1,200 pre-registered runs (branch `phase4-cpu`)
- [ ] Phase 5 onward: awaiting approval
- [ ] Optional: normalised-sparsity reward, as a separately labelled experiment (not run)

---

## Environment

Python 3.13.13 · torch 2.7.1+cpu · torchvision 0.22.1 · gymnasium 1.2.3 ·
stable-baselines3 2.8.0 · numpy 2.2.2 · pandas 2.2.3 · CPU only.
Pinned in `requirements.txt`. From Phase 2 on, `ptflops==0.7.5` is added (`requirements_locked_phase2.txt`).

Two machines are involved:

| Work | Machine | Python |
|---|---|---|
| Main notebook, matched-sparsity analysis, original storage/latency | not recorded | 3.13.13 |
| Fine-tuning notebook, PPO ablation and sweep, fine-tune GM baseline, controlled latency | Intel Core i5-6400 @ 2.70 GHz, 4 threads, 8 GB, no GPU, Windows 11 | 3.11.9 |

The baseline, layer sensitivities and all four recorded PPO runs reproduced
exactly on the second machine.

## Reproduction

```
python evaluation/matched_sparsity_analysis.py     # matched baselines + McNemar
python evaluation/storage_and_latency.py           # storage + latency (idle machine)
python evaluation/ppo_experiments.py               # 28 PPO runs, resumable (~2.3 h on the i5-6400)
python evaluation/aggregate_experiments.py         # ablation/sweep tables, stats, figure data
python evaluation/finetune_matched_baselines.py    # matched GM + fine-tune (~4 min)
python evaluation/latency_benchmark.py             # controlled latency (idle machine; refuses if busy)
```

`ppo_experiments.py` skips runs whose `results/runs/<id>.json` already exists;
delete a file to re-run it.

Refined sensitivity analysis:

```
python evaluation/sensitivity_diagnostic.py                          # diagnosis of the original feature
python evaluation/refined_sensitivity.py probe                       # refuses if values are already registered
python evaluation/refined_sensitivity.py run --threads 1 --worker K --workers 4   # K = 0..3, resumable
python evaluation/refined_aggregate.py                               # tables, statistics, state usage, figures
python evaluation/reward_landscape.py && python evaluation/reward_landscape.py --rank-learned
```

Sensitivity-Aware Constrained PPO (needs `sb3-contrib`, in `requirements.txt`):

```
python evaluation/constrained_masks.py                               # curves, masks, state values, heuristic
python evaluation/policy_landscape.py --worker K --workers 4         # K = 0..3, then:
python evaluation/policy_landscape.py --merge
python evaluation/predictive_validity.py
python evaluation/constrained_ppo.py verify                          # must print "identical ... = True" twice
python evaluation/constrained_ppo.py run --worker K --workers 4 --seeds 10   # K = 0..3, resumable
python evaluation/constrained_aggregate.py
```

Soft Sensitivity-Prior PPO:

```
python evaluation/soft_prior.py check                                # must report ALL CHECKS PASS
python evaluation/soft_prior.py run --worker K --workers 4 --seeds 20  # K = 0..3, resumable
python evaluation/soft_prior_aggregate.py                            # tables, statistics, classification, figures
```

Confirmatory exploration study:

```
python evaluation/exploration_confirm.py check                       # must report ALL CHECKS PASS
python evaluation/exploration_confirm.py run --worker K --workers 4  # K = 0..3, resumable
python evaluation/exploration_confirm_aggregate.py                   # refuses to run with fewer than 120 runs
```

Archive-confirmatory experiment:

```
python evaluation/archive_confirm.py prepare                         # refuses if the split/sensitivity already exist
python evaluation/archive_confirm.py check                           # must report ALL CHECKS PASS
python evaluation/archive_confirm.py run --worker K --workers 4      # K = 0..3, resumable
python evaluation/archive_aggregate.py                               # refuses to run with fewer than 160 runs
```

`matched_sparsity_analysis.py` asserts that it reproduces the notebook's recorded
seed-42 sparsity and accuracy exactly before computing anything new.

Phase 2 (CPU-only; run with `venv/Scripts/python.exe` from the repository root):

```
python experiments/phase2/verify_library.py static --threads 4           # 21 checks vs historical results
python experiments/phase2/verify_library.py ppo main42                   # also main1, main2, main3, explore_E1_100
python experiments/phase2/verify_library.py report                       # -> library_verification.json
python experiments/phase2/prepare_splits.py                              # CIFAR-100 split; refuses to overwrite
python experiments/phase2/benchmark_epoch.py --dataset cifar10 --archs simplecnn lenet5 resnet8
python experiments/phase2/benchmark_epoch.py --dataset cifar100 --archs simplecnn lenet5 resnet8
python experiments/phase2/train_dense.py --dataset cifar10 --arch lenet5 --seeds 0 1 2 --threads 4   # etc.; resumable
python experiments/phase2/architecture_tables.py units | dense | efficiency   # efficiency needs an idle machine
python experiments/phase2/phase2_checks.py                               # -> cpu_phase2_checks.md
```
