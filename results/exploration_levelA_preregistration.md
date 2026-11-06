# Confirmatory study: sensitivity-guided exploration — pre-registration

Written 2026-09-28, before any implementation check or training run of this
study, and committed before any fresh-seed run. This file is not modified
afterwards; a necessary correction would be a new, separately committed
amendment made before any rerun.

## Hypothesis (exploration claim, not a test-accuracy claim)

A correct layer-sensitivity prior improves PPO exploration quality and reduces
destructive pruning behaviour during training, relative to no prior, a
shuffled prior and a non-layer-specific constant prior.

Origin: the Soft Sensitivity-Prior PPO study (Level B) found trajectory-level
exploration advantages for the correct prior (seeds 42, 1–19). This study
tests whether they replicate on fresh seeds. All earlier experiments are
unchanged.

## Fixed protocol

| Item | Value |
|---|---|
| Baseline | `checkpoints/cnn_baseline_FIXED.pth`, sha256 `ca28f8345c23365ee7b92686e780ae0a9befa76c9d6bb9b3bb0b9e42272e111c`; val 77.32%, test 77.03% (asserted at start) |
| Split | CIFAR-10, seeded permutation (seed 42): 45,000 train / 5,000 validation / 10,000 official test; RL reward probe = first 1,000 validation images |
| Data use | validation only for the reward, exploration metrics, policy evaluation during training and policy selection; test only for the single evaluation of each frozen final policy; the test loader raises if requested during training |
| Sensitivity (fixed, not recomputed) | loss-based refined vector from `results/refined_sensitivity_values.csv` (sha256 `3efd306f…`), layers features.0, features.3, features.6, classifier.1: **S = [1.0, 0.032278303448596, 0.0133578022446062, 0.0]** (≈ [1.000, 0.032, 0.013, 0.000]) |
| Prior | adjusted_logit(ℓ, a) = raw_logit(ℓ, a) − β · S_ℓ · a_norm(a), a_norm = ratio/60 ∈ {0, 1/6, 1/3, 1/2, 2/3, 1}; applied to the policy logits before the categorical distribution is formed (sampling, PPO log-probabilities and entropy, deterministic final policy). No masking, clipping, remapping or reward penalty; all six actions always available |
| β | **0.50 only** |
| Reward | R = 1.5 · acc_probe / 77.32 + 0.01 · S(%) + 0.05 · (number of distinct actions); λ_s = 0.01; unchanged throughout |
| PPO | stable-baselines3 2.8.0 PPO with `SoftPriorPolicy` (`evaluation/soft_prior.py`, sha256 `85425e13…`, unmodified): MlpPolicy network [64, 64] tanh, orthogonal init, Adam lr 3e-4, n_steps 64, batch 32, n_epochs 10, γ 0.99, GAE λ 0.95, clip 0.2, ent_coef 0.01, vf_coef 0.5, max_grad_norm 0.5, `seed` = run seed |
| Budget | 2,000 timesteps requested → 2,048 trained = 512 episodes of 4 steps, every run |
| Environment | the verified memoised environment of `evaluation/constrained_ppo.py` (sha256 `80e5fe5a…`, unmodified), all actions valid |
| Final policy | deterministic policy after training |
| Hardware | Intel Core i5-6400, 4 worker processes × 1 torch thread, Python 3.11.9, torch 2.7.1+cpu (1-, 2- and 4-thread execution were previously verified bit-identical) |

## Seeds (locked)

**100, 101, …, 129 (30 seeds).** None was used by any earlier experiment
(checked: earlier runs used only 1–19 and 42). No seed is removed, replaced or
added. No early stopping. The experimental unit is the seed.

## Conditions (exactly four)

| | State sensitivity | Prior | β |
|---|---|---|---|
| E0 no prior | 0 | none | 0 |
| E1 correct prior (proposed) | 0 | S | 0.50 |
| E2 shuffled prior | 0 | S permuted among layers (below) | 0.50 |
| E3 constant prior | 0 | every layer = mean(S) = 0.26140902642330055 | 0.50 |

E2 permutation per seed, fixed now (from `soft_prior.shuffled`, a
`numpy.random.default_rng(seed)` permutation in which every layer receives
another layer's, different, value; notation `layer<-source`):

| Seed | Prior vector (f.0, f.3, f.6, c.1) | Mapping |
|---|---|---|
| 100 | 0.032278, 0.013358, 0.0, 1.0 | f.0<-f.3, f.3<-f.6, f.6<-c.1, c.1<-f.0 |
| 101 | 0.013358, 1.0, 0.0, 0.032278 | f.0<-f.6, f.3<-f.0, f.6<-c.1, c.1<-f.3 |
| 102 | 0.013358, 0.0, 0.032278, 1.0 | f.0<-f.6, f.3<-c.1, f.6<-f.3, c.1<-f.0 |
| 103 | 0.0, 1.0, 0.032278, 0.013358 | f.0<-c.1, f.3<-f.0, f.6<-f.3, c.1<-f.6 |
| 104 | 0.013358, 0.0, 1.0, 0.032278 | f.0<-f.6, f.3<-c.1, f.6<-f.0, c.1<-f.3 |
| 105 | 0.013358, 0.0, 1.0, 0.032278 | f.0<-f.6, f.3<-c.1, f.6<-f.0, c.1<-f.3 |
| 106 | 0.032278, 0.013358, 0.0, 1.0 | f.0<-f.3, f.3<-f.6, f.6<-c.1, c.1<-f.0 |
| 107 | 0.0, 1.0, 0.032278, 0.013358 | f.0<-c.1, f.3<-f.0, f.6<-f.3, c.1<-f.6 |
| 108 | 0.013358, 0.0, 1.0, 0.032278 | f.0<-f.6, f.3<-c.1, f.6<-f.0, c.1<-f.3 |
| 109 | 0.032278, 1.0, 0.0, 0.013358 | f.0<-f.3, f.3<-f.0, f.6<-c.1, c.1<-f.6 |
| 110 | 0.032278, 1.0, 0.0, 0.013358 | f.0<-f.3, f.3<-f.0, f.6<-c.1, c.1<-f.6 |
| 111 | 0.0, 0.013358, 0.032278, 1.0 | f.0<-c.1, f.3<-f.6, f.6<-f.3, c.1<-f.0 |
| 112 | 0.013358, 1.0, 0.0, 0.032278 | f.0<-f.6, f.3<-f.0, f.6<-c.1, c.1<-f.3 |
| 113 | 0.0, 0.013358, 1.0, 0.032278 | f.0<-c.1, f.3<-f.6, f.6<-f.0, c.1<-f.3 |
| 114 | 0.0, 0.013358, 0.032278, 1.0 | f.0<-c.1, f.3<-f.6, f.6<-f.3, c.1<-f.0 |
| 115 | 0.032278, 0.0, 1.0, 0.013358 | f.0<-f.3, f.3<-c.1, f.6<-f.0, c.1<-f.6 |
| 116 | 0.013358, 0.0, 1.0, 0.032278 | f.0<-f.6, f.3<-c.1, f.6<-f.0, c.1<-f.3 |
| 117 | 0.013358, 1.0, 0.0, 0.032278 | f.0<-f.6, f.3<-f.0, f.6<-c.1, c.1<-f.3 |
| 118 | 0.032278, 1.0, 0.0, 0.013358 | f.0<-f.3, f.3<-f.0, f.6<-c.1, c.1<-f.6 |
| 119 | 0.013358, 0.0, 1.0, 0.032278 | f.0<-f.6, f.3<-c.1, f.6<-f.0, c.1<-f.3 |
| 120 | 0.032278, 0.0, 1.0, 0.013358 | f.0<-f.3, f.3<-c.1, f.6<-f.0, c.1<-f.6 |
| 121 | 0.032278, 0.013358, 0.0, 1.0 | f.0<-f.3, f.3<-f.6, f.6<-c.1, c.1<-f.0 |
| 122 | 0.0, 0.013358, 0.032278, 1.0 | f.0<-c.1, f.3<-f.6, f.6<-f.3, c.1<-f.0 |
| 123 | 0.013358, 1.0, 0.0, 0.032278 | f.0<-f.6, f.3<-f.0, f.6<-c.1, c.1<-f.3 |
| 124 | 0.013358, 0.0, 0.032278, 1.0 | f.0<-f.6, f.3<-c.1, f.6<-f.3, c.1<-f.0 |
| 125 | 0.0, 1.0, 0.032278, 0.013358 | f.0<-c.1, f.3<-f.0, f.6<-f.3, c.1<-f.6 |
| 126 | 0.013358, 0.0, 1.0, 0.032278 | f.0<-f.6, f.3<-c.1, f.6<-f.0, c.1<-f.3 |
| 127 | 0.013358, 0.0, 0.032278, 1.0 | f.0<-f.6, f.3<-c.1, f.6<-f.3, c.1<-f.0 |
| 128 | 0.032278, 1.0, 0.0, 0.013358 | f.0<-f.3, f.3<-f.0, f.6<-c.1, c.1<-f.6 |
| 129 | 0.032278, 0.0, 1.0, 0.013358 | f.0<-f.3, f.3<-c.1, f.6<-f.0, c.1<-f.6 |

(Vector entries are rounded here to six decimals; the exact values of S are used.)

## Evaluation checkpoints (identical for every run)

A checkpoint is the end of each training episode k = 1, …, 512, at timestep
t_k = 4k. At checkpoint k the evaluated policy is the four-action policy
sampled in episode k. Its **validation accuracy** is its accuracy on the full
5,000-image validation split, read from the fixed 1,296-policy validation
landscape `results/policy_landscape_validation.csv` (sha256 `a90f33c0…`),
which evaluated every policy with the same pruning code. No additional or
condition-specific evaluation occurs; the schedule is asserted identical for
all 120 runs.

AUC(y) = trapezoidal integral of y_k over t_k from t_1 to t_512, divided by
(t_512 − t_1), so AUC is on the scale of y.

## Endpoints

**Primary — Validation Accuracy AUC:** AUC of the checkpoint validation
accuracies (higher = better exploration).

Secondary (each its own family):

- **S1 reward-rank AUC:** AUC of the rank score (1296 − rank_k)/1295, where rank_k is
  the sampled policy's rank in the landscape's λ_s = 0.01 reward (`reward_rank_0.01`;
  1 = best). Score 1 = best policy; higher is better.
- **S2 utility AUC:** AUC of U_k = 1.5 · val_acc_k / 77.32 + 0.01 · sparsity_k(%),
  exactly as in the soft-prior study. Higher is better.
- **S3 destructive features.0 frequency:** (number of the 512 episodes whose
  features.0 action is ≥ 40%) / 512. Lower is better.
- **S4 worst trajectory validation accuracy:** min over the 512 checkpoints of
  val_acc_k. Higher is safer.
- **S5 final validation policy quality:** the final deterministic policy's
  full-validation accuracy (tested); its sparsity, reward rank and Pareto
  status are reported descriptively.

Descriptive (not tested for classification):

- **Safety analysis:** per run, the per-layer action distribution over all
  sampled episodes; features.0 destructive count, mean features.0 pruning
  and maximum features.0 pruning sampled.
- **Sample efficiency:** first episode and timestep at which the sampled
  policy has val_acc ≥ 76.0% **and** sparsity ≥ 55%, or NOT REACHED (never
  imputed). Paired descriptive comparison: per seed, which condition reached
  it first (both not reached = tie); exact sign test on non-ties.
- **Final test-set models:** test accuracy, sparsity, accuracy of global
  magnitude pruning over the same four layers at the exact sparsity,
  McNemar tests vs E0 (same seed) and vs matched global magnitude.

## Statistics

- Unit: the seed; N = 30 paired seeds per comparison. Episodes and timesteps
  are never observations. The analysis code asserts one row per (condition,
  seed) with exactly the 30 locked seeds.
- Comparisons: E1 − E0, E1 − E2, E1 − E3 (paired by seed).
- Test: **two-sided exact sign-flip permutation test** on the mean paired
  difference, over all 2³⁰ sign patterns (computed exactly by a
  meet-in-the-middle split into two halves of 15).
- Reported for every comparison: paired mean, median, SD of paired
  differences, **95% t confidence interval** (df = 29, t = 2.045;
  pre-registered interval for the Level-A rule), 95% bootstrap percentile CI
  (10,000 resamples, `default_rng(0)`, descriptive), d_z = mean/SD,
  exact Wilcoxon signed-rank p (descriptive).
- Multiplicity: Holm over the three primary comparisons; separately, Holm
  over the three comparisons within each secondary endpoint (S1–S5).
- For S3, "favours E1" means E1 − Ex < 0; for all others E1 − Ex > 0.

## Classification (precedence A, then B, then C)

**Level A — confirmed exploration benefit** only if, for all three primary
comparisons, the mean paired difference > 0, the Holm-adjusted p < 0.05 and
the 95% t CI excludes 0; **and** at least two of S1 (rank AUC), S2 (utility
AUC), S3 (destructive frequency) favour E1 over E0 in the expected direction
with their within-endpoint Holm-adjusted p < 0.05.

**Level B — partial replication** if not A and at least one of:
(B1) E1 − E0 on the primary endpoint has mean > 0, Holm p < 0.05 and a CI
excluding 0; (B2) all three primary mean differences are > 0; (B3) at least
one of S1, S2, S3 favours E1 over E0 in the expected direction with
within-endpoint Holm p < 0.05.

**Level C — no confirmation** otherwise.

Final test accuracy cannot change the classification. After this study, no
further sensitivity experiment without a genuinely new hypothesis.

## Conduct

- **Stopping:** all 30 seeds × 4 conditions = 120 runs are completed before
  any inferential statistic or aggregate performance summary is computed.
  Progress reports contain only runs completed, runtime, errors and
  hardware state; per-run results are not printed to the progress log.
- **Exclusions:** none.
- **Infrastructure failures** (crash, power loss, corrupted file): the run is
  repeated with the same seed and configuration, and the failure is
  documented. The pipeline is deterministic, so a repeat yields the
  identical run.
- **Genuine numerical/model failures:** retained in the record, never
  replaced. If an endpoint cannot be computed for a run, the primary analysis
  uses complete pairs, and a worst-case sensitivity analysis (a failed E1 run
  given the minimum observed E1 value; a failed control given the maximum
  observed control value) is reported.
- **Implementation checks** run before the study, on seed 42 (non-
  experimental) and synthetic data only; no fresh-seed outcome is inspected.

## Planned outputs (`results/`)

`exploration_levelA_preregistration.md` (this file),
`exploration_levelA_implementation_checks.md`, `exploration_levelA_all_runs.csv`,
`exploration_levelA_trajectory.csv`, `exploration_levelA_auc.csv`,
`exploration_levelA_safety.csv`, `exploration_levelA_sample_efficiency.csv`,
`exploration_levelA_final_models.csv`, `exploration_levelA_statistics.csv`,
`exploration_levelA_summary.csv`, `exploration_levelA_classification.json`,
`exploration_levelA_predictions/`, figure data
`exploration_levelA_fig_A…D_*.csv`, per-run records `exploration_levelA_runs/`
and `exploration_levelA_training_curves_raw/`, and a section in
`experiment_log.md`.
