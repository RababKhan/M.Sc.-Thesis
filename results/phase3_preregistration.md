# Phase 3 pre-registration — architecture and dataset generalisation of PPO pruning and sensitivity-guided exploration

Written 2026-09-29. At that point the frozen inputs (sensitivity vectors, destructive-action rules,
shuffled and constant priors) had been computed, and no Phase-3 landscape, implementation check or
PPO run had been started. The file is committed before any of them and never modified afterwards;
a necessary correction would be a separately committed amendment made before any affected run.
Governing documents: `rocksolid_cpu_master_protocol.md`, `cpu_runtime_budget.md`.

This study is not designed to make PPO or the sensitivity prior win. Negative and mixed results are
reported as they are.

## 1. Questions (four separate claims)

| | Question | Primary comparison |
|---|---|---|
| A | Does adaptive PPO beat uniform per-layer pruning at matched sparsity? | P0 vs uniform, same exact zero count |
| B | How does PPO compare with global magnitude, LAMP, ERK and random pruning? | P0 vs each, same exact zero count |
| C | Does the correct sensitivity soft prior improve PPO **exploration**? | P1 vs P0 (and P1 vs P2, P3) on trajectory metrics |
| D | Does the correct prior improve **final** pruned-model accuracy? | P1 vs P0, sparsity-adjusted |

A result for one question never changes the classification of another.

## 2. Settings and frozen inputs

**Settings.** Six architecture-dataset settings. Each uses the reference checkpoint chosen by the
Phase-2 rule: seed 0, or `cnn_baseline_FIXED.pth` for SimpleCNN-C10.
- No other checkpoint is used, whatever its accuracy.
- The CIFAR-10 LeNet-5 reference is kept although its seed is below its three-seed mean.

**Frozen inputs.** All inputs are in
`results/architecture_generalization/phase3_frozen_inputs.json`, committed with this file (content
sha256 `7a323f87…`; file sha256 `3887ffd1…`). The file contains:
- reference hashes;
- the V_RL reward denominators;
- unit definitions;
- raw and normalised sensitivity vectors;
- rankings;
- destructive rules;
- the per-seed shuffled priors;
- the constant priors.

**Sensitivity procedure.** All six vectors were computed by the same procedure:
- `src.sensitivity.loss_sensitivity`, on V_RL only, at 1 thread;
- ratios 10/20/40/60%;
- min-max normalisation.

The SimpleCNN-C10 vector agrees with the archive vector (computed at 4 threads) within 3.2e-8.

| Setting | Reference sha256 | V_RL base acc. (denominator) | Normalised S (unit order) | Most sensitive unit | Constant prior |
|---|---|---|---|---|---|
| C10 SimpleCNN | `ca28f834…` | 76.9667 | 1.000, 0.031603, 0.009350, 0.000 | features.0 | 0.260238 |
| C10 LeNet-5 | `5ca6e0fc…` | 64.3000 | 1.000, 0.422236, 0.046116, 0.000 | conv1 | 0.367088 |
| C10 ResNet-8 | `41fc5ee5…` | 78.5333 | 1.000, 0.137730, 0.330767, 0.000 | stem | 0.367124 |
| C100 SimpleCNN | `d75dd19c…` | 50.3667 | 1.000, 0.102728, 0.014311, 0.000 | features.0 | 0.279260 |
| C100 LeNet-5 | `8e1b1aee…` | 33.3000 | 1.000, 0.587416, 0.000, 0.085762 | conv1 | 0.418295 |
| C100 ResNet-8 | `8b84bfad…` | 43.7667 | 0.987086, 0.310050, 1.000, 0.000 | stage2 | 0.574284 |

### Pruning units

The units are frozen in Phase 2. Four per architecture; the output classifier is protected. The
table gives weights and their % of the prunable weights.

| Architecture | Unit 1 | Unit 2 | Unit 3 | Unit 4 |
|---|---|---|---|---|
| SimpleCNN | features.0: 864 (0.14%) | features.3: 18,432 (2.99%) | features.6: 73,728 (11.94%) | classifier.1: 524,288 (84.93%) |
| LeNet-5 | conv1: 450 (0.74%) | conv2: 2,400 (3.94%) | fc1: 48,000 (78.78%) | fc2: 10,080 (16.54%) |
| ResNet-8 | stem: 432 (0.56%) | stage1: 4,608 (6.01%) | stage2: 14,336 (18.69%) | stage3: 57,344 (74.74%) |

- **Action set.** {0, 10, 20, 30, 40, 60}% in every setting, so each setting has 6^4 = 1,296
  policies.
- **Action semantics.** An action applies L1 unstructured pruning to every tensor of its unit.

### Destructive-action rule

`phase3_destructive_action_rules.json`, fixed before any PPO run:
- a sampled policy is destructive if it prunes the setting's **most sensitive unit** (highest raw
  V_RL loss sensitivity; ties go to the earliest unit) at ≥ 40%, i.e. action index ≥ 4;
- this reproduces the historical rule ("features.0 ≥ 40%") for SimpleCNN.

## 3. Data use

| Split | Use in Phase 3 |
|---|---|
| Train | none (no fine-tuning in Phase 3) |
| V_RL (3,000, stratified) | sensitivity, PPO reward, the policy landscape, all exploration/trajectory metrics, final-policy validation accuracy |
| V_SELECT (2,000) | **not used**: no final-selection mechanism is pre-registered |
| Test (10,000) | one evaluation of each frozen final policy and of each baseline model; never enumerated over the policy space |

- **Guards.** `DataBundle` guards stay active: test access raises inside `training()` and before
  `freeze()`.
- **Test predictions.** They are deterministic per pruned model, so they are cached by (setting,
  policy) or (setting, method, zero count, random seed). They are computed after the model is frozen.

## 4. PPO (identical in every run)

- **Implementation.** stable-baselines3 2.8.0 PPO on `src.rl.PruningEnv`. The environment was
  verified bit-identical to the historical environments.
  - P0 uses `MlpPolicy`.
  - P1–P3 use `src.rl.SoftPriorPolicy`, the historical soft prior generalised to n units. With a
    zero prior it equals `MlpPolicy`.
- **Hyper-parameters.**
  - network [64, 64] tanh, orthogonal init;
  - Adam, lr 3e-4;
  - n_steps 64, batch 32, n_epochs 10;
  - γ 0.99, GAE λ 0.95, clip 0.2;
  - ent_coef 0.01, vf_coef 0.5, max_grad_norm 0.5;
  - `seed` = run seed.
- **Budget.** `total_timesteps = 2000` → **2,048 trained = 512 episodes of 4 steps**, the same for
  every run, condition and setting.
- **Reward.** R = 1.5 · A/A_base + 0.01 · S(%) + 0.05 · (number of distinct actions), where:
  - A is the pruned model's V_RL accuracy;
  - A_base is the reference's V_RL accuracy (table above);
  - S is the total sparsity over all Conv2d/Linear weights.
  No coefficient sweep.
- **State.** The 7 historical features per unit, with the sensitivity feature set to **0** in all
  conditions.
- **Evaluation cache.** Terminal evaluations come from the setting's exact landscape (§6). On every
  terminal step the environment creates the V_RL DataLoader iterator, so the global random stream
  is identical to an uncached run. This is verified in the implementation checks.
- **Final policy.** The deterministic policy after training, with no selection among episodes.
- **Threads.** Every PPO run and every evaluation runs at `torch.set_num_threads(1)`, in 4 worker
  processes.

### Conditions

All actions are always available; there are no masks. The prior is
`adjusted_logit(u, a) = raw_logit(u, a) − β · prior_u · ratio(a)/60`, with β fixed at 0.50.

| | Prior | β |
|---|---|---|
| P0 plain PPO | none (MlpPolicy) | 0 |
| P1 correct prior | the setting's normalised S | 0.50 |
| P2 shuffled prior | S permuted per seed (below) | 0.50 |
| P3 constant prior | every unit = mean(S) | 0.50 |

**P2 permutations.** For each seed, `numpy.random.default_rng(seed)` permutations are drawn until
every unit receives another unit's, different, value. This is the `soft_prior.shuffled` algorithm
of the confirmed exploration study. The resulting vectors and mappings are fixed in
`phase3_frozen_inputs.json`.

### Seeds

**300, 301, …, 319 (20 seeds)**, the same for all 4 conditions and 6 settings, giving
**6 × 4 × 20 = 480 runs**.
- **Freshness.** A scan of every run record and agent checkpoint found none of these seeds in any
  earlier PPO experiment (highest used: 239).
- **Disclosure.** Seed 300 was used once, in a 64-timestep PPO smoke test inside the Phase-2
  implementation checks. That test had no experimental purpose, and no outcome was recorded or
  looked at.
- **No changes.** No seed is removed, replaced or added.

## 5. Baselines (matched sparsity)

- **Methods.** The five `src.pruning` baselines, whose formulas are in
  `architecture_generalization/cpu_phase2_checks.md`:
  - uniform per-layer magnitude;
  - global magnitude over the prunable tensors;
  - random: uniform allocation with random masks, **10 seeds 1000–1009**, summarised per run by the
    mean test accuracy of the 10;
  - LAMP;
  - ERK.
- **Exact matching.** For every PPO final model, each baseline is built at **exactly the same number
  of zeroed weights**. This is asserted, so all comparisons are at identical sparsity. Deterministic
  baselines at an identical zero count are evaluated once and reused.
- **Descriptive curves.** Accuracy-sparsity curves use the same five methods at a fixed grid of
  total sparsity {10, 20, 30, 40, 50, 60}%. They are never compared as if matched to PPO.

## 6. Exact policy landscape (per setting, V_RL)

All 1,296 policies are recorded with:
- actions and per-unit ratios;
- total sparsity and per-unit sparsity;
- V_RL accuracy and mean V_RL loss;
- reward and its components (λ_s = 0.01);
- reward rank (1 = best, ties get the minimum rank);
- rank score (1296 − rank)/1295;
- utility U = 1.5 · A/A_base + 0.01 · S;
- the destructive flag;
- Pareto efficiency in (sparsity ↑, accuracy ↑) and in (sparsity ↑, loss ↓).

The landscape is never enumerated on test.

## 7. Endpoints

### Trajectory (per run, seed = unit)

- **Checkpoints.** The end of each episode k = 1…512 is a checkpoint, at timestep t_k = 4k. Its
  values are the landscape values of the policy sampled in episode k.
- **AUC.** AUC(y) is the trapezoidal integral over t_k divided by (t_512 − t_1). This is identical
  to the historical study.

### Question-specific endpoints

**A and B.** For each P0 run:

  d = test accuracy(P0 final model) − test accuracy(method at the identical zero count)

**C, primary.** The V_RL Accuracy AUC.

**C, secondary.** Each is its own family:
- rank-score AUC;
- utility AUC;
- destructive frequency (fraction of the 512 sampled policies that are destructive; lower is better).

**C, descriptive.**
- worst trajectory V_RL accuracy;
- V_RL accuracy of the final policy.

**D, primary.** The excess test accuracy over matched global magnitude:

  e = test(final model) − test(global magnitude at the identical zero count)

  compared as P1 − P0.

**D, secondary** (descriptive family):
- raw test-accuracy difference P1 − P0;
- the paired sparsity differences (mean |Δ|, share within 0.5 pp).

**Descriptive, per setting and condition.**
- test and V_RL accuracy;
- sparsity;
- accuracy drop and retention vs the dense reference (test, 1 thread);
- the learned policies and per-unit ratio distributions;
- reward, reward rank and Pareto status;
- number of unique final policies;
- final-policy entropy (Shannon entropy in bits of the 20 final policies);
- mean action entropy of the trained policy along its deterministic rollout;
- McNemar counts (P1 vs P0 same seed; P0 vs matched global magnitude). These are supportive only.

## 8. Statistics and families

**Tests.** All comparisons are paired by seed, N = 20, with one row per (setting, condition, seed)
asserted. For each comparison the analysis reports:
- the **exact two-sided sign-flip p** over all 2^20 patterns;
- mean, median and SD of the paired differences;
- the **95% t-CI** (df 19);
- the bootstrap 95% CI (10,000 resamples, `default_rng(0)`);
- d_z;
- the exact Wilcoxon p;
- wins, ties and losses.

**Holm families** (each corrected separately):

| Family | Comparisons | Size |
|---|---|---|
| A | P0 − uniform, one per setting | 6 |
| B | P0 − {global, LAMP, ERK, random} per setting | 24 |
| C | P1 − P0 on V_RL Accuracy AUC, per setting | 6 |
| C-specificity | P1 − P2 and P1 − P3 on V_RL Accuracy AUC, per setting | 12 |
| C-rank, C-utility, C-destructive | P1 − P0 per setting, one family per endpoint | 6 each |
| D | P1 − P0 on excess over matched global magnitude, per setting | 6 |

**Per-comparison outcome** (computed mechanically). The hypothesised direction is:
- A and B: PPO better;
- C and D: P1 better (for destructive frequency, lower is better).

The outcome is:
- **supported**: the mean difference is in the hypothesised direction, the Holm-adjusted p is below
  0.05, **and** the 95% t-CI excludes 0;
- **opposite**: the same criteria hold in the reverse direction;
- **inconclusive**: otherwise.

**Replication of the historical Level-A/B/C rule, per setting** (reported, but not a new family).
The confirmed study's classification is recomputed within each setting with its own rule:
- Holm over that setting's three primary comparisons P1−P0, P1−P2 and P1−P3;
- Holm over the three comparisons within each of rank AUC, utility AUC and destructive frequency.

## 9. Cross-setting classification (per statement, mechanical)

| Class | Rule |
|---|---|
| **STRONG** | supported in ≥ 5 of 6 settings and opposite in none |
| **MODERATE** | supported in 3–4 of 6 and opposite in none |
| **LIMITED** | supported in 1–2 of 6, or supported in ≥ 3 with at least one opposite |
| **NOT SUPPORTED** | supported in 0 of 6 |

- **Reversal.** A statement that is opposite in ≥ 3 of 6 settings is additionally reported as
  **REVERSED**, e.g. "global magnitude > PPO".
- **No universal claims.** A result that holds only in a subset of settings is never described as
  general.

The statements are:
1. PPO > uniform (A).
2. PPO > global magnitude (B).
3. PPO > LAMP (B).
4. PPO > ERK (B).
5. PPO > random (B).
6. The correct prior improves exploration (C).
7. The correct prior's exploration gain is layer-specific (C-specificity): P1 > P2 **and** P1 > P3
   in the same setting count as support.
8. The correct prior improves final accuracy (D).

## 10. Conduct

- **Execution.** The 480 runs are executed in the fixed order (setting, seed, condition) by 4 worker
  processes (job i goes to worker i mod 4).
- **Run records.** Every run writes:
  - its training curve (512 episodes);
  - its SB3 agent;
  - its record, written atomically and last. The record holds the setting, condition, seed, the
    checkpoint, sensitivity, landscape, pre-registration and code hashes, the policy, per-unit
    ratios, sparsity, V_RL and test accuracy, reward and components, runtime and completion status.
- **Restarts.** A restart skips only runs with a verified complete record. A completed stochastic
  run is never re-run or overwritten unless corruption is detected and documented.
- **Infrastructure failures.** A failure (power loss, crash) repeats the run with the same seed and
  configuration, and the failure is documented. The pipeline is deterministic.
- **Genuine failures.** A genuine numerical or model failure is retained, never replaced.
- **No interim peeking.**
  - Until all 480 runs finish, progress reports contain only run counts, runtime, failures and the
    estimated time remaining.
  - No aggregate or comparative result is computed.
  - Baselines are evaluated after all PPO runs finish.
- **Integrity checks before any aggregate analysis.** These are written to
  `phase3_integrity_checks.md`:
  - 480 records;
  - all seeds;
  - hashes of checkpoints, sensitivity file and landscapes unchanged;
  - equal budgets;
  - unit definitions unchanged;
  - 1 thread in every record;
  - exact float parsing;
  - guards active;
  - the reference baselines unchanged.
- **Implementation checks.** Run before the study, using only the non-experimental seed 42 and
  synthetic inputs. They are written to `phase3_implementation_checks.md`.

## 11. Deviations from the historical exploration study (declared)

| Item | Historical study | This study |
|---|---|---|
| Reward accuracy / denominator | V_RL 3,000 and its accuracy (archive protocol) | same |
| Exploration landscape | full 5,000 validation (included the reward probe) | V_RL (the reward data; V_SELECT is not used) |
| Destructive rule | features.0 ≥ 40% | the setting's most sensitive unit ≥ 40% (the same rule for SimpleCNN) |
| Seeds per condition | 30 | 20 (master protocol) |
| Settings | SimpleCNN-C10 | 6 settings |

## 12. Budget (`cpu_runtime_budget.md` R4, R5)

- **Per-run and per-setting limits.**
  - Each landscape must take ≤ 1.5 machine-hours per setting.
  - Each PPO run must take ≤ 15 machine-minutes at 1 thread.
  - The implementation checks measure both before the runs start.
- **Phase projection.** Well below the 150-machine-hour phase limit.

## 13. Outputs

**`results/architecture_generalization/`**
- `phase3_all_runs.csv`
- `phase3_summary.csv`
- `phase3_sensitivity_vectors.csv`
- `phase3_policy_stats.csv`
- `phase3_training_curves.csv`
- `phase3_trajectory.csv.gz`
- `phase3_exploration_auc.csv`
- `phase3_statistics.csv`
- `phase3_cross_setting_summary.csv`
- `phase3_destructive_action_rules.json`
- `phase3_frozen_inputs.json`
- `phase3_implementation_checks.md`
- `phase3_integrity_checks.md`
- `phase3_classification.json`
- `phase3_policy_landscapes.csv`
- `phase3_landscapes/`
- `phase3_runs/`
- `phase3_predictions/`
- the figure data `phase3_fig_A…F_*.csv`

**`results/stronger_baselines/`**
- `phase3_all_runs.csv`
- `phase3_summary.csv`
- `phase3_grid.csv`

**Other**
- a Phase-3 section in `experiment_log.md`
