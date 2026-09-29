# Phase 4 pre-registration: redesigned reward, sensitivity-specific prior, elite-archive search

Written 2026-09-29 on branch `phase4-cpu` (from `67f4270`). No Phase-4 landscape, configuration value,
implementation check or PPO run existed at that point. Committed before all of them and never modified
afterwards; a correction would be a separately committed amendment made before any affected run.
Phase 3 is frozen evidence: its files, checkpoints, splits, sensitivity vectors and V_RL landscapes are
read only, and their hashes are verified.

Goal: diagnose and improve the mechanism, not search for a favourable number. Negative results are
reported.

## 0. Two corrections to the Phase-4 brief (declared)

1. **R0 uses λ_s = 0.01.** The Phase-3 reward used λ_s = 0.01 (Phase-3 pre-registration §4), not 0.04;
   0.04 was the original notebook value. R0 therefore reproduces the **Phase-3** reward exactly:
   `1.5·A/A_b + 0.01·ρ + 0.05·D`.
2. **The proposed prior `b(l,a) = −β·S_l·r_a` is the Phase-3 prior itself.** Phase 3 subtracted
   `β·S_u·ratio/60`, so the literal formula would reproduce P1 and cannot address problem B.
   - The Phase-3 controls (correct, shuffled, constant) all carried the same *total* conservatism
     Σ_u c_u = β·Σ S. That shared "prune less" pressure is what they had in common.
   - A prior whose benefit can only come from layer-specific information must carry **zero net
     conservatism**: c_u = β·(S_u − S̄). It penalises aggressive actions on sensitive units, favours them
     on insensitive units, and leaves low actions almost unchanged (ratio 0 gets exactly 0).
   - This centred interaction prior is P2. The pure-conservatism control P4 is magnitude-matched to it.

Also found before designing: the feasible sparsity range is 0 to 52.7–59.8% in every setting,
because it is set by the action set. The accuracy retained at maximal pruning, however, ranges from 11%
to 53%. The cross-setting inconsistency of Phase 3 therefore comes from **accuracy fragility, not
sparsity scale**. A ρ-only normalisation would merely rescale λ by a factor of 1.7–1.9 in every setting,
so R2 normalises both axes.

## 1. Fixed inputs (Phase 3, frozen)

**Settings.** The six settings, their reference checkpoints (hashes asserted), the CIFAR splits
(V_RL 3,000 / V_SELECT 2,000 / test 10,000) and the four pruning units. The actions are {0, 10, 20,
30, 40, 60}% on every unit, giving 1,296 policies.

**Sensitivity and rules.** The normalised V_RL loss-sensitivity vectors S and the destructive-action
rules come from `phase3_frozen_inputs.json`. The exact V_RL landscapes come from
`architecture_generalization/phase3_landscapes/`. Everything is evaluated at 1 thread.

**PPO.** The Phase-3 configuration throughout:
- SB3 2.8.0; MlpPolicy [64, 64] tanh; Adam 3e-4;
- n_steps 64, batch 32, n_epochs 10;
- γ 0.99, GAE λ 0.95, clip 0.2;
- entropy 0.01, value 0.5, max grad norm 0.5;
- 2,048 timesteps = 512 episodes in **every** run;
- the exact landscape cache;
- state sensitivity 0;
- the final policy is the deterministic policy.

## 2. New inputs, computed after this commit by fixed rules (`experiments/phase4/prepare.py`)

- **V_SELECT landscape.** All 1,296 policies on V_SELECT, 1 thread (`landscape_vselect.py`). It is used
  only for:
  - archive selection;
  - held-out ranks;
  - the Stage-C selection.

  It is never used in any reward or in training.
- **R1b target.** ρ*_s = max{ρ(π) : A_VRL(π) ≥ 0.95·A_b} over the V_RL landscape.
- **R1a appropriateness (reported).** Checks whether any landscape policy with ρ ∈ [47.5, 52.5] retains
  ≥ 95%. R1a is run in all settings regardless, so the pure sparsity-control test is complete.
- **κ (R1a and R1b separately).** The smallest value in {0.005, 0.01, 0.02, 0.04, 0.08} whose V_RL
  reward optimum lies within ±5 pp of the target in all six settings. If none qualifies, 0.08 is used and
  the failure is reported.
- **R2 anchors.** A_60 and ρ_60 = V_RL accuracy and sparsity of the maximal policy [5,5,5,5]. V_SELECT
  scores use the V_SELECT anchors.
- **Rank tables.** For every reward, the V_RL reward, rank (1 = best; ties get the minimum rank) and
  utility (the reward without the diversity term). Also the V_SELECT score and rank, and the V_RL frontier
  regret max{A(π') : ρ(π') ≥ ρ(π)} − A(π).

All of these are written to `phase4_config.json` and `landscapes/<setting>_rewards.json`, hashed, and
committed before any PPO run.

## 3. Rewards (A = pruned-model accuracy, A_b = reference accuracy on the same split, ρ = total sparsity %, D = number of distinct actions)

| | Reward |
|---|---|
| R0 | 1.5·A/A_b + 0.01·ρ + 0.05·D (Phase-3 control) |
| R1a | 1.5·A/A_b − κ_a·\|ρ − 50\| + 0.05·D (common target 50%) |
| R1b | 1.5·A/A_b − κ_b·\|ρ − ρ*_s\| + 0.05·D (95%-retention target) |
| R2 | 1.5·(A − A_60)/(A_b − A_60) + 1.5·ρ/ρ_60 + 0.05·D (anchor-normalised; both axes on [0, 1]) |

R2 needs only two anchor evaluations (dense and maximal) and is indifferent between those two anchors,
so its optimum is each landscape's "knee". The reward is always computed on V_RL.

## 4. Priors (subtracted from the PPO logits of unit u: c_u·ratio(a)/60; all actions always available; no masks)

| | c_u |
|---|---|
| P0 | none (MlpPolicy) |
| P1 | 0.5·S_u (Phase-3 prior) |
| P2 | 1.0·(S_u − S̄) (**new: centred sensitivity-action interaction**) |
| P3 | P2 with S − S̄ permuted per seed: `default_rng(seed)` permutations until every unit gets another unit's, different, value |
| P4 | 1.0·mean_u \|S_u − S̄\| for every unit (pure conservatism, magnitude-matched to P2) |
| P5 | −1.0·(S_u − S̄) (sign-reversed) |

## 5. Search variants

| | Training | Final selection |
|---|---|---|
| S0 | standard PPO | deterministic final policy |
| S1 | the S0 runs (no new training) | archive rule |
| S2 | PPO + elite behavioural cloning: every minibatch loss adds η·L_elite, L_elite = −mean log π(a_e \| s_e) over the four (state, action) pairs of each of the **K = 5** best unique V_RL-reward policies seen so far (ties: earlier discovery); active once ≥ 5 unique policies are seen; **η = 0.1**; the PPO network remains the decision maker | deterministic final policy |
| S2A | the S2 runs | archive rule |

- **Archive rule.** Take the top 10 unique sampled policies by V_RL reward (ties: earlier discovery).
  Choose the one with the highest **V_SELECT** score under the same reward (ties: higher V_RL reward, then
  earlier).
- **S2 implementation.** A copy of SB3's `PPO.train()` with the single added term. It is verified
  bit-identical to SB3 PPO when η = 0.

## 6. Staged design, seeds and selection (all validation-only; mechanical)

- **Seeds.** **400–419** (fresh; verified unused), paired across every condition and stage.
- **Stage A (reward).** R0, R1a, R1b, R2 with P0 and S0: 6 × 4 × 20 = 480 runs.
  - **Winner:** among R1a, R1b and R2, those meeting **A1** and **A2** qualify; the one with the
    smallest I(R) wins. If none qualifies, **R0** is carried forward. The winner is R*.
  - **A1:** I(R)/I(R0) ≤ 0.5, and the 95% seed-bootstrap upper bound of that ratio is < 1. I(R) is the
    SD over the 6 settings of the setting-mean final sparsity. The bootstrap uses 10,000 resamples of the
    20 seeds, with the same resample in every setting and reward, `default_rng(0)`.
  - **A2:** the setting-mean final V_RL retention A/A_b is ≥ 0.95 in ≥ 5 of 6 settings.
- **Stage B (prior).** P0–P5 under R*, S0: 600 new runs, since P0 is Stage A's run of R*.
  - **Carry-forward rule:** P1 and P2 are each tested against P0 on the reward-rank AUC (family of 12).
    A prior is eligible if it is supported in ≥ 3 settings and opposite in none.
  - **Winner:** the eligible prior supported in more settings (tie: larger mean d_z); else P0. The
    winner is P*.
- **Stage C (search).** Under (R*, P*): S0 reuses Stage B, S2 is new (120 runs), S1 and S2A are the
  selection rules.
  - **Winner:** the variant with the lowest mean over settings of the per-setting median **V_SELECT**
    rank of the final-selected policy (ties: S0, S1, S2, S2A order). The winner is S*.
- **Pipelines.**
  - **Final pipeline:** F = (R*, P*, S*).
  - **Control:** K = (R0, P0, S0), i.e. the Phase-3 configuration on fresh seeds.
- **Test set.**
  - The test set is **locked until all three selections are frozen**. No runner touches it.
  - The list of final-selected policies of every run (final and archive) is then written and hashed.
  - Only after that are test predictions computed, once per distinct pruned model, together with the
    matched baselines:
    - LAMP and global magnitude at every distinct zero count;
    - uniform, ERK and random (10 seeds, 1000–1009) for F and K.

## 7. Endpoints

**Search quality**, computed on the exact V_RL landscape of the run's reward:
- final-selected rank;
- best-seen rank (the minimum over the 512 sampled policies; equal to the "minimum rank during
  training");
- median rank over training;
- reward-rank AUC ((1296 − rank)/1295);
- V_RL accuracy AUC;
- utility AUC;
- destructive frequency;
- reach of the top 100, 50, 20 and 10 and of the optimum (best-seen and final);
- first episode entering the top 100, 50 and 20 (censored at 513);
- V_SELECT rank of the final-selected policy.

**Sparsity control:**
- the setting-mean final sparsity and I(R);
- control error |ρ − target| (R1a, R1b);
- V_RL retention;
- V_RL frontier regret.

**Final models** (test, after the freeze):
- test accuracy;
- **excess over matched LAMP** e = test(π) − test(LAMP at the identical zero count), and the same over
  global magnitude;
- absolute sparsity.

## 8. Statistics and families

**Tests and reporting.** Tests are exact two-sided sign-flip tests paired by seed (N = 20; 2^20
patterns). The analysis reports:
- mean ± SD and median;
- the paired mean difference;
- the 95% t-CI;
- the bootstrap CI;
- d_z;
- the Wilcoxon p;
- wins, ties and losses.

**Per-comparison outcome.** Holm correction is applied per family.
- **Supported:** the direction is as hypothesised, the Holm p < 0.05, **and** the 95% t-CI excludes 0.
- **Opposite:** the same criteria hold in reverse.
- **Inconclusive:** otherwise.

"Lower is better" endpoints are hypothesised negative.

| Family | Comparisons (per setting) | Size |
|---|---|---|
| A-regret (secondary) | R1a−R0, R1b−R0, R2−R0 on final V_RL frontier regret (lower) | 18 |
| **B-specific (primary)** | **P2−P3, P2−P4 on reward-rank AUC** | 12 |
| B-vsP0 | P1−P0, P2−P0 on reward-rank AUC (also the carry-forward rule) | 12 |
| B-reverse | P2−P5 on reward-rank AUC | 6 |
| B-secondary (one family each) | P2−P3, P2−P4 on V_RL AUC; utility AUC; destructive frequency (lower); first top-100 (lower); first top-50 (lower); best-seen rank (lower); final rank (lower) | 12 each |
| **C-final (primary)** | S1−S0, S2−S0, S2A−S0 on final-selected V_RL rank (lower) | 18 |
| C-reach | S2−S0 on best-seen rank (lower) | 6 |
| C-heldout | S1−S0, S2−S0, S2A−S0 on final-selected V_SELECT rank (lower) | 18 |
| **D (primary)** | F−K on test excess over matched LAMP | 6 |
| D-GM | F−K on test excess over matched global magnitude | 6 |
| **E (primary)** | F − LAMP (test, identical zero count) | 6 |
| E-NI | F non-inferior to LAMP, margin 0.5 pp: one-sided sign-flip on d + 0.5 (Holm) and t-CI lower bound > −0.5 | 6 |

**Cross-setting class per statement.**

| Class | Rule |
|---|---|
| STRONG | ≥ 5 of 6 supported, none opposite |
| MODERATE | 3–4 of 6 supported, none opposite |
| LIMITED | 1–2 supported, or ≥ 3 supported with ≥ 1 opposite |
| NOT SUPPORTED | 0 supported |

A statement opposite in ≥ 3 settings is additionally REVERSED. The sensitivity-specific statement is
supported in a setting only if both P2−P3 **and** P2−P4 are supported there.

## 9. Success criteria (pre-registered; Phase 4 counts as successful if ≥ 1 holds)

- **A. Reward:** some redesigned reward meets A1 and A2.
- **B. Sensitivity-specific:** "P2 > P3 and P2 > P4 on reward-rank AUC" is MODERATE or STRONG.
- **C. Search:** for some X ∈ {S1, S2, S2A}:
  - "X better than S0 on final-selected V_RL rank" is MODERATE or STRONG; **and**
  - X's median final-selected V_RL rank is ≤ 20 in ≥ 3 settings.
- **D. Final model:** "F > K on excess over matched LAMP" is MODERATE or STRONG.
- **E. LAMP competitiveness:** either:
  - "F > LAMP" is MODERATE or STRONG; or
  - F is non-inferior to LAMP (margin 0.5 pp) in ≥ 3 settings.

An improved mean alone is never called a success.

## 10. Conduct and integrity

- **Execution.**
  - 4 worker processes × 1 thread.
  - Every run writes its curve, agent and record durably (fsync + atomic rename); the record is
    written last.
  - Restarts skip only verified records.
  - Infrastructure failures are repeated with the same seed and configuration and logged; genuine
    failures are kept.
- **No peeking.** During runs, progress reports show counts, runtime and failures only. The mechanical
  stage selections are the only aggregate computed before the end, and they use validation data only.
- **Implementation checks** (`phase4_implementation_checks.md`, non-experimental seed 42 only, before
  Stage A):
  - rewards reproduce the tables;
  - R0 is bit-identical to the Phase-3 reward and training;
  - the environment under every reward is identical cached and uncached;
  - ElitePPO with η = 0 equals SB3 PPO;
  - the elite states equal the environment's observations;
  - the prior matrices are correct (P1 = the Phase-3 matrix);
  - the selection rules work on synthetic inputs;
  - the guards are active;
  - the budget is equal.
- **Integrity checks before the final analysis** (`phase4_integrity_checks.md`):
  - all expected runs, and all seeds;
  - hashes of the Phase-3 files, references, landscapes, config and pre-registration unchanged;
  - no test access in any training record;
  - the selections were frozen before the test-freeze file;
  - matched baselines have identical zero counts;
  - 1 thread everywhere;
  - commit, environment and package versions recorded.

## 11. Outputs (`results/phase4/`)

- **Protocol and configuration:** `phase4_preregistration.md`, `phase4_config.json`,
  `phase4_stage_selection.json`.
- **Checks:** `phase4_implementation_checks.md`, `phase4_integrity_checks.md`.
- **Run tables and curves:** `phase4_all_runs.csv`, `phase4_training_curves.csv`,
  `phase4_policy_rank_trajectory.csv`.
- **Comparisons:** `phase4_reward_comparison.csv`, `phase4_prior_comparison.csv`,
  `phase4_search_quality.csv`, `phase4_baseline_comparison.csv`.
- **Statistics and classification:** `phase4_statistics.csv`, `phase4_cross_setting_summary.csv`,
  `phase4_classification.json`.
- **Summary:** `phase4_summary.md`.
- **Figures:** `phase4_fig_A…E_*.csv/png`.
- **Run data:** `runs/`, `landscapes/`, `predictions/`.

## 12. Budget

About 1,200 PPO runs, each ≤ 2 machine-minutes (measured in Phase 3), plus:
- the V_SELECT landscapes, about 1.5 h;
- the test and baseline evaluation, about 2 h.

That totals ≈ 8–10 machine-hours, within the 150 h phase limit (`cpu_runtime_budget.md` R5).
