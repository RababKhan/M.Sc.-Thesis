# Phase 5 pre-registration: reward–objective alignment (accuracy-constrained pruning)

Branch `phase5-cpu`. Part A (`phase5_design_commitments.md`, commit `aff2f3e`) fixed the following
before any Phase-5 data existed:
- the split rule;
- the reward form and its constants;
- the τ candidates and decision rule;
- the oracle definition;
- the seeds, conditions and archive rules.

This document completes the pre-registration. It was written after the validation-only τ diagnostic and
**before any confirmatory (seed 500–519) run**, is committed with its hash, and is never edited. A
correction would be a separately committed amendment made before any affected run.

**Purpose.** To test whether an explicit accuracy constraint aligns RL search with the real goal: remove
as many parameters as safely possible while retaining predictive accuracy.
- Equal sparsity across architectures is **not** a goal.
- Higher sparsity, a better reward or a better reward rank is **not** assumed to mean a better model.
- Negative results are kept.

## 1. Frozen inputs

| Item | Value |
|---|---|
| Phase 3 / Phase 4 | unchanged (zero `git diff` vs `67f4270` / `a3653fd`, re-verified before and after the runs) |
| Dense references | sha256 `ca28f834`, `5ca6e0fc`, `41fc5ee5`, `d75dd19c`, `8e1b1aee`, `8b84bfad` |
| Old splits | `archive_validation_split.npz` `55b7ff40`; `cifar100_split_indices.npz` `d7351925` (train 45,000 / validation 5,000 / test 10,000 unchanged) |
| New split | `splits/cifar10_phase5_split.npz` `eccd7522…`, `splits/cifar100_phase5_split.npz` `131cf534…`; VAL-RL / VAL-SELECT 2,500 each; per class they differ by ≤ 1 image |
| Landscapes | all 1,296 policies on VAL-RL and VAL-SELECT, 1 thread (`landscapes/<setting>_landscape.json`, hashed in the config) |
| Configuration | `phase5_config.json` sha256 `357551dc…`: τ, dense accuracies, sensitivity vectors, P2 coefficients, oracles, table hashes |

- **Sensitivity (VAL-RL, 1 thread).** The normalised loss-based S is recomputed per setting. The most
  sensitive unit, which defines the destructive action, is:
  - features.0 in the SimpleCNN settings;
  - conv1 in the LeNet-5 settings;
  - stem in the ResNet-8 settings (C100: stem 1.000 vs stage2 0.997).
- **Prior P2.** c_u = 1.0·(S_u − mean S). Its coefficients sum to 0 in every setting.

## 2. τ (chosen by the Part-A rule on VAL-RL; test never used)

| τ | ≥ 20 feasible and ≥ 1 feasible with ≥ 10% sparsity in all six settings |
|---|---|
| 0.95 | yes |
| 0.97 | yes |
| **0.98** | **yes (chosen: the largest passing value)** |
| 0.99 | no: C100 ResNet-8 has only 11 feasible policies |

At τ = 0.98 (full table in `phase5_tau_diagnostic.csv`; q = VAL-RL accuracy / dense VAL-RL accuracy):

| Setting | Dense VAL-RL | Feasible | Max feasible sparsity | Oracle (constrained optimum) | Oracle q |
|---|---|---|---|---|---|
| C10 SimpleCNN | 76.88 | 515 | 59.09% | [1,4,5,5] 59.09% | 0.9906 |
| C10 LeNet-5 | 64.20 | 395 | 51.91% | [0,1,5,3] 51.91% | 0.9807 |
| C10 ResNet-8 | 78.92 | 46 | 25.34% | [1,2,1,3] 25.34% | 0.9812 |
| C100 SimpleCNN | 48.96 | 254 | 54.39% | [1,3,4,5] 54.39% | 0.9845 |
| C100 LeNet-5 | 32.76 | 279 | 34.61% | [1,3,4,4] 34.61% | 0.9817 |
| C100 ResNet-8 | 43.76 | 21 | 16.08% | [0,1,1,2] 16.08% | 0.9890 |

A feasible policy with ≥ 20% sparsity exists in 5 of 6 settings; C100 ResNet-8 is the exception.

## 3. Rewards, conditions, seeds (from Part A)

- **R_constrained** (on VAL-RL, ρ as a fraction):
  - if q ≥ 0.98: R = 1 + ρ + 0.01·q;
  - otherwise: R = −10·(0.98 − q).
  - There is no diversity term.
- **R0** (C0): 1.5·A/A_d + 0.01·ρ(%) + 0.05·D, with λ = 0.01.
- **PPO.** The Phase-3/4 PPO throughout:
  - SB3 2.8.0; MlpPolicy [64, 64] tanh; Adam 3e-4;
  - n_steps 64, batch 32, 10 epochs;
  - γ 0.99, GAE λ 0.95, clip 0.2;
  - entropy 0.01, value 0.5, max grad norm 0.5;
  - 2,048 timesteps = 512 episodes;
  - the exact VAL-RL landscape cache;
  - state sensitivity 0;
  - 1 thread;
  - no behavioural cloning.

| Condition | Reward | Prior | Archive + VAL-SELECT selection |
|---|---|---|---|
| C0 (Phase-4 pipeline, corrected protocol) | R0 | P2 | yes: max R0 score on VAL-SELECT |
| C1 (proposed) | R_constrained, τ = 0.98 | P2 | yes: VAL-SELECT-feasible, max sparsity |
| C2 (no prior) | R_constrained, τ = 0.98 | none | yes: as C1 |

- **Seeds.** 500–519 (fresh), the same in every condition and setting.
- **Run matrix.** **6 settings × 3 conditions × 20 seeds = 360 runs**, executed in the order
  (setting, seed, condition) by 4 workers (job i goes to worker i mod 4).

## 4. Archive, selection, test (from Part A)

1. **Archive.** Every unique policy sampled during training is recorded, with its VAL-RL reward and first
   episode. The candidates are the top 10 by VAL-RL reward (ties: earlier discovery). The archive of all
   360 runs is frozen and hashed (`phase5_archive.json`) **before** any VAL-SELECT evaluation.
2. **Selection.** The frozen candidates are evaluated live on VAL-SELECT, 1 thread, and must equal the
   VAL-SELECT landscape.
   - **C1, C2:** among candidates with q_SELECT ≥ 0.98, the highest sparsity; ties go to higher
     VAL-SELECT accuracy, then higher VAL-RL reward, then earlier discovery. If no candidate is feasible,
     the highest q_SELECT; fallbacks are reported.
   - **C0:** the highest R0 score on VAL-SELECT; ties go to higher VAL-RL reward, then earlier discovery.
3. **Freeze.** The final-policy list of all 360 runs is written, hashed and **committed**.
4. **Test.** Only then is the test set read, once per distinct pruned model, together with:
   - the dense references;
   - **LAMP** and **global magnitude** at exactly the selected policy's zero count (sparsity mismatch 0).

## 5. Metrics

Ranks are the **constrained rank**: the rank of a policy under R_constrained on the VAL-RL landscape
(1 = the oracle). It is the common yardstick for C0, C1 and C2.

**Search quality (per run):**
- best-seen rank;
- final archive-selected rank;
- rank of the deterministic final policy (descriptive);
- median rank over training;
- reach of the top 100, 50, 20, 10 and 5 and of the optimum (best-seen and selected);
- first episode entering the top 20 and top 10 (censored at 513);
- reward-rank AUC (AUC of (1296 − rank)/1295 over the 512 sampled policies);
- VAL-RL accuracy AUC;
- destructive-action rate;
- feasible-sample rate.

**Constraint and accuracy:**
- q on VAL-RL and on VAL-SELECT;
- test accuracy, and q_test = test / dense test;
- feasibility of the final policy on each split;
- dense-to-pruned test drop;
- the share of final policies that are feasible.

**Sparsity:**
- total and per-unit sparsity;
- the maximum feasible sparsity;
- the gap to the oracle.

**Baselines.** Test accuracy of PPO minus LAMP and minus global magnitude at identical sparsity.

## 6. Statistics

**Tests.** Exact two-sided sign-flip tests paired by seed (N = 20; 2^20 patterns). Each comparison reports:
- n; mean ± SD; median;
- the paired mean difference;
- the 95% t-CI and the bootstrap CI;
- d_z;
- the raw and Holm-adjusted p;
- wins, ties and losses.

**Per-comparison outcome.** Holm correction is applied within each family.
- **Supported:** the direction is as hypothesised, the Holm p < 0.05 **and** the t-CI excludes 0.
- **Opposite:** the same criteria hold in reverse.
- **Inconclusive:** otherwise.

| Family | Comparisons (per setting) | Size |
|---|---|---|
| H1/H4 | C1 − C0 on final test accuracy (positive) | 6 |
| H3 primary | C1 − C2 on reward-rank AUC (+), VAL-RL accuracy AUC (+), first episode in the top 20 (−), first episode in the top 10 (−) | 24 |
| H3 secondary | C1 − C2 on destructive rate (−), best-seen rank (−), final selected rank (−) | 18 |
| H5 | C1 − LAMP (test, identical sparsity) (+) | 6 |
| H5 gap | C1 − C0 on (test − matched LAMP) (+) | 6 |
| descriptive | C1 − global (+); C1 − C0 on selected rank (−); C1 − C0 on selected sparsity | 6 each |

## 7. Criteria (each reported as MET / PARTIALLY MET / NOT MET with the exact numbers)

- **A. Accuracy safety.**
  - **MET:** H1 (C1 − C0 test accuracy) is supported in **both** ResNet-8 settings **and** opposite in at
    most 1 of the other four settings.
  - **PARTIALLY MET:** supported in at least one ResNet-8 setting, but not MET.
  - **NOT MET:** otherwise.
- **B. Non-trivial compression.**
  - **Sub-condition 1:** C1's median final sparsity is ≥ 10% in all six settings.
  - **Sub-condition 2:** C1's median final sparsity is ≥ 20% in at least min(4, number of settings where
    a ≥ 20% feasible policy exists) settings. That number is 5, so the requirement is **4**.
  - **MET:** both sub-conditions hold. **PARTIALLY MET:** one holds. **NOT MET:** neither holds.
- **C. Search quality.**
  - **Sub-condition 1:** C1's median final selected constrained rank is ≤ 20 in ≥ 5 of 6 settings.
  - **Sub-condition 2:** ≥ 80% of the 120 C1 runs have a final selected constrained rank ≤ 20. The
    best-seen reach is also reported.
  - **MET:** both hold. **PARTIALLY MET:** one holds.
- **D. Sensitivity contribution.**
  - **MET:** at least one H3-primary metric is supported (C1 better than C2) in ≥ 4 settings, **and** at
    most 1 of the 24 H3-primary outcomes is opposite.
  - **PARTIALLY MET:** at least one setting has a supported metric, but not MET.
- **E. Final model improvement.**
  - **MET:** H1 is supported in ≥ 2 settings **and** opposite in none.
  - **PARTIALLY MET:** supported in ≥ 1 setting, but not MET.
- **F. LAMP competitiveness.** Reported as two separate criteria.
  - **F strong, MET:** H5 (C1 − LAMP) is supported in ≥ 1 setting and opposite in none.
  - **F weak, MET:** the mean C1 − LAMP is ≥ −0.5 pp in ≥ 4 settings **and** ≥ −1.0 pp in every
    setting. These are point estimates at identical sparsity.
  - **F weak, PARTIALLY MET:** within 0.5 pp in ≥ 1 setting, but not MET.

## 8. Interpretation rules (fixed now)

- **C1 searches well but gives worse final models.** State that reward alignment remains unresolved.
- **C1 gives up nearly all sparsity** (B not met). State that the constraint is too conservative.
- **Architecture dependence.** If C1 improves ResNet-8 but hurts SimpleCNN or LeNet-5, report the
  architecture dependence.
- **Sensitivity and final accuracy.** Sensitivity is never claimed to improve final accuracy unless
  statistically supported.

## 9. Conduct and integrity

- **No interim peeking.** Progress reports show run counts, runtime and failures only. No aggregate is
  computed before the integrity checks.
- **Resumability.** Runs are resumable and written durably (fsync + atomic rename). An infrastructure
  failure is logged; the analysis verifies that no result of the failed run was saved, and the run is
  repeated with the same seed.
- **Implementation checks** (`phase5_implementation_checks.md`, seed 42 and synthetic inputs only) run
  before the confirmatory runs.
- **Integrity checks** (`phase5_integrity_checks.md`) run after them, covering:
  - all 360 runs, and no dropped seed;
  - TEST and VAL-SELECT never accessed in training;
  - the archive frozen before VAL-SELECT, and the final list frozen before TEST;
  - τ, rewards and seeds unchanged;
  - Phase-3/4 files unchanged;
  - checkpoints, splits and landscapes unchanged;
  - identical-sparsity baselines;
  - commit, CPU, threads and packages recorded;
  - interruptions logged.

## 10. Budget

- **Per run.** A run takes ≤ 2 machine-minutes at 1 thread (measured in the implementation checks).
- **Total.** 360 runs ≈ 1–1.5 wall-hours on 4 workers, plus a few minutes for the archive and selection
  and about 20–30 minutes for the test evaluation.
