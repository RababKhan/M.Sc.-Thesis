# Result freeze: all phases

Frozen on 2026-11-05, branch `thesis-final-integration`, base `6b275096`.

This file is the single reference for thesis numbers. Every number below was re-read from the frozen result files,
not from memory or summaries. Nothing in any phase folder was changed. Where the thesis narrative and the files
disagree, the files win; known differences are listed in section 11.

Conventions:
- **pp** = percentage points.
- **Test** = official CIFAR test set (10,000 images), read once per frozen model.
- **V_RL / V_SELECT** (Phases 1–4) = 3,000 / 2,000 split of the 5,000-image validation set.
- **VAL-RL / VAL-SELECT** (Phase 5 onward) = a different 2,500 / 2,500 split (seed 20260930, commit `fb9d5fbb`).
- **Settings** = CIFAR-10/100 × SimpleCNN / LeNet-5 / ResNet-8 (6 settings) unless stated otherwise.
- **"supported"** (Phases 3–6) = correct direction, Holm p < 0.05, and 95% CI excluding 0.

---

## 1. Phase 1: audit and master protocol

- **Commit:** `e47edf49`.
- **Sources:** `results/rocksolid_experiment_audit.md`, `results/rocksolid_master_protocol.md`, `results/rocksolid_compute_plan.md`, `results/rocksolid_audit_data.json`, `requirements_locked.txt`.
- **Frozen:**
  - The pre-audit (legacy) pipeline is labelled "SUPERSEDED — test-set leakage". In it, the test set was used for reward, sensitivity and epoch selection. The leakage fix is commit `985c8043`.
  - Every later phase uses the guarded protocol:
    - train is used for weight updates only;
    - full validation selects the dense training epoch;
    - V_RL drives reward and sensitivity;
    - V_SELECT is used only to choose among already-generated candidates;
    - test is read once per frozen, hashed model.
- **Legacy SimpleCNN/CIFAR-10 studies** (soft prior, exploration Level A, constrained PPO, archive confirmatory): labelled VERIFIED, but they reuse validation data for both reward and evaluation. Use them only as motivation or history, not as thesis evidence.
- **Allowed:** "All reported experiments follow a leakage-guarded protocol, adopted after an audit that superseded an earlier leaky pipeline."
- **Forbidden:** citing any number from the pre-audit pipeline or the old `README.md`.

## 2. Phase 2: datasets, splits, dense references, library

- **Commits:** `366d21ae` (protocols, `src/` library), `c156f623` (CIFAR-100 split), `2055dacf` (CPU benchmark), `ab14e460` (dense baselines, 69/69 checks).
- **Sources:** `results/cpu_dense_training_protocol.md`, `results/architecture_generalization/cpu_baselines.csv`, `results/architecture_generalization/cpu_phase2_checks.md`, `results/reproducibility/*_split_manifest.json`.
- **Splits:** 45,000 train / 5,000 validation / 10,000 test, for both datasets.
- **Dense training:** Adam lr 1e-3, cosine schedule, 30 epochs, batch 128, crop + flip, best epoch on full validation, seeds 0–2. Seed 0 is the pruning reference.
  - Exception: the CIFAR-10 SimpleCNN reference reuses an older 10-epoch model (`cnn_baseline_FIXED.pth`). Disclose this as a recipe difference.
- **Search space:** 4 prunable units per architecture × 6 ratios {0, 10, 20, 30, 40, 60}% = 1,296 policies. The output classifier is never pruned.

| Dense reference (seed 0) | Validation | Test |
|---|---|---|
| CIFAR-10 SimpleCNN | 77.32 | 77.03 |
| CIFAR-10 LeNet-5 | 64.32 | 64.96 |
| CIFAR-10 ResNet-8 | 78.72 | 78.69 |
| CIFAR-100 SimpleCNN | 49.08 | 50.50 |
| CIFAR-100 LeNet-5 | 32.58 | 32.28 |
| CIFAR-100 ResNet-8 | 43.48 | 43.99 |

- **Literature review:** not part of the repository. It is a writing task, not a frozen result.

## 3. Phase 3: multi-architecture PPO vs baselines

- **Commits:** `2dbcc770` (pre-registration), `a2cf5e51` (landscapes, 57/57 checks), `67f42700` (480 runs, statistics, classification; integrity 22/22).
- **Sources:**
  - `results/phase3_preregistration.md`;
  - `results/architecture_generalization/phase3_cross_setting_summary.csv`, `phase3_classification.json`, `phase3_statistics.csv`, `phase3_integrity_checks.md`, `phase3_figures/`;
  - `results/stronger_baselines/phase3_*`.
- **Design:**
  - 6 settings × 4 conditions × 20 seeds (300–319) = 480 runs.
  - Conditions: P0 plain PPO; P1 correct sensitivity prior; P2 shuffled prior; P3 constant prior.
  - 2,048 timesteps per run; pruning only, no fine-tuning.
  - Every baseline is built at **the zero count of each PPO run**. There is no fixed sparsity target; mean PPO sparsity is 9.5–55.1% by setting.

| Pre-registered statement | Supported / opposite (of 6) | Class |
|---|---|---|
| PPO > Uniform | 5 / 0 | **STRONG** |
| PPO > Random | 6 / 0 | **STRONG** |
| PPO > Global magnitude | 3 / 0 | **MODERATE** |
| PPO > ERK | 3 / 1 | **LIMITED** |
| PPO > LAMP | 0 / 5 | **NOT SUPPORTED, REVERSED** |
| Correct prior improves exploration (V_RL accuracy AUC) | 3 / 0 | **MODERATE** (secondary rank, utility and destructive endpoints STRONG, 6/6) |
| Exploration gain is layer-specific | 1 / 0 | **LIMITED** |
| Correct prior improves final accuracy | 0 / 0 | **NOT SUPPORTED** |

- **Key numbers** (P0 minus baseline, test pp, mean of 20 seeds):
  - **vs Uniform:** +31.36 (C10 SimpleCNN), +16.29 (C10 LeNet-5), +14.17 (C10 ResNet-8), +29.69 (C100 SimpleCNN), +3.67 (C100 LeNet-5). C100 ResNet-8 is +0.23, inconclusive: PPO chose the uniform policy [1,1,1,1] in 19 of 20 runs.
  - **vs Random:** +27.89 to +64.36 in every setting.
  - **vs LAMP:** −0.55 to −1.64 in the 5 significant settings; C10 ResNet-8 is +0.15, inconclusive.
- **Allowed:**
  - "PPO learned non-uniform allocations that strongly outperformed uniform and random allocation at the same sparsity."
  - "LAMP was significantly better than PPO in five of six settings."
- **Forbidden:**
  - "PPO beats LAMP".
  - "at matched high sparsity" (sparsity was PPO's own choice).
  - "PPO beats heuristic baselines" (global and ERK were only MODERATE and LIMITED).
  - "the sensitivity prior improves accuracy".

## 4. Phase 4: centred prior, archive selection, reward redesign

- **Commits:**
  - `f52e87d9` (pre-registration and code), `e5e9a375` (frozen configuration);
  - stage selections: `f6f85dd8` (A: R0), `c42976be` (B: P2), `a13c828d` (C: S2A);
  - `bf5f3fb0` (policy freeze); `a3653fdc` (1,200 runs, test evaluation, classification).
- **Sources:** `results/phase4/phase4_preregistration.md`, `phase4_summary.md`, `phase4_classification.json`, `phase4_cross_setting_summary.csv`, `phase4_search_quality.csv`, `phase4_prior_comparison.csv`, `phase4_reward_comparison.csv`, `phase4_baseline_comparison.csv`, `phase4_fig_*.png`.
- **Design:**
  - Seeds 400–419; 1,200 runs (Stage A 480, Stage B 600, Stage C 120).
  - **R0** = Phase-3 reward, kept because no redesigned reward met criterion A.
  - **P2** = centred sensitivity prior, c_u = 1.0·(S_u − S̄).
  - **S2A** = PPO + elite behavioural cloning + archive selection: the top 10 policies by V_RL reward, re-ranked on V_SELECT.
  - Control K = R0/P0/S0; final pipeline F = R0/P2/S2A.

| Criterion | Verdict |
|---|---|
| A. Reward | **Not met** |
| B. Sensitivity-specific prior | **Met (STRONG)** |
| C. Search | **Met (STRONG, S1/S2A)** |
| D. Final model vs control | **Not met (LIMITED)** |
| E. LAMP competitiveness | **Met only as non-inferiority**; "F > LAMP" NOT SUPPORTED |

- **Retention** (R0/P2):
  - Without the archive (S0): the best policy seen during search reaches the top 10 in 95–100% of runs, but the final selected policy reaches the top 10 in **0% of runs, in all 6 settings**. Median selected V_RL rank is 65–173.
  - With the archive (S2A): the selected policy reaches the top 10 in 75–100% of runs and the top 20 in 100%. Median selected rank is 1–10.
- **Archive effect on test accuracy (S2A − S0), descriptive:**

| Setting | Δ test (pp) |
|---|---|
| C10 SimpleCNN | +0.14 |
| C10 LeNet-5 | +0.90 |
| C10 ResNet-8 | **−5.66** |
| C100 SimpleCNN | +0.22 |
| C100 LeNet-5 | +0.45 |
| C100 ResNet-8 | **−4.00** |

  The Stage-C criterion favours the archive by construction.
- **Prior, exploration:** reward-rank AUC STRONG (5/6); destructive frequency MODERATE (4/6).
- **Prior, final outcomes:** V_RL accuracy AUC **REVERSED** (opposite in 4/6); final rank NOT SUPPORTED (0/6).
- **Misalignment and over-pruning:** R0's own optimum lies at 23.5–56.4% sparsity. For ResNet-8, moving from K to F:

| Setting | Sparsity K → F | Test K → F | Dense test | Drop vs dense |
|---|---|---|---|---|
| CIFAR-10 ResNet-8 | 29.0% → 47.8% | 76.82 → 70.83 | 78.69 | 7.86 pp |
| CIFAR-100 ResNet-8 | 8.3% → 27.7% | 43.11 → 39.22 | 43.99 | 4.77 pp |

- **F vs baselines** (matched zero count):
  - LAMP: −0.08 to −1.62 pp; LAMP better in 6/6.
  - Global magnitude: +0.01 to +18.57 pp.
  - Uniform: +12.05 to +36.04 pp.
- **Allowed:**
  - "Without archive selection, PPO discovered top-ranked policies but did not retain them; archive selection restored retention on validation ranking."
  - "Stronger search exposed a misaligned reward that drove ResNet-8 to over-prune."
- **Forbidden:**
  - "archive selection improves final test accuracy" (it is mixed, with large ResNet-8 losses).
  - "the centred prior improves accuracy-based exploration in Phase 4" (that endpoint was reversed).

## 5. Phase 5: validation-aware accuracy-constrained reward

- **Commits:** `aff2f3e9` (design), `fb9d5fbb` (VAL-RL/VAL-SELECT split), `3cbaec1a` (pre-registration, τ = 0.98), `6c4676f4` (archive freeze), `ddaf72c1` (final-list freeze), `da061e65` (360 runs, test evaluation, criteria A–F).
- **Sources:** `results/phase5/phase5_preregistration.md`, `phase5_tau_selection.md`, `phase5_summary.md`, `phase5_classification.json`, `phase5_constraint_results.csv`, `phase5_c1_vs_c2.csv`, `phase5_lamp_comparison.csv`, `phase5_global_comparison.csv`, `phase5_statistics.csv`, `phase5_fig_*.png`.
- **Reward:**
  - R = 1 + ρ + 0.01·q if q ≥ 0.98, otherwise −10·(0.98 − q).
  - q = pruned / dense accuracy on VAL-RL.
  - τ = 0.98 was the largest candidate giving ≥ 20 feasible policies in every setting.
- **Conditions:** C0 = R0 + P2 + archive; C1 = constrained reward + P2; C2 = constrained reward, no prior. Seeds 500–519; 360 runs.

| Criterion | Verdict |
|---|---|
| A. Accuracy safety | **PARTIALLY MET** |
| B. Non-trivial compression | **MET** |
| C. Search quality | **PARTIALLY MET** (76.7% of runs select a top-20 policy, against ≥ 80% required) |
| D. Sensitivity contribution | **MET** |
| E. Final model improvement | **PARTIALLY MET** (4 supported, 2 opposite) |
| F strong | **NOT MET** |
| F weak | **MET** |

- **Over-pruning control, ResNet-8 (C0 → C1):**

| Setting | Sparsity | Accuracy drop vs dense | C1 − C0 [95% CI] |
|---|---|---|---|
| CIFAR-10 ResNet-8 | 48.8% → 24.7% | 7.32 → 1.18 pp | +6.14 pp [4.92, 7.36] |
| CIFAR-100 ResNet-8 | 23.5% → 16.1% | 4.05 → 1.16 pp | +2.89 pp [2.30, 3.48] |

  SimpleCNN becomes slightly worse (−0.31 and −0.21 pp; classed as opposite).
- **C1 − LAMP:** −0.49, −0.32, −0.53, −0.29, −0.26, −0.51 pp. LAMP is significantly better in 6/6; 4/6 are within 0.5 pp.
- **C1 vs matched global magnitude:** supported in 5/6 (+0.32 to +6.51 pp); C10 SimpleCNN −0.09, inconclusive. This comparison was **descriptive** in the pre-registration.
- **Sensitivity (C1 vs C2):**
  - reward-rank AUC 5/6, VAL-RL accuracy AUC 5/6, destructive rate 6/6;
  - 0 of 24 primary tests opposite;
  - test accuracy C1 is 0.05–0.16 pp below C2 in all 6 settings (descriptive).
- **Feasibility of the selected policies:**

| Condition | VAL-RL | VAL-SELECT | Test |
|---|---|---|---|
| C1 | 100% | 99.2% | **67.5%** |

  C100 ResNet-8 had 0% test feasibility. Of the C1 runs feasible on VAL-SELECT, 39 of 119 were infeasible on test.
- **Allowed:**
  - "The accuracy-constrained reward sharply reduced ResNet-8 over-pruning."
  - "Validation feasibility did not guarantee test feasibility."
- **Forbidden:**
  - "the constraint guarantees accuracy".
  - "PPO beats global magnitude" stated as a confirmatory result.
  - "LAMP only slightly better" without stating 6/6.

## 6. Phase 6: equal fixed-mask fine-tuning

- **Commits:** `74e5f457` (design), `fdd08ff7` (pilot), `b1b5396f` (pre-registration, E = 5), `31811465` (freeze of 480 models), `3f7a658e` (test evaluation, integrity 20/20, criteria), `e6e09aaa` (summary).
- **Sources:** `results/phase6/phase6_preregistration.md`, `phase6_summary.md`, `phase6_integrity_checks.md`, `phase6_ppo_vs_lamp.csv`, `phase6_ppo_vs_global.csv`, `phase6_pre_post_accuracy.csv`, `phase6_cross_setting_summary.csv`, `phase6_final_models.json`, `figures/phase6_fig1–5`.
- **Design:**
  - Methods: PPO (frozen Phase-5 C1 policy), LAMP and global magnitude at exactly PPO's zero count (mismatch 0 in 120/120), plus a dense control.
  - **No Uniform method.**
  - Fine-tuning: Adam lr 1e-4, cosine schedule, 5 epochs, batch 128, fixed mask, no checkpoint selection. Seeds 600–619 (n = 20 per setting).
  - 480 runs.
  - Equivalence = 95% t-CI inside (−0.5, +0.5) pp. This is a CI-inclusion rule, not a formal equivalence test.
- **Criteria A–F:** all **MET**. Criterion C states that LAMP is still ahead after fine-tuning in 3/6 settings, by 0.14–0.22 pp.
- **Mean test accuracy, unweighted over the 6 settings:**

| Method | Before fine-tuning | After fine-tuning |
|---|---|---|
| PPO | 57.08 | 58.38 |
| LAMP | 57.48 | 58.48 |
| Global magnitude | 55.08 | 58.36 |
| Dense | 57.91 | 58.51 |

- **PPO − LAMP gap (pp):**

| Setting | Before | After [95% CI] | Inside ±0.5? |
|---|---|---|---|
| C10 SimpleCNN | −0.49 | −0.22 [−0.28, −0.15] | yes |
| C10 LeNet-5 | −0.32 | −0.18 [−0.28, −0.09] | yes |
| C10 ResNet-8 | −0.53 | +0.00 [−0.02, 0.03] | yes |
| C100 SimpleCNN | −0.29 | −0.05 [−0.09, −0.01] | yes |
| C100 LeNet-5 | −0.26 | −0.14 [−0.19, −0.08] | yes |
| C100 ResNet-8 | −0.51 | −0.04 [−0.07, −0.01] | yes |

  The gap shrank significantly in 5/6 settings (C10 LeNet-5 inconclusive). After fine-tuning, LAMP is still significantly ahead in 3/6, and PPO is ahead nowhere.
- **PPO vs global magnitude after fine-tuning:** −0.16 to +0.12 pp, all inside ±0.5. Before fine-tuning PPO led by up to +6.51 pp.
- **Other checks:**
  - Regrowth: none (360/360 runs, zero counts identical).
  - Reproducibility: 6/6 repeats bit-identical.
  - Worst post-fine-tuning shortfall vs dense: 0.133 pp.
- **Allowed:**
  - "After identical 5-epoch fixed-mask fine-tuning, the PPO–LAMP difference fell inside ±0.5 pp in all six settings; LAMP remained significantly ahead in three."
  - "PPO's advantage over global magnitude was mainly before fine-tuning."
- **Forbidden:**
  - "PPO is equivalent to or better than LAMP" without the CI-rule qualification.
  - "PPO beats Uniform after fine-tuning" (not run).

## 7. Paper 3: structured CPU-latency analysis

- **Branch:** `paper3-hardware-aware`.
- **Commits:** `cc25545c` (H0 GO), `2bd9770e` (H1 50/50), `de006d85` (H2-A tables), `7bfb144a` (H2-B), `5f19f117` (H2-C stopped for compute), `b37375d1` (proxy validation), `f9a013e1` (ResNet-8 threshold / latency room: PPO NO-GO), `52764425` (final manuscript v3).
- **Sources:**
  - `results/paper3/h0_decision.json`, `h1_validation_report.md`, `h2_e2_latency_correlations.csv`, `paper3_proxy_decision.json`;
  - `resnet8_threshold_calibration_decision.json`, `resnet8_threshold_calibration_report.md`;
  - `paper/paper3_hardware_analysis_final_v3.{docx,pdf}`;
  - `paper_figures_revised/`, `paper_figures_final_v3/`.
- **Manuscript title:** "Sparsity Is Not Speed: An Exhaustive CPU-Latency Analysis of Structured Neural Network Pruning."
- **Measurement configuration:** batch 1, 1 thread, pinned to CPU 2; i7-1260P.
- **Latency vs cost metrics** (exhaustive 1,296-policy tables, Spearman ρ):

| Table | ρ (param reduction vs latency gain) | ρ (MAC reduction vs latency gain) | Sparser-but-slower pairs | Max speed-up |
|---|---|---|---|---|
| SimpleCNN C10 | 0.32 | 0.89 | 39.05% | 2.61× |
| SimpleCNN C100 | 0.31 | 0.89 | 39.4% | 2.59× |
| ResNet-8 C10 | 0.55 | 0.98 | 30.61% | 1.67× |
| ResNet-8 C100 | 0.54 | 0.98 | 30.8% | 1.72× |

- **GO/NO-GO verdicts:**
  - **SimpleCNN:** proxy validity rho = 0.844 < 0.9 → **NO-GO**.
  - **ResNet-8:** proxy rho = 0.980 → GO; τ_proxy = 0.9601 → GO; latency-room rule (≥ 1 proxy-feasible policy at ≥ 1.10×) → **NOT MET**, best 1.098×, so **"PPO: NO-GO. STOP"**.
  - No hardware-aware RL was run.
- **Headroom:**
  - SimpleCNN: up to 2.61× only **without** an accuracy constraint. No accuracy-feasible SimpleCNN headroom was established.
  - ResNet-8: under the accuracy constraint, < 10% latency headroom (best 1.098×).
- **Allowed:**
  - "Parameter reduction is a weak proxy for CPU latency (ρ 0.31–0.55); MAC reduction is a strong one (ρ 0.89–0.98); in 31–39% of policy pairs the sparser policy is slower."
  - "Measured latency remains the deployment-grounded metric."
- **Forbidden:**
  - "SimpleCNN has large accuracy-feasible headroom".
  - "hardware-aware RL improved latency".
  - Any Pearson value (none is recorded).

## 8. Phase 7A: sustainability evaluation (structured transfer of the Phase-5 policy)

- **Branch:** `phase7a-sustainability`.
- **Commits:** `b8c27665` (pre-registration), `aaec0dbe` (integrity, 5-epoch recovery, frozen models), `64de5104` (RAPL backend verified; pilot CV 2.30% / 1.74% → N = 100,000), `24fefd32` (sessions, statistics, interpretation; frozen).
- **Sources:** `results/phase7a_sustainability/preregistration.json`, `selected_policies.json`, `accuracy_results.csv`, `integrity_checks.json`, `energy_summary.csv`, `latency_summary.csv`, `pairwise_comparisons.csv`, `sustainability_summary.json`, `primary_table.md`, `interpretation.md`, `figures/fig_A_accuracy_vs_energy.png`, `figures/fig_B_normalized_efficiency.png`.
- **Design:**
  - CIFAR-10 only, SimpleCNN and ResNet-8.
  - PPO-Structured = the modal Phase-5 C1 policy translated to channel counts (L1 structured, Paper-3 backend). SimpleCNN [0,2,5,5] → widths [32,51,51,102]; ResNet-8 [1,1,1,3]; the ResNet-8 mapping is approximate.
  - Comparators: Uniform (MAC-matched) and LAMP-structured.
  - Recovery: equal 5-epoch Phase-6 protocol.
  - Energy: RAPL package via Windows EMI; 5 sessions × N = 100,000; gates AC power and CPU load ≤ 15%.
- **Accuracy:** the criterion split is **VAL-RL** (±0.5 pp). The test set was not used.

| Architecture | Method | VAL-RL / VAL-SELECT | ΔVAL-RL (pp) | Params ↓ | MACs ↓ | Latency (ms) | J per 1,000 | Energy ↓ | EDP |
|---|---|---|---|---|---|---|---|---|---|
| SimpleCNN | Dense | 80.60 / 80.00 | 0 | – | – | 0.7313 | 11.054 | – | 1.000 |
| SimpleCNN | PPO | 75.52 / 76.00 | −5.08 | 80.1% | 42.6% | 0.5711 | 8.541 | 22.8% (5/5) | 0.603 |
| SimpleCNN | Uniform | 79.12 / 78.48 | −1.48 | 44.7% | 42.5% | 0.5341 | 7.906 | 27.9% (5/5) | 0.518 |
| SimpleCNN | LAMP | 78.84 / 79.44 | −1.76 | 65.5% | 19.7% | 0.6579 | 9.760 | 12.1% (5/5) | 0.783 |
| ResNet-8 | Dense | 78.40 / 78.52 | 0 | – | – | 0.6984 | 10.882 | – | 1.000 |
| ResNet-8 | PPO | 75.16 / 74.72 | −3.24 | 25.0% | 21.5% | 0.6400 | 9.592 | 13.4% (5/5) | 0.791 |
| ResNet-8 | Uniform | 75.60 / 76.00 | −2.80 | 16.1% | 19.3% | 0.6459 | 9.645 | 11.8% (5/5) | 0.818 |
| ResNet-8 | LAMP | 75.24 / 75.72 | −3.16 | 25.0% | 18.2% | 0.6490 | 9.819 | 8.9% (5/5) | 0.828 |

- **Verdicts:**
  - SimpleCNN: **Case B + Case D (PPO vs Uniform)**. Uniform dominated PPO: +3.60 pp accuracy and lower energy.
  - ResNet-8: **Case B**; PPO, Uniform and LAMP show "no clear difference".
  - Primary Case A: **not supported** for either architecture.
  - **No pruned model of any method met ±0.5 pp.**
- **Allowed** (from the interpretation): "Physically compacting networks along the layer-wise allocation learned by the adaptive pruning policy reduced measured CPU package energy per inference: by 22.8% for SimpleCNN and 13.4% for ResNet-8, consistently across five sessions. After an identical 5-epoch recovery, however, these models lost 5.1 and 3.2 percentage points of validation accuracy."
- **Forbidden:**
  - "accuracy-preserving";
  - "green", "environmentally friendly", carbon wording;
  - claims about the masked Phase-5/6 models;
  - "PPO is more energy-efficient than Uniform";
  - test-set claims.

## 9. Phase 7B: energy-aware structured PPO (SimpleCNN, CIFAR-10)

- **Branch:** `phase7b-green-structured-ppo`. In thesis prose, call this "energy-aware structured PPO", not "green".
- **Commits:** `f5abe039` (design), `4de26d90` (samples), `ca0a48da` (validation tooling, sensitivity), `eac53bb1` (proxy and surrogate accepted), `0e101d55` (PPO, landscape, α, frozen pre-registration), `b77c5e19` (20 confirmatory seeds, 5-epoch recovery), `5600d091` (energy-pilot STOP, close-out).
- **Sources:** `results/phase7b_green_pruning/phase7b_final_preregistration.json`, `accuracy_proxy_metrics.json`, `energy_surrogate_metrics.json`, `selected_policies.json`, `final_summary.csv`, `memory_results.csv`, `recovery_results.csv`, `pareto_results.csv`, `statistics.json`, `interpretation.md`, `development_log.md`, `figures/fig_accuracy_planes.png`.
- **Method:**
  - PPO chooses per-unit structured ratios r ∈ {0, .1, .2, .3, .4, .6} directly.
  - State has 8 features, including centred sensitivity; a soft sensitivity prior is applied.
  - Reward is constraint-first: feasible R = 1 + 4·ŝ/s_max + 0.01·q; infeasible R = −10·(τ − q).
  - Feasibility uses the 2-epoch accuracy proxy (ρ 0.948, τ 0.98849, precision 0.655).
  - The energy surrogate is E ≈ 13.762·L − 0.117 (ρ 0.996, MdAPE 1.05%).
  - 20 confirmatory seeds (700–719).
- **Accuracy:** the criterion split is **VAL-SELECT**; 5-epoch recovery for every model.

| Model | Widths | VAL-SELECT (Δ pp) | VAL-RL | Param memory ↓ | MACs ↓ |
|---|---|---|---|---|---|
| Dense | [32,64,128,256] | 80.00 (0) | 80.60 | – (2,481,448 B) | – (10.85 M) |
| **PPO [2,2,2,5]** (5/20 seeds) | [26,51,102,102] | **78.88 (−1.12)** | 80.00 | **63.4%** | **36.0%** |
| Uniform r = 0.208 | [25,51,101,203] | 78.76 (−1.24) | 79.76 | 37.3% | 36.2% |
| LAMP s = 56.2% | [32,58,103,94] | 79.64 (**−0.36**) | 79.12 | 63.3% | 19.3% |
| Old PPO (7A) [0,2,5,5] | [32,51,51,102] | 76.00 (−4.00) | 75.52 | 80.1% | 42.6% |

- **All seed selections:** −1.08 to −1.28 pp VAL-SELECT.
- **Verdicts:**
  - **GREEN-SUCCESS = NO** (accuracy fails); **0/20 seeds**.
  - The case is undetermined between GREEN-C and "none", because energy was not measured.
  - Baseline-comparison verdicts are not assessable.
- **Energy pilot:** CV 11.36% (N = 100,000), then 4.01% (N = 200,000), against a 3% limit → **hardware validation STOPPED**. One interrupted attempt was declared void by user decision.
- **Allowed:** "The redesigned structured PPO reduced parameter memory by 63.4% and MACs by 36.0% at a 1.12 pp VAL-SELECT accuracy cost, improving on the 4.00 pp loss of the Phase-7A structured transfer at similar MACs."
- **Forbidden:**
  - "energy-efficient", "lower energy", "lower latency" for Phase 7B;
  - GREEN-SUCCESS;
  - "accuracy-preserving";
  - "beats Uniform or LAMP".

## 10. Phase 7B energy confirmation

- **Branch:** `phase7b-energy-confirmation`.
- **Commits:** `df233be0` (protocol, checkpoint verification), `0761ce1f` (qualification NOT qualified), `6b275096` (analysis).
- **Sources:** `results/phase7b_energy_confirmation/protocol.json`, `checkpoint_verification.json`, `environment.json`, `measurement_qualification.json`, `statistics.json`, `interpretation.md`.
- **Frozen:**
  - All 5 checkpoint hashes match.
  - Dense-only qualification (3 runs × 2 blocks, 180 s soak, N = 200,000): 11.42, 14.28, 12.96, 15.22, 9.51, 9.47 J per 1,000 → CV **19.92%** > 3% → **not qualified**.
  - No Dense-vs-PPO session was run. **ENERGY-CONFIRMED = NO.**
  - Diagnostic only: latency stayed within 1.042–1.052 ms while package power ranged 9.1–14.5 W, tracking a 72–92 °C thermal zone.
- **Allowed:** "Direct energy confirmation remains unresolved because the available measurement setup did not satisfy the stability requirement."
- **Forbidden:** any measured-energy or latency claim for Phase 7B, and any estimate from latency or the surrogate presented as energy.

## 11. Known differences between the earlier narrative and the frozen files (corrected here)

| Narrative item | Frozen fact |
|---|---|
| Phase 3 "at matched high sparsity" | Matched to each PPO run's own sparsity (9.5–55.1% mean). Uniform gain was large in 4 settings, +3.67 in C100 LeNet-5, inconclusive in C100 ResNet-8. |
| Phase 3 "several heuristic baselines" | Baselines were uniform, random, global magnitude, ERK and LAMP. Global MODERATE, ERK LIMITED. |
| "PPO does not consistently outperform LAMP" | LAMP significantly better: 5/6 (Phase 3), 6/6 (Phase 4), 6/6 (Phase 5); after fine-tuning still ahead in 3/6. |
| Phase 4 "archive selection solves retention" | True on validation ranking (0% → 75–100% top-10). Test effect mixed (−5.66 / −4.00 pp on ResNet-8); criterion D not met. |
| Phase 4 "relative sensitivity improves exploration" | Reward-rank AUC yes. V_RL-accuracy AUC reversed. |
| Phase 6 "±0.5 pp equivalence" | CI-inside-margin rule; 3/6 were already inside before fine-tuning; LAMP still significantly ahead in 3/6. |
| Phase 6 includes Uniform | It does not. |
| Paper 3 "SimpleCNN large latency headroom" | Unconstrained only (2.61×). No accuracy-feasible SimpleCNN headroom; its proxy was rejected (ρ 0.844). |
| Paper 3 "GO criterion failed" | SimpleCNN failed proxy validity; ResNet-8 passed proxy and threshold but failed latency room (1.098× vs 1.10×). |
| Phase 7A "≈ 5.1 / 3.2 pp" | VAL-RL (5.08 / 3.24). On VAL-SELECT: 4.00 / 3.80. No method met ±0.5 pp. |
| Phase 7B accuracy "validation" | VAL-SELECT 78.88 vs 80.00 (−1.12); VAL-RL 80.00 vs 80.60 (−0.60). |
| Phase 7A/7B dense SimpleCNN 80.60 / 80.00 | These are the 5-epoch recovered dense model on VAL-RL / VAL-SELECT, not the Phase-2 reference (77.32 validation / 77.03 test). |
| Phases 1–2 "literature review" | Not in the repository; a thesis writing task. |
| Folder and branch names containing "green" | Names only; thesis prose must not use "green". |

## 12. Branch heads at freeze

| Branch | Head |
|---|---|
| `main` | `52df18e9` |
| `paper3-hardware-aware` | `52764425` |
| `phase7a-sustainability` | `24fefd32` |
| `phase7b-green-structured-ppo` | `5600d091` |
| `phase7b-energy-confirmation` | `6b275096` |

The integration branch `thesis-final-integration` starts at `6b275096`, which contains all phases in its history.
