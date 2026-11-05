# Experimental design audit (internal support for Chapters 5 and 6)

- **Supports:** `chapter_04_experimental_setup.md`.
- **Use:** Chapters 5 and 6 must quote the split, matching rule and test status of every number from the row of its
  phase below.
- **Sources:** all paths are relative to `results/`. Every entry was read from the frozen file named, not from a
  summary. The numeric entries are re-checked mechanically by `thesis_final_writing/tools/audit_ch4.py`
  (100/100 PASS).

## Phase 2: dense references

| Field | Value |
|---|---|
| Dataset | CIFAR-10, CIFAR-100; train 45,000 / validation 5,000 / test 10,000 (randperm seed 42) |
| Architecture | SimpleCNN, LeNet-5, ResNet-8 |
| Split | full 5,000 validation selects the epoch; no VAL-RL / VAL-SELECT use |
| Methods | dense training: Adam 1e-3, cosine to 0 over 30 epochs, batch 128, crop + flip, best validation epoch |
| Seeds | 0, 1, 2 (seed 0 = reference); C10 SimpleCNN = historical `cnn_baseline_FIXED.pth` (10 epochs, final epoch) |
| Reward / selection | — / best full-validation epoch, ties earliest |
| Recovery | — |
| Metrics | validation and test accuracy; parameter counts; hashes |
| Test use | once per frozen, hashed checkpoint |
| Source | `cpu_dense_training_protocol.md`; `architecture_generalization/cpu_baselines.csv`; `reproducibility/*_split_manifest.json` |

## Phase 3: PPO vs baselines; sensitivity prior

| Field | Value |
|---|---|
| Dataset | CIFAR-10, CIFAR-100 |
| Architecture | SimpleCNN, LeNet-5, ResNet-8 (6 settings) |
| Split | configuration I: V_RL 3,000 (sensitivity, reward, landscape, trajectory metrics); V_SELECT not used |
| Methods | P0 plain PPO; P1 correct prior (0.5·S); P2 shuffled; P3 constant (mean S); baselines Uniform, Random (10 masks, seeds 1000–1009), Global, ERK, LAMP at PPO's exact zero count |
| Seeds | 300–319; 6 × 4 × 20 = 480 runs |
| Reward | R0 = 1.5·A/A_b + 0.01·ρ + 0.05·D (λ_s = 0.01) |
| Selection | terminal deterministic policy |
| Recovery | none (pruning only) |
| Metrics | test accuracy differences vs matched baselines; V_RL accuracy AUC (primary C); rank-score, utility AUC, destructive frequency; excess over matched global (D) |
| Statistics | exact sign-flip (2²⁰), 95% t-CI, bootstrap (10,000), d_z, Wilcoxon; Holm per family; STRONG / MODERATE / LIMITED / NOT SUPPORTED, REVERSED |
| Test use | once per frozen final policy and per matched baseline, after all runs; confirmatory |
| Source | `phase3_preregistration.md`; `architecture_generalization/phase3_frozen_inputs.json`, `phase3_all_runs.csv`, `phase3_statistics.csv`, `phase3_classification.json` |

## Phase 4: reward redesign, centred prior, archive

| Field | Value |
|---|---|
| Dataset / architecture | as Phase 3 |
| Split | configuration I: V_RL (reward, training, Stage A/B selections); V_SELECT 2,000 (archive re-ranking, held-out ranks, Stage-C selection) |
| Methods | Stage A rewards R0, R1a, R1b, R2 (P0, S0); Stage B priors P0–P5 (P2 = centred 1.0·(S − S̄)); Stage C S0, S1, S2, S2A (S2 = elite behavioural cloning, K = 5, η = 0.1) |
| Stage outcomes | R0 retained (no qualifying reward); P2 (P1 and P2 both eligible, tie → larger mean d_z); S2A. Final F = (R0, P2, S2A); control K = (R0, P0, S0) |
| Seeds | 400–419; 480 + 600 + 120 = 1,200 runs |
| Reward | R0 (final), R1a/R1b/R2 (Stage A only) |
| Selection | terminal (S0, S2) or archive: top 10 by V_RL reward, maximum V_SELECT score (S1, S2A) |
| Recovery | none |
| Metrics | final-selected / best-seen / median rank, reward-rank AUC, V_RL AUC, utility AUC, destructive frequency, top-k reach, first entry, V_SELECT rank, frontier regret; test excess over matched LAMP and global |
| Statistics | as Phase 3, plus Stage-A bootstrap ratio and one-sided non-inferiority (margin 0.5 pp) of F vs LAMP |
| Matching | LAMP and Global at every distinct zero count; Uniform, ERK and Random for F and K |
| Test use | locked until all stage selections and final lists were frozen; then once per distinct model; confirmatory |
| Source | `phase4/phase4_preregistration.md`, `phase4_stage_selection.json`, `phase4_all_runs.csv`, `phase4_config.json` |

## Phase 5: accuracy-constrained reward

| Field | Value |
|---|---|
| Dataset / architecture | as Phase 3 |
| Split | **configuration II**: VAL-RL 2,500 (sensitivity recomputed, τ choice, reward, training, archive); VAL-SELECT 2,500 (frozen-candidate selection only) |
| Methods | C0 = R0 + P2 + archive (max R0 on VAL-SELECT); C1 = R_c (τ = 0.98) + P2 + constrained selection; C2 = R_c, no prior; no behavioural cloning |
| Seeds | 500–519; 6 × 3 × 20 = 360 runs |
| Reward | R_c = 1 + ρ/100 + 0.01·q if q ≥ 0.98, else −10·(0.98 − q) (C1, C2); R0 (C0) |
| Selection | archive of 360 runs frozen before VAL-SELECT; C1 and C2: maximum sparsity among q_SELECT ≥ 0.98 (ties, fallback); final list committed before test |
| Recovery | none |
| Metrics | constrained rank (best-seen, selected), reach, first entry, reward-rank AUC, VAL-RL AUC, destructive rate, feasible-sample rate; q on VAL-RL, VAL-SELECT and test; test accuracy vs LAMP and global |
| Statistics | exact sign-flip, Holm per family (H1/H4, H3-primary 24, H3-secondary 18, H5 6 + 6); C1 − global descriptive |
| Matching | LAMP and Global at the selected policy's exact zero count |
| Test use | after the final-policy list was committed; confirmatory; feasibility on test recorded separately |
| Source | `phase5/phase5_design_commitments.md`, `phase5_preregistration.md`, `phase5_config.json`, `splits/phase5_split_manifest.json`, `phase5_all_runs.csv` |

## Phase 6: equal fixed-mask recovery

| Field | Value |
|---|---|
| Dataset / architecture | as Phase 3 |
| Split | training 45,000 updates weights; VAL-RL and VAL-SELECT (II) recorded per epoch, no selection |
| Methods | PPO (frozen Phase-5 C1), LAMP, Global at PPO's zero count (mismatch 0 in 120/120), dense control. **No Uniform, ERK, Random, C0 or C2** |
| Seeds | policy seeds 500–519; fine-tuning seed = policy seed + 100 (600–619), shared within each four-run group; 360 + 120 = 480 runs, plus 6 reproduction runs |
| Reward / selection | — / none: the epoch-5 model |
| Recovery | Adam 1e-4, cosine to 0 over E = 5 (timing rule: projected 29.44 h ≤ 30 h), batch 128, crop + flip, BN train mode, fixed mask asserted every epoch |
| Metrics | A_pre, A_post, recovery, dense gap, G_pre, G_post, ΔG, retention, retention gap, pruning-specific recovery |
| Statistics | exact sign-flip, Holm per family; ±0.5 pp CI-inclusion rule (not a formal equivalence test); H4 unpaired permutation (100,000) + Welch CI; cluster-level sensitivity analysis |
| Test use | after the 480 checkpoints were frozen and committed; pre-fine-tuning test must equal the Phase-5 record; confirmatory, but not fully independent (tests read in Phases 2–5) |
| Source | `phase6/phase6_design_commitments.md`, `phase6_preregistration.md`, `phase6_config.json`, `phase6_matched_masks.csv`, `phase6_all_runs.csv` |

## Paper 3: structured CPU-latency analysis

| Field | Value |
|---|---|
| Dataset | latency tables: CIFAR-10 and CIFAR-100 output sizes; accuracy and proxy: CIFAR-10 only |
| Architecture | SimpleCNN, ResNet-8 (structured units; ResNet-8 stem = coupled residual pruning group) |
| Split | VAL-RL (II): recovered accuracy, proxy validation, threshold calibration; VAL-SELECT stored, not analysed |
| Methods | H0 calibration (configurations A/B/C); H1 50 checks; H2-A four exhaustive tables, 4 × 1,296 = 5,184 shapes; 1-epoch proxy validation (156 SimpleCNN, 31 ResNet-8 pairs); ResNet-8 threshold calibration (48 policies); H2-C 5-epoch landscape stopped for compute (126 of 5,184) |
| Seeds | recovery seed 600 (data generator 600 + 10⁶) for every policy; table order seed 20261024 |
| Reward / selection | hardware-aware PPO planned but **not run** (proxy NO-GO for SimpleCNN; latency room not met for ResNet-8) |
| Recovery | Phase-6 protocol, E = 5 (proxy: first epoch of the same run) |
| Metrics | latency (median of session medians), p95, IQR, throughput, speed-up, latency gain g_T, parameter reduction, MAC reduction, FLOPs |
| Statistics | Spearman correlations; sparser-but-slower pair fractions; H0 SNR rule; proxy Spearman, MAE, bias, confusion |
| Latency protocol | batch 1, 1 + 1 threads, CPU 2, 50 warm-up / 300 timed, 5 fresh sessions, anchor every 25 shapes, drift ≤ 10% (4 attempts repeated), AC, load ≤ 15% |
| Test use | not read |
| Source | `paper3/h0_protocol.md`, `paper3_preregistration.md`, `paper3_proxy_protocol.md`, `resnet8_threshold_calibration_protocol.md`, `h2_latency_freeze.json`, `paper3_proxy_decision.json`, `h2c_stop_record.md` |

## Phase 7A: structured transfer and measured energy

| Field | Value |
|---|---|
| Dataset / architecture | CIFAR-10; SimpleCNN, ResNet-8 |
| Split | VAL-RL (II) primary accuracy criterion (±0.5 pp); VAL-SELECT secondary; known bias: the PPO policy was trained on VAL-RL and selected on VAL-SELECT |
| Methods | Dense (recovered); PPO-Structured (modal Phase-5 C1 policy, ties → smaller vector); Uniform-Structured (MAC-matched, 0.001 grid); LAMP-Structured (Phase-6 LAMP mask → per-unit sparsity → structured ratios) |
| Seeds | recovery seed 600, generator 1000600; energy-session order seeded |
| Reward / selection | — (no new search) |
| Recovery | Phase-6 protocol, 5 epochs, identical for all 8 models |
| Metrics | VAL-RL / VAL-SELECT accuracy, parameter reduction, MACs, latency, E (J per 1,000), energy saving, EDP, storage |
| Energy protocol | RAPL package via EMI; warm-up 1,000; N = 100,000; 5 fresh sessions per architecture; 30 s idle; pilot CV ≤ 3% (met at N = 100,000) |
| Statistics | exact one-sided Wilcoxon vs dense (n = 5, min p 0.031); noise threshold 2 × dense CV; Cases A–D; comparison rule |
| Test use | not used |
| Source | `phase7a_sustainability/preregistration.json`, `selected_policies.json`, `environment.json`, `energy_pilot/`, `accuracy_results.csv` |

## Phase 7B: structured, energy-aware PPO

| Field | Value |
|---|---|
| Dataset / architecture | CIFAR-10; SimpleCNN only |
| Split | VAL-RL (II): structured sensitivity, proxy, landscape, reward, archive; VAL-SELECT: held out from search and selection, final accuracy criterion |
| Methods | new structured PPO (8-feature state, centred structured prior β = 1.0); comparators Dense, Uniform-Structured (MAC-matched), LAMP-Structured (parameter-count-matched), old PPO (Phase-7A checkpoint, read only) |
| Seeds | development 900–904 (α tuning); confirmatory 700–719 (20, all reported) |
| Reward | R_7B: 1 + 4·ŝ/s_max + 0.01·q̂ if q̂ ≥ τ̂ = 0.98849, else −10·(τ̂ − q̂); ŝ from Ê = −0.117 + 13.762·T_table |
| Selection | feasible archive; lowest Ê with 1% tie band → higher q̂, fewer MACs, lower memory, smaller vector; primary = modal selection over 20 seeds |
| Recovery | proxy: separate 2-epoch run; final: 5-epoch Phase-6 protocol for 11 final models |
| Metrics | VAL-SELECT accuracy (criterion), VAL-RL, parameter memory (bytes), serialised size, MACs, FLOPs; latency and energy planned |
| Success rule | joint deployment-efficiency criterion (internal name GREEN-SUCCESS): VAL-SELECT within 0.5 pp; memory ↓; MACs ↓; latency ↓ (median, 5/5); energy ↓ beyond noise (5/5, median saving > 2 × CV); no override |
| Development acceptance | proxy: Spearman ≥ 0.90, recall 1.00, precision ≥ 0.50, ≥ 5 feasible (2-epoch accepted, 49 policies); surrogate: Spearman ≥ 0.95, MdAPE ≤ 5%, low-quartile MdAPE ≤ 5% (candidate 1 failed, candidate 2 accepted; 200 calibration policies; calibration pilot CV ≤ 4% met at N = 20,000) |
| Energy stage | pilot CV ≤ 3%; N = 100,000 → 11.36%; doubled-N attempt void (load gate, user decision); rerun N = 200,000 → 4.01% → **STOP**; no session run |
| Test use | never read |
| Source | `phase7b_green_pruning/phase7b_design.md`, `phase7b_final_preregistration.json`, `accuracy_proxy_metrics.json`, `energy_surrogate_metrics.json`, `_energy_calibration/`, `_energy_final/`, `development_log.md`, `final_summary.csv` |

## Phase 7B energy confirmation

| Field | Value |
|---|---|
| Dataset / architecture | CIFAR-10; SimpleCNN (frozen 7B models; 5 checkpoint hashes verified) |
| Split | none measured; accuracy taken read only from Phase 7B (VAL-SELECT) |
| Methods | primary Dense vs new PPO [2,2,2,5]; secondary Uniform, LAMP, old PPO (only if qualified) |
| Seeds | session-order permutation seeds 20261103 (primary) and 20261104 (secondary) |
| Protocol | 180 s soak; warm-up 2,000; N = 200,000; 5 s pause; fresh child process per run; void rules (load > 15%, off AC; never on values); > 3 voids → STOP |
| Qualification | Dense only, 3 runs × 2 blocks, CV ≤ 3%, one attempt → **CV 19.92%, not qualified → STOP** before any paired session |
| Planned statistics | exact Wilcoxon (one- and two-sided), exact sign test; t-test excluded; ENERGY-CONFIRMED rule (qualified; ≥ 8/10; median > noise; leave-one-out) |
| Test use | none |
| Source | `phase7b_energy_confirmation/protocol.json`, `measurement_qualification.json`, `checkpoint_verification.json`, `environment.json` |

## Cross-phase cautions for Chapters 5 and 6

1. **Splits.**
   - Phases 3–4 accuracies are on configuration I; Phase 5 onward on configuration II. Never compare across them.
   - Phase 7A's criterion split is VAL-RL; Phase 7B's is VAL-SELECT.
2. **Labels.** "P2" means the shuffled prior in Phase 3 and the centred prior in Phases 4–5.
3. **Matching.**
   - Phases 3–6 match exact zero counts.
   - Phase 7A Uniform is MAC-matched; Phase 7A LAMP is zero-count-derived.
   - Phase 7B Uniform is MAC-matched; Phase 7B LAMP is parameter-count-matched.
4. **The Phase 6 ±0.5 pp rule** is CI inclusion, not formal equivalence. Phase 4 has a separate, formal one-sided
   non-inferiority test.
5. **Phase 7B has no measured latency or energy.** Ê is a search surrogate only.
6. **ResNet-8 units differ between the masked and structured definitions.** The Phase-7A mapping is approximate.
