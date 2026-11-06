# Appendices

The appendices give only the supporting values an examiner may need to verify statements in Chapters 3–6
without opening the code repository. Each table is generated from the frozen phase records by
`tools/build_appendices.py` and names its source record.

## Appendix A — Architectures, Hyperparameters and Reproducibility

### A.1 Pruning units

**Table A.1. Pruning units of the three networks: weights, share of prunable weights and MACs per image (CIFAR-10 output layer).**

| Network | Unit | Role | Weights | Share of prunable | MACs per image |
|---|---|---|---|---|---|
| SimpleCNN | features.0 | pruned (PPO decision) | 864 | 0.14% | 884,736 |
| SimpleCNN | features.3 | pruned (PPO decision) | 18,432 | 2.99% | 4,718,592 |
| SimpleCNN | features.6 | pruned (PPO decision) | 73,728 | 11.94% | 4,718,592 |
| SimpleCNN | classifier.1 | pruned (PPO decision) | 524,288 | 84.93% | 524,288 |
| SimpleCNN | classifier.3 | never pruned | 2,560 | — | 2,560 |
| LeNet-5 | conv1 | pruned (PPO decision) | 450 | 0.74% | 352,800 |
| LeNet-5 | conv2 | pruned (PPO decision) | 2,400 | 3.94% | 240,000 |
| LeNet-5 | fc1 | pruned (PPO decision) | 48,000 | 78.78% | 48,000 |
| LeNet-5 | fc2 | pruned (PPO decision) | 10,080 | 16.54% | 10,080 |
| LeNet-5 | classifier.5 | never pruned | 840 | — | 840 |
| ResNet-8 | stem | pruned (PPO decision) | 432 | 0.56% | 442,368 |
| ResNet-8 | stage1 | pruned (PPO decision) | 4,608 | 6.01% | 4,718,592 |
| ResNet-8 | stage2 | pruned (PPO decision) | 14,336 | 18.69% | 3,670,016 |
| ResNet-8 | stage3 | pruned (PPO decision) | 57,344 | 74.74% | 3,670,016 |
| ResNet-8 | fc | never pruned | 640 | — | 640 |

*Source: `cpu_prunable_units.csv`. The PPO hyper-parameters are in Table 4.6 and the equal recovery protocol in Section 4.13.*

### A.2 Reproducibility

Every experiment was pre-registered and frozen before its confirmatory runs; Table 4.10 lists the governing
records and the commits that first added them. The complete per-seed experimental records, per-session
measurement records, machine-readable result files (CSV and JSON), checkpoint hashes and audit scripts are
retained in the associated research repository, organised by phase: `architecture_generalization` (Phases 2–3),
`phase4`, `phase5`, `phase6`, `paper3` (latency), `phase7a_sustainability`, the Phase-7B folder and
`phase7b_energy_confirmation`; the thesis sources and audits are in `thesis_final_writing`.

## Appendix B — Supplementary Core-Pruning Results

### B.1 Phase 3 absolute accuracies

**Table B.1. Phase 3 mean test accuracy (%) of PPO (P0) and the five baselines built at each PPO run's zero count (20 runs).**

| Setting (dense test) | P0 sparsity (%) | PPO (P0) | Uniform | Random | Global | ERK | LAMP |
|---|---|---|---|---|---|---|---|
| C10 SimpleCNN (77.03) | 55.06 | 76.44 | 45.08 | 16.84 | 76.51 | 75.70 | 76.99 |
| C10 LeNet-5 (64.96) | 51.54 | 61.67 | 45.38 | 12.99 | 58.94 | 59.74 | 63.31 |
| C10 ResNet-8 (78.69) | 28.35 | 76.95 | 62.79 | 12.59 | 69.57 | 77.01 | 76.80 |
| C100 SimpleCNN (50.50) | 53.28 | 49.66 | 19.96 | 2.37 | 49.67 | 45.31 | 50.30 |
| C100 LeNet-5 (32.28) | 34.59 | 31.19 | 27.51 | 3.29 | 31.32 | 30.88 | 31.86 |
| C100 ResNet-8 (43.99) | 9.46 | 43.01 | 42.78 | 4.25 | 42.74 | 43.81 | 43.76 |

*Source: `phase3_summary.csv`. Paired differences and their statistics are in Table 5.2.*

### B.2 Sensitivity vectors

**Table B.2. Normalised loss sensitivity S_l of the four units by setting and the Phase-5 centred prior coefficients.**

| Setting | Configuration I (Phases 3–4) | Configuration II (Phase 5) | Phase-5 coefficients c_l = S_l − S̄ |
|---|---|---|---|
| C10 SimpleCNN | 1.000, 0.032, 0.009, 0.000 | 1.000, 0.030, 0.012, 0.000 | 0.740, −0.230, −0.249, −0.260 |
| C10 LeNet-5 | 1.000, 0.422, 0.046, 0.000 | 1.000, 0.440, 0.042, 0.000 | 0.629, 0.070, −0.329, −0.371 |
| C10 ResNet-8 | 1.000, 0.138, 0.331, 0.000 | 1.000, 0.145, 0.320, 0.000 | 0.634, −0.221, −0.046, −0.366 |
| C100 SimpleCNN | 1.000, 0.103, 0.014, 0.000 | 1.000, 0.119, 0.010, 0.000 | 0.718, −0.164, −0.272, −0.282 |
| C100 LeNet-5 | 1.000, 0.587, 0.000, 0.086 | 1.000, 0.564, 0.000, 0.059 | 0.594, 0.159, −0.406, −0.347 |
| C100 ResNet-8 | 0.987, 0.310, 1.000, 0.000 | 1.000, 0.309, 0.997, 0.000 | 0.423, −0.268, 0.421, −0.577 |

*Sources: `phase3_frozen_inputs.json`; `phase5_config.json`.*

## Appendix C — Archive, Reward and Recovery Diagnostics

### C.1 Recovery gains under equal fine-tuning

**Table C.1. Phase 6 mean recovery (test accuracy after minus before equal fine-tuning, pp; 20 runs) by setting and method.**

| Setting (sparsity) | PPO | LAMP | Global magnitude | Dense control |
|---|---|---|---|---|
| C10 SimpleCNN (56.9%) | +3.28 | +3.01 | +3.35 | +3.05 |
| C10 LeNet-5 (38.6%) | +1.04 | +0.91 | +2.42 | +0.14 |
| C10 ResNet-8 (24.5%) | +1.23 | +0.70 | +7.67 | +0.14 |
| C100 SimpleCNN (53.1%) | +0.46 | +0.22 | +0.76 | +0.10 |
| C100 LeNet-5 (30.6%) | +0.57 | +0.44 | +0.81 | +0.11 |
| C100 ResNet-8 (15.6%) | +1.20 | +0.72 | +4.69 | +0.09 |

*Source: `phase6_pre_post_accuracy.csv`. Accuracies before and after recovery and the PPO–LAMP and PPO–Global gaps are in Table 5.7. Phase-4 search quality and Phase-5 per-run selections are summarised in Tables 5.3, 5.4 and 5.5; their run-level records are in the repository.*

## Appendix D — Hardware and Structured-Pruning Protocol

### D.1 Latency-measurement calibration

**Table D.1. H0 calibration: decision quantities of the three candidate measurement configurations.**

| Configuration | Signal (ms) | Noise (ms) | SNR | Relative reduction | Positive in all sessions | Usable |
|---|---|---|---|---|---|---|
| A | 0.4127 | 0.0335 | 12.32 | 61.6% | True | True |
| B | 0.1222 | 0.0287 | 4.26 | 31.8% | True | False |
| C | 15.4531 | 1.6070 | 9.62 | 63.2% | True | True |

*Source: `h0_decision.json`. Rule: SNR ≥ 5, reduction ≥ 10%, positive in every session; preference order A, C, B.*

### D.2 Latency-table sessions

**Table D.2. Exhaustive latency tables: accepted sessions, repeated attempts and anchor drift (limit 10%).**

| Latency table | Accepted sessions | Repeated attempts | Largest accepted drift | Largest rejected drift |
|---|---|---|---|---|
| ResNet-8, CIFAR-10 | 5 | 2 | 9.7% | 27.1% |
| ResNet-8, CIFAR-100 | 5 | 0 | 9.6% | — |
| SimpleCNN, CIFAR-10 | 5 | 0 | 9.4% | — |
| SimpleCNN, CIFAR-100 | 5 | 2 | 9.1% | 40.6% |

*Source: `h2_latency_freeze.json` (`session_log`); the per-session log is in the repository.*

## Appendix E — Energy Measurement and Qualification

### E.1 Phase 7A energy sessions

**Table E.1. Phase 7A energy: session summaries (five sessions per architecture).**

| Network | Method | Median J / 1,000 | CV over sessions | Median saving | Min–max saving | Sessions below dense | One-sided Wilcoxon p |
|---|---|---|---|---|---|---|---|
| SimpleCNN | dense | 11.054 | 2.14% | 0.0% | 0.0–0.0% | 0 | — |
| SimpleCNN | ppo | 8.541 | 1.83% | 22.8% | 19.5–25.7% | 5 | 0.031 |
| SimpleCNN | uniform | 7.906 | 1.54% | 27.9% | 27.3–31.2% | 5 | 0.031 |
| SimpleCNN | lamp | 9.760 | 3.59% | 12.1% | 6.8–15.1% | 5 | 0.031 |
| ResNet-8 | dense | 10.882 | 1.84% | 0.0% | 0.0–0.0% | 0 | — |
| ResNet-8 | ppo | 9.592 | 2.84% | 13.4% | 5.6–14.5% | 5 | 0.031 |
| ResNet-8 | uniform | 9.645 | 3.63% | 11.8% | 4.7–13.9% | 5 | 0.031 |
| ResNet-8 | lamp | 9.819 | 1.46% | 8.9% | 7.4–12.3% | 5 | 0.031 |

*Source: `energy_summary.csv`; per-block session records are in the repository.*

### E.2 Energy-measurement qualification

**Table E.2. Dense-only stability pilots and qualification (energy per 1,000 inferences, J; six dense blocks each).**

| Measurement | N per block | Dense blocks | CV | Limit | Decision |
|---|---|---|---|---|---|
| 7A pilot, SimpleCNN | 100,000 | 11.518, 11.150, 11.081, 11.253, 10.963, 10.768 | 2.30% | 3% | passed |
| 7A pilot, ResNet-8 | 100,000 | 10.793, 10.489, 10.555, 10.667, 10.358, 10.312 | 1.74% | 3% | passed |
| 7B surrogate-calibration pilot | 20,000 | 12.200, 11.198, 11.107, 11.729, 11.433, 11.420 | 3.47% | 4% | stable |
| 7B confirmatory pilot (N = 100,000) | 100,000 | 13.088, 11.414, 11.204, 15.125, 12.265, 12.273 | 11.36% | 3% | unstable |
| 7B confirmatory pilot (N = 200,000) | 200,000 | 11.814, 12.074, 11.039, 11.539, 11.025, 11.033 | 4.01% | 3% | unstable |
| 7B energy-confirmation qualification | 200,000 | 11.417, 14.278, 12.962, 15.221, 9.514, 9.473 | 19.92% | 3% | not qualified |

*Sources: `pilot_n100000.json` (Phase 7A); `pilot_n20000.json`, `statistics.json` (Phase 7B); `measurement_qualification.json` (energy confirmation). A doubled-N Phase-7B attempt stopped by the load gate was declared void by a recorded decision of the author and is not listed. The per-block diagnostics and the eleven recovered Phase-7B models (summarised in Table 6.4) are in the repository.*

