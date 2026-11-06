# Table shortlist

Main text has 11 tables. The others go to an appendix or are omitted. Source files and commits are in
`result_freeze.md`.

## Main text

| ID | Section | Content | Source |
|---|---|---|---|
| **T1** | 4.2 | Dense reference accuracy (validation and test, seed 0; 3-seed mean ± SD), 6 settings | `architecture_generalization/cpu_baselines.csv` |
| **T2** | 4.4 | Phase overview: purpose, runs, seeds, settings, primary criterion, verdict (Phases 3–7B-EC) | phase summaries |
| **T3** | 5.2 | Phase 3 PPO vs Uniform, Random, Global, ERK, LAMP: supported/opposite of 6, class, mean pp range; one row per baseline | `phase3_cross_setting_summary.csv`, `phase3_classification.json`, `phase3_statistics.csv` |
| **T4** | 5.3 | Sensitivity across phases: exploration endpoints vs final-accuracy endpoints (Phase 3 P1−P0, Phase 4 P2, Phase 5 C1−C2) | `phase3_*`, `phase4_prior_comparison.csv`, `phase5_c1_vs_c2.csv` |
| **T5** | 5.4 | Retention: S0 vs S2A (best-seen top-10 reach, selected top-10/top-20 reach, median rank, Δ test) per setting | `phase4_search_quality.csv` |
| **T6** | 5.5 | Reward alignment: ResNet-8 sparsity and accuracy drop under K/F (Phase 4) and C0/C1 (Phase 5); feasibility of C1 on VAL-RL / VAL-SELECT / test | `phase4_*`, `phase5_constraint_results.csv` |
| **T7** | 5.6 | Phase 6: PPO−LAMP gap before → after fine-tuning [95% CI], inside ±0.5?; PPO−Global after; mean pre/post accuracy per method | `phase6_ppo_vs_lamp.csv`, `phase6_ppo_vs_global.csv`, `phase6_pre_post_accuracy.csv` |
| **T8** | 6.2 | Paper 3: Spearman ρ (param vs latency, MAC vs latency), sparser-but-slower pairs, max speed-up; 4 tables | `paper3/h2_e2_latency_correlations.csv`, manuscript Table II |
| **T9** | 6.3 | Phase 7A: per architecture × method, VAL-RL/VAL-SELECT, Δ, params↓, MACs↓, latency, J/1,000, energy↓ (sessions lower), EDP | `phase7a_sustainability/primary_table.md`, `energy_summary.csv` |
| **T10** | 6.4–6.5 | Phase 7B: Dense / new PPO / Uniform / LAMP / old PPO, with widths, VAL-SELECT (Δ), VAL-RL, memory↓, MACs↓, and an energy-status column ("not measured: qualification CV 19.92%") | `phase7b_green_pruning/final_summary.csv`, `memory_results.csv`, `phase7b_energy_confirmation/statistics.json` |
| **T11** | 6.6 | Three-level assessment: compression / computation / operational energy × Phase 7A / 7B, each marked established, measured with trade-off, or unresolved | synthesis of T9 and T10 |

## Appendix

| ID | Content |
|---|---|
| A1 | Phase 3 per-setting statistics (all baselines, Holm p, win/tie/loss, McNemar) |
| A2 | Phase 4 criteria A–E per setting; Stage A/B/C selection tables (R0, P2, S2A) |
| A3 | Phase 4 F vs baselines per setting (LAMP, Global, Uniform, ERK) |
| A4 | Phase 5 criteria A–F per setting; τ selection table |
| A5 | Phase 5 C1 vs LAMP and vs Global per setting |
| A6 | Phase 6 per-method, per-setting pre/post accuracy; integrity 20/20 |
| A7 | Paper 3 GO/NO-GO decisions (H0 SNR, proxy ρ, τ_proxy, latency room) |
| A8 | Phase 7A pairwise comparisons (PPO vs Uniform/LAMP) and integrity checks |
| A9 | Phase 7B proxy validation (ρ, τ, recall, precision), energy surrogate candidates, α tuning, per-seed selections (20 seeds) |
| A10 | Energy pilots and qualification: 7A pilot, 7B pilots (N = 100k, 200k), 7B-EC qualification blocks with power, PP0, temperature, latency |
| A11 | Legacy pre-audit timeline (one line per study, leakage status) |

## Omit

- Legacy pre-audit result tables as evidence (reward sweep, soft prior, Level A/B/C studies): history only (A11).
- Paper 3 H2-C partial recovery runs (stopped for compute).
- Phase 7B surrogate-predicted energies as a results table. Mention them only in the A9 method validation.
- Raw training curves and per-run CSVs.
- Any carbon or kWh projection table.
