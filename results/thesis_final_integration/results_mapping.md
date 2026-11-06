# Result-to-chapter mapping

Each Results subsection lists the RQ it answers, its source phase, the strongest numbers to report, the negative or
partial result that must stay visible, and the table and figure IDs (from `table_shortlist.md` and
`figure_shortlist.md`). Exact sources and commits are in `result_freeze.md`.

## Overview

| Phase | RQ | Chapter / section |
|---|---|---|
| 1 audit, 2 splits and references | setup | 4.1–4.2, 5.1 |
| 3 multi-architecture PPO vs baselines | RQ1, RQ2 | 5.2, 5.3 |
| 4 centred prior, archive, reward redesign | RQ2, RQ3, RQ4 | 5.3, 5.4, 5.5 |
| 5 constrained reward | RQ1, RQ2, RQ4 | 5.2 (LAMP/global), 5.3, 5.5 |
| 6 equal fine-tuning | RQ1, RQ5 | 5.6 |
| Paper 3 latency analysis | RQ6 | 6.2 |
| 7A structured transfer and energy | RQ6 | 6.3 |
| 7B energy-aware structured PPO | RQ6 (and RQ2/RQ4 inside the extension) | 6.4 |
| 7B energy confirmation | RQ6 | 6.5 |
| Legacy pre-audit studies | none | Appendix F only |

## Chapter 5: core framework

### 5.1 Baseline models (setup; T1)
- Dense references, seed 0 test accuracy:
  - CIFAR-10: SimpleCNN 77.03, LeNet-5 64.96, ResNet-8 78.69;
  - CIFAR-100: SimpleCNN 50.50, LeNet-5 32.28, ResNet-8 43.99.
- 3-seed spread is small (for example ResNet-8 C10 78.69 ± 0.51).
- **Must show:** the CIFAR-10 SimpleCNN reference uses the older 10-epoch recipe.

### 5.2 PPO vs Uniform and other baselines (RQ1; Phase 3, with Phase 4–5 LAMP rows; T3, F3, F4)
- **Strongest:**
  - PPO > Uniform STRONG (5/6): +31.36, +29.69, +16.29, +14.17, +3.67 pp.
  - PPO > Random STRONG (6/6): +27.9 to +64.4 pp.
  - Allocations are visibly non-uniform (F4).
- **Must show:**
  - Global magnitude: MODERATE (3/6).
  - ERK: LIMITED, 1 opposite.
  - LAMP better in 5/6 in Phase 3 (−0.55 to −1.64 pp) and 6/6 in Phases 4–5 (Phase 5: −0.26 to −0.53 pp).
  - C100 ResNet-8: PPO chose the uniform policy in 19/20 runs.
- **Wording:** compare "at the same sparsity as each PPO run", not "at matched high sparsity".

### 5.3 Sensitivity-guided exploration (RQ2; Phases 3–5; T4, F5)
- **Strongest:**
  - Phase 3: secondary exploration endpoints 6/6 STRONG.
  - Phase 4: centred prior P2 reward-rank AUC STRONG (5/6).
  - Phase 5: C1 vs C2 exploration supported in 5/6, with 0 of 24 primary tests opposite; destructive-policy rate
    lower in 6/6.
- **Must show:**
  - Final accuracy not improved: Phase 3 0/6; Phase 4 final rank 0/6; Phase 5 C1 is 0.05–0.16 pp below C2 on test.
  - Phase 4 V_RL-accuracy AUC reversed.
  - On LeNet-5 (Phase 3), shuffled and constant priors helped as much as the correct prior.

### 5.4 Policy retention and archive selection (RQ3; Phase 4; T5, F6)
- **Strongest:**
  - Without the archive, the best policy seen reaches the top 10 in 95–100% of runs, but the terminal policy reaches
    it in 0% of settings (median selected rank 65–173).
  - With the archive, 75–100% reach the top 10 and 100% the top 20 (median rank 1–10).
- **Must show:**
  - Test effect of the archive: +0.14, +0.90, −5.66, +0.22, +0.45, −4.00 pp (ResNet-8 losses).
  - Criterion D not met.
  - The Stage-C criterion favours the archive by construction.

### 5.5 Reward alignment and accuracy constraints (RQ4; Phases 4–5; T6, F7)
- **Strongest:**
  - Phase 4 diagnosis: no redesigned reward met criterion A. With F, ResNet-8 sparsity went 29.0% → 47.8% and the
    test drop reached 7.86 pp (C10).
  - Phase 5 constraint: ResNet-8 drop 7.32 → 1.18 pp (C10) and 4.05 → 1.16 pp (C100); C1 − C0 = +6.14 and
    +2.89 pp.
  - C1 vs matched global magnitude: 5/6 (descriptive).
- **Must show:**
  - Criteria A and E PARTIALLY MET; SimpleCNN slightly worse (−0.31 / −0.21 pp).
  - Test feasibility 67.5% vs 99.2% on VAL-SELECT; C100 ResNet-8 0%.
  - Search criterion C only PARTIALLY MET (76.7% vs ≥ 80%).

### 5.6 Equal fine-tuning and final comparison (RQ5; Phase 6; T7, F8)
- **Strongest:**
  - Criteria A–F all MET.
  - Mean test accuracy before → after fine-tuning: PPO 57.08 → 58.38, LAMP 57.48 → 58.48, Global 55.08 → 58.36
    (dense 58.51).
  - PPO–LAMP CI inside ±0.5 pp in 6/6; gap shrank significantly in 5/6.
  - No regrowth (360/360); 6/6 repeats bit-identical.
- **Must show:**
  - LAMP still significantly ahead in 3/6 (0.14–0.22 pp).
  - The rule is CI-inside-margin, not formal equivalence.
  - 3/6 settings were already inside ±0.5 before fine-tuning.
  - PPO's lead over Global (up to +6.51 pp before fine-tuning) vanishes (−0.16 to +0.12 after).
  - No Uniform in Phase 6.

## Chapter 6: deployment and sustainability extension

### 6.2 Hardware efficiency: sparsity is not speed (RQ6; Paper 3; T8, F9)
- **Strongest:**
  - Spearman ρ, parameter reduction vs latency gain: 0.31–0.55.
  - Spearman ρ, MAC reduction vs latency gain: 0.89–0.98.
  - Sparser-but-slower pairs: 31–39%.
  - Maximum speed-up: SimpleCNN 2.61× (unconstrained); ResNet-8 1.67–1.72×.
- **Must show:**
  - ResNet-8 accuracy-feasible headroom only 1.098×, so the latency-room rule was NOT MET.
  - SimpleCNN proxy rejected (ρ 0.844), so no accuracy-feasible SimpleCNN headroom exists.
  - No hardware-aware RL was run.

### 6.3 Structured transfer and measured energy (RQ6; Phase 7A; T9, F10)
- **Strongest:**
  - Measured RAPL package energy −22.8% (SimpleCNN) and −13.4% (ResNet-8), lower than dense in 5/5 sessions
    (one-sided Wilcoxon p = 0.031).
  - EDP 0.603 / 0.791.
  - Pilot CV 2.30% / 1.74% (stable).
- **Must show:**
  - VAL-RL accuracy −5.08 / −3.24 pp; Case B, not A.
  - SimpleCNN Case D: Uniform at matched MACs was 3.60 pp more accurate and used less energy (27.9% saving).
  - No method met ±0.5 pp.

### 6.4 Energy-aware structured PPO (Phase 7B; T10, F11)
- **Strongest:**
  - [2,2,2,5] → widths [26,51,102,102]: memory −63.4%, MACs −36.0%, VAL-SELECT −1.12 pp, against −4.00 pp for the
    Phase-7A structured PPO.
  - On the accuracy–memory Pareto front.
  - Proxy ρ 0.948; energy surrogate ρ 0.996 (as model-selection tools only).
- **Must show:**
  - 0/20 seeds reach GREEN-SUCCESS; every selection is −1.08 to −1.28 pp.
  - LAMP-structured is the only model within 0.5 pp (−0.36 pp, memory −63.3%, MACs −19.3%).
  - Selections sit on the proxy feasibility boundary (q ≈ 0.989 vs τ 0.98849; proxy precision 0.655).

### 6.5 Energy-measurement qualification (Phase 7B and 7B-EC; part of T10 or appendix table E)
- **Strongest** (as an honest finding): a reproducible, pre-registered measurement-qualification procedure.
- **Must show:**
  - Pilot CV 11.36% → 4.01% (Phase 7B).
  - Qualification CV 19.92% (7B-EC), so no Dense-vs-PPO energy measurement exists.
  - Diagnostic: latency constant within 1% while package power ranged 9.1–14.5 W with temperature.
  - Phase 7A's pilot was stable on the same machine under its earlier conditions; state this contrast without
    over-explaining.

### 6.6 Three-level sustainability assessment (T11)
- Compression: established for 7A and 7B.
- Computation: MACs established for 7A and 7B; latency measured for 7A only.
- Operational energy: measured for 7A only (with accuracy loss); unresolved for 7B.
- See `sustainability_positioning.md`.

## What is deliberately not reported as a result
- Legacy pre-audit numbers (old README, leaky pipeline).
- Phase 7B surrogate energy predictions presented as savings.
- Paper 3 H2-C partial runs, which were stopped for compute.
- Any test-set number for Phases 7A and 7B (none exists).
