# Chapter structure, thesis story and methodology flow

## 1. Thesis story (narrative spine)

1. **Problem.** Uniform pruning applies one ratio to every layer, but layers differ in how much accuracy they lose
   when pruned.
2. **Initial solution.** Learn a layer-wise allocation with PPO, one action per layer unit.
3. **First result** (Phase 3). PPO allocations strongly beat uniform and random allocation at the same sparsity, but
   LAMP, a cheap layer-adaptive heuristic, is better in 5 of 6 settings.
4. **Sensitivity.** A sensitivity prior makes the search more efficient. The better search does not raise final
   accuracy (Phases 3–5).
5. **Problem found: retention** (Phase 4). PPO finds top-ranked policies but does not end on them; the terminal
   policy is in the top 10 in 0% of settings.
6. **Solution: archive selection.** Retention is restored on validation ranking (75–100% top 10).
7. **Next problem: reward misalignment** (Phase 4). With better search and retention, the accuracy–sparsity reward
   now reliably finds its own optimum, which over-prunes. ResNet-8 lost 7.86 pp.
8. **Solution: an accuracy-constrained, validation-aware reward** (Phase 5). The ResNet-8 drop falls to about 1.2 pp.
   Validation feasibility still does not guarantee test feasibility (67.5%).
9. **Fair comparison** (Phase 6). After equal 5-epoch fine-tuning, every method recovers most of its loss. PPO–LAMP
   falls inside ±0.5 pp in all 6 settings, with LAMP still ahead in 3. PPO's advantage is before fine-tuning.
10. **Deployment question** (Paper 3). Does compression become real CPU benefit? Parameter sparsity is a weak latency
    proxy; MACs are a strong one; measured latency is the deployment-grounded metric.
11. **Sustainability question** (Phase 7A). Structured realisation of the PPO allocation reduces measured CPU energy
    (−22.8% / −13.4%), but loses 5.1 / 3.2 pp accuracy. Uniform dominates on SimpleCNN.
12. **Redesign** (Phase 7B). PPO that learns structured, energy-aware policies directly cuts memory by 63.4% and MACs
    by 36.0% at −1.12 pp. That is much better accuracy retention than Phase 7A, but it still misses 0.5 pp, and
    direct energy could not be confirmed on the available hardware.
13. **Message.** Adaptive RL pruning is useful, but search quality alone is not enough. Reward alignment, policy
    retention, equal recovery, deployment-aware objectives and stable hardware measurement are all needed before a
    compression result can be called a deployment-efficiency result.

## 2. Methodology flow

### A. Core thesis framework (Phases 1–6; unstructured masks)

```
Dense reference model (Phase 2; leakage-guarded 45k/5k/10k split; V_RL / V_SELECT)
  → layer sensitivity analysis on V_RL (loss/accuracy change per unit and ratio)
  → PPO environment: state = per-unit features (+ sensitivity), action = ratio per unit
    from {0, 10, 20, 30, 40, 60}%, 4 units → 1,296 policies
  → centred sensitivity prior  c_u = β·(S_u − S̄)  (Phase 4: P2; Phase 3 tested correct/shuffled/constant)
  → pruning environment: magnitude masks applied per unit; the classifier is never pruned
  → reward: Phase 3–4 R0 (accuracy–sparsity trade-off)
           → Phase 5 constraint-first: R = 1 + ρ + 0.01·q if q ≥ 0.98, else −10·(0.98 − q)
  → archive selection: top-10 policies by V_RL reward, re-ranked on V_SELECT (Phase 4 S2A; Phase 5 archive)
  → accuracy-constrained selection: highest-sparsity candidate with q_VAL-SELECT ≥ 0.98 (Phase 5)
  → equal fixed-mask recovery: 5-epoch Adam fine-tuning, identical for every method (Phase 6)
  → evaluation: test accuracy once per frozen model; matched-sparsity baselines (Uniform, Random,
    Global, ERK, LAMP); pre-registered statistics
```

### B. Deployment and sustainability extension (Paper 3, Phases 7A–7B; added after the core framework)

```
Structured search space: channel/neuron removal keeping max(1, round((1 − r)·C)) units by L1 norm
  → exhaustive CPU latency tables (1,296 structured shapes; batch 1, 1 thread, pinned core)      [Paper 3]
  → cost-metric analysis: parameter reduction vs MAC reduction vs measured latency               [Paper 3]
  → structured transfer of the Phase-5 PPO allocation + equal 5-epoch recovery                   [Phase 7A]
  → direct RAPL package energy, J per 1,000 inferences, paired sessions, EDP                     [Phase 7A]
  → structured, energy-aware PPO: structured actions, centred sensitivity in state and prior,
    2-epoch accuracy proxy, latency-based energy surrogate, constraint-first reward              [Phase 7B]
  → memory (parameter bytes) / MACs / accuracy / Pareto; energy measurement qualification        [Phase 7B, 7B-EC]
```

**Presentation rule.** Introduce B as an extension motivated by the core framework's results. It reuses the
framework's ingredients (sensitivity, constraint-first reward, archive-style selection, equal recovery), but it was
not part of the original design. Do not draw a single pipeline figure that suggests energy-aware structured PPO was
planned from Phase 1.

## 3. Chapters

Total of about 105–120 pages, plus appendices. The suggested 8-chapter layout is kept, with one change. Results are
split into **Chapter 5 (core framework, RQ1–RQ5)** and **Chapter 6 (deployment and sustainability extension, RQ6)**.
Chapter 6 carries its own short method subsection, which keeps the core/extension boundary visible. Discussion
becomes Chapter 7.

### Chapter 1: Introduction (8–10 pages)
- **Purpose:** motivate layer-wise adaptive pruning and deployment efficiency; state RQ1–RQ6 and contributions
  C1–C6.
- **Subsections:**
  - 1.1 Motivation: model size, inference cost, deployment;
  - 1.2 Problem: uniform vs layer-adaptive pruning;
  - 1.3 Why reinforcement learning, and its known difficulties;
  - 1.4 Research questions;
  - 1.5 Contributions;
  - 1.6 Scope (CIFAR-scale, CPU, pre-registered design);
  - 1.7 Thesis outline.
- **Figures:** F1 (overall framework, conceptual).
- **Phases:** none (framing only).

### Chapter 2: Literature review (15–18 pages)
- **Purpose:** position the thesis.
- **Subsections:**
  - 2.1 Pruning taxonomy: unstructured vs structured; magnitude, global and layer-adaptive (LAMP, ERK);
  - 2.2 Sensitivity-based pruning;
  - 2.3 RL and AutoML for compression (AMC-style agents, PPO);
  - 2.4 Fine-tuning and recovery, and fair comparison;
  - 2.5 Hardware-aware compression: latency, MACs, sparsity–speed mismatch;
  - 2.6 Energy measurement and sustainable ML (RAPL, measurement pitfalls, carbon reporting caveats);
  - 2.7 Research gap.
- **Note:** this chapter is not in the repository yet. It is a writing task.

### Chapter 3: Proposed framework (14–16 pages)
- **Purpose:** define the core method (flow A).
- **Subsections:**
  - 3.1 Problem formulation (units, ratio set, 1,296-policy space);
  - 3.2 Sensitivity analysis and centring;
  - 3.3 MDP: state, action, episode;
  - 3.4 Sensitivity prior (correct/shuffled/constant; centred P2);
  - 3.5 Reward family: R0 and the constraint-first reward;
  - 3.6 Archive selection and accuracy-constrained selection;
  - 3.7 Equal fixed-mask recovery;
  - 3.8 Baselines (Uniform, Random, Global, ERK, LAMP) and matched-sparsity construction.
- **Figures:** F1 (framework, repeated or detailed), F2 (sensitivity profile); optional O1 (detailed MDP flow).
- **Phases:** method elements from Phases 3–6.

### Chapter 4: Experimental setup (10–12 pages)
- **Purpose:** make everything reproducible and auditable.
- **Subsections:**
  - 4.1 Datasets and leakage-guarded splits (Phase 1 audit, Phase 2; V_RL/V_SELECT vs VAL-RL/VAL-SELECT);
  - 4.2 Architectures and dense references (T1);
  - 4.3 PPO configuration and budgets;
  - 4.4 Pre-registration, seeds and decision rules (sign-flip/Holm, CI rule, equivalence margin);
  - 4.5 Integrity checks and hashing;
  - 4.6 Compute environment (CPU only).
- **Tables:** T1 (dense references), T2 (phase overview: runs, seeds, settings, primary criteria).
- **Phases:** 1, 2, plus the setup of 3–6.

### Chapter 5: Results of the core framework (22–26 pages)
- **Purpose:** answer RQ1–RQ5 in story order (see `results_mapping.md` for numbers).
- **Subsections:**
  - 5.1 Dense baselines;
  - 5.2 PPO vs Uniform, Random, Global, ERK and LAMP (Phase 3);
  - 5.3 Sensitivity-guided exploration (Phases 3–5);
  - 5.4 Policy retention and archive selection (Phase 4);
  - 5.5 Reward alignment and accuracy constraints (Phases 4–5);
  - 5.6 Equal fine-tuning and final comparison (Phase 6);
  - 5.7 Summary of the core findings.
- **Tables:** T3–T7.
- **Figures:** F3–F8.

### Chapter 6: Deployment-efficiency and sustainability analysis (16–20 pages)
- **Purpose:** answer RQ6 as the extension (flow B).
- **Subsections:**
  - 6.1 From masks to physical structure (method extension);
  - 6.2 Sparsity is not speed: latency vs parameters and MACs (Paper 3);
  - 6.3 Structured transfer and measured energy (Phase 7A);
  - 6.4 Energy-aware structured PPO (Phase 7B: method and results);
  - 6.5 Energy-measurement qualification and its failure (Phase 7B confirmation);
  - 6.6 Three-level sustainability assessment (compression / computation / operational energy).
- **Tables:** T8–T11.
- **Figures:** F9–F11; optional O6–O8.

### Chapter 7: Discussion (8–10 pages)
- **Purpose:** interpret across phases.
- **Subsections:**
  - 7.1 What RL adds over heuristics, and what it does not (LAMP);
  - 7.2 Exploration vs final accuracy;
  - 7.3 Retention and reward alignment as the real bottlenecks;
  - 7.4 Why recovery equalises methods;
  - 7.5 Compression ≠ deployment efficiency;
  - 7.6 Threats to validity (validation-to-test transfer, single recovery seed in 7A/7B, CPU-only, measurement
    noise);
  - 7.7 What a negative or partial pre-registered result means.
- **Uses:** `claim_matrix.md`, `limitations_future_work.md`.

### Chapter 8: Conclusion and future work (4–5 pages)
- **Subsections:**
  - 8.1 Answers to RQ1–RQ6;
  - 8.2 Contributions restated within their boundaries;
  - 8.3 Future work (`limitations_future_work.md`).
- **Uses:** `conceptual_conclusion.md`.

### Appendices (unrestricted)
- A. Pre-registrations and decision-rule summaries per phase.
- B. Full per-setting tables (Phases 3–6).
- C. Integrity checks.
- D. Phase 7B proxy and surrogate validation.
- E. Energy-measurement logs (pilots, qualification).
- F. Legacy pre-audit history (brief; leakage note).
