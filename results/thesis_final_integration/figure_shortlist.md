# Figure shortlist

There are **11 essential** and **9 optional** figures. Existing figures can be re-rendered in a common thesis style.
Re-rendering may only change styling; the plotted data must stay the same. Paths are relative to `results/`.

## Essential (main text)

| ID | Section | Content | Status / source |
|---|---|---|---|
| **F1** | 1 and 3 | Overall thesis framework. Two visually separated blocks: the **core framework** (sensitivity → PPO → reward → archive → constrained selection → equal recovery → evaluation) and the later **deployment and sustainability extension** (structured realisation → latency → energy → energy-aware structured PPO). | **New.** Draw from `chapter_structure.md` §2. Do not reuse the Paper-3 flowchart. |
| **F2** | 3.2 | Layer-sensitivity profile per architecture (unit × ratio) | `architecture_generalization/phase3_figures/phase3_fig_D_sensitivity_tolerance.png` |
| **F3** | 5.2 | Accuracy vs sparsity: PPO vs Uniform, Random, Global, ERK, LAMP | `architecture_generalization/phase3_figures/phase3_fig_A_accuracy_sparsity.png` |
| **F4** | 5.2 | PPO allocation examples (per-unit ratio heatmap) | `architecture_generalization/phase3_figures/phase3_fig_B_policy_heatmap.png` |
| **F5** | 5.3 | Sensitivity-guided exploration (C1 vs C2) | `phase5/phase5_fig_D_sensitivity_exploration.png` |
| **F6** | 5.4 | Archive selection effect: top-k reach, terminal vs archive | `phase4/phase4_fig_D_topk_reach_rate.png` |
| **F7** | 5.5 | ResNet-8 reward alignment (R0 vs constrained reward: sparsity and accuracy) | `phase5/phase5_fig_B_resnet_reward_alignment.png` |
| **F8** | 5.6 | PPO vs LAMP before vs after equal fine-tuning, with the ±0.5 pp band | `phase6/figures/phase6_fig2_ppo_vs_lamp.png` |
| **F9** | 6.2 | Latency gain vs parameter reduction and vs MAC reduction | `paper3/paper_figures_revised/fig1_latency_vs_reduction.png` |
| **F10** | 6.3 | Phase 7A accuracy vs measured energy | `phase7a_sustainability/figures/fig_A_accuracy_vs_energy.png` |
| **F11** | 6.4 | Phase 7B accuracy vs MACs and vs parameter memory (Pareto planes; dashed line = dense − 0.5 pp) | `phase7b_green_pruning/figures/fig_accuracy_planes.png` |

## Optional (main text if space allows, otherwise appendix)

| ID | Content | Source |
|---|---|---|
| O1 | Detailed MDP / methodology flow (state, action, reward, archive) | **New** |
| O2 | Policy-rank trajectory during training (terminal vs best seen) | `phase4/phase4_fig_C_policy_rank_trajectory.png` |
| O3 | Phase 3 exploration curves (correct / shuffled / constant prior) | `architecture_generalization/phase3_figures/phase3_fig_C_exploration.png` |
| O4 | Validation vs test feasibility gap | `phase5/phase5_fig_E_feasible_sparsity_gap.png` |
| O5 | Phase 6 recovery of all methods (pre vs post) | `phase6/figures/phase6_fig1_recovery.png` |
| O6 | ResNet-8 accuracy-feasible latency room | `paper3/paper_figures_revised/fig3_resnet8_latency_room.png` |
| O7 | Phase 7A normalised efficiency (memory, MACs, latency, energy, EDP; dense = 1) | `phase7a_sustainability/figures/fig_B_normalized_efficiency.png` |
| O8 | Energy-measurement stability: dense J/1,000 per block for the 7A pilot, 7B pilots and 7B-EC qualification, with temperature | **New**, from the frozen pilot and qualification JSONs (descriptive) |
| O9 | Phase 6 compute cost | `phase6/figures/phase6_fig5_cost.png` |

## Not to be produced

- Phase 7B energy-vs-session or energy Pareto figures. No measured data exists, so do not plot surrogate energy as
  if it were measured.
- Normalised Phase 7B deployment metrics including energy or latency.
- Any "green" or carbon infographic.
