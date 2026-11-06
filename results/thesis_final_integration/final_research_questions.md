# Final research questions and expected relationships

Every question below was actually tested. Each comes with the relationship that was expected *before* the
experiments and the frozen outcome. Numbers are in `result_freeze.md`.

## Research questions

**RQ1: Learning layer-wise allocation.**
Can a PPO agent learn non-uniform layer-wise pruning allocations that outperform simple allocation rules at the same
sparsity, and how do they compare with a strong layer-adaptive heuristic (LAMP)?
*Phases 3, 4, 5 (comparisons before fine-tuning), Phase 6 (after equal fine-tuning).*

**RQ2: Sensitivity guidance.**
Does layer-sensitivity information, given to the agent as a prior or as a state feature, improve PPO's search, and
does any search improvement carry through to final accuracy?
*Phase 3 (correct, shuffled and constant priors), Phase 4 (centred prior P2), Phase 5 (C1 vs C2).*

**RQ3: Policy retention.**
Does PPO retain the high-quality policies it discovers during search, and does archive-based selection improve
retention?
*Phase 4 (S0 vs S2A), Phase 5 (archive candidates re-ranked on VAL-SELECT).*

**RQ4: Reward alignment.**
How does reward design control pruning aggressiveness, and does a validation-aware, accuracy-constrained reward
prevent over-pruning?
*Phase 4 (criterion A, R0 misalignment), Phase 5 (constrained reward, τ = 0.98).*

**RQ5: Equal recovery.**
How does identical post-pruning fine-tuning change the comparison between pruning methods?
*Phase 6 (fixed-mask, 5-epoch fine-tuning; PPO vs LAMP vs global magnitude).*

**RQ6: Deployment efficiency.**
Do compression gains translate into deployment efficiency (CPU latency, parameter memory and measured operational
energy) when pruning is realised as physically structured networks?
*Paper 3 (latency), Phase 7A (structured transfer, RAPL energy), Phase 7B (structured, energy-aware PPO), Phase 7B
energy confirmation.*

RQ1–RQ5 belong to the **core framework**, which works with unstructured pruning masks. RQ6 belongs to the
**deployment and sustainability extension**, added after the core framework was complete (see
`chapter_structure.md`, the methodology flow).

## Expected relationships (hypotheses) and frozen outcomes

| Hypothesis | Expected relationship | Frozen outcome |
|---|---|---|
| **H1a** (RQ1) | PPO allocation beats uniform and random allocation at equal sparsity | **Supported.** Phase 3 STRONG: Uniform 5/6, Random 6/6. |
| **H1b** (RQ1) | PPO beats global magnitude, ERK and LAMP at equal sparsity | **Partly or not supported.** Global MODERATE (3/6); ERK LIMITED (3/6, 1 opposite); LAMP NOT SUPPORTED, REVERSED (LAMP better 5/6 in Phase 3, 6/6 in Phases 4–5). |
| **H2a** (RQ2) | Sensitivity guidance improves search quality | **Supported on search metrics.** Phase 3: secondary endpoints STRONG, primary AUC MODERATE. Phase 4: reward-rank AUC STRONG, but V_RL-accuracy AUC reversed. Phase 5: 5/6, with 0 of 24 tests opposite. |
| **H2b** (RQ2) | The search improvement raises final accuracy | **Not supported.** Phase 3: 0/6. Phase 4: final rank 0/6. Phase 5: C1 is 0.05–0.16 pp below C2 on test. |
| **H3** (RQ3) | Terminal PPO policies under-retain the best discovered policies; an archive restores them | **Supported on validation ranking.** Terminal selection reaches the top 10 in 0% of settings vs 75–100% with the archive. Test effect mixed: ResNet-8 −5.66 / −4.00 pp; Phase 4 criterion D not met. |
| **H4a** (RQ4) | A reward that trades accuracy against sparsity is misaligned, so better search leads to over-pruning | **Supported.** Phase 4: no redesigned reward met criterion A. ResNet-8 sparsity rose to 47.8% with a 7.86 pp test drop. |
| **H4b** (RQ4) | A hard accuracy constraint on validation controls over-pruning | **Supported for ResNet-8 and LeNet-5, partial overall.** ResNet-8 drop fell 7.32 → 1.18 and 4.05 → 1.16 pp. Criteria A and E PARTIALLY MET; test feasibility 67.5%. |
| **H5** (RQ5) | Equal fine-tuning recovers most pruning loss and narrows method differences | **Supported.** Criteria A–F all MET. PPO–LAMP 95% CI inside ±0.5 pp in 6/6, but LAMP still ahead in 3/6. PPO's lead over global magnitude disappears. |
| **H6a** (RQ6) | Parameter sparsity predicts CPU latency | **Not supported.** ρ = 0.31–0.55, and 31–39% of policy pairs are sparser but slower. MACs are a strong predictor (ρ = 0.89–0.98). |
| **H6b** (RQ6) | A structured realisation of the PPO allocation reduces measured CPU energy without material accuracy loss | **Energy reduced, but accuracy not preserved.** Phase 7A: −22.8% / −13.4% energy at −5.08 / −3.24 pp VAL-RL (Case B). On SimpleCNN, Uniform dominated PPO. |
| **H6c** (RQ6) | PPO that learns structured, energy-aware policies directly meets every deployment criterion within 0.5 pp | **Not supported.** Phase 7B: −63.4% memory, −36.0% MACs, −1.12 pp VAL-SELECT; 0/20 seeds meet GREEN-SUCCESS. Measured energy unresolved (pilot CV 4.01%; confirmation CV 19.92%). |

## Questions deliberately *not* posed (untested)

- Superiority of PPO over LAMP as a research claim.
- Accuracy-feasible latency gains from hardware-aware RL. It was never run because Paper 3's GO rule failed.
- Carbon, lifecycle or whole-system energy.
- GPU, mobile or embedded deployment, and ImageNet-scale models.
