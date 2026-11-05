# QA report: Chapter 3 (Proposed Framework)

- **File:** `chapter_03_framework.md`.
- **Status:** **FROZEN (v1, 2026-11-05).**

## 1. Word count

**7,937** words of prose, excluding citation keys and the pseudocode blocks. That is about 19 pages of text, or about
21–22 pages with tables, equations and algorithms. It is within the 6,000–8,000 target.

## 2. Subsections

The 19 sections follow the requested structure:

| § | Title |
|---|---|
| 3.1 | Chapter Overview (with Table 3.1 timeline, Figure 3.1 = F1, Table 3.2 notation) |
| 3.2 | Problem Formulation |
| 3.3 | Layer-Wise Pruning Representation |
| 3.4 | Reinforcement Learning Formulation |
| 3.5 | State Representation |
| 3.6 | Action Space |
| 3.7 | Initial Reward Function |
| 3.8 | Layer Sensitivity Analysis |
| 3.9 | Sensitivity-Guided Action Prior |
| 3.10 | Centred Sensitivity Prior |
| 3.11 | Policy-Retention Problem |
| 3.12 | Archive-Based Policy Selection (3.12.1 archive, 3.12.2 Phase-4 rule, 3.12.3 elite behavioural cloning, 3.12.4 Phase-5 rule) |
| 3.13 | Reward Misalignment and Accuracy-Constrained Reward (3.13.1, 3.13.2) |
| 3.14 | Final Core Framework |
| 3.15 | Equal Recovery and Fine-Tuning Protocol |
| 3.16 | Deployment-Oriented Structured Extension (3.16.1 masks to structure, 3.16.2 three levels, 3.16.3 Phases 7A and 7B) |
| 3.17 | Overall Algorithm |
| 3.18 | Computational Workflow and Reproducibility |
| 3.19 | Chapter Summary |

## 3–4. Equations and algorithms

- **Equations:** 20, numbered (3.1)–(3.20).
- **Algorithms:** 2. Algorithm 3.1 covers the core framework. Algorithm 3.2 covers deployment-oriented structured
  evaluation. They are kept separate.
- **Tables:** 3.1–3.7.

## 5. Citations

12 distinct references, all from the verified bibliography:
- PyTorch;
- PPO and GAE;
- Stable-Baselines3;
- reward misspecification (2);
- rewinding and retraining (2);
- sparsity survey and structured survey;
- filter and channel pruning (2).

No new reference was added. Citations are used only for established concepts, never for the thesis' own
implementation.

## 6. Source verification of equations

**Yes, all 20.** See `equation_audit_chapter_03.md`. 18 are exact to the code or pre-registration. Two are standard
published forms whose constants are exact: the PPO objective (3.20) and the cosine schedule (3.16). (3.3) is labelled
as a conceptual objective.

## 7. PPO state

**Core framework, Phases 3–5 (`src/rl` `unit_state`):**

[(l−1)/(N_L−1), log(1+n_l), n_l/max_j n_j, mean|w_l|, σ(w_l), Σ_{j<l} r_j, sensitivity slot]

The **sensitivity slot is 0 in every Phase 3–5 condition**. Sensitivity entered only through the action prior.

**Phase 7B (`ppo.py` `_state`), eight features:**

[(l−1)/(N_L−1), log(1+C_l), parameter share, MAC share, S̃_l, previous ratio, mean ratio so far, Ê(partial)/Ê(dense)]

The two definitions are listed separately in Table 3.4.

## 8. Action set

r(a) ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.6} for a ∈ {0, …, 5}; one action per unit; 6⁴ = 1,296 policies. The chapter states
explicitly that structural meaning differs across architectures, for example a single layer in SimpleCNN versus a
residual stage in ResNet-8.

## 9. Initial reward

R0 = 1.5·A_p/A_b + 0.01·ρ + 0.05·D, paid at episode end, where:
- ρ is the percentage sparsity over all Conv2d/Linear weights, including the never-pruned classifier;
- D is the number of distinct actions;
- λ_s = 0.01 in every frozen phase. The library default of 0.04 is the original notebook value, and this is
  disclosed.

## 10. Sensitivity prior (Phase 3)

z̃_l(a) = z_l(a) − c_l · r(a)/0.6, with c_l = 0.5·S_l. S_l is the min–max normalised mean relative loss increase
over single-unit probes at {0.1, 0.2, 0.4, 0.6} on VAL-RL. Controls: no prior, shuffled, constant.

## 11. Centred prior (Phase 4 onward)

c_l = 1.0·(S_l − S̄). Its coefficients sum to zero, so it carries no net conservatism. Controls: permuted,
magnitude-matched conservative, sign-reversed.

## 12. Constrained reward (Phase 5)

R_c = 1 + ρ/100 + 0.01·q if q ≥ 0.98, otherwise −10·(0.98 − q), with q = A_p/A_b on VAL-RL. There is no diversity
term. τ was chosen by a pre-registered rule on the VAL-RL landscapes.

## 13. Archive selection

The archive holds every unique sampled policy, with its VAL-RL reward and first episode.

- **Phase 4:** candidates are the top 10 by VAL-RL reward. The selected policy is the candidate with the highest
  VAL-SELECT score (ties: higher VAL-RL reward, then earlier discovery).
- **Phase 5:** the archive is frozen before any VAL-SELECT evaluation. The selected policy is the highest-sparsity
  candidate with q_VAL-SELECT ≥ 0.98 (ties: VAL-SELECT accuracy, then VAL-RL reward, then earlier discovery;
  fallback: highest q_SELECT, reported).
- Elite behavioural cloning (K = 5, η = 0.1) is documented as Phase-4-only.

## 14. Recovery protocol

- Adam, learning rate 1e-4, β = (0.9, 0.999), ε = 1e-8, weight decay 0.
- Cosine schedule to 0, stepped per epoch; E = 5 epochs.
- Batch 128; crop with padding 4, plus horizontal flip.
- Mask frozen and checked every epoch.
- BN in training mode during fine-tuning.
- Fine-tuning seed = policy seed + 100; shared seed and data stream within each comparison triple.
- No checkpoint selection.

It is an **evaluation** component, not part of PPO's decision making (Table 3.5, Algorithm 3.1 steps 19–20).

## 15. Core versus extension

Section 3.16 opens with a boxed statement that it is a later extension. Its other safeguards:
- Algorithm 3.2 is separate from Algorithm 3.1.
- Table 3.1 tags every component with its phase.
- The Phase-7B agent is described as "a separate, deployment-oriented variant … not the final form of the core
  method".

## 16. Consistency with F1 (Figure 3.1)

| F1 box | Chapter 3 section | Phase tag in F1 | Implementation source | Consistent? |
|---|---|---|---|---|
| Dense CNN | 3.2 | Phase 2 | `cpu_baselines.csv` | yes |
| Layer sensitivity analysis (VAL-RL) | 3.8 | Phase 3 | `src/sensitivity` | yes |
| Sensitivity-aware PPO policy — soft prior; centred from Phase 4 | 3.9–3.10 | Phases 3–4 | `src/rl` `SoftPriorPolicy`; `phase4_lib` | yes (prior, not state) |
| Layer-wise pruning actions {0, .1, .2, .3, .4, .6} | 3.6 | Phase 3 | `ACTION_TO_PRUNE` | yes |
| Accuracy/compression reward R0 | 3.7 | Phase 3 | `reward_components` | yes |
| Policy search (PPO), 2,048 timesteps | 3.4, 3.17 | Phase 3 | `PPO_KWARGS`, budget | yes |
| Archive selection (top-10, held-out re-ranking) | 3.12 | Phase 4 | `phase4_metrics.py` | yes |
| Accuracy-constrained reward/selection, τ = 0.98 | 3.12.4, 3.13 | Phase 5 | `phase5_lib.r_constrained`, `select_archive` | yes |
| Final pruned policy (unstructured masks) | 3.3, 3.14 | Phases 3–5 | `src/pruning` | yes |
| Equal recovery (5 epochs, fixed mask) | 3.15 | Phase 6 | `phase6_lib` | yes |
| Accuracy–compression evaluation | 3.14 (Table 3.5, stage 10) | Phases 3–6 | Chapter 4 | yes |
| Extension boxes (structured pruning, memory, MACs, latency, energy, trade-off) | 3.16 | Paper 3, 7A, 7B | `structured_units.py`; 7A/7B scripts | yes |

**One intentional omission, not an error.** F1 does not show the Phase-4 elite behavioural-cloning term, because it
was not retained after Phase 4. Chapter 3 documents it in 3.12.3 as historical. No change to F1 is required.

## 17. Dangerous-claim audit

The automated scan (`tools/qa_chapter.py`) plus a manual grep covered:
- *improves accuracy, beats / better than / outperform LAMP, archive improves test, accuracy-preserving*;
- *green, environmentally friendly, energy-efficient, carbon, Phase-7B measured energy*;
- result numbers (22.8, 13.4, 63.4, 36.0, 1.12, top-10 reach).

Hits and verdicts:
- "equivalently the action vector" (notation): benign.
- "not a guaranteed optimum": benign; it is a deliberate disclaimer.
- "LAMP": appears only as the name of a comparison method in the recovery protocol.
- "better": appears only in definitions, for example "more sparsity is better among feasible policies".
- **No result value appears in the chapter.**
- The energy surrogate is stated three times not to be a measurement.

## 18. Unresolved methodological ambiguities

None affects correctness. Three points are disclosed in the text:

1. **λ_s history.** The library default (0.04) differs from the frozen value (0.01). The frozen value is used and
   the difference is explained in 3.7.
2. **Process definition.** The process is described as a finite-horizon sequential decision process, *not* claimed
   to be Markov in the state (3.4).
3. **Phase-7A ResNet-8 mapping.** The unstructured-to-structured mapping is approximate because the unit definitions
   differ (3.16.1). This is a known limitation already recorded in Phase 7A.

**Item to confirm in Chapter 4.** VAL-RL and VAL-SELECT are used as one name pair with two versions: 3,000/2,000 for
Phases 3–4, and 2,500/2,500 from Phase 5 on. Chapter 4 will define both versions explicitly.

## Freeze

Chapter 3 v1 is frozen. Later changes are limited to citation-number assembly or factual corrections, and will be
logged here.
