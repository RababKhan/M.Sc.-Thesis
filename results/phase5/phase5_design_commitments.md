# Phase 5 design commitments (Part A of the pre-registration)

Written 2026-09-30 on branch `phase5-cpu` (from `a3653fd`), **before** the Phase-5 split, landscapes,
sensitivity vectors and τ diagnostic were computed. This file is committed first and never edited. The
full pre-registration (`phase5_preregistration.md`) follows after the τ diagnostic and before any PPO run.
It must be consistent with this file.

## Frozen evidence

- Phase 3 (`67f4270`) and Phase 4 (`a3653fd`) are read only. Their files are verified unchanged (zero
  `git diff` lines) at the start of Phase 5.
- Dense references (sha256 prefixes): `ca28f834`, `5ca6e0fc`, `41fc5ee5`, `d75dd19c`, `8e1b1aee`,
  `8b84bfad`.
- Old splits: `archive_validation_split.npz` `55b7ff40`, `cifar100_split_indices.npz` `d7351925`.
- The training set (45,000), official test set (10,000) and 5,000-image validation set are unchanged.

## Clarification of the Phase-4 bias

- **What Phase 4 did.** The Phase-4 archive already selected on V_SELECT, which is disjoint from V_RL.
- **The actual bias.** The *Stage-C method choice* was scored by the V_SELECT rank of policies that the
  archive itself had selected on V_SELECT.
- **How Phase 5 removes it:**
  1. a new balanced split, VAL-RL 2,500 / VAL-SELECT 2,500;
  2. **no data-driven method selection at all** (C0, C1 and C2 are fixed in advance);
  3. no inference about condition differences from VAL-SELECT metrics of VAL-SELECT-selected policies.
     Search quality is measured on VAL-RL; final quality on test.

## Splits

**Rule.** `src.data.stratified_split(val_labels, n_first = 2500, seed = 20260930)` on each dataset's
frozen 5,000-image validation positions:
- the first part is **VAL-RL** (2,500);
- the second is **VAL-SELECT** (2,500).

The split is class-stratified, with largest-remainder allocation. Files are
`results/phase5/splits/<dataset>_phase5_split.npz` plus a manifest with hashes.

**Use of each split:**
- **VAL-RL:** sensitivity, rewards, PPO training, exploration and rank metrics.
- **VAL-SELECT:** evaluating the frozen archive candidates and final selection only.
- **Test:** only after the final-policy list of all 360 runs is frozen and committed.

## Sensitivity and prior

- **Sensitivity.** S is **recomputed on VAL-RL**; the Phase-3 vector used the old V_RL, which overlaps
  the new VAL-SELECT. The procedure is unchanged:
  - `src.sensitivity.loss_sensitivity`, ratios 10/20/40/60%;
  - min-max normalised;
  - 1 thread.
- **Destructive action.** The most sensitive unit (highest raw S; ties: earliest) pruned at ≥ 40%.
- **Prior P2.** The Phase-4 centred prior: c_u = 1.0·(S_u − mean S), subtracted from the logits as
  c_u·ratio(a)/60. There are no masks.

## Rewards (on VAL-RL; A_d = dense VAL-RL accuracy; q = A/A_d; ρ = total sparsity as a fraction in [0, 1])

- **R_constrained** (C1, C2):
  - if q ≥ τ: R = 1 + ρ + ε·q;
  - otherwise: R = −κ·(τ − q).
  - The constants are **ε = 0.01** and **κ = 10**. There is no diversity term.
  - Every feasible reward exceeds 1; every infeasible reward is below 0. So accuracy feasibility comes
    first, then sparsity, then accuracy as a small tie-breaker (a 0.01 difference in q is worth 0.0001
    in ρ).
- **R0** (C0): the Phase-3/4 reward 1.5·A/A_d + 0.01·ρ(%) + 0.05·D, with λ = 0.01, on the new VAL-RL.

## τ (one global value, chosen by rule on VAL-RL landscapes; never on test)

- **Candidates:** {0.95, 0.97, 0.98, 0.99}.
- **What counts as "meaningful non-trivial" (fixed now).** A setting has at least **20** policies with
  q ≥ τ **and** at least **one** policy with q ≥ τ and total sparsity ≥ **10%**.
- **Rule:** τ = the **largest** candidate that meets this definition in **all six** settings.
- **Reporting:** the diagnostic (`phase5_tau_diagnostic.csv`) lists, per τ and setting:
  - feasible count;
  - max and median feasible sparsity;
  - best feasible VAL-RL accuracy and that policy's sparsity;
  - the oracle's sparsity and accuracy.

## Analysis oracle (never given to PPO)

- **Oracle.** The constrained optimum of a setting: the highest sparsity among policies with q ≥ τ on
  VAL-RL (ties: higher VAL-RL accuracy, then lower policy index).
- **Constrained rank.** The rank of a policy under R_constrained on the VAL-RL landscape (1 = oracle).
  It is the common yardstick for C0, C1 and C2.

## Design

**Seeds.** 500–519 (fresh; highest earlier seed 419), paired across conditions.

**Scale.** 6 settings × 3 conditions × 20 seeds = **360 runs**, each with 2,048 timesteps (512
episodes), 1 thread, the Phase-3/4 PPO hyper-parameters and the exact VAL-RL landscape cache. There is no
behavioural cloning.

**Conditions:**

| | Reward | Prior | Archive |
|---|---|---|---|
| C0 | R0 (λ 0.01) | P2 | yes |
| C1 | R_constrained | P2 | yes |
| C2 | R_constrained | none | yes |

**Archive** (identical mechanism for all conditions; VAL-RL only until frozen):
1. **Record.** Every unique policy sampled during training is recorded, with its VAL-RL reward and first
   episode.
2. **Candidates.** After training, the **top 10 unique policies by VAL-RL reward** (ties: earlier
   discovery) are frozen.
3. **Evaluation.** They are then evaluated on VAL-SELECT.
4. **Selection:**
   - **C1, C2:** among candidates with q_SELECT ≥ τ (q_SELECT = accuracy / dense accuracy, both on
     VAL-SELECT), take the highest sparsity; ties go to higher VAL-SELECT accuracy, then higher VAL-RL
     reward, then earlier discovery. If no candidate is feasible on VAL-SELECT, take the candidate with
     the highest q_SELECT (reported).
   - **C0:** the highest R0 score computed with VAL-SELECT accuracy and dense VAL-SELECT accuracy; ties go
     to higher VAL-RL reward, then earlier discovery. This is the Phase-4 archive rule on the new
     independent split.
5. **Freeze.** The final-selected policies of all 360 runs are written, hashed and committed; only then is
   the test set read.
