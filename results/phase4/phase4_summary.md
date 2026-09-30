# Phase 4 summary: redesigned reward, sensitivity-specific prior, elite-archive search

- **Branch:** `phase4-cpu`.
- **Pre-registration:** `phase4_preregistration.md` (sha256 `914b0418…`), committed in `f52e87d`
  before any Phase-4 computation.
- **Frozen configuration:** `e5e9a37`.
- **Stage selections:** `f6f85dd` (A), `c42976b` (B), `a13c828` (C).
- **Final-policy freeze:** `bf5f3fb`, before any test access.
- **Scale:** 1,200 PPO runs on seeds 400–419, 1 thread each, 2,048 timesteps per run.
- **Integrity checks:** 16/16 (`phase4_integrity_checks.md`).
- **Implementation checks:** 21/21.

Phase 3 remains frozen; its files are verified unchanged by `git diff` against `67f4270`.

## Pipelines

- **Control** K = R0 / P0 / S0: the Phase-3 configuration on fresh seeds.
- **Final pipeline** F = R0 / P2 / S2A, chosen by the pre-registered validation-only rules:
  - Stage A kept **R0**, because no redesigned reward met both criteria;
  - Stage B chose **P2**, the centred sensitivity-action prior;
  - Stage C chose **S2A**, elite behavioural cloning plus archive selection.

## Pre-registered criteria

| Criterion | Result | Qualification |
|---|---|---|
| A. Reward | **Not met** | R1a (common 50%) cut the cross-setting sparsity SD to 0.29× of R0 (CI 0.23–0.35) but kept ≥ 95% V_RL retention in only 2/6 settings (C100 ResNet-8: 67.5%). R1b (95%-retention targets) kept retention 6/6 but did not reduce inconsistency (1.05×). R2 reached 0.66× and 4/6. |
| B. Sensitivity-specific prior | **Met (STRONG)** | P2 > shuffled (P3) **and** > magnitude-matched constant (P4) on reward-rank AUC in 5/6 settings (C100 ResNet-8: P2 = P4); P2 > sign-reversed P5 in 6/6. It does not carry over to best-seen rank (1/6) or final rank (0/6), and P2's V_RL-accuracy AUC is *lower* than P4's in 4/6, because it prunes insensitive units harder. |
| C. Search | **Met (STRONG, S1/S2A)** | The archive rule raises the median final-selected rank from 65–173 (S0) to 1–10, with top-20 reach 100% in all six settings. It is mainly a selection effect: plain PPO already *visits* near-optimal policies (best-seen median rank 1–4.5, also in Phase 3) but does not converge to them. Behavioural cloning alone (S2) improves the final rank in 3/6 settings and the best-seen rank in 1/6. |
| D. Final model vs control (test excess over matched LAMP) | **Not met (LIMITED)** | F is better in C10 SimpleCNN (+0.50 pp), C10 LeNet-5 (+0.95) and C100 LeNet-5 (+0.67); worse in C10 ResNet-8 (−0.95) and C100 ResNet-8 (−0.92); C100 SimpleCNN is inconclusive. |
| E. LAMP competitiveness | **Met only as non-inferiority** | LAMP is significantly better than F in **all 6** settings. F is within the pre-registered 0.5 pp margin in 3/6 (C10 SimpleCNN −0.11, C10 LeNet-5 −0.11, C100 LeNet-5 −0.08 pp). "F > LAMP": NOT SUPPORTED. |

## Mechanistic findings

1. **Phase 3 misdiagnosed the search problem.**
   - PPO's final deterministic policy is poor: its median rank is 65–199.
   - Yet every setting's best-seen policy has a median rank of 1–4.5.
   - An archive of visited policies recovers top-10 policies.
2. **Better search exposes reward miscalibration.**
   - R0's own V_RL optimum lies at 23.5–56.4% sparsity (SD 11.1 pp across settings). Plain PPO's
     finals lie further apart (SD 16.4 pp), undershooting the optimum on the ResNet-8 settings and
     C100 LeNet-5.
   - Finding the optimum on ResNet-8 means 48% and 28% sparsity with 4–6 pp lower test accuracy,
     where LAMP's advantage is largest.
3. **Sparsity inconsistency is mostly structural.**
   - Feasible sparsity is 0–53/60% in every setting. What differs is accuracy fragility: retention at
     maximal pruning ranges from 11% to 53%.
   - No reward reached comparable sparsity without losing accuracy on the fragile models: optimum
     retention in C100 ResNet-8 is 64% (R1a) and 66% (R2).
4. **Layer-specific guidance works once net conservatism is removed.** The Phase-3 priors shared a
   common "prune less" component. Centring the sensitivity makes the average quality of explored
   policies depend on correct layer assignment, but not the best or final policy.
5. **LAMP remains the strongest method at matched sparsity.**

## Caveat (declared)

The pre-registered Stage-C rule ranks variants by the V_SELECT rank of the selected policy. The archive
rules S1 and S2A choose their policy by V_SELECT score, so that criterion favours them by construction.
The unbiased comparison is on test (D, E).

## Files

**Tables**
- `phase4_all_runs.csv`
- `phase4_reward_comparison.csv`
- `phase4_prior_comparison.csv`
- `phase4_search_quality.csv` (includes Phase-3 reference rows)
- `phase4_baseline_comparison.csv`
- `phase4_statistics.csv`
- `phase4_cross_setting_summary.csv`
- `phase4_classification.json`

**Curves and trajectories**
- `phase4_training_curves.csv`
- `phase4_policy_rank_trajectory.csv` (full per-run version in `…_full.csv.gz`)

**Figures**
- `phase4_fig_A…E_*.csv` and `.png`

**Evaluations and frozen inputs**
- `phase4_test_and_baseline_evaluations.csv`
- `phase4_final_policies.json`
- `phase4_config.json`
- `phase4_stage_selection.json`

**Raw data**
- `runs/`
- `landscapes/`
- `predictions/`
- `runs/infrastructure_log.txt`: file-lock crash and power interruption during the test evaluation;
  both were resumed with nothing lost.
