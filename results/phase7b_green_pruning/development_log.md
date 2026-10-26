# Phase 7B development log (Stage A)

Each entry is dated. Stage-A results support design decisions only. No final claim comes from Stage A.

## 2026-10-26: start

- **Git status before implementation.** Branch `phase7a-sustainability` at `24fefd3`. The only working-tree
  changes were untracked items that stay local and are never staged: `checkpoints/phase6_finetuned/`,
  `checkpoints/phase7a/`, `template/` and the Phase-7A lock directory. `checkpoints/baseline_clean.pth`
  shows as modified and is also never staged.
- **New branch.** `phase7b-green-structured-ppo`, created from `24fefd3`.
- **Design.** `phase7b_design.md` was written before any Phase-7B computation.
- **Machine state.** On battery at the start. Sleep timeouts are 0 (never) on both AC and battery, Balanced
  plan. Energy measurements will run only on AC (the scripts refuse otherwise).

## 2026-10-26: sample construction (structural only; nothing measured)

- **Proxy-validation sample.** The design asked for 10 policies with MAC reduction in [0, 5%), but the space
  holds only 7 non-dense policies there. All 7 are taken, and the shortfall of 3 is carried to the next bin,
  [5, 15%), which therefore gets 13. The total stays at 49 (dense plus 48). This was decided from policy
  counts alone, before any measurement.
- **Energy-calibration set.** Bins with fewer than 20 available policies contribute all of them. The
  similar-MAC pairs and the remainder rule fill the set to 200, as designed.

## 2026-10-26: interruption of the proxy-validation run

- **What happened.** The recovery workers stopped when the controlling Claude Code session ended, with 41 of
  147 jobs complete. Eight 5-epoch jobs were in progress.
- **Resumption.** Their lock files were removed. Their per-epoch work states, which `phase6_lib.finetune`
  resumes bit-identically (verified in Phase 6), were kept. The workers were then relaunched as independent
  Windows processes on AC power.
- **Data.** No data was inspected or discarded.

## 2026-10-26: accuracy-proxy decision (design section 4, applied unchanged)

- **Completion.** All 147 jobs (49 policies × E ∈ {2, 3, 5}) completed. Each budget used a single data stream.
- **Dense VAL-RL:** 79.88 (E=2), 80.48 (E=3), 80.60 (E=5). 19 of the 49 sample policies are feasible at
  5 epochs (within 0.5 pp of the dense model).
- **2-epoch proxy:** Spearman 0.948, τ = 0.98849, recall 1.00, precision 0.66 (29 selected).
  **Accepted.**
- **3-epoch proxy:** Spearman 0.939, recall 1.00, precision 0.61. It is recorded for completeness only;
  the rule tests the 3-epoch proxy only if the 2-epoch proxy fails.
- **Consequence.** The landscape cache and PPO use E = 2 and τ = 0.98849.

## 2026-10-26: RAPL calibration

- **Pilot.** Dense only, N = 20,000, 2 fresh processes × 3 blocks, on AC.
  E_1000 = 11.515 J, CV 3.47% (≤ 4%): stable, so N = 20,000 is kept.
- **Sessions.** The three calibration sessions were launched as a detached process.
- **Session 1 repeats.** Session 1 was repeated twice under the pre-registered drift rule: dense-anchor drift was
  12.1%, then 11.7%, both above 10%. The third attempt drifted 6.7% and was kept. Sessions 2 and 3 drifted 4.3%
  and 3.9%. CPU load at each gate was 2.4–3.2%, on AC power. Rejected sessions are kept as `session1_rejected*.json`
  and are not used.
- **Dataset.** `energy_surrogate_dataset.csv` has 200 policies. E_1000 is the median of the 3 sessions; median
  per-policy CV is 0.69% and maximum CV 3.7%. Dense E_1000 = 10.82 J.

## 2026-10-27: energy-surrogate decision (design section 7, applied unchanged)

- **Results.** 5-fold CV, seed 20261029, candidates tested in the fixed order:

  | Candidate | Spearman | MdAPE | Low-quartile MdAPE | Result |
  |---|---|---|---|---|
  | 1 (MACs) | 0.946 | 4.1% | 8.5% | fail |
  | 2 (L) | 0.996 | 1.05% | 0.95% | pass |
  | 3 | 0.998 | 0.73% | 0.81% | pass |
  | 4 | 0.999 | 0.50% | 0.58% | pass |

- **Accepted: candidate 2**, the first to pass. Ê_1000 = 13.762 · L_table − 0.117 J; the dense prediction
  is 10.77 J and the maximum predicted saving over the 1,296 policies is 62.4%. Candidates 3–4 are recorded only.
- **Next.** The 1,296-policy landscape at E = 2.
