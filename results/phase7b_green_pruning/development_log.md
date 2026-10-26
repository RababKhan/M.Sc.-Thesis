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
