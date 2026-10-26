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
