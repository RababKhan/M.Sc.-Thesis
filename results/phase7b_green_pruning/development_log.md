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
- **Landscape launched.** On AC, 8 detached workers started on 2026-10-27, 1,296 policies × E = 2.

## 2026-10-27: Stage-B tooling (written during the landscape run, before any Stage-B data)

- **`recover.py`.** The `final` queue now takes `final_jobs.json` entries `{id, ratios, policy, roles}`, so
  that the off-grid Uniform and LAMP ratio vectors are realized through the same slimming function
  (`structured.simplecnn_structured`, which `structured_units.slim` calls). The action-policy path used by the
  proxy and landscape queues is unchanged.
- **`final_models.py`.** Builds the primary new-PPO model, the per-seed selections, the enumeration
  reference, the MAC-matched Uniform and the parameter-matched LAMP (0.1% sparsity grid), and reuses the old
  Phase-7A PPO checkpoint.
  - Smoke test on a placeholder target (the old PPO's counts): Uniform reproduces Phase 7A's r = 0.256.
- **`energy_final.py`.** Phase-7A hardware protocol, with Dense measured in 3 blocks per session.
- **`final_analysis.py`.** Implements the GREEN rules, cases and Pareto analysis.
- **`write_prereg.py`.** Writes the frozen Stage-B file, with input and code hashes.
- **Check: dense recovery matches Phase 7A.** The Stage-A 5-epoch dense recovery is bit-identical to the
  Phase-7A dense model (final state `8c285b…`, data stream `2cc04b…`). The old PPO checkpoint therefore
  comes from the same protocol.

## 2026-10-27: landscape run paused at the user's request

- **Pause.** The 8 workers were stopped at 1136 of 1,296 records, at the user's request.
- **Interrupted jobs.** The lock files of the jobs in progress were removed: 5132_E2 5133_E2 5134_E2 5135_E2 5140_E2 5141_E2 5142_E2 5143_E2.
- **Kept.** Their per-epoch work states, which resume bit-identically, and all completed records. No data
  was inspected or discarded.
- **Resume.** Relaunch with the same `launch_workers.ps1 landscape` command.
- **Resumed.** On 2026-10-27, at the user's request, 8 workers were relaunched on battery power.
  Recovery is deterministic, so the power source does not affect results; it is not an energy measurement.

## 2026-10-27: landscape complete; α tuning (Stage A)

- **Landscape.** All 1,296 policies have 2-epoch recovery records, all from one data stream.
  - Dense A_2 = 79.88. 202 of 1,296 policies are proxy-feasible (q ≥ τ = 0.98849).
  - The 49 Stage-A policies reproduce their proxy-validation 2-epoch accuracies exactly.
- **Windows Smart App Control (enforcing) blocked matplotlib.** From 15:46 on 2026-10-27,
  `matplotlib/_c_internal_utils.pyd` (and apparently `ft2font.pyd`) were blocked from loading
  (Code Integrity events 3033/3077). The same files loaded earlier in this phase.
  - **`ppo.py`.** SB3's logger imports `matplotlib.figure` only for an output type that is never used. If
    that import is blocked, a placeholder module stands in. No computation reads it.
  - **`final_analysis.py`.** matplotlib is imported only inside the figure step. A blocked import leaves
    every number intact and records that the figures are pending. The verified exact Wilcoxon function is
    copied verbatim from Phase 7A's `analyze.py`, which imports matplotlib at module level.
  - **Not done.** The security setting was not changed; that is the user's decision.
- **α tuning** (`ppo.py dev`, seeds 900–904, pre-registered rule: lowest median selected Ê, ties → smaller α):

  | α | Median selected Ê | Empty archives |
  |---|---|---|
  | 0.5 | 8.3631 J | 0 |
  | 1 | 8.3631 J | 0 |
  | 2 | 8.3631 J | 0 |
  | 4 | 8.3562 J | 0 |

  **Chosen: α = 4.** Selections were [2,2,2,1], [2,2,2,4] and [2,2,2,5]. The differences between α
  values are small (< 0.1%), and the rule is applied as written.

## 2026-10-27: procedural error in the Stage-B dry run (disclosed)

- **What happened.** The dry run of the Stage-B tooling ran in a scratch sandbox, with a throwaway
  pre-registration and nothing written to `results/` or `checkpoints/`. By mistake, it ran the confirmatory
  PPO search with the real confirmatory seeds 700–719, before the real pre-registration was committed. The
  20 per-seed selected policies were therefore printed and seen.
- **Stopped.** The run was stopped during final-model derivation: no model was recovered and no accuracy
  or energy was computed for these policies.
- **Why no rule is affected.** Every pre-registered quantity was already fixed by Stage-A rules before
  this happened:
  - α = 4 (development seeds);
  - τ, E = 2 and the surrogate (validation rules);
  - the selection, baseline, statistics and GREEN rules (design);
  - the confirmatory seeds (design).

  Nothing is changed after it. PPO on the cache is deterministic, so the real confirmatory run will
  reproduce these selections.
- **Fix.** The dry run is repeated with placeholder seeds 9000–9019.

## 2026-10-27: Stage-B dry run passed; pre-registration frozen

- **Dry run.** A scratch sandbox with placeholder seeds 9000–9019, 1-epoch recovery and n = 200 RAPL
  sessions ran every Stage-B step end to end:
  - pre-registration, confirmatory PPO and final-model derivation (including the LAMP sweep);
  - 10 recoveries;
  - 5 energy sessions with Dense measured 3 times per session;
  - the full analysis.

  Only completion and errors were checked; no numbers were read.
- **Bugs it found, fixed before freezing:**
  - `recover.py` built the record path from `policy`, which is None for Uniform and LAMP; it now uses the job.
  - The `ppo.py` matplotlib guard now runs when `ppo` is imported, before `write_prereg` imports `src.rl`.
  - The figure step recognises the placeholder module and raises ImportError.
- **Integrity checks work.** On the 1-epoch dry run, the checks "dense identical to Phase 7A" and "same
  data stream as Phase 7A" correctly returned False.
- **Not exercised.** The figure code (matplotlib blocked).
- **Frozen.** `phase7b_final_preregistration.json` was written: α = 4, τ = 0.98849, E = 2, surrogate 2,
  confirmatory seeds 700–719. It records the SHA-256 of every Stage-A input and of the Stage-B code.

## 2026-10-27: Stage B, confirmatory search and final recovery

- **Confirmatory PPO** (seeds 700–719, frozen pre-registration). Every seed has a selected policy.
  - The selections are identical to those seen in the disclosed dry-run error, as expected: PPO on the
    cache is deterministic.
  - Primary new PPO: [2,2,2,5], selected by 5 of 20 seeds. Enumeration reference: [3,1,2,5], also seed
    700's selection.
- **Baselines** (`final_models.py`):
  - Uniform: r = 0.208; MAC gap 11,934.
  - LAMP: 56.2% global sparsity; unit ratios [0.010, 0.089, 0.192, 0.634]; parameter gap 254.
  - 10 recovery jobs in total.
- **Final 5-epoch recovery.** All 10 jobs are complete, on one data stream (`2cc04b…`, as in Phase 7A).
  Dense is bit-identical to Phase 7A and to its Stage-A run. The test set was not used.
