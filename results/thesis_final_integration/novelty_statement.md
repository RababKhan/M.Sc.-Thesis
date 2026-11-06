# Novelty statement

## Short version (for the abstract and introduction)

This thesis contributes an integrated, pre-registered analysis of reinforcement-learning-based layer-wise pruning
that follows the method from search to deployment, rather than a new leaderboard result. The framework combines a
sensitivity-aware PPO agent with archive-based policy retention, a validation-aware accuracy-constrained reward and
equal post-pruning recovery. It then tests whether the resulting compression becomes measured CPU latency and energy
benefit. Each stage isolates a separate failure mode of RL pruning:

- search efficiency (sensitivity);
- policy retention (archive);
- reward misalignment and over-pruning (constraint);
- recovery-dependent rankings (equal fine-tuning);
- the gap between compression metrics and deployment efficiency (structured latency and energy analysis).

## Why it is methodological and analytical, not a "best method" claim

1. **The baseline result is used, not hidden.** PPO strongly beats uniform and random allocation, but LAMP, a cheap
   heuristic, stays as good or better. The thesis therefore does not rest on PPO being the best pruning algorithm.
   It shows *where* RL adds value (pre-fine-tuning allocation quality, flexible objectives) and where it does not.
2. **It separates exploration from outcome.** Sensitivity guidance measurably improves search but not final
   accuracy. This dissociation is rarely reported, and it changes how such priors should be evaluated.
3. **It names and fixes a specific RL-pruning failure.** PPO finds top-ranked policies but does not retain them
   (terminal top-10 reach 0% vs archive 75–100%). Fixing retention then exposes reward misalignment, which is
   addressed with a constraint-first reward. The diagnosis chain is itself a contribution.
4. **It controls for recovery.** Under identical fine-tuning, method differences shrink to within ±0.5 pp. Many
   pruning comparisons are therefore recovery comparisons.
5. **It grounds deployment claims in measurement.** It shows from exhaustive measurement that sparsity is not speed.
   It shows directly measured energy reductions from structured realisation, together with their accuracy cost.
   And it reports where measurement itself was not reliable enough to support a claim.
6. **Every confirmatory result was pre-registered.** Frozen thresholds and leakage-guarded splits were in place before
   data, and negative verdicts are kept. The conclusions are therefore trustworthy precisely where they are modest.
