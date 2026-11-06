# Paper 3 pre-registration (final for H2; amendment 1: recovery decision)

*Hardware-Aware Reinforcement Learning for Neural Network Pruning.* Branch `paper3-hardware-aware`.
Written 2026-10-23, **before any H2 result exists** (no latency table, no recovered accuracy), and
committed before H2 starts.
- **Draft.** `paper3_preregistration_draft.md` (commit `3fc1b4e`) is kept unchanged for provenance.
  This document supersedes it wherever they differ. Everything frozen in the draft stays frozen.
- **What this amendment adds:** the recovery decision made by the PI (option c), and the settings H2
  needs.

## 1. Recovery decision (PI decision, 2026-10-23)

Every structured policy is evaluated as follows:

dense reference → physical compaction (`structured_units.slim`) → **fixed 5-epoch recovery** →
validation accuracy.

The recovery is the frozen Phase-6 fine-tuning protocol (`results/phase6/phase6_config.json`,
`experiments/phase6/phase6_lib.finetune`), reused unchanged. It was chosen for methodological
continuity with Phase 6, **not tuned for Paper 3**.

| Item | Value (Phase 6) |
|---|---|
| Epochs | 5, no early stopping, no checkpoint selection: the model after epoch 5 |
| Optimiser | Adam, lr 1e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0; a new optimiser per policy |
| Schedule | CosineAnnealingLR(T_max = 5, eta_min = 0), stepped per epoch |
| Batch / data | 128; the 45,000 training images only; random crop 32 (zero pad 4) + horizontal flip |
| BatchNorm (ResNet-8) | train mode during recovery; eval mode for every evaluation; no separate recalibration |
| Seeds | **one recovery seed for every policy:** 600 (`torch.manual_seed(600)`); data order and augmentation from `torch.Generator(600 + 10^6)`. Every policy sees the identical minibatch sequence. 600 is the Phase-6 fine-tuning seed of policy seed 500. |
| Threads | 1 per recovery process; results must be bit-identical regardless of which core runs them (checked in H2-B) |
| Implementation | `phase6_lib.finetune` loaded by file path, called with `validate=False`. The per-epoch validation in Phase 6 selected nothing, and no model in Phase 6 or Paper 3 draws from the global random stream during training, so the trained weights are unchanged. |

**Compatibility with slimmed models.** A compacted model has ordinary Conv2d/Linear/BatchNorm layers
and no pruning masks. In Phase 6 this was the dense-control path. The Phase-6 per-epoch assertions
(mask buffers unchanged, zero count not decreasing) hold trivially. Paper 3 adds a topology assertion:
the state-dict shapes after recovery equal those before, so no channel can return.

- **Same budget for everyone.** Every policy gets exactly 5 epochs of 352 minibatches with the same
  seed. No policy gets extra epochs, a different learning rate or early stopping.
- **Nothing tuned on test.** No Paper-3 hyperparameter is selected from test performance, and none is
  changed after seeing H2 results.

## 2. Accuracy used by the reward

- **A_recovered(p):** the VAL-RL accuracy (2,500 images) of policy p after the recovery.
- **A_dense:** the VAL-RL accuracy of the identity policy [0,0,0,0] **after the same recovery**, i.e.
  the dense control.
  - This means q measures the accuracy lost to pruning, not the accuracy gained from 5 more epochs.
    Phase 6 showed that extra training alone moves the CIFAR-10 SimpleCNN reference by +3 pp.
  - The VAL-RL accuracy of the un-retrained frozen reference is recorded alongside, and q against it
    is reported as secondary.
- **q(p) = A_recovered(p) / A_dense.** Feasible ⇔ q ≥ τ.
- **VAL-SELECT.** The VAL-SELECT accuracy of every recovered policy is computed in the same run and
  stored, for the frozen selection step in H4. It is **not used in any H2 analysis**: the Pareto front,
  correlations and feasibility rule all use VAL-RL only.
- **Test set.** Not read in H2.

## 3. τ rule (the Phase-5 rule, carried over; fixed before the landscape exists)

- **Phase 5.** Its rule (`phase5_tau_selection.md`) was: the largest τ in {0.95, 0.97, 0.98, 0.99}
  such that every setting has ≥ 20 feasible policies and ≥ 1 feasible policy with ≥ 10% sparsity.
- **Paper-3 form.** The efficiency criterion becomes ≥ 1 feasible policy with **latency gain
  g_L ≥ 10%** in configuration A. A non-trivial hardware gain is what H1 needs.
- **Starting value.** τ = 0.98, kept if every one of the four settings has ≥ 20 feasible policies and
  ≥ 1 feasible policy with g_L ≥ 0.10.
- **If 0.98 fails:** **STOP before PPO.** Report the feasible counts, the latency distribution of the
  feasible policies, the best feasible speed-up, and the fallback the rule would select: the largest of
  {0.97, 0.95} that passes. The PI approves before anything continues.
- **No raising.** τ is never raised above 0.98, which is the Phase-5 value. For transparency, the
  report also states what the literal Phase-5 rule (which allows 0.99) would give.

## 4. Latency tables (H2-A)

- **Protocol.** The H0 primary configuration A, unchanged (`h0_protocol.md`):
  - batch 1; 1 intra-op and 1 inter-op thread;
  - the process pinned to logical CPU 2 before torch is imported;
  - float32 NCHW, `inference_mode`, garbage collection off;
  - `perf_counter_ns`; 50 warm-up and 300 timed passes per shape.
- **Machine.** The i7-1260P laptop, AC power, Balanced plan, Python 3.11.9, torch 2.7.1+cpu. No other
  CPU-intensive job may run; the CPU load must be ≤ 15% before and after every session (H0 rule).
- **Tables.** One per (architecture, classes) pair: SimpleCNN-10, SimpleCNN-100, ResNet-8-10 and
  ResNet-8-100. Each covers all 1,296 shapes, built from the frozen dense reference of that setting.
- **Sessions.** **5 independent sessions** per table, each in a fresh process.
  - Each session times every shape once (one block of 300 passes) in a random order drawn from
    `numpy.random.default_rng` (seed recorded).
  - A dense anchor block of 300 passes is timed before the first shape and after every 25 shapes.
- **Latency of a shape:** the median over the 5 sessions of the per-session median. Also recorded:
  pooled p95, IQR, throughput, speed-up versus the identity policy of the same table, and
  g_L = 1 − L / L_identity.
- **Drift.** Anchors monitor drift: the limit is 10% from the session's anchor median (H0 rule), and a
  session that exceeds it is repeated once. Anchor-normalised latency is reported as a secondary column
  only. The primary value is raw, exactly as in H0.
- **Freeze.** The tables are frozen (sha256) and committed before any PPO run. Latency depends only on
  shape, so recovery never changes it.

## 5. Recovered-accuracy landscapes (H2-B, H2-C)

- **H2-B sanity check** (implementation only, not tuning):
  - Representative policies on CIFAR-10 SimpleCNN and ResNet-8: identity, 10% everywhere, a moderate
    mixed policy, fc-heavy or stem-heavy, conv-heavy, and aggressive.
  - Checks: training runs; shapes unchanged; no regrowth; a bit-identical rerun on a different core
    type; checkpoint reload.
  - The 5-epoch budget is not changed whatever the accuracies are. A software bug would be fixed, given
    a regression test, documented, and the sanity run repeated.
- **H2-C landscapes:** all 1,296 policies × 4 settings, each recovered once with the protocol above.
  - **Stored per policy:** action tuple; widths; parameters; MACs; VAL-RL and VAL-SELECT accuracy
    before and after recovery; training loss per epoch; the data-stream hash; the final state hash.
  - **Shared metadata:** dense reference hash, importance rule, recovery seed and settings, code
    commit.
  - Recovered checkpoints are not kept. Any policy can be regenerated deterministically from the
    stored metadata.
- **Parallelism.** Many 1-thread worker processes. Which worker or core runs a policy does not change
  its result, by the determinism requirement checked in H2-B.

## 6. H2 analysis (VAL-RL only)

- **Feasibility:** the τ rule (§3).
- **Pareto front:** the exhaustive Pareto front of (A_recovered, L) per setting, with its parameter
  and MAC reductions.
- **Correlations** (all 1,296 policies; Spearman):
  - parameter reduction ρ_s versus g_L;
  - MAC reduction g_M versus g_L;
  - the number and fraction of policy pairs where higher ρ_s comes with higher latency.
- **Also reported:** the same over the feasible policies only.

## 7. Open item not decided here: structured sensitivity

- **What the draft said.** It named `structured_loss_sensitivity` (sensitivity immediately after
  pruning, before any recovery) but listed it under *Open*. Whether the PPO prior should use
  immediate or post-recovery sensitivity is therefore **not frozen**.
- **Consequence.** As instructed, H2 does **not** compute final sensitivity vectors. The choice goes to
  the PI before H3. Whichever is chosen, all PPO conditions (C1-S, M1, H1) use the same prior, so the
  reward stays the only manipulated variable.

## 8. Unchanged from the draft

Machine and latency configuration A; action semantics, units and the importance rule; definitions of
ρ_s, g_M and g_L; settings; seeds 700–719; conditions C1-S, M1 and H1 with the Phase-5 constrained
reward form; archive and selection rule; test protection; the confirmation-latency session;
hypotheses H1–H5.
