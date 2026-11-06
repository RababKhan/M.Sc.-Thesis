# Paper 3 pre-registration — DRAFT (after H0 and H1, before H2)

*Hardware-Aware Reinforcement Learning for Neural Network Pruning.* Branch `paper3-hardware-aware`.
Written 2026-10-23. This is a **draft**: the items under "Frozen now" rest on validated H0/H1 facts;
the items under "Open" must be decided, and the final pre-registration committed, before H2 starts.
No Paper-3 accuracy landscape, latency table, PPO run or test evaluation exists yet.

## Research question

Can PPO learn structured layer-wise pruning policies that keep validation accuracy within a fixed
tolerance while minimising **measured CPU inference latency**, rather than structured sparsity alone?

## Frozen now (validated in H0/H1)

### Machine and latency protocol

- **Machine.** All Paper-3 latency is measured on one machine: Intel Core i7-1260P laptop, Windows 11,
  torch 2.7.1+cpu (oneDNN 3.7.1, MKL), Python 3.11.9.
- **Conditions.** AC power is required; the power plan is recorded; no other experiment may run.
- **Not transferable.** These numbers do not carry over to the Phase 1–6 i5-6400 desktop or to any
  other CPU.
- **Primary latency metric.** Configuration **A** (`h0_protocol.md`, selected mechanically in H0):
  - batch 1, 1 intra-op thread, 1 inter-op thread;
  - process pinned to logical CPU 2, a P-core thread;
  - float32 NCHW, `inference_mode`, `perf_counter_ns`, garbage collection off during timing;
  - 50 warm-up and 300 timed passes per model;
  - a model's latency is the median of its passes in one session.
- **Secondary metric.** Configuration C (batch 128, 4 threads on logical CPUs 0, 2, 4, 6), reported but
  never used in a reward.
- **H0 evidence.** In configuration A the dense SimpleCNN takes 0.680 ms and the aggressive [5,5,5,5]
  model 0.261 ms (2.60×). SNR is 12.3; the dense between-session CV is 2.1% and anchor drift at most
  1.6%.

### Action semantics and units

- **Actions.** a ∈ {0, …, 5} removes {0, 10, 20, 30, 40, 60}% of a unit's channels, keeping
  `max(1, round((1 − r)·C))` channels with the largest L1 norm (ties: lower index). All scores are
  computed on the dense model. Code: `src/pruning/structured_units.slim` over the verified Phase-2
  `src/pruning/structured.py`.
- **SimpleCNN units:** conv1 (32), conv2 (64), conv3 (128) output filters and fc1 hidden neurons (256).
  The output classifier is never pruned.
- **ResNet-8 units:**
  - **stem**: the coupled stage-1 residual width (16). The same channel set is removed from the stem
    conv and bn, layer1.conv1 inputs, layer1.conv2 and bn2 outputs, layer2.conv1 inputs and
    layer2.shortcut inputs; the layer1 shortcut stays an identity. Score = L1 of the two filters that
    write the channel.
  - **stage1, stage2, stage3**: the block-internal channels (16, 32, 64).
  - The stage-2/3 output widths and the classifier are not pruned.
- **Search space.** 6⁴ = 1,296 policies per architecture, each a distinct physical shape (verified for
  both architectures).

### Definitions

- **Physical parameters:** all parameters of the compacted model (`count_parameters`).
- **Structured parameter reduction:** ρ_s = 1 − params / params_dense.
- **MACs:** conv/linear multiply-accumulates per image, from analytic forward hooks. FlopCounter FLOPs
  equal 2 × MACs; ptflops (which adds BN, ReLU and pooling) is reported alongside.
- **MAC gain:** g_M = 1 − MACs / MACs_dense.
- **Latency gain:** g_L = 1 − L / L_dense, both from the same latency table (configuration A).

### Settings and seeds

- **Settings.** CIFAR-10 and CIFAR-100 × SimpleCNN and ResNet-8 (audit Option B), from the frozen
  Phase-2 dense references (sha256 asserted).
- **Seeds.** 20 PPO seeds per condition and setting (proposed 700–719, disjoint from earlier phases).
  PPO keeps the Phase-5 settings: 512 episodes, centred structured-sensitivity prior.

### Conditions

All three use the Phase-5 accuracy-constrained form with q = A / A_dense on VAL-RL:
R = 1 + gain + 0.01·q if q ≥ τ, otherwise −10·(τ − q).

| Condition | Gain in the reward |
|---|---|
| C1-S | ρ_s (structured parameter reduction) |
| M1 | g_M (MAC reduction) |
| H1 | g_L (measured latency reduction, from the table) |

The latency and accuracy tables are precomputed exhaustively, so every reward is deterministic. There
is no live timing inside PPO.

### Data and test protection

- **Splits.** Phase-5 VAL-RL and VAL-SELECT (2,500 / 2,500) for reward and selection; the 45,000
  training images only for any training or BatchNorm re-estimation.
- **Test set.** Read once, for the frozen final models only. It was read in earlier phases, which will
  be disclosed.

### Selection and confirmation

- **Archive.** The top-10 unique policies of each run by VAL-RL reward.
- **Selection.** On VAL-SELECT: among candidates with q_SELECT ≥ τ, the one with the highest gain of
  the run's own condition; ties go to higher VAL-SELECT accuracy, then earlier discovery.
- **Freeze.** The final-policy list is frozen and committed before test access.
- **Confirmation-latency session.** The final models are re-measured under configuration A in fresh
  processes on a different day from the table sessions. These are the reported latencies; the table
  values feed the reward only.

### Hypotheses (to be tested in H5)

- **H1.** Sparsity is an imperfect proxy for CPU latency. Tested on the exhaustive tables, without
  seeds: Spearman ρ of ρ_s versus g_L and of g_M versus g_L, plus the count of policy pairs where one
  has the higher sparsity but also the higher latency. H0 already shows the pattern: fc_heavy removes
  51% of parameters for a 1.03× speed-up, while conv_heavy removes 63% for 1.47×.
- **H2.** The objective changes the per-unit allocation: the selected policies differ between C1-S and
  H1 (per-unit actions, permutation test).
- **H3.** At validation-feasible accuracy, H1 reaches a lower confirmed latency than C1-S (paired by
  seed), with test accuracy reported and compared with an equivalence margin.
- **H4.** H1 lies closer to the exhaustive accuracy–latency Pareto front (per-seed regret and
  hypervolume).
- **H5.** Measured latency adds information beyond MACs (H1 versus M1). If they select the same
  policies, that is reported as MACs being a sufficient proxy on this CPU.

## Open — must be decided before H2 (not frozen)

1. **Accuracy-recovery treatment** (critical; H1 sanity data, VAL-RL, CIFAR-10).

   | Policy | SimpleCNN retention | ResNet-8 retention |
   |---|---|---|
   | [1,1,1,1] (10% of channels) | 0.965 | 0.27 |
   | conv_heavy (SimpleCNN only) | 0.34 | — |
   | fc_heavy (SimpleCNN only) | 0.987 | — |

   Without recovery, almost every latency-relevant policy is infeasible at τ = 0.98. Options:
   - (a) no recovery, with a lower τ chosen by rule;
   - (b) BatchNorm re-estimation on training images for every policy (cheap; ResNet-8 only, since
     SimpleCNN has no BN);
   - (c) a short fixed fine-tuning budget per policy (expensive for 1,296 policies);
   - (d) a combination.

   This choice defines what "accuracy" means in the reward and must be fixed first.
2. **τ.** Chosen by a pre-registered rule on the H2 VAL-RL landscape, as in Phase 5 (minimum number
   of feasible policies and of efficiency spread), after item 1 is fixed.
3. **Final structured sensitivity vectors.** Computed in H2 on VAL-RL with
   `structured_loss_sensitivity` (implemented and unit-tested in H1), then the centred prior.
4. **Latency-table session layout.** 5 independent sessions per (architecture, classes), with dense
   anchors every 25 shapes and a recorded random shape order. To be confirmed in the final
   pre-registration.
5. **Baselines.** Uniform L1-filter pruning and a global L1-filter ranking, matched on confirmed latency
   and on MACs, plus the exhaustive oracle.
