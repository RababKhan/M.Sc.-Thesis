# Rock-solid roadmap — CPU master protocol

Written at the start of Phase 2 (2026-09-28), before any Phase-2 benchmark or
training run. It adapts `rocksolid_master_protocol.md` (Phase 1, unchanged and
kept for provenance) to the project's permanent hardware: a 4-core desktop CPU
with no CUDA GPU. Every rule of the Phase-1 protocol that does not depend on
hardware still applies and is cited here by section number rather than
repeated. Where the two documents differ, this one governs from Phase 2 on.

## 1. Scope: compact CNNs on commodity CPUs

The project studies reinforcement-learning-based layer-wise pruning of
**compact convolutional networks on commodity CPU hardware**. This is a
deliberate scope, not a workaround:

- **Deployment relevance.** Small CNNs that run on CPUs without accelerators
  are the networks most often deployed on CPU-class edge and desktop devices.
- **Reproducibility.** Every experiment can be repeated end to end on an
  ordinary desktop (Intel Core i5-6400, 4 cores, 7.9 GB RAM, no GPU). No
  cloud credits or special hardware are needed.
- **Complete evaluation.** Every setting has a small action space
  (4 pruning units × 6 ratios = 1,296 policies). The full validation
  landscape of every setting can therefore be computed exactly. PPO is judged
  against the true optimum and against every one-shot baseline, not against
  a sample.
- **Statistical power.** The per-run cost is low, which lets the study keep
  the full seed counts (20 PPO seeds per condition, 10 random-baseline seeds,
  3 dense seeds) in every setting instead of trading seeds for model size.

Claims are limited to this regime: CIFAR-scale images, networks under
≈ 1 M parameters and ≈ 40 M multiply-accumulates per image, and CPU
inference. The paper does not extrapolate to ImageNet-scale data, to large
networks (ResNet-18/50, MobileNetV2) or to GPU latency. It states this
boundary as a scope, not as a missing experiment.

## 2. Research goals (unchanged from Phase 1)

1. Architecture generalisation: three genuinely different CNN families.
2. Dataset generalisation: CIFAR-10 (primary) and CIFAR-100.
3. Fair and strong baselines: uniform, global magnitude, random, LAMP and
   ERK, all at exactly matched sparsity.
4. The sensitivity mechanism: whether the per-unit sensitivity helps, with
   correct, shuffled and constant controls.
5. Reward-design analysis.
6. Structured pruning with real channel removal.
7. Real CPU efficiency: parameters, MACs/FLOPs, storage, CPU latency and
   peak RAM.
8. Fair fine-tuning.
9. Robustness (CIFAR-10-C/100-C) and calibration.
10. Reproducibility.

## 3. Settings

| | CIFAR-10 (primary) | CIFAR-100 (second) |
|---|---|---|
| A SimpleCNN | existing `cnn_baseline_FIXED.pth` (reused as is) | new, 3 seeds |
| B LeNet-5 (CIFAR) | new, 3 seeds | new, 3 seeds |
| C ResNet-8 (fallback: SmallVGG) | new, 3 seeds | new, 3 seeds |

- **Fashion-MNIST** is optional. It is a supplementary, clearly labelled
  *easier* dataset, considered only after CIFAR-100 is complete. It never
  replaces CIFAR-100, unless CIFAR-100 infeasibility is documented before any
  CIFAR-100 result exists.

## 4. Architectures and pruning units

Definitions are in `src/models/__init__.py` and the unit registry in
`src/pruning/__init__.py`. Every architecture has **exactly four pruning
units**. The output classifier is always excluded (as `classifier.3` was for
SimpleCNN), and biases and BatchNorm parameters are never pruned.

| Arch | Structure | Units (decision order) |
|---|---|---|
| A SimpleCNN | 3 × (3×3 conv, ReLU, max-pool) 32/64/128, FC 2048→256→C | `features.0`, `features.3`, `features.6`, `classifier.1` |
| B LeNet-5 (CIFAR) | 2 × (5×5 valid conv, ReLU, max-pool) 6/16, FC 400→120→84→C | `conv1`, `conv2`, `fc1`, `fc2` |
| C ResNet-8 | He et al. (2016) CIFAR ResNet, n = 1: 3×3 stem (16), 3 stages of one BasicBlock (16/32/64; strides 1/2/2; 1×1 projection shortcuts), BN, GAP, FC | `stem`, `stage1` (2 convs), `stage2` (2 convs + shortcut), `stage3` (2 convs + shortcut) |
| C′ SmallVGG (fallback) | 3 stages × 2 (3×3 conv, BN, ReLU) 32/64/128 with max-pool, FC 2048→256→C | `stage1`, `stage2`, `stage3`, `fc1` |

- **Architecture fallback.** C′ replaces C only under the timing rule of
  `results/cpu_runtime_budget.md`. That rule uses measured seconds per epoch
  and never accuracy.
- **Unit action.** A PPO action applies one ratio from
  {0, 10, 20, 30, 40, 60}% to every tensor of the unit (L1 unstructured, per
  tensor).
- **Unit state.** Statistics are computed over the unit's concatenated
  weights.
- **Parameter table.** `results/architecture_generalization/cpu_prunable_units.csv`
  lists the parameter counts, fractions and unit types.

## 5. Data (Phase-1 §2–3 apply unchanged)

| Dataset | Train / validation | V_RL / V_SELECT (of validation) | Test |
|---|---|---|---|
| CIFAR-10 | seed-42 `randperm`, 45,000 / 5,000 (val hash `9d3648af…`) | `archive_validation_split.npz`, 3,000 / 2,000, stratified | official 10,000 |
| CIFAR-100 | the same rule, 45,000 / 5,000 | stratified 3,000 / 2,000, the same algorithm and seed (20260928); `results/reproducibility/cifar100_split_indices.npz` | official 10,000 |

- **Normalisation.** Per-dataset channel mean and pixel standard deviation
  of the official 50,000-image training set (no test images).
- **Augmentation** (training only): random crop 32 with 4-pixel zero padding,
  and random horizontal flip.
- **Data use.**
  - Train updates weights.
  - Full validation selects the dense training epoch.
  - V_RL provides the PPO reward, sensitivity and training diagnostics.
  - V_SELECT is used only for post-hoc choice among already generated
    candidates.
  - Test is read once per frozen model or policy.
- **Enforcement.** `src.data.DataBundle` enforces the data-use rules. Test
  access raises during `training()` and before `freeze()`. V_SELECT access
  raises during `training()`.

## 6. Dense training

The recipe is pre-registered in `results/cpu_dense_training_protocol.md`: the
same Adam + cosine recipe for every architecture, 30 epochs, and validation
checkpoint selection.

- **Seeds and reference model.** Training seeds are 0, 1 and 2. The reference
  model of each new setting is **seed 0**, fixed by rule. It is not selected by
  validation or test accuracy.
- **SimpleCNN-C10 exception.** Its reference remains
  `cnn_baseline_FIXED.pth`, because every verified historical result depends
  on it. It was trained with the historical recipe (Adam 1e-3, 10 epochs,
  final epoch), and this is reported as a recipe difference.

## 7. Methods

- **PPO.** The historical formulation, unchanged (Phase-1 §5):
  - reward 1.5·acc/base + λ_s·S(%) + 0.05·distinct actions;
  - 7-feature state per unit;
  - SB3 2.8.0 PPO with the historical hyper-parameters;
  - 512 episodes per run, which is 2,048 timesteps for four units.

  Reward accuracy is measured on V_RL, and the baseline denominator is the
  reference model's V_RL accuracy.
- **Exact memoisation.** Every setting's 1,296 policies are deterministic
  functions of the frozen reference model. The terminal evaluation of an
  episode is memoised: an action tuple is evaluated on V_RL once per setting
  and then reused. On a cache hit, the environment still creates the
  DataLoader iterator, so the global torch random stream, and therefore the
  whole PPO run, is bit-identical to an uncached run. This was verified
  historically and again in Phase 2
  (`results/reproducibility/library_verification.json`). Memoisation is the
  main reason PPO stays at SimpleCNN-scale cost on every architecture.
- **One-shot baselines.** Uniform, global magnitude, random (10 seeds), LAMP
  and ERK, each calibrated to the exact zero count of the compared model. The
  formulas are in the `src/pruning/__init__.py` docstring and are summarised
  in `results/architecture_generalization/cpu_phase2_checks.md`.
- **Sensitivity conditions.** No prior, correct prior, shuffled prior and
  constant prior (Phase-1 §5), for the loss-based sensitivity measured on
  V_RL.
- **Structured pruning.** Physical filter/neuron removal:
  - SimpleCNN: 4 units;
  - ResNet-8: 3 block-internal units, since the residual additions tie the
    stage output widths.

  `src/pruning/structured.py` implements it; its experiments start in a
  later phase.
- **Fine-tuning** follows the Phase-1 §8 fairness rules unchanged: a fixed
  mask, an identical budget, and a dense fine-tuned control.

## 8. Seeds (Phase-1 §6 unchanged)

- **Counts.**
  - dense: 3 per setting;
  - PPO: 20 per condition per setting;
  - random baseline: 10;
  - fine-tuning of deterministic baselines: 5.
- **No silent reduction.** CPU feasibility comes from the design (compact
  networks, four units, exact memoisation), **not from reducing seeds**. Any
  reduction would have to be decided and justified in a pre-registration
  before the runs.
- **Seed registry.**
  - PPO seeds already used: 1–19, 42, 100–129, 200–239. The next free block
    is 300+.
  - Dense training seeds form a separate namespace (0–2).
  - Random-baseline seeds are drawn from 1000+.

## 9. Matched sparsity, fine-tuning fairness, statistics

Phase-1 §7, §8 and §9 apply unchanged:
- excess over matched global magnitude;
- the matched-sparsity tolerance rule;
- the exact sign-flip test with the seed as the unit of analysis;
- Holm correction within families;
- mechanically computed pre-registered classifications;
- no interim peeking.

The statistics are implemented once, in `src/statistics`, and verified
against the historical implementations.

## 10. Efficiency (replaces Phase-1 §10)

`results/cpu_efficiency_protocol.md` governs efficiency measurement:
parameters (dense and non-zero), raw and gzip storage, MACs/FLOPs from a
pinned counter, CPU latency at batch 1/32/128 and peak RAM.

- **No GPU latency** is measured or claimed.
- **Unstructured sparsity** is never reported as a speed-up. Real speed
  claims are made only for structurally slimmed networks.

## 11. Reproducibility (replaces the hardware parts of Phase-1 §11)

- **Environment.** The Phase-1 lock (`requirements_locked.txt`, Python
  3.11.9, torch 2.7.1+cpu, SB3 2.8.0) plus `ptflops==0.7.5`. The Phase-2 lock
  is `requirements_locked_phase2.txt`.
- **Threads.** Each run records its thread count. A resumed run must use the
  same count.
  - PPO runs were verified bit-identical at 1, 2 and 4 threads.
  - Dense training is reproducible at a fixed thread count; its phase check
    is in `cpu_phase2_checks.md`.
- **Evaluation thread count.** Different thread counts change float
  reduction order in the last bits:
  - losses differ at the 1e-8 level;
  - an accuracy can change when two logits tie exactly.

  Example: SimpleCNN policy `[3,4,2,2]` on validation image 3771 gives 73.24%
  at 1 thread and 73.26% at 2 or 4 threads, in both the historical and the new
  code. The rules are therefore:
  - evaluations that are compared with one another (landscapes, PPO rewards,
    one-shot baselines, V_SELECT, test) use **1 thread**, as every historical
    landscape and PPO evaluation did;
  - loss-based sensitivity is computed at the thread count recorded with it
    (the archive vector used 4 threads);
  - dense training and its per-epoch validation use the run's recorded count.
- **Power cuts.** Every runner is resumable:
  - dense training at epoch granularity with atomic state files;
  - PPO at run granularity.
  - Integrity checks follow any interruption.
- **Library.** New work uses `src/`. It reproduced the historical SimpleCNN
  results exactly before any new run
  (`results/reproducibility/library_verification.json`). Historical scripts
  stay frozen.
- **Provenance.** Every run record holds the sha256 of its checkpoint, split
  file, protocol and code; the git commit; the environment; the thread count;
  and the runtime.

## 12. Compute budget

`results/cpu_runtime_budget.md` pre-registers:
- the per-run limits;
- the fallback rule;
- the phase budgets.

A phase whose projected cost exceeds its budget is re-designed before it
starts; seeds are never cut silently.
