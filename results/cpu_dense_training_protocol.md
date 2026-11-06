# Dense training protocol (Phase 2, pre-registered)

Written and committed before the one-epoch benchmark and before any dense
training run. It fixes every choice in advance. No hyper-parameter is
selected using validation or test accuracy.

## Runs

| Dataset | Architecture | Seeds | Status |
|---|---|---|---|
| CIFAR-10 | SimpleCNN | — | reuse `checkpoints/cnn_baseline_FIXED.pth` (not retrained) |
| CIFAR-10 | LeNet-5 | 0, 1, 2 | new |
| CIFAR-10 | ResNet-8 (or SmallVGG under budget rule R3) | 0, 1, 2 | new |
| CIFAR-100 | SimpleCNN | 0, 1, 2 | new |
| CIFAR-100 | LeNet-5 | 0, 1, 2 | new |
| CIFAR-100 | ResNet-8 (or SmallVGG) | 0, 1, 2 | new |

That is 15 new runs.

- **Pruning reference.** The **pruning reference** of each new setting is
  its **seed-0** run, fixed by this rule now. It is never chosen by
  validation or test accuracy.
- **Seeds 1–2.** These exist only to report dense-training variance.

## Recipe (identical for every architecture and dataset)

| Item | Value |
|---|---|
| Optimiser | Adam, lr 1e-3, betas (0.9, 0.999), eps 1e-8, weight decay 0 (the optimiser and learning rate of the SimpleCNN-C10 reference) |
| Schedule | cosine annealing from 1e-3 to 0 over 30 epochs, stepped once per epoch |
| Epochs | 30; no early stopping |
| Batch | 128; a new permutation every epoch; last partial batch kept (352 steps per epoch) |
| Augmentation (train only) | random crop 32 with 4-pixel zero padding + random horizontal flip |
| Normalisation | per-dataset channel mean / pixel std (`src/data`) |
| Loss | cross-entropy, no label smoothing |
| Initialisation | PyTorch defaults; ResNet-8 convs Kaiming-normal (fan-out), BN weight 1 and bias 0 (`src/models`) |
| Randomness | `torch.manual_seed(s)` before construction (initial weights); `torch.Generator(s + 10^6)` for permutations and augmentation |
| Threads | fixed per run and recorded; a resumed run must use the same count |

- **Stability exception.** This is the only permitted architecture-specific
  learning rate. It is decided in the one-epoch benchmark from **training
  loss only**.
- **Trigger.** It applies to an architecture if either:
  - any training loss is non-finite; or
  - the benchmark epoch's mean training loss is not below ln(C), the loss of
    uniform guessing (2.303 for C10, 4.605 for C100).
- **Effect.** That architecture uses lr 3e-4 for all its runs.
- **Record.** The decision is recorded before any full run.

## Checkpoint selection and the test set

- **Validation each epoch.** After every epoch, the full 5,000-image
  validation split is evaluated (accuracy and mean loss).
- **Selected checkpoint.** The epoch with the **highest validation
  accuracy**; ties go to the earliest epoch.
- **Freeze, then test.**
  1. The selected weights are written to
     `checkpoints/phase2_dense/<dataset>/<arch>_seed<s>.pth` and hashed.
  2. Only then is the test set read, once
     (`DataBundle.freeze` → `test_loader`).
  3. Test predictions are stored in the run record for later paired tests.
- **Guard.** Test access raises during training (`DataBundle.training()`).
- **Validation reuse.** Dense epoch selection uses the whole validation
  split, which later is partitioned into V_RL and V_SELECT. This affects every
  pruning method compared on a setting identically, so it cannot favour any
  method. It is reported as a limitation.

## Resumability and records

- **Resumability.** After every epoch the complete state is written
  atomically: model, optimiser, scheduler, data generator, history and best
  checkpoint. An interrupted run resumes at the next epoch, bit-identical to
  an uninterrupted run at the same thread count.
- **Run records.** Each run writes
  `results/architecture_generalization/dense_runs/<dataset>_<arch>_seed<s>.json`
  holding:
  - the per-epoch history;
  - the selected epoch;
  - validation and test accuracy;
  - checkpoint and state-dict sha256;
  - the initial-weights hash;
  - parameter counts;
  - timing, threads, environment, git commit;
  - the sha256 of this protocol, of the split files and of the code.
- **Summaries.**
  - `cpu_baseline_runs.csv`: one row per run.
  - `cpu_baselines.csv`: per setting, mean ± SD over seeds 0–2 and the
    reference (seed 0) row.
