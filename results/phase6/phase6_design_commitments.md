# Phase 6 design commitments (Part A of the pre-registration)

Written 2026-10-01 on branch `phase6-cpu` (from `da061e6`), **before** the timing pilot and before any
fine-tuning run. This file is committed first and never edited. The full pre-registration follows the
pilot and must be consistent with it.

## Frozen evidence

- **Phases 3–5.** Unchanged, verified by `git diff` (zero lines) against `67f4270`, `a3653fd` and
  `da061e6`.
- **Dense references** (sha256 prefixes): `ca28f834`, `5ca6e0fc`, `41fc5ee5`, `d75dd19c`, `8e1b1aee`,
  `8b84bfad`.
- **Splits.**
  - Train: 45,000 images.
  - VAL-RL / VAL-SELECT: 2,500 / 2,500 (`cifar10_phase5_split.npz` `eccd7522`,
    `cifar100_phase5_split.npz` `131cf534`).
  - Official test: 10,000 images.
- **Phase-5 final policies.** `phase5_final_policies.json` (`d04c8556`): the 120 C1 selected policies
  (20 policy seeds 500–519 × 6 settings). Nothing is re-trained or re-selected.

## Methods and matching

- **M0 PPO.** The frozen Phase-5 C1 policy of each (setting, policy seed).
- **M1 LAMP** and **M2 global magnitude.** The existing `src.pruning` implementations, applied to the
  same dense checkpoint at **exactly the PPO model's zero count** (asserted). The scope is the same
  prunable tensors; the output classifier is dense in all three.
- **Dense control.** The dense reference fine-tuned with the same protocol and seed, with no mask.

## Fine-tuning protocol (identical for every method and setting)

| Item | Value |
|---|---|
| Start | the frozen dense checkpoint with the method's mask applied; a new optimiser for every run |
| Optimiser | Adam, lr 1e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0 (one tenth of the dense-training learning rate; the project's earlier fine-tuning learning rate) |
| Schedule | cosine annealing to 0 over E epochs, stepped per epoch |
| Batch / data | 128; the 45,000 training images; random crop (pad 4) + horizontal flip |
| Seeds | fine-tuning seed = policy seed + 100 (600–619); `torch.manual_seed(seed)`; data order and augmentation from `torch.Generator(seed + 10^6)`; identical for the four runs of a (setting, policy seed) |
| BatchNorm | train mode during fine-tuning for every method; eval mode for every evaluation; no separate recalibration |
| Mask | fixed: PyTorch pruning reparameterisation (effective weight = weight_orig × mask); mask hash, mask violations and zero count are verified after every epoch |
| Selection | **none**: the model after the final epoch E is the result for every method |
| Threads | 1 per run; 4 worker processes |

**Epoch count E.** Decided by timing only, after the pilot:
- **E = 5** if the projected wall time of the whole confirmatory experiment (480 runs + 6 reproduction
  runs on 4 workers) is ≤ 30 hours;
- otherwise **E = 3**.

No accuracy from the pilot is used.

## Pilot (non-confirmatory)

- **Inputs.** Seed 42 and the uniform policy `[2,2,2,2]`, which is not a Phase-5 policy.
- **Test set.** Never read.
- **Measured.** Time per epoch under 4 concurrent 1-thread workers, and peak memory.
- **Checked.**
  - mask preservation;
  - bit-reproducibility;
  - resume equivalence;
  - checkpoint serialisation;
  - reproduction of Phase-5 pruned models, by VAL-RL accuracy and zero count only.

## Test-set disclosure

The official test sets were read in Phases 2–5 for frozen models. Phase 6 therefore cannot claim a fully
independent confirmation.
- No Phase-6 design choice (learning rate, epochs, schedule, methods) uses test accuracy.
- Test is read only for the pre-fine-tuning model and the final-epoch model of each run.
