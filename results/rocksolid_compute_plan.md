# Rock-solid roadmap — compute plan (Phase 1)

Estimates for Phases 2–8. The CPU numbers come from a measured
micro-benchmark on the current machine (2026-09-28): random tensors,
forward/backward only, 4 threads, data loading excluded, so they are lower
bounds. The GPU numbers are **assumptions** for a single mid-range NVIDIA GPU
(T4 / RTX 3060 class, fp32). They must be replaced by a measured benchmark at
the start of Phase 2. **No seed count below is reduced to fit the hardware**;
where the hardware cannot support a design, this is stated as a blocker.

## Hardware available now

Intel Core i5-6400 (4 cores / 4 threads), 7.9 GB RAM, **no NVIDIA GPU**
(Intel HD 530), CPU-only torch 2.7.1, 76.8 GB free disk, unstable mains
power (27 unexpected shutdowns in 60 days).

## Measured CPU speed (batch 128)

| Model | Params | Conv/Linear layers | Train step | ≈ min / epoch (45k) | 3,000-image eval |
|---|---|---|---|---|---|
| SimpleCNN | 0.62 M | 5 | 0.15 s | 0.9 | 1.2 s |
| ResNet-18 (CIFAR stem: 3×3 conv, no max-pool) | 11.17 M | 21 | 3.41 s | 20.0 | 23.5 s |
| MobileNetV2 (CIFAR: first stride 1) | 2.24 M | 53 | 1.42 s | 8.3 | 8.5 s |

GPU assumptions (to be measured): ResNet-18 ≈ 25 s/epoch, MobileNetV2 ≈ 25 s/epoch,
3,000-image evaluation ≈ 0.3 s.

## Settings

Six (architecture, dataset) settings: SimpleCNN-C10 (existing, verified),
SimpleCNN-C100, ResNet-18-C10, ResNet-18-C100, MobileNetV2-C10,
MobileNetV2-C100. Five are new.

Seed policy used below (defined in `rocksolid_master_protocol.md`): 3 dense
training seeds per setting; 20 PPO seeds per PPO condition per setting;
deterministic baselines need no seeds; stochastic baselines 10 seeds.

---

## Phase 2 — dense baselines

| Task | Training runs | Inference-only | GPU h | CPU h | Disk |
|---|---|---|---|---|---|
| ResNet-18 C10 + C100 (3 seeds each, 200 epochs) | 6 | per-epoch validation | 8–12 | ≈ 400+ (infeasible) | 0.3 GB |
| MobileNetV2 C10 + C100 (3 seeds each, 200 epochs) | 6 | same | 8–12 | ≈ 170 | 0.06 GB |
| SimpleCNN C100 (3 seeds; recipe to be fixed) | 3 | same | < 0.1 | 0.5–1.5 | < 0.01 GB |
| Optional: SimpleCNN C10 seeds 2–3 (dense variance only; FIXED stays the reference) | 2 | same | < 0.1 | 0.3 | — |
| Baseline cards (hash, val/test, params, FLOPs) | — | 17 evaluations | < 0.1 | 1–2 | — |
| **Total** | **15–17** | | **≈ 16–25** | CPU-only: ≈ 600 h (not viable) | ≈ 0.5 GB + CIFAR-100 0.16 GB |

Reuse: none (new models). Dependencies: GPU access, CIFAR-100 download, a
fixed training recipe (Phase-2 pre-registration).

## Phase 3 — architecture / dataset PPO and stronger baselines

PPO budget is defined in **episodes** (512 per run, as in all SimpleCNN
studies); timesteps = 512 × number of decision steps (21 for ResNet-18; 53
for MobileNetV2 unless a layer-grouping rule is pre-registered).
Memoisation is ineffective for large networks (6²¹ policies), so every
episode costs one validation evaluation.

| Task | Training runs | Inference-only | GPU h | CPU h |
|---|---|---|---|---|
| PPO, 2 conditions (no prior, correct prior) × 20 seeds × 5 new settings | 200 PPO | 512 evals / run | ≈ 20–30 | ≈ 200 × 3.3 h ≈ 660 (infeasible) |
| + shuffled and constant prior controls (needed to keep sensitivity claims layer-specific) | +200 PPO | same | +20–30 | +660 |
| One-shot baselines (uniform, global magnitude, LAMP, ERK, layer-adaptive magnitude) at matched sparsities | — | ≈ 300 evaluations | ≈ 1 | ≈ 3–10 |
| Random pruning (10 seeds × matched sparsities) | — | ≈ 150 evaluations | < 1 | ≈ 2–5 |
| Iterative magnitude pruning + fine-tuning at the matched target (3 seeds × 5 settings) | 15 | — | ≈ 15–25 | infeasible |
| **Total** | **215–415** | ≈ 250k evaluations | **≈ 55–85** | not viable on CPU |

Reuse: sensitivity curves and baseline sweeps are deterministic per
reference model (compute once). Test evaluations of frozen policies are
cached by policy. Dependencies: Phase-2 reference models and hashes.

## Phase 4 — sensitivity mechanism and reward analysis

| Task | Training runs | Inference-only | GPU h | CPU h |
|---|---|---|---|---|
| Sensitivity curves + folds + predictive validity per new setting (≤ 21–53 layers × 5 ratios × 6 scopes) | — | ≈ 1,500 evaluations | ≈ 1–2 | ≈ 10 (ResNet on CPU) |
| Reward-design grid on SimpleCNN (diversity coefficient {0, 0.05} × λ_s {0, 0.01, 0.02, 0.04, 0.08} × 20 seeds) | 200 PPO | memoised (exhaustive landscape exists) | — | ≈ 5–8 |
| Reward sub-study on ResNet-18 C10 (diversity {0, 0.05} × λ_s {0.01, 0.04} × 10 seeds) | 40 PPO | 512 evals / run | ≈ 5 | infeasible |
| **Total** | **240** | | **≈ 6–7** | **≈ 15–20** |

SimpleCNN parts can run on this machine; reuse the 1,296-policy landscape.
Dependencies: Phase 3 for the ResNet sub-study.

## Phase 5 — structured pruning

| Task | Training runs | Inference-only | GPU h | CPU h |
|---|---|---|---|---|
| Channel/filter pruning PPO (2 conditions × 20 seeds × ResNet-18 and MobileNetV2 on C10) | 80 PPO | 512 evals / run | ≈ 10–15 | infeasible |
| Structured baselines (uniform channel, global L1 channel, network-slimming-style) | — | ≈ 100 evaluations | < 1 | — |
| Fine-tuning of every final structured model (3 epochs-equivalent budget per protocol) | ≈ 120 FT | — | ≈ 25–40 | infeasible |
| SimpleCNN structured (full design, CPU) | ≈ 60 PPO + 60 FT | memoisable | — | ≈ 10–20 |
| **Total** | **≈ 320** | | **≈ 35–55** | **≈ 10–20** |

Dependencies: Phase 2 models; a dependency-aware channel-pruning
implementation (new dependency, see master protocol).

## Phase 6 — latency, FLOPs, memory

| Task | Inference-only | GPU h | CPU h |
|---|---|---|---|
| FLOP / parameter counts for every final model | ≈ 400 counts | — | < 1 |
| CPU latency (this machine, existing protocol: 100 warm-up + 600 timed passes, batches 1 and 128) for dense / unstructured / structured per setting | ≈ 60 model × batch combinations | — | ≈ 5–10 (idle machine required) |
| GPU latency (CUDA synchronised) | same | ≈ 1–2 | — |
| Peak memory (activations + weights) | same | < 1 | < 1 |
| **Total** | | **≈ 2–3** | **≈ 6–12** |

Reuse: storage and CPU latency of SimpleCNN are already verified.
Dependencies: Phase 3 and Phase 5 final models.

## Phase 7 — fair fine-tuning

| Task | Training runs | GPU h | CPU h |
|---|---|---|---|
| Identical FT protocol for every final model: dense-FT control, uniform, global magnitude, LAMP, ERK (5 FT seeds each), PPO and PPO+prior (the 20 seed-specific models each) — 6 settings | ≈ 6 × (5 × 5 + 2 × 20) ≈ 390 | ≈ 35–45 | SimpleCNN part ≈ 5 |
| **Total** | **≈ 390** | **≈ 35–45** | **≈ 5** |

Dependencies: Phase 3 (unstructured) and Phase 5 (structured) final models.

## Phase 8 — robustness and calibration

| Task | Inference-only | GPU h | CPU h | Disk |
|---|---|---|---|---|
| CIFAR-10-C / CIFAR-100-C (19 corruptions × 5 severities) on every final model | ≈ 100 models × 950k images | ≈ 5–8 | SimpleCNN ≈ 3; ResNet infeasible | ≈ 6 GB datasets |
| Calibration (ECE, NLL, Brier) on clean test | ≈ 400 models | < 1 | ≈ 2 | — |
| **Total** | | **≈ 6–9** | **≈ 5** | **≈ 6 GB** |

Dependencies: final models of Phases 3, 5 and 7.

---

## Summary

| Phase | Training runs | GPU h (assumed) | CPU h on this machine | Viable here? |
|---|---|---|---|---|
| 2 | 15–17 | 16–25 | ≈ 600 | **No** (SimpleCNN-C100 part only) |
| 3 | 215–415 | 55–85 | > 1,300 | **No** (SimpleCNN-C100 part only) |
| 4 | 240 | 6–7 | 15–20 | SimpleCNN parts yes |
| 5 | ≈ 320 | 35–55 | 10–20 (SimpleCNN) | SimpleCNN only |
| 6 | — | 2–3 | 6–12 | CPU latency yes |
| 7 | ≈ 390 | 35–45 | ≈ 5 (SimpleCNN) | SimpleCNN only |
| 8 | — | 6–9 | ≈ 5 (SimpleCNN) | SimpleCNN only |
| **All** | **≈ 1,200–1,400** | **≈ 155–230** | **> 2,000** | **needs a GPU** |

**Disk:** about 20 GB in total, assuming dense and selected final models are
stored in full (ResNet-18 45 MB, MobileNetV2 9 MB each), other models as
policies/masks, plus about 6 GB of corruption datasets. 76.8 GB is free.

**Deterministic reuse:** reference-model sensitivity curves, one-shot
baseline sweeps, validation landscapes (exhaustive only for SimpleCNN), test
evaluations of frozen policies, CIFAR-C evaluations of frozen models, and
every existing SimpleCNN-C10 result listed as VERIFIED in the audit.

**Main blocker:** GPU access (university cluster, Kaggle or Colab GPU
sessions, or a rented GPU) with a CUDA build of the pinned stack. Without
it, only the SimpleCNN parts of Phases 4–8 and SimpleCNN-C100 are feasible.
