# Appendices

These appendices give supporting detail for Chapters 3–6, generated from the frozen phase records by
`tools/build_appendices.py`. Each table names its source record. Complete per-seed, per-session and
per-comparison records (all statistical families, raw energy blocks, per-seed policies) are preserved in the
repository and are summarised, not reproduced, here. The central results and conclusions are in the main chapters.

## Appendix A — Architectures, Hyperparameters and Reproducibility

### A.1 Pruning units

**Table A.1. Pruning units, weight shapes, parameters and MACs per image (CIFAR-10 output layer).**

| Architecture | Unit | Role | Modules | Weight shapes | Weights | Share of prunable | MACs per image |
|---|---|---|---|---|---|---|---|
| simplecnn | features.0 | prunable (PPO decision step) | features.0 | 32x3x3x3 | 864 | 0.14% | 884,736 |
| simplecnn | features.3 | prunable (PPO decision step) | features.3 | 64x32x3x3 | 18,432 | 2.99% | 4,718,592 |
| simplecnn | features.6 | prunable (PPO decision step) | features.6 | 128x64x3x3 | 73,728 | 11.94% | 4,718,592 |
| simplecnn | classifier.1 | prunable (PPO decision step) | classifier.1 | 256x2048 | 524,288 | 84.93% | 524,288 |
| simplecnn | classifier.3 | excluded (never pruned) | classifier.3 | 10x256 | 2,560 | — | 2,560 |
| lenet5 | conv1 | prunable (PPO decision step) | features.0 | 6x3x5x5 | 450 | 0.74% | 352,800 |
| lenet5 | conv2 | prunable (PPO decision step) | features.3 | 16x6x5x5 | 2,400 | 3.94% | 240,000 |
| lenet5 | fc1 | prunable (PPO decision step) | classifier.1 | 120x400 | 48,000 | 78.78% | 48,000 |
| lenet5 | fc2 | prunable (PPO decision step) | classifier.3 | 84x120 | 10,080 | 16.54% | 10,080 |
| lenet5 | classifier.5 | excluded (never pruned) | classifier.5 | 10x84 | 840 | — | 840 |
| resnet8 | stem | prunable (PPO decision step) | conv1 | 16x3x3x3 | 432 | 0.56% | 442,368 |
| resnet8 | stage1 | prunable (PPO decision step) | layer1.conv1;layer1.conv2 | 16x16x3x3;16x16x3x3 | 4,608 | 6.01% | 4,718,592 |
| resnet8 | stage2 | prunable (PPO decision step) | layer2.conv1;layer2.conv2;layer2.shortcut.0 | 32x16x3x3;32x32x3x3;32x16x1x1 | 14,336 | 18.69% | 3,670,016 |
| resnet8 | stage3 | prunable (PPO decision step) | layer3.conv1;layer3.conv2;layer3.shortcut.0 | 64x32x3x3;64x64x3x3;64x32x1x1 | 57,344 | 74.74% | 3,670,016 |
| resnet8 | fc | excluded (never pruned) | fc | 10x64 | 640 | — | 640 |

*Source: `cpu_prunable_units.csv`.*

### A.2 Dense reference models

**Table A.2. Dense references over three training seeds.**

| Setting | Seeds | Reference | Val. (ref.) | Test (ref.) | Val. mean ± SD | Test mean ± SD | Parameters | Recipe |
|---|---|---|---|---|---|---|---|---|
| cifar10 simplecnn | 1 | cnn_baseline_FIXED.pth | 77.32 | 77.03 | 77.32 | 77.03 | 620,362 | historical: Adam 1e-3, 10 epochs, final epoch (not retrained) |
| cifar10 lenet5 | 3 | lenet5_seed0.pth | 64.32 | 64.96 | 65.63 ± 1.94 | 66.21 ± 1.51 | 62,006 | Phase-2 protocol: Adam 0.001 cosine, 30 epochs, best-validation epoch |
| cifar10 resnet8 | 3 | resnet8_seed0.pth | 78.72 | 78.69 | 78.62 ± 0.23 | 78.69 ± 0.51 | 78,042 | Phase-2 protocol: Adam 0.001 cosine, 30 epochs, best-validation epoch |
| cifar100 lenet5 | 3 | lenet5_seed0.pth | 32.58 | 32.28 | 32.15 ± 0.42 | 31.42 ± 0.75 | 69,656 | Phase-2 protocol: Adam 0.001 cosine, 30 epochs, best-validation epoch |
| cifar100 resnet8 | 3 | resnet8_seed0.pth | 43.48 | 43.99 | 43.07 ± 0.54 | 43.58 ± 0.53 | 83,892 | Phase-2 protocol: Adam 0.001 cosine, 30 epochs, best-validation epoch |
| cifar100 simplecnn | 3 | simplecnn_seed0.pth | 49.08 | 50.5 | 48.78 ± 0.95 | 49.84 ± 0.65 | 643,492 | Phase-2 protocol: Adam 0.001 cosine, 30 epochs, best-validation epoch |

*Source: `cpu_baselines.csv`.*

### A.3 Recovery protocol

**Table A.3. Equal recovery protocol (Phase 6; reused in Paper 3, Phases 7A and 7B).**

| Configuration key | Value |
|---|---|
| `optimizer` | Adam |
| `lr` | 0.0001 |
| `betas` | [0.9, 0.999] |
| `eps` | 1e-08 |
| `weight_decay` | 0.0 |
| `schedule` | CosineAnnealingLR(T_max=epochs, eta_min=0), stepped per epoch |
| `batch_size` | 128 |
| `augmentation` | random crop 32 (zero pad 4) + horizontal flip |
| `batchnorm` | train mode during fine-tuning (running statistics updated identically for every method); eval mode for every evaluation; no separate recalibration |
| `selection` | none: the model after the final epoch |

*Source: `phase6_config.json` (`protocol`). The PPO hyper-parameters are given in Sections 3.17 and 4.8.*

### A.4 Pre-registration commits

**Table A.4. Pre-registration and freeze commits.**

| Experiment | Pre-registration record | Commit that first added it |
|---|---|---|
| Phase 2 dense training | `cpu_dense_training_protocol.md` | `366d21ae` |
| Phase 3 | `phase3_preregistration.md` | `2dbcc770` |
| Phase 4 | `phase4_preregistration.md` | `f52e87d9` |
| Phase 5 | `phase5_preregistration.md` | `3cbaec1a` |
| Phase 6 | `phase6_preregistration.md` | `b1b5396f` |
| Paper 3 H0 | `h0_protocol.md` | `c2dcb22a` |
| Paper 3 H2 | `paper3_preregistration.md` | `45936846` |
| Paper 3 proxy | `paper3_proxy_protocol.md` | `ec7a7f27` |
| Paper 3 ResNet-8 calibration | `resnet8_threshold_calibration_protocol.md` | `5b127804` |
| Phase 7A | `preregistration.json` | `b8c27665` |
| Phase 7B design | `phase7b_design.md` | `f5abe039` |
| Phase 7B final | `phase7b_final_preregistration.json` | `0e101d55` |
| 7B energy confirmation | `protocol.json` | `df233be0` |

*Commits verified with `git log --diff-filter=A` (Section 4.19).*

### A.5 Software environments

**Table A.5. Software and hardware environments.**

| Item | Core framework (Phases 2–6) | Deployment extension (Paper 3, 7A, 7B, energy confirmation) |
|---|---|---|
| CPU | Intel(R) Core(TM) i5-6400 CPU @ 2.70GHz | 12th Gen Intel(R) Core(TM) i7-1260P |
| OS | Windows-10-10.0.22621-SP0 (Windows 11 build 22621) | Microsoft Windows 11 Home Single Language 10.0.26200 |
| Python | 3.11.9 | 3.11.9 |
| PyTorch | 2.7.1+cpu | 2.7.1+cpu |
| Stable-Baselines3 | 2.8.0 | 2.8.0 (pinned environment) |
| Gymnasium / NumPy | 1.2.3 / 2.2.2 | pinned environment |
| CUDA | False | not used |

*Sources: `phase5_environment.json`; `environment.json`; `requirements_locked.txt`.*

## Appendix B — Core PPO and Sensitivity Supplementary Results

### B.1 Phase 3 results by setting and method

**Table B.1. Phase 3 mean test accuracy and sparsity by setting and method (20 runs; baselines at each PPO run's zero count).**

| Setting | Method | Test mean | Test SD | Sparsity mean (%) | Drop vs dense (pp) |
|---|---|---|---|---|---|
| C10 SimpleCNN | PPO P0 | 76.44 | 0.49 | 55.06 | 0.59 |
| C10 SimpleCNN | PPO P1 | 76.51 | 0.41 | 56.87 | 0.52 |
| C10 SimpleCNN | PPO P2 | 76.24 | 0.61 | 55.95 | 0.79 |
| C10 SimpleCNN | PPO P3 | 76.36 | 0.80 | 56.04 | 0.67 |
| C10 SimpleCNN | uniform (matched to P0) | 45.08 | 10.44 | 55.06 | 31.95 |
| C10 SimpleCNN | global (matched to P0) | 76.51 | 0.19 | 55.06 | 0.52 |
| C10 SimpleCNN | lamp (matched to P0) | 76.99 | 0.03 | 55.06 | 0.04 |
| C10 SimpleCNN | erk (matched to P0) | 75.70 | 0.32 | 55.06 | 1.34 |
| C10 SimpleCNN | random (matched to P0) | 16.84 | 5.34 | 55.06 | 60.19 |
| C10 LeNet-5 | PPO P0 | 61.67 | 2.64 | 51.54 | 3.29 |
| C10 LeNet-5 | PPO P1 | 62.17 | 1.04 | 53.20 | 2.79 |
| C10 LeNet-5 | PPO P2 | 62.54 | 1.31 | 48.59 | 2.42 |
| C10 LeNet-5 | PPO P3 | 62.83 | 1.36 | 47.45 | 2.13 |
| C10 LeNet-5 | uniform (matched to P0) | 45.38 | 9.56 | 51.54 | 19.58 |
| C10 LeNet-5 | global (matched to P0) | 58.94 | 2.35 | 51.54 | 6.03 |
| C10 LeNet-5 | lamp (matched to P0) | 63.31 | 0.64 | 51.54 | 1.65 |
| C10 LeNet-5 | erk (matched to P0) | 59.74 | 2.10 | 51.54 | 5.22 |
| C10 LeNet-5 | random (matched to P0) | 12.99 | 2.92 | 51.54 | 51.97 |
| C10 ResNet-8 | PPO P0 | 76.95 | 0.78 | 28.35 | 1.74 |
| C10 ResNet-8 | PPO P1 | 76.87 | 0.69 | 30.31 | 1.82 |
| C10 ResNet-8 | PPO P2 | 77.00 | 1.26 | 26.96 | 1.69 |
| C10 ResNet-8 | PPO P3 | 76.49 | 1.79 | 29.20 | 2.20 |
| C10 ResNet-8 | uniform (matched to P0) | 62.79 | 4.18 | 28.35 | 15.90 |
| C10 ResNet-8 | global (matched to P0) | 69.57 | 2.82 | 28.35 | 9.12 |
| C10 ResNet-8 | lamp (matched to P0) | 76.80 | 1.37 | 28.35 | 1.89 |
| C10 ResNet-8 | erk (matched to P0) | 77.01 | 0.94 | 28.35 | 1.68 |
| C10 ResNet-8 | random (matched to P0) | 12.59 | 1.67 | 28.35 | 66.10 |
| C100 SimpleCNN | PPO P0 | 49.66 | 0.59 | 53.28 | 0.84 |
| C100 SimpleCNN | PPO P1 | 49.48 | 0.53 | 54.32 | 1.02 |
| C100 SimpleCNN | PPO P2 | 49.52 | 0.54 | 53.39 | 0.98 |
| C100 SimpleCNN | PPO P3 | 49.50 | 0.53 | 54.21 | 1.00 |
| C100 SimpleCNN | uniform (matched to P0) | 19.96 | 1.33 | 53.28 | 30.54 |
| C100 SimpleCNN | global (matched to P0) | 49.67 | 0.23 | 53.28 | 0.84 |
| C100 SimpleCNN | lamp (matched to P0) | 50.30 | 0.04 | 53.28 | 0.20 |
| C100 SimpleCNN | erk (matched to P0) | 45.31 | 1.26 | 53.28 | 5.19 |
| C100 SimpleCNN | random (matched to P0) | 2.37 | 0.30 | 53.28 | 48.13 |
| C100 LeNet-5 | PPO P0 | 31.19 | 0.23 | 34.59 | 1.10 |
| C100 LeNet-5 | PPO P1 | 30.81 | 1.04 | 36.98 | 1.47 |
| C100 LeNet-5 | PPO P2 | 31.15 | 0.58 | 34.37 | 1.13 |
| C100 LeNet-5 | PPO P3 | 31.05 | 0.81 | 35.03 | 1.23 |
| C100 LeNet-5 | uniform (matched to P0) | 27.51 | 1.85 | 34.59 | 4.77 |
| C100 LeNet-5 | global (matched to P0) | 31.32 | 0.40 | 34.59 | 0.96 |
| C100 LeNet-5 | lamp (matched to P0) | 31.86 | 0.20 | 34.59 | 0.42 |
| C100 LeNet-5 | erk (matched to P0) | 30.88 | 0.97 | 34.59 | 1.40 |
| C100 LeNet-5 | random (matched to P0) | 3.29 | 0.31 | 34.59 | 28.99 |
| C100 ResNet-8 | PPO P0 | 43.01 | 0.09 | 9.46 | 0.98 |
| C100 ResNet-8 | PPO P1 | 43.13 | 0.38 | 8.76 | 0.86 |
| C100 ResNet-8 | PPO P2 | 43.01 | 0.07 | 9.23 | 0.98 |
| C100 ResNet-8 | PPO P3 | 43.09 | 0.31 | 8.62 | 0.90 |
| C100 ResNet-8 | uniform (matched to P0) | 42.78 | 0.93 | 9.46 | 1.21 |
| C100 ResNet-8 | global (matched to P0) | 42.74 | 0.02 | 9.46 | 1.25 |
| C100 ResNet-8 | lamp (matched to P0) | 43.76 | 0.09 | 9.46 | 0.23 |
| C100 ResNet-8 | erk (matched to P0) | 43.81 | 0.14 | 9.46 | 0.18 |
| C100 ResNet-8 | random (matched to P0) | 4.25 | 0.36 | 9.46 | 39.74 |

*Source: `phase3_summary.csv`.*

### B.2 Sensitivity vectors

**Table B.2. Normalised loss sensitivity S_l by setting and the Phase-5 centred prior coefficients.**

| Setting | Config. I VAL-RL (Phases 3–4) | Config. II VAL-RL (Phase 5) | Phase-5 P2 coefficients c_l = S_l − S̄ |
|---|---|---|---|
| C10 SimpleCNN | 1.000, 0.032, 0.009, 0.000 | 1.000, 0.030, 0.012, 0.000 | 0.740, −0.230, −0.249, −0.260 |
| C10 LeNet-5 | 1.000, 0.422, 0.046, 0.000 | 1.000, 0.440, 0.042, 0.000 | 0.629, 0.070, −0.329, −0.371 |
| C10 ResNet-8 | 1.000, 0.138, 0.331, 0.000 | 1.000, 0.145, 0.320, 0.000 | 0.634, −0.221, −0.046, −0.366 |
| C100 SimpleCNN | 1.000, 0.103, 0.014, 0.000 | 1.000, 0.119, 0.010, 0.000 | 0.718, −0.164, −0.272, −0.282 |
| C100 LeNet-5 | 1.000, 0.587, 0.000, 0.086 | 1.000, 0.564, 0.000, 0.059 | 0.594, 0.159, −0.406, −0.347 |
| C100 ResNet-8 | 0.987, 0.310, 1.000, 0.000 | 1.000, 0.309, 0.997, 0.000 | 0.423, −0.268, 0.421, −0.577 |

Phase 7B structured sensitivity (C10 SimpleCNN, VAL-RL (2,500 images), units conv1, conv2, conv3, fc1): normalised 0.898, 1.000, 0.341, 0.000; centred 0.338, 0.440, −0.219, −0.560.

*Sources: `phase3_frozen_inputs.json` (configuration I); `phase5_config.json` (configuration II); `structured_sensitivity.json` (Phase 7B).*

## Appendix C — Archive, Reward and Recovery Diagnostics

### C.1 Phase 4 search quality and retention

**Table C.1. Phase 4 search quality by variant (VAL-RL (I) ranks among 1,296 policies; medians over 20 runs).**

| Setting | Variant | Selected rank | Selected V_SELECT rank | Best-seen rank | Best-seen top-10 | Selected top-10 | Selected top-20 | Test mean | Sparsity mean |
|---|---|---|---|---|---|---|---|---|---|
| C10 SimpleCNN | S0 | 92 | 78 | 1 | 1.00 | 0.00 | 0.00 | 76.62 | 57.74 |
| C10 SimpleCNN | S1 | 3 | 1 | 1 | 1.00 | 1.00 | 1.00 | 76.77 | 56.06 |
| C10 SimpleCNN | S2 | 46 | 38 | 1 | 1.00 | 0.10 | 0.10 | 76.76 | 56.25 |
| C10 SimpleCNN | S2A | 3 | 1 | 1 | 1.00 | 1.00 | 1.00 | 76.76 | 56.06 |
| C10 LeNet-5 | S0 | 69 | 66 | 1 | 1.00 | 0.00 | 0.00 | 62.32 | 54.89 |
| C10 LeNet-5 | S1 | 5.5 | 1.5 | 1 | 1.00 | 1.00 | 1.00 | 63.20 | 52.94 |
| C10 LeNet-5 | S2 | 69 | 66 | 1 | 1.00 | 0.00 | 0.05 | 62.08 | 56.78 |
| C10 LeNet-5 | S2A | 6 | 1 | 1 | 1.00 | 1.00 | 1.00 | 63.22 | 52.79 |
| C10 ResNet-8 | S0 | 96 | 87 | 2.5 | 0.95 | 0.00 | 0.00 | 76.48 | 32.20 |
| C10 ResNet-8 | S1 | 3.5 | 2.5 | 2.5 | 0.95 | 0.90 | 1.00 | 71.52 | 45.09 |
| C10 ResNet-8 | S2 | 42 | 28 | 1 | 1.00 | 0.35 | 0.40 | 71.77 | 43.58 |
| C10 ResNet-8 | S2A | 1 | 1 | 1 | 1.00 | 1.00 | 1.00 | 70.83 | 47.78 |
| C100 SimpleCNN | S0 | 65 | 70 | 1 | 1.00 | 0.00 | 0.00 | 49.62 | 52.86 |
| C100 SimpleCNN | S1 | 6 | 2 | 1 | 1.00 | 1.00 | 1.00 | 49.89 | 54.07 |
| C100 SimpleCNN | S2 | 62 | 61 | 1 | 1.00 | 0.00 | 0.10 | 49.69 | 53.66 |
| C100 SimpleCNN | S2A | 6 | 1 | 1 | 1.00 | 1.00 | 1.00 | 49.84 | 54.20 |
| C100 LeNet-5 | S0 | 173 | 140 | 1 | 1.00 | 0.05 | 0.05 | 30.28 | 41.32 |
| C100 LeNet-5 | S1 | 7 | 2 | 1 | 1.00 | 0.80 | 1.00 | 30.71 | 46.46 |
| C100 LeNet-5 | S2 | 48.5 | 34.5 | 1 | 1.00 | 0.00 | 0.00 | 29.70 | 48.76 |
| C100 LeNet-5 | S2A | 7 | 2 | 1 | 1.00 | 0.85 | 1.00 | 30.72 | 46.30 |
| C100 ResNet-8 | S0 | 114 | 137 | 2 | 0.95 | 0.00 | 0.00 | 43.22 | 8.53 |
| C100 ResNet-8 | S1 | 10 | 2.5 | 2 | 0.95 | 0.75 | 1.00 | 39.41 | 27.26 |
| C100 ResNet-8 | S2 | 45 | 55.5 | 2 | 0.90 | 0.00 | 0.05 | 42.82 | 13.78 |
| C100 ResNet-8 | S2A | 10 | 2 | 2 | 0.90 | 0.80 | 1.00 | 39.22 | 27.71 |
| C10 SimpleCNN | Phase-3 P0 (reference, seeds | 100.5 | — | 1.5 | — | 0.00 | 0.00 | 76.44 | — |
| C10 LeNet-5 | Phase-3 P0 (reference, seeds | 74.5 | — | 2 | — | 0.00 | 0.00 | 61.67 | — |
| C10 ResNet-8 | Phase-3 P0 (reference, seeds | 122 | — | 4 | — | 0.00 | 0.00 | 76.95 | — |
| C100 SimpleCNN | Phase-3 P0 (reference, seeds | 65 | — | 1 | — | 0.00 | 0.00 | 49.66 | — |
| C100 LeNet-5 | Phase-3 P0 (reference, seeds | 199 | — | 4.5 | — | 0.00 | 0.00 | 31.19 | — |
| C100 ResNet-8 | Phase-3 P0 (reference, seeds | 114 | — | 2 | — | 0.00 | 0.00 | 43.01 | — |

*Source: `phase4_search_quality.csv`.*

### C.2 Phase 5 selected policies

**Table C.2. Phase 5 selected policies by setting and condition (20 runs each).**

| Cond. | Setting | Sparsity mean | Sparsity median | Test mean | Drop vs dense | Feasible VAL-RL/VAL-SELECT/test (of 20) | Mean q_test | Fallbacks |
|---|---|---|---|---|---|---|---|---|
| C0 | C10 SimpleCNN | 56.10 | 56.10 | 76.75 | 0.28 | 20/20/20 | 0.9964 | 0 |
| C0 | C10 LeNet-5 | 53.89 | 54.00 | 63.14 | 1.83 | 0/0/0 | 0.9719 | 0 |
| C0 | C10 ResNet-8 | 46.23 | 48.78 | 71.37 | 7.32 | 0/0/0 | 0.9070 | 0 |
| C0 | C100 SimpleCNN | 52.90 | 52.66 | 50.28 | 0.22 | 20/20/20 | 0.9956 | 0 |
| C0 | C100 LeNet-5 | 46.32 | 46.25 | 30.67 | 1.61 | 0/4/0 | 0.9501 | 0 |
| C0 | C100 ResNet-8 | 25.88 | 23.53 | 39.94 | 4.05 | 0/0/0 | 0.9079 | 0 |
| C1 | C10 SimpleCNN | 56.93 | 57.59 | 76.45 | 0.58 | 20/20/20 | 0.9924 | 0 |
| C1 | C10 LeNet-5 | 38.57 | 38.42 | 63.83 | 1.13 | 20/20/9 | 0.9827 | 0 |
| C1 | C10 ResNet-8 | 24.50 | 24.74 | 77.51 | 1.18 | 20/20/20 | 0.9850 | 0 |
| C1 | C100 SimpleCNN | 53.09 | 53.67 | 50.07 | 0.43 | 20/20/20 | 0.9915 | 0 |
| C1 | C100 LeNet-5 | 30.60 | 31.01 | 31.79 | 0.49 | 20/19/12 | 0.9848 | 1 |
| C1 | C100 ResNet-8 | 15.62 | 16.08 | 42.83 | 1.16 | 20/20/0 | 0.9736 | 0 |
| C2 | C10 SimpleCNN | 56.21 | 56.26 | 76.50 | 0.53 | 20/20/20 | 0.9931 | 0 |
| C2 | C10 LeNet-5 | 36.92 | 36.17 | 63.99 | 0.97 | 20/20/15 | 0.9851 | 0 |
| C2 | C10 ResNet-8 | 23.92 | 24.09 | 77.57 | 1.12 | 20/20/20 | 0.9857 | 0 |
| C2 | C100 SimpleCNN | 51.86 | 53.24 | 50.13 | 0.37 | 20/20/20 | 0.9926 | 0 |
| C2 | C100 LeNet-5 | 30.13 | 30.77 | 31.86 | 0.42 | 20/19/14 | 0.9870 | 1 |
| C2 | C100 ResNet-8 | 15.45 | 16.08 | 42.90 | 1.09 | 20/20/1 | 0.9752 | 0 |

*Source: `phase5_constraint_results.csv`.*

### C.3 Phase 6 recovery by setting and method

**Table C.3. Phase 6 test accuracy before and after equal recovery (means of 20 runs).**

| Setting | Method | Sparsity | Test before | Test after | Recovery | Gap to dense after | Retention ≥ 0.98 before / after | Zero-count change |
|---|---|---|---|---|---|---|---|---|
| C100 LeNet-5 | dense | 0.00 | 32.28 | 32.39 | 0.11 | −0.114 | 1.00 / 1.00 | 0 |
| C100 LeNet-5 | global | 30.60 | 31.43 | 32.24 | 0.81 | 0.042 | 0.20 / 1.00 | 0 |
| C100 LeNet-5 | lamp | 30.60 | 32.05 | 32.49 | 0.44 | −0.213 | 1.00 / 1.00 | 0 |
| C100 LeNet-5 | ppo | 30.60 | 31.79 | 32.36 | 0.57 | −0.077 | 0.60 / 1.00 | 0 |
| C100 ResNet-8 | dense | 0.00 | 43.99 | 44.08 | 0.09 | −0.086 | 1.00 / 1.00 | 0 |
| C100 ResNet-8 | global | 15.62 | 39.35 | 44.04 | 4.69 | −0.050 | 0.00 / 1.00 | 0 |
| C100 ResNet-8 | lamp | 15.62 | 43.34 | 44.06 | 0.72 | −0.071 | 1.00 / 1.00 | 0 |
| C100 ResNet-8 | ppo | 15.62 | 42.83 | 44.02 | 1.20 | −0.032 | 0.00 / 1.00 | 0 |
| C100 SimpleCNN | dense | 0.00 | 50.50 | 50.60 | 0.10 | −0.102 | 1.00 / 1.00 | 0 |
| C100 SimpleCNN | global | 53.09 | 49.75 | 50.51 | 0.76 | −0.005 | 1.00 / 1.00 | 0 |
| C100 SimpleCNN | lamp | 53.09 | 50.36 | 50.58 | 0.22 | −0.081 | 1.00 / 1.00 | 0 |
| C100 SimpleCNN | ppo | 53.09 | 50.07 | 50.53 | 0.46 | −0.031 | 1.00 / 1.00 | 0 |
| C10 LeNet-5 | dense | 0.00 | 64.96 | 65.10 | 0.14 | −0.143 | 1.00 / 1.00 | 0 |
| C10 LeNet-5 | global | 38.57 | 62.41 | 64.83 | 2.42 | 0.133 | 0.00 / 1.00 | 0 |
| C10 LeNet-5 | lamp | 38.57 | 64.15 | 65.06 | 0.91 | −0.103 | 1.00 / 1.00 | 0 |
| C10 LeNet-5 | ppo | 38.57 | 63.83 | 64.88 | 1.04 | 0.082 | 0.45 / 1.00 | 0 |
| C10 ResNet-8 | dense | 0.00 | 78.69 | 78.83 | 0.14 | −0.143 | 1.00 / 1.00 | 0 |
| C10 ResNet-8 | global | 24.50 | 71.00 | 78.67 | 7.67 | 0.025 | 0.00 / 1.00 | 0 |
| C10 ResNet-8 | lamp | 24.50 | 78.03 | 78.73 | 0.70 | −0.044 | 1.00 / 1.00 | 0 |
| C10 ResNet-8 | ppo | 24.50 | 77.51 | 78.74 | 1.23 | −0.048 | 1.00 / 1.00 | 0 |
| C10 SimpleCNN | dense | 0.00 | 77.03 | 80.08 | 3.05 | −3.050 | 1.00 / 1.00 | 0 |
| C10 SimpleCNN | global | 56.93 | 76.54 | 79.89 | 3.35 | −2.861 | 1.00 / 1.00 | 0 |
| C10 SimpleCNN | lamp | 56.93 | 76.94 | 79.95 | 3.01 | −2.920 | 1.00 / 1.00 | 0 |
| C10 SimpleCNN | ppo | 56.93 | 76.45 | 79.73 | 3.28 | −2.701 | 1.00 / 1.00 | 0 |

*Source: `phase6_pre_post_accuracy.csv`.*

## Appendix D — Structured Pruning and Hardware Protocol

### D.1 Latency-measurement calibration (H0)

**Table D.1. H0 calibration: per-configuration decision quantities.**

| Config. | Signal (ms) | Noise (ms) | SNR | Relative reduction | Positive in all sessions | Usable |
|---|---|---|---|---|---|---|
| A | 0.4127 | 0.0335 | 12.32 | 61.6% | True | True |
| B | 0.1222 | 0.0287 | 4.26 | 31.8% | True | False |
| C | 15.4531 | 1.6070 | 9.62 | 63.2% | True | True |

*Source: `h0_decision.json`. Rule: SNR ≥ 5, reduction ≥ 10%, positive in every session; order A, C, B.*

### D.2 Latency-table session log

**Table D.2. Latency-table session log (anchor drift; limit 10%).**

| Table | Session | Attempt | Load before (%) | Load after (%) | Anchor drift | Accepted |
|---|---|---|---|---|---|---|
| resnet8_c10 | 1 | 1 | 4.1 | 3.0 | 27.1% | False |
| resnet8_c10 | 1 | 2 | 2.6 | 2.6 | 9.0% | True |
| resnet8_c100 | 1 | 1 | 2.7 | 2.2 | 9.5% | True |
| simplecnn_c10 | 1 | 1 | 2.8 | 2.4 | 7.5% | True |
| simplecnn_c100 | 1 | 1 | 2.2 | 5.1 | 40.6% | False |
| simplecnn_c100 | 1 | 2 | 4.7 | 2.6 | 7.1% | True |
| resnet8_c100 | 2 | 1 | 2.1 | 3.3 | 8.8% | True |
| resnet8_c10 | 2 | 1 | 2.3 | 2.4 | 7.6% | True |
| simplecnn_c10 | 2 | 1 | 2.3 | 2.2 | 8.8% | True |
| simplecnn_c100 | 2 | 1 | 3.1 | 2.0 | 8.4% | True |
| resnet8_c100 | 3 | 1 | 2.3 | 3.4 | 8.7% | True |
| simplecnn_c10 | 3 | 1 | 2.7 | 2.3 | 7.9% | True |
| resnet8_c10 | 3 | 1 | 2.5 | 2.3 | 9.7% | True |
| simplecnn_c100 | 3 | 1 | 2.7 | 2.6 | 8.4% | True |
| resnet8_c100 | 4 | 1 | 2.1 | 3.1 | 9.6% | True |
| resnet8_c10 | 4 | 1 | 2.7 | 3.1 | 10.6% | False |
| resnet8_c10 | 4 | 2 | 5.4 | 2.4 | 7.4% | True |
| simplecnn_c100 | 4 | 1 | 2.6 | 3.4 | 14.0% | False |
| simplecnn_c100 | 4 | 2 | 3.1 | 1.9 | 8.6% | True |
| simplecnn_c10 | 4 | 1 | 2.2 | 2.8 | 8.9% | True |
| resnet8_c10 | 5 | 1 | 3.3 | 2.8 | 8.4% | True |
| simplecnn_c10 | 5 | 1 | 2.6 | 2.6 | 9.4% | True |
| simplecnn_c100 | 5 | 1 | 2.7 | 3.0 | 9.1% | True |
| resnet8_c100 | 5 | 1 | 3.2 | 4.2 | 6.8% | True |

*Source: `h2_latency_freeze.json` (`session_log`).*

### D.3 Accuracy-proxy validation and ResNet-8 calibration

**Table D.3. Accuracy-proxy validation (1-epoch against 5-epoch recovery).**

| Architecture | Validation pairs | Spearman ρ(A₁, A₅) | MAE of q | Top-10 overlap | q₅-feasible pairs | GO |
|---|---|---|---|---|---|---|
| cifar10_simplecnn | 156 | 0.844 | 0.0088 | 3 | 112 | False |
| cifar10_resnet8 | 31 | 0.980 | 0.0679 | 10 | 2 | True |

**Table D.4. ResNet-8 threshold calibration and latency-room decision.**

| Item | Value |
|---|---|
| Calibration policies / truly feasible | 48 / 14 |
| τ_proxy | 0.9601 |
| Recall / precision (TP, FP, FN, TN) | 1.00 / 0.875 (14, 2, 0, 32) |
| Proxy-feasible landscape policies | 52 |
| Best proxy-feasible speed-up | 1.098× |
| Policies ≥ 1.10× | 0 |
| GO to PPO | False |

*Sources: `paper3_proxy_decision.json`, `resnet8_threshold_calibration_decision.json`. The best confirmed five-epoch-feasible calibration policy reached 1.089× ([0, 3, 1, 1]; `resnet8_threshold_calibration_report.md`).*

## Appendix E — Energy Evaluation and Qualification Records

### E.1 Phase 7A energy session summaries

**Table E.1. Phase 7A energy: session summaries (five sessions per architecture).**

| Architecture | Method | Median J / 1,000 | CV over sessions | Median saving | Min–max saving | Sessions below dense | One-sided Wilcoxon p | Consistent reduction |
|---|---|---|---|---|---|---|---|---|
| simplecnn | dense | 11.054 | 2.14% | 0.0% | 0.0–0.0% | 0 | — | False |
| simplecnn | ppo | 8.541 | 1.83% | 22.8% | 19.5–25.7% | 5 | 0.031 | True |
| simplecnn | uniform | 7.906 | 1.54% | 27.9% | 27.3–31.2% | 5 | 0.031 | True |
| simplecnn | lamp | 9.760 | 3.59% | 12.1% | 6.8–15.1% | 5 | 0.031 | True |
| resnet8 | dense | 10.882 | 1.84% | 0.0% | 0.0–0.0% | 0 | — | False |
| resnet8 | ppo | 9.592 | 2.84% | 13.4% | 5.6–14.5% | 5 | 0.031 | True |
| resnet8 | uniform | 9.645 | 3.63% | 11.8% | 4.7–13.9% | 5 | 0.031 | True |
| resnet8 | lamp | 9.819 | 1.46% | 8.9% | 7.4–12.3% | 5 | 0.031 | True |

*Source: `energy_summary.csv`.*

### E.2 Phase 7B recovered final models

**Table E.2. Phase 7B: all recovered final models (5-epoch recovery).**

| Id | Roles | Widths | VAL-SELECT | Δ VAL-SELECT | VAL-RL | Memory ↓ | MACs ↓ | Joint criterion |
|---|---|---|---|---|---|---|---|---|
| 0000 | dense | [32, 64, 128, 256] | 80.00 | 0.00 | 80.60 | 0.0% | 0.0% | NO (memory, MACs failed; latency and energy not measured) |
| 2225 | new_ppo, seed701, seed702, seed707, seed712, seed718 | [26, 51, 102, 102] | 78.88 | −1.12 | 80.00 | 63.4% | 36.0% | NO (accuracy failed; latency and energy not measured) |
| uniform_r0.208 | uniform | [25, 51, 101, 203] | 78.76 | −1.24 | 79.76 | 37.3% | 36.2% | NO (accuracy failed; latency and energy not measured) |
| lamp_s56.2 | lamp | [32, 58, 103, 94] | 79.64 | −0.36 | 79.12 | 63.3% | 19.3% | not assessable (hardware validation stopped) |
| 3125 | seed700, enumeration_reference | [22, 58, 102, 102] | 78.84 | −1.16 | 79.64 | 62.4% | 34.3% | NO (accuracy failed; latency and energy not measured) |
| 3132 | seed703, seed704, seed709 | [22, 58, 90, 205] | 78.92 | −1.08 | 79.60 | 42.5% | 36.8% | NO (accuracy failed; latency and energy not measured) |
| 2224 | seed705, seed714, seed716 | [26, 51, 102, 154] | 78.84 | −1.16 | 79.84 | 49.6% | 35.3% | NO (accuracy failed; latency and energy not measured) |
| 3130 | seed706, seed708 | [22, 58, 90, 256] | 78.80 | −1.20 | 79.56 | 30.6% | 36.2% | NO (accuracy failed; latency and energy not measured) |
| 2221 | seed710, seed711, seed713, seed719 | [26, 51, 102, 230] | 78.72 | −1.28 | 79.68 | 29.5% | 34.1% | NO (accuracy failed; latency and energy not measured) |
| 3134 | seed715, seed717 | [22, 58, 90, 154] | 78.92 | −1.08 | 79.52 | 54.4% | 37.5% | NO (accuracy failed; latency and energy not measured) |
| phase7a_ppo | old_ppo | [32, 51, 51, 102] | 76.00 | −4.00 | 75.52 | 80.1% | 42.6% | NO (accuracy failed; latency and energy not measured) |

*Source: `final_summary.csv`. The last column is the pre-registered joint deployment-efficiency criterion (internal file name GREEN-SUCCESS). Latency and energy were not measured.*

### E.3 Energy-measurement qualification

**Table E.3. Dense-only stability pilots and qualification (energy per 1,000 inferences, J).**

| Measurement | N per block | Six dense blocks | CV | Limit | Decision |
|---|---|---|---|---|---|
| 7A pilot, simplecnn | 100,000 | 11.518, 11.150, 11.081, 11.253, 10.963, 10.768 | 2.30% | 3% | passed |
| 7A pilot, resnet8 | 100,000 | 10.793, 10.489, 10.555, 10.667, 10.358, 10.312 | 1.74% | 3% | passed |
| 7B surrogate-calibration pilot | 20,000 | 12.200, 11.198, 11.107, 11.729, 11.433, 11.420 | 3.47% | 4% | stable |
| 7B confirmatory pilot (pilot_n100000) | 100,000 | 13.088, 11.414, 11.204, 15.125, 12.265, 12.273 | 11.36% | 3% | unstable |
| 7B confirmatory pilot (pilot_n200000) | 200,000 | 11.814, 12.074, 11.039, 11.539, 11.025, 11.033 | 4.01% | 3% | unstable |
| 7B energy-confirmation qualification | 200,000 | 11.417, 14.278, 12.962, 15.221, 9.514, 9.473 | 19.92% | 3% | not qualified |

*Sources: `pilot_n100000.json` (Phase 7A); `pilot_n20000.json` and `statistics.json` (Phase 7B); `measurement_qualification.json` (energy confirmation). A doubled-N Phase-7B pilot attempt stopped by the load gate was declared void by a recorded decision of the author and is not listed.*

### E.4 Energy-confirmation qualification blocks

**Table E.4. Energy-confirmation qualification blocks (descriptive diagnostics).**

| Run | Position | J / 1,000 | Package W | Median latency (ms) | System load (%) | CPU-2 performance (%) | Thermal zone start → end (°C) |
|---|---|---|---|---|---|---|---|
| qual1 | 0 | 11.417 | 11.16 | 1.0456 | 8.4 | 130 | 86 → 76 |
| qual1 | 1 | 14.278 | 13.76 | 1.0513 | 9.2 | 133 | 77 → 83 |
| qual2 | 0 | 12.962 | 12.72 | 1.0479 | 8.9 | 131 | 86 → 84 |
| qual2 | 1 | 15.221 | 14.51 | 1.0517 | 10.2 | 135 | 79 → 92 |
| qual3 | 0 | 9.514 | 9.40 | 1.0420 | 7.6 | 129 | 76 → 81 |
| qual3 | 1 | 9.473 | 9.09 | 1.0430 | 7.9 | 125 | 72 → 73 |

*Source: `measurement_qualification.json` (`blocks`). No cause of the instability is claimed, and no Dense-versus-PPO session was run.*

