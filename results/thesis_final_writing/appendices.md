# Appendices

These appendices contain supporting detail for Chapters 3–6. They are generated from the frozen phase files by
`tools/build_appendices.py`, and the source file is named under each table. Seed-level and session-level
records beyond these summaries remain in the repository files listed in Appendix K. The central results and
conclusions are in the main chapters.

## Appendix A — Network Architectures and Hyperparameters

### A.1 Pruning units, shapes, parameters and MACs per image (CIFAR-10 output layer)

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

*Source: `results/architecture_generalization/cpu_prunable_units.csv`.*

### A.2 Dense references: three training seeds

| Setting | Seeds | Reference | Val. (ref.) | Test (ref.) | Val. mean ± SD | Test mean ± SD | Parameters | Recipe |
|---|---|---|---|---|---|---|---|---|
| cifar10 simplecnn | 1 | cnn_baseline_FIXED.pth | 77.32 | 77.03 | 77.32 | 77.03 | 620,362 | historical: Adam 1e-3, 10 epochs, final epoch (not retrained) |
| cifar10 lenet5 | 3 | lenet5_seed0.pth | 64.32 | 64.96 | 65.63 ± 1.94 | 66.21 ± 1.51 | 62,006 | Phase-2 protocol: Adam 0.001 cosine, 30 epochs, best-validation epoch |
| cifar10 resnet8 | 3 | resnet8_seed0.pth | 78.72 | 78.69 | 78.62 ± 0.23 | 78.69 ± 0.51 | 78,042 | Phase-2 protocol: Adam 0.001 cosine, 30 epochs, best-validation epoch |
| cifar100 lenet5 | 3 | lenet5_seed0.pth | 32.58 | 32.28 | 32.15 ± 0.42 | 31.42 ± 0.75 | 69,656 | Phase-2 protocol: Adam 0.001 cosine, 30 epochs, best-validation epoch |
| cifar100 resnet8 | 3 | resnet8_seed0.pth | 43.48 | 43.99 | 43.07 ± 0.54 | 43.58 ± 0.53 | 83,892 | Phase-2 protocol: Adam 0.001 cosine, 30 epochs, best-validation epoch |
| cifar100 simplecnn | 3 | simplecnn_seed0.pth | 49.08 | 50.5 | 48.78 ± 0.95 | 49.84 ± 0.65 | 643,492 | Phase-2 protocol: Adam 0.001 cosine, 30 epochs, best-validation epoch |

*Source: `results/architecture_generalization/cpu_baselines.csv`.*

### A.3 Recovery protocol (Phase 6; reused in Paper 3, Phases 7A and 7B)

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

*Source: `results/phase6/phase6_config.json` (`protocol`). The PPO hyper-parameters are listed in Table 3.7 and Table 4.6.*

## Appendix B — Full Phase 3 Results

### B.1 Mean test accuracy and sparsity by setting and method (20 runs; baselines at each PPO run's zero count)

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

*Source: `results/architecture_generalization/phase3_summary.csv`.*

### B.2 All pre-registered Phase 3 comparisons (paired by seed, n = 20; exact sign-flip test, Holm within family)

| Setting | Comparison | Endpoint | Mean diff. | 95% t-CI | Holm p | +/−/tie | Outcome |
|---|---|---|---|---|---|---|---|
| C10 SimpleCNN | P0 - uniform | test_accuracy | 31.365 | [26.568, 36.162] | 1.1e-05 | 20/0/0 | supported |
| C10 SimpleCNN | P0 - global | test_accuracy | −0.066 | [−0.261, 0.129] | 1 | 15/5/0 | inconclusive |
| C10 SimpleCNN | P0 - lamp | test_accuracy | −0.547 | [−0.770, −0.323] | 5.3e-05 | 1/19/0 | opposite |
| C10 SimpleCNN | P0 - erk | test_accuracy | 0.748 | [0.556, 0.940] | 1.5e-04 | 17/3/0 | supported |
| C10 SimpleCNN | P0 - random | test_accuracy | 59.603 | [57.186, 62.020] | 4.6e-05 | 20/0/0 | supported |
| C10 SimpleCNN | P1 - P0 | vrl_auc | 0.461 | [0.254, 0.669] | 0.002 | 16/4/0 | supported |
| C10 SimpleCNN | P1 - P2 | vrl_auc | 0.413 | [0.193, 0.634] | 0.016 | 15/5/0 | supported |
| C10 SimpleCNN | P1 - P3 | vrl_auc | 0.327 | [0.135, 0.519] | 0.025 | 16/4/0 | supported |
| C10 SimpleCNN | P1 - P0 | rank_auc | 0.032 | [0.022, 0.042] | 1.1e-05 | 20/0/0 | supported |
| C10 SimpleCNN | P1 - P0 | utility_auc | 0.029 | [0.020, 0.038] | 1.1e-05 | 20/0/0 | supported |
| C10 SimpleCNN | P1 - P0 | destructive_frequency | −0.032 | [−0.048, −0.016] | 0.0012 | 2/18/0 | supported |
| C10 SimpleCNN | P1 - P0 | excess_over_global | 0.081 | [−0.124, 0.286] | 0.94 | 5/6/9 | inconclusive |
| C10 SimpleCNN | P1 - P0 | test_accuracy | 0.068 | [−0.180, 0.316] | 1 | 6/5/9 | inconclusive |
| C10 LeNet-5 | P0 - uniform | test_accuracy | 16.291 | [12.297, 20.284] | 1.1e-05 | 20/0/0 | supported |
| C10 LeNet-5 | P0 - global | test_accuracy | 2.736 | [1.683, 3.789] | 0.0015 | 18/2/0 | supported |
| C10 LeNet-5 | P0 - lamp | test_accuracy | −1.643 | [−2.754, −0.531] | 4.6e-05 | 0/20/0 | opposite |
| C10 LeNet-5 | P0 - erk | test_accuracy | 1.934 | [0.923, 2.944] | 0.0087 | 19/1/0 | supported |
| C10 LeNet-5 | P0 - random | test_accuracy | 48.685 | [47.326, 50.044] | 4.6e-05 | 20/0/0 | supported |
| C10 LeNet-5 | P1 - P0 | vrl_auc | 0.068 | [−0.101, 0.237] | 0.91 | 12/8/0 | inconclusive |
| C10 LeNet-5 | P1 - P2 | vrl_auc | −0.215 | [−0.406, −0.023] | 0.27 | 6/14/0 | inconclusive |
| C10 LeNet-5 | P1 - P3 | vrl_auc | −0.207 | [−0.406, −0.009] | 0.31 | 9/11/0 | inconclusive |
| C10 LeNet-5 | P1 - P0 | rank_auc | 0.027 | [0.014, 0.039] | 5.5e-04 | 18/2/0 | supported |
| C10 LeNet-5 | P1 - P0 | utility_auc | 0.020 | [0.011, 0.029] | 3.6e-04 | 18/2/0 | supported |
| C10 LeNet-5 | P1 - P0 | destructive_frequency | −0.038 | [−0.054, −0.021] | 0.0012 | 3/17/0 | supported |
| C10 LeNet-5 | P1 - P0 | excess_over_global | 0.861 | [−0.173, 1.894] | 0.41 | 8/3/9 | inconclusive |
| C10 LeNet-5 | P1 - P0 | test_accuracy | 0.504 | [−0.565, 1.572] | 1 | 6/5/9 | inconclusive |
| C10 ResNet-8 | P0 - uniform | test_accuracy | 14.165 | [12.369, 15.961] | 1.1e-05 | 20/0/0 | supported |
| C10 ResNet-8 | P0 - global | test_accuracy | 7.379 | [6.293, 8.464] | 4.6e-05 | 20/0/0 | supported |
| C10 ResNet-8 | P0 - lamp | test_accuracy | 0.153 | [−0.408, 0.713] | 1 | 8/12/0 | inconclusive |
| C10 ResNet-8 | P0 - erk | test_accuracy | −0.061 | [−0.371, 0.249] | 1 | 11/9/0 | inconclusive |
| C10 ResNet-8 | P0 - random | test_accuracy | 64.361 | [63.691, 65.032] | 4.6e-05 | 20/0/0 | supported |
| C10 ResNet-8 | P1 - P0 | vrl_auc | 0.195 | [−0.185, 0.575] | 0.91 | 13/7/0 | inconclusive |
| C10 ResNet-8 | P1 - P2 | vrl_auc | −0.019 | [−0.369, 0.330] | 1 | 11/9/0 | inconclusive |
| C10 ResNet-8 | P1 - P3 | vrl_auc | 0.044 | [−0.330, 0.419] | 1 | 11/9/0 | inconclusive |
| C10 ResNet-8 | P1 - P0 | rank_auc | 0.024 | [0.018, 0.029] | 1.1e-05 | 20/0/0 | supported |
| C10 ResNet-8 | P1 - P0 | utility_auc | 0.018 | [0.014, 0.022] | 1.1e-05 | 20/0/0 | supported |
| C10 ResNet-8 | P1 - P0 | destructive_frequency | −0.016 | [−0.025, −0.007] | 0.0022 | 3/17/0 | supported |
| C10 ResNet-8 | P1 - P0 | excess_over_global | 0.847 | [−0.315, 2.008] | 0.57 | 8/4/8 | inconclusive |
| C10 ResNet-8 | P1 - P0 | test_accuracy | −0.078 | [−0.259, 0.104] | 1 | 5/7/8 | inconclusive |
| C100 SimpleCNN | P0 - uniform | test_accuracy | 29.693 | [29.332, 30.054] | 1.1e-05 | 20/0/0 | supported |
| C100 SimpleCNN | P0 - global | test_accuracy | −0.009 | [−0.176, 0.159] | 1 | 8/12/0 | inconclusive |
| C100 SimpleCNN | P0 - lamp | test_accuracy | −0.647 | [−0.905, −0.388] | 0.002 | 6/14/0 | opposite |
| C100 SimpleCNN | P0 - erk | test_accuracy | 4.350 | [4.021, 4.678] | 4.6e-05 | 20/0/0 | supported |
| C100 SimpleCNN | P0 - random | test_accuracy | 47.288 | [47.145, 47.431] | 4.6e-05 | 20/0/0 | supported |
| C100 SimpleCNN | P1 - P0 | vrl_auc | 0.350 | [0.145, 0.554] | 0.0098 | 16/4/0 | supported |
| C100 SimpleCNN | P1 - P2 | vrl_auc | 0.343 | [0.160, 0.526] | 0.016 | 15/5/0 | supported |
| C100 SimpleCNN | P1 - P3 | vrl_auc | 0.216 | [0.002, 0.430] | 0.31 | 14/6/0 | inconclusive |
| C100 SimpleCNN | P1 - P0 | rank_auc | 0.024 | [0.016, 0.033] | 6.3e-05 | 17/3/0 | supported |
| C100 SimpleCNN | P1 - P0 | utility_auc | 0.026 | [0.017, 0.035] | 2.9e-05 | 18/2/0 | supported |
| C100 SimpleCNN | P1 - P0 | destructive_frequency | −0.015 | [−0.021, −0.010] | 1.7e-04 | 1/19/0 | supported |
| C100 SimpleCNN | P1 - P0 | excess_over_global | −0.104 | [−0.309, 0.101] | 0.94 | 2/8/10 | inconclusive |
| C100 SimpleCNN | P1 - P0 | test_accuracy | −0.179 | [−0.502, 0.144] | 1 | 2/8/10 | inconclusive |
| C100 LeNet-5 | P0 - uniform | test_accuracy | 3.672 | [2.828, 4.515] | 1.1e-05 | 20/0/0 | supported |
| C100 LeNet-5 | P0 - global | test_accuracy | −0.134 | [−0.343, 0.076] | 0.96 | 1/19/0 | inconclusive |
| C100 LeNet-5 | P0 - lamp | test_accuracy | −0.678 | [−0.821, −0.534] | 5.3e-05 | 1/19/0 | opposite |
| C100 LeNet-5 | P0 - erk | test_accuracy | 0.302 | [−0.145, 0.749] | 0.19 | 11/9/0 | inconclusive |
| C100 LeNet-5 | P0 - random | test_accuracy | 27.891 | [27.731, 28.051] | 4.6e-05 | 20/0/0 | supported |
| C100 LeNet-5 | P1 - P0 | vrl_auc | 0.007 | [−0.132, 0.145] | 0.92 | 13/7/0 | inconclusive |
| C100 LeNet-5 | P1 - P2 | vrl_auc | −0.083 | [−0.215, 0.049] | 0.81 | 11/9/0 | inconclusive |
| C100 LeNet-5 | P1 - P3 | vrl_auc | −0.086 | [−0.207, 0.034] | 0.81 | 9/11/0 | inconclusive |
| C100 LeNet-5 | P1 - P0 | rank_auc | 0.023 | [0.013, 0.033] | 6.3e-05 | 18/2/0 | supported |
| C100 LeNet-5 | P1 - P0 | utility_auc | 0.014 | [0.008, 0.020] | 2.9e-05 | 18/2/0 | supported |
| C100 LeNet-5 | P1 - P0 | destructive_frequency | −0.029 | [−0.042, −0.016] | 0.0012 | 4/16/0 | supported |
| C100 LeNet-5 | P1 - P0 | excess_over_global | 0.772 | [0.111, 1.432] | 0.094 | 9/2/9 | inconclusive |
| C100 LeNet-5 | P1 - P0 | test_accuracy | −0.380 | [−0.895, 0.136] | 0.75 | 7/5/8 | inconclusive |
| C100 ResNet-8 | P0 - uniform | test_accuracy | 0.229 | [−0.250, 0.708] | 1 | 1/0/19 | inconclusive |
| C100 ResNet-8 | P0 - global | test_accuracy | 0.275 | [0.223, 0.327] | 4.6e-05 | 20/0/0 | supported |
| C100 ResNet-8 | P0 - lamp | test_accuracy | −0.750 | [−0.834, −0.665] | 5.3e-05 | 1/19/0 | opposite |
| C100 ResNet-8 | P0 - erk | test_accuracy | −0.798 | [−0.907, −0.688] | 5.3e-05 | 1/19/0 | opposite |
| C100 ResNet-8 | P0 - random | test_accuracy | 38.757 | [38.547, 38.968] | 4.6e-05 | 20/0/0 | supported |
| C100 ResNet-8 | P1 - P0 | vrl_auc | 0.536 | [0.360, 0.713] | 5.7e-05 | 18/2/0 | supported |
| C100 ResNet-8 | P1 - P2 | vrl_auc | 0.152 | [−0.056, 0.359] | 0.81 | 13/7/0 | inconclusive |
| C100 ResNet-8 | P1 - P3 | vrl_auc | 0.127 | [−0.042, 0.297] | 0.81 | 11/9/0 | inconclusive |
| C100 ResNet-8 | P1 - P0 | rank_auc | 0.013 | [0.011, 0.016] | 1.1e-05 | 20/0/0 | supported |
| C100 ResNet-8 | P1 - P0 | utility_auc | 0.019 | [0.014, 0.023] | 1.5e-05 | 19/1/0 | supported |
| C100 ResNet-8 | P1 - P0 | destructive_frequency | −0.009 | [−0.012, −0.007] | 1.4e-04 | 1/18/1 | supported |
| C100 ResNet-8 | P1 - P0 | excess_over_global | 0.153 | [−0.190, 0.495] | 0.94 | 2/2/16 | inconclusive |
| C100 ResNet-8 | P1 - P0 | test_accuracy | 0.116 | [−0.060, 0.291] | 1 | 3/1/16 | inconclusive |

*Source: `results/architecture_generalization/phase3_statistics.csv`.*

## Appendix C — Sensitivity Vectors and Phase 4 Prior Diagnostics

### C.1 Normalised loss sensitivity S_l by setting

| Setting | Config. I VAL-RL (Phases 3–4) | Config. II VAL-RL (Phase 5) | Phase-5 P2 coefficients c_l = S_l − S̄ |
|---|---|---|---|
| C10 SimpleCNN | 1.000, 0.032, 0.009, 0.000 | 1.000, 0.030, 0.012, 0.000 | 0.740, −0.230, −0.249, −0.260 |
| C10 LeNet-5 | 1.000, 0.422, 0.046, 0.000 | 1.000, 0.440, 0.042, 0.000 | 0.629, 0.070, −0.329, −0.371 |
| C10 ResNet-8 | 1.000, 0.138, 0.331, 0.000 | 1.000, 0.145, 0.320, 0.000 | 0.634, −0.221, −0.046, −0.366 |
| C100 SimpleCNN | 1.000, 0.103, 0.014, 0.000 | 1.000, 0.119, 0.010, 0.000 | 0.718, −0.164, −0.272, −0.282 |
| C100 LeNet-5 | 1.000, 0.587, 0.000, 0.086 | 1.000, 0.564, 0.000, 0.059 | 0.594, 0.159, −0.406, −0.347 |
| C100 ResNet-8 | 0.987, 0.310, 1.000, 0.000 | 1.000, 0.309, 0.997, 0.000 | 0.423, −0.268, 0.421, −0.577 |

Phase 7B structured sensitivity (C10 SimpleCNN, VAL-RL (2,500 images), units conv1, conv2, conv3, fc1): normalised 0.898, 1.000, 0.341, 0.000; centred 0.338, 0.440, −0.219, −0.560.

*Sources: `results/architecture_generalization/phase3_frozen_inputs.json`; `results/phase5/phase5_config.json`; `results/phase7b_green_pruning/structured_sensitivity.json`.*

### C.2 Phase 4 prior comparisons (reward-rank AUC and secondary endpoints)

| Setting | Comparison | Endpoint | Mean diff. | 95% t-CI | Holm p | +/−/tie | Outcome |
|---|---|---|---|---|---|---|---|
| C10 SimpleCNN | P2 - P3 | rank_auc | 0.097 | [0.074, 0.120] | 2.3e-05 | 20/0/0 | supported |
| C10 SimpleCNN | P2 - P4 | rank_auc | 0.070 | [0.052, 0.088] | 3.8e-05 | 19/1/0 | supported |
| C10 LeNet-5 | P2 - P3 | rank_auc | 0.071 | [0.049, 0.093] | 8.4e-05 | 18/2/0 | supported |
| C10 LeNet-5 | P2 - P4 | rank_auc | 0.052 | [0.035, 0.069] | 5.7e-05 | 19/1/0 | supported |
| C10 ResNet-8 | P2 - P3 | rank_auc | 0.044 | [0.031, 0.057] | 2.3e-05 | 20/0/0 | supported |
| C10 ResNet-8 | P2 - P4 | rank_auc | 0.028 | [0.017, 0.039] | 3.4e-04 | 19/1/0 | supported |
| C100 SimpleCNN | P2 - P3 | rank_auc | 0.070 | [0.045, 0.095] | 3.8e-05 | 19/1/0 | supported |
| C100 SimpleCNN | P2 - P4 | rank_auc | 0.047 | [0.029, 0.064] | 5.3e-05 | 19/1/0 | supported |
| C100 LeNet-5 | P2 - P3 | rank_auc | 0.044 | [0.026, 0.062] | 3.4e-04 | 17/3/0 | supported |
| C100 LeNet-5 | P2 - P4 | rank_auc | 0.032 | [0.023, 0.041] | 4.6e-05 | 19/1/0 | supported |
| C100 ResNet-8 | P2 - P3 | rank_auc | 0.018 | [0.011, 0.025] | 5.3e-05 | 18/2/0 | supported |
| C100 ResNet-8 | P2 - P4 | rank_auc | −0.002 | [−0.005, 0.001] | 0.13 | 9/11/0 | inconclusive |
| C10 SimpleCNN | P1 - P0 | rank_auc | 0.030 | [0.022, 0.038] | 2.3e-05 | 20/0/0 | supported |
| C10 SimpleCNN | P2 - P0 | rank_auc | 0.061 | [0.045, 0.076] | 2.3e-05 | 20/0/0 | supported |
| C10 LeNet-5 | P1 - P0 | rank_auc | 0.029 | [0.018, 0.041] | 1.1e-04 | 19/1/0 | supported |
| C10 LeNet-5 | P2 - P0 | rank_auc | 0.057 | [0.042, 0.073] | 2.3e-05 | 19/1/0 | supported |
| C10 ResNet-8 | P1 - P0 | rank_auc | 0.020 | [0.013, 0.028] | 1.1e-04 | 19/1/0 | supported |
| C10 ResNet-8 | P2 - P0 | rank_auc | 0.033 | [0.024, 0.042] | 2.3e-05 | 19/1/0 | supported |
| C100 SimpleCNN | P1 - P0 | rank_auc | 0.039 | [0.029, 0.049] | 2.3e-05 | 20/0/0 | supported |
| C100 SimpleCNN | P2 - P0 | rank_auc | 0.051 | [0.038, 0.065] | 2.3e-05 | 19/1/0 | supported |
| C100 LeNet-5 | P1 - P0 | rank_auc | 0.026 | [0.018, 0.034] | 2.3e-05 | 20/0/0 | supported |
| C100 LeNet-5 | P2 - P0 | rank_auc | 0.044 | [0.036, 0.053] | 2.3e-05 | 20/0/0 | supported |
| C100 ResNet-8 | P1 - P0 | rank_auc | 0.014 | [0.011, 0.017] | 2.3e-05 | 20/0/0 | supported |
| C100 ResNet-8 | P2 - P0 | rank_auc | 0.010 | [0.007, 0.013] | 2.3e-05 | 20/0/0 | supported |
| C10 SimpleCNN | P2 - P5 | rank_auc | 0.145 | [0.124, 0.165] | 1.1e-05 | 20/0/0 | supported |
| C10 LeNet-5 | P2 - P5 | rank_auc | 0.102 | [0.083, 0.121] | 1.1e-05 | 20/0/0 | supported |
| C10 ResNet-8 | P2 - P5 | rank_auc | 0.065 | [0.054, 0.077] | 1.1e-05 | 20/0/0 | supported |
| C100 SimpleCNN | P2 - P5 | rank_auc | 0.110 | [0.094, 0.126] | 1.1e-05 | 20/0/0 | supported |
| C100 LeNet-5 | P2 - P5 | rank_auc | 0.073 | [0.066, 0.081] | 1.1e-05 | 20/0/0 | supported |
| C100 ResNet-8 | P2 - P5 | rank_auc | 0.021 | [0.014, 0.028] | 1.1e-05 | 20/0/0 | supported |
| C10 SimpleCNN | P2 - P3 | vrl_auc | 0.880 | [0.656, 1.104] | 6.9e-05 | 19/1/0 | supported |
| C10 SimpleCNN | P2 - P4 | vrl_auc | 0.230 | [0.045, 0.414] | 0.12 | 17/3/0 | inconclusive |
| C10 LeNet-5 | P2 - P3 | vrl_auc | −0.165 | [−0.341, 0.011] | 0.26 | 8/12/0 | inconclusive |
| C10 LeNet-5 | P2 - P4 | vrl_auc | −0.377 | [−0.558, −0.197] | 6.7e-04 | 3/17/0 | opposite |
| C10 ResNet-8 | P2 - P3 | vrl_auc | 0.215 | [−0.382, 0.812] | 1 | 10/10/0 | inconclusive |
| C10 ResNet-8 | P2 - P4 | vrl_auc | −0.760 | [−1.196, −0.324] | 0.012 | 5/15/0 | opposite |
| C100 SimpleCNN | P2 - P3 | vrl_auc | 0.591 | [0.318, 0.863] | 0.0016 | 17/3/0 | supported |
| C100 SimpleCNN | P2 - P4 | vrl_auc | 0.057 | [−0.146, 0.260] | 1 | 13/7/0 | inconclusive |
| C100 LeNet-5 | P2 - P3 | vrl_auc | −0.018 | [−0.152, 0.116] | 1 | 8/12/0 | inconclusive |
| C100 LeNet-5 | P2 - P4 | vrl_auc | −0.259 | [−0.382, −0.137] | 3.1e-04 | 1/19/0 | opposite |
| C100 ResNet-8 | P2 - P3 | vrl_auc | 0.419 | [0.042, 0.796] | 0.12 | 15/5/0 | inconclusive |
| C100 ResNet-8 | P2 - P4 | vrl_auc | −0.513 | [−0.680, −0.346] | 1.9e-04 | 2/18/0 | opposite |
| C10 SimpleCNN | P2 - P3 | utility_auc | 0.088 | [0.068, 0.108] | 2.3e-05 | 20/0/0 | supported |
| C10 SimpleCNN | P2 - P4 | utility_auc | 0.066 | [0.049, 0.083] | 3.8e-05 | 19/1/0 | supported |
| C10 LeNet-5 | P2 - P3 | utility_auc | 0.053 | [0.036, 0.070] | 1.1e-04 | 17/3/0 | supported |
| C10 LeNet-5 | P2 - P4 | utility_auc | 0.040 | [0.027, 0.053] | 4.6e-05 | 19/1/0 | supported |
| C10 ResNet-8 | P2 - P3 | utility_auc | 0.034 | [0.025, 0.043] | 2.3e-05 | 20/0/0 | supported |
| C10 ResNet-8 | P2 - P4 | utility_auc | 0.022 | [0.014, 0.030] | 1.7e-04 | 18/2/0 | supported |
| C100 SimpleCNN | P2 - P3 | utility_auc | 0.075 | [0.050, 0.101] | 3.8e-05 | 19/1/0 | supported |
| C100 SimpleCNN | P2 - P4 | utility_auc | 0.051 | [0.033, 0.069] | 3.8e-05 | 19/1/0 | supported |
| C100 LeNet-5 | P2 - P3 | utility_auc | 0.024 | [0.014, 0.034] | 3.3e-04 | 18/2/0 | supported |
| C100 LeNet-5 | P2 - P4 | utility_auc | 0.018 | [0.013, 0.024] | 4.0e-05 | 19/1/0 | supported |
| C100 ResNet-8 | P2 - P3 | utility_auc | 0.023 | [0.012, 0.034] | 8.6e-05 | 18/2/0 | supported |
| C100 ResNet-8 | P2 - P4 | utility_auc | −0.004 | [−0.009, −0.000] | 0.038 | 8/12/0 | opposite |
| C10 SimpleCNN | P2 - P3 | destructive_frequency | −0.073 | [−0.090, −0.055] | 4.6e-05 | 0/19/1 | supported |
| C10 SimpleCNN | P2 - P4 | destructive_frequency | −0.034 | [−0.046, −0.022] | 2.6e-04 | 1/19/0 | supported |
| C10 LeNet-5 | P2 - P3 | destructive_frequency | −0.083 | [−0.102, −0.064] | 4.6e-05 | 1/19/0 | supported |
| C10 LeNet-5 | P2 - P4 | destructive_frequency | −0.046 | [−0.066, −0.027] | 3.7e-04 | 2/18/0 | supported |
| C10 ResNet-8 | P2 - P3 | destructive_frequency | −0.034 | [−0.052, −0.016] | 0.002 | 3/17/0 | supported |
| C10 ResNet-8 | P2 - P4 | destructive_frequency | −0.010 | [−0.018, −0.001] | 0.062 | 5/15/0 | inconclusive |
| C100 SimpleCNN | P2 - P3 | destructive_frequency | −0.037 | [−0.049, −0.024] | 7.6e-05 | 1/19/0 | supported |
| C100 SimpleCNN | P2 - P4 | destructive_frequency | −0.017 | [−0.026, −0.008] | 0.0034 | 2/16/2 | supported |
| C100 LeNet-5 | P2 - P3 | destructive_frequency | −0.040 | [−0.057, −0.023] | 8.4e-04 | 2/18/0 | supported |
| C100 LeNet-5 | P2 - P4 | destructive_frequency | −0.017 | [−0.028, −0.006] | 0.014 | 3/17/0 | supported |
| C100 ResNet-8 | P2 - P3 | destructive_frequency | −0.013 | [−0.021, −0.006] | 0.0034 | 3/16/1 | supported |
| C100 ResNet-8 | P2 - P4 | destructive_frequency | 0.000 | [−0.001, 0.002] | 0.68 | 7/6/7 | inconclusive |
| C10 SimpleCNN | P2 - P3 | first_episode_top100 | −24.100 | [−75.695, 27.495] | 1 | 2/3/15 | inconclusive |
| C10 SimpleCNN | P2 - P4 | first_episode_top100 | −2.100 | [−6.291, 2.091] | 1 | 1/3/16 | inconclusive |
| C10 LeNet-5 | P2 - P3 | first_episode_top100 | −3.600 | [−14.474, 7.274] | 1 | 1/7/12 | inconclusive |
| C10 LeNet-5 | P2 - P4 | first_episode_top100 | −2.300 | [−12.110, 7.510] | 1 | 2/7/11 | inconclusive |
| C10 ResNet-8 | P2 - P3 | first_episode_top100 | −2.450 | [−5.782, 0.882] | 0.75 | 1/7/12 | inconclusive |
| C10 ResNet-8 | P2 - P4 | first_episode_top100 | −0.700 | [−2.343, 0.943] | 1 | 3/4/13 | inconclusive |
| C100 SimpleCNN | P2 - P3 | first_episode_top100 | −8.150 | [−20.003, 3.703] | 1 | 0/4/16 | inconclusive |
| C100 SimpleCNN | P2 - P4 | first_episode_top100 | −5.750 | [−15.037, 3.537] | 1 | 1/4/15 | inconclusive |
| C100 LeNet-5 | P2 - P3 | first_episode_top100 | 3.400 | [−6.573, 13.373] | 1 | 2/6/12 | inconclusive |
| C100 LeNet-5 | P2 - P4 | first_episode_top100 | −3.850 | [−15.246, 7.546] | 1 | 2/5/13 | inconclusive |
| C100 ResNet-8 | P2 - P3 | first_episode_top100 | −2.850 | [−9.824, 4.124] | 1 | 4/6/10 | inconclusive |
| C100 ResNet-8 | P2 - P4 | first_episode_top100 | 2.950 | [−0.096, 5.996] | 1 | 4/1/15 | inconclusive |
| C10 SimpleCNN | P2 - P3 | first_episode_top50 | −25.950 | [−77.331, 25.431] | 0.44 | 1/5/14 | inconclusive |
| C10 SimpleCNN | P2 - P4 | first_episode_top50 | −11.650 | [−23.063, −0.237] | 0.043 | 0/9/11 | supported |
| C10 LeNet-5 | P2 - P3 | first_episode_top50 | −25.700 | [−51.233, −0.167] | 0.21 | 3/9/8 | inconclusive |
| C10 LeNet-5 | P2 - P4 | first_episode_top50 | −20.250 | [−34.057, −6.443] | 0.043 | 1/10/9 | supported |
| C10 ResNet-8 | P2 - P3 | first_episode_top50 | −11.550 | [−24.897, 1.797] | 0.44 | 1/5/14 | inconclusive |
| C10 ResNet-8 | P2 - P4 | first_episode_top50 | −6.600 | [−14.559, 1.359] | 0.44 | 0/5/15 | inconclusive |
| C100 SimpleCNN | P2 - P3 | first_episode_top50 | −29.150 | [−66.102, 7.802] | 0.14 | 0/7/13 | inconclusive |
| C100 SimpleCNN | P2 - P4 | first_episode_top50 | −7.200 | [−16.140, 1.740] | 0.44 | 1/6/13 | inconclusive |
| C100 LeNet-5 | P2 - P3 | first_episode_top50 | −18.400 | [−41.841, 5.041] | 0.44 | 3/8/9 | inconclusive |
| C100 LeNet-5 | P2 - P4 | first_episode_top50 | −26.950 | [−53.612, −0.288] | 0.023 | 0/10/10 | supported |
| C100 ResNet-8 | P2 - P3 | first_episode_top50 | −3.250 | [−14.492, 7.992] | 0.56 | 4/8/8 | inconclusive |
| C100 ResNet-8 | P2 - P4 | first_episode_top50 | 4.350 | [−0.368, 9.068] | 0.44 | 5/1/14 | inconclusive |
| C10 SimpleCNN | P2 - P3 | best_seen_rank | −9.300 | [−23.708, 5.108] | 0.07 | 1/9/10 | inconclusive |
| C10 SimpleCNN | P2 - P4 | best_seen_rank | −1.400 | [−3.127, 0.327] | 0.11 | 1/6/13 | inconclusive |
| C10 LeNet-5 | P2 - P3 | best_seen_rank | −3.550 | [−8.142, 1.042] | 0.079 | 1/13/6 | inconclusive |
| C10 LeNet-5 | P2 - P4 | best_seen_rank | −0.400 | [−0.890, 0.090] | 0.16 | 1/6/13 | inconclusive |
| C10 ResNet-8 | P2 - P3 | best_seen_rank | −4.400 | [−8.459, −0.341] | 0.037 | 1/12/7 | supported |
| C10 ResNet-8 | P2 - P4 | best_seen_rank | −1.750 | [−3.181, −0.319] | 0.095 | 2/10/8 | inconclusive |
| C100 SimpleCNN | P2 - P3 | best_seen_rank | −2.500 | [−4.607, −0.393] | 0.082 | 2/8/10 | inconclusive |
| C100 SimpleCNN | P2 - P4 | best_seen_rank | −1.750 | [−3.769, 0.269] | 0.15 | 3/7/10 | inconclusive |
| C100 LeNet-5 | P2 - P3 | best_seen_rank | −8.350 | [−12.520, −4.180] | 0.0088 | 2/12/6 | supported |
| C100 LeNet-5 | P2 - P4 | best_seen_rank | −4.400 | [−7.105, −1.695] | 0.016 | 1/11/8 | supported |
| C100 ResNet-8 | P2 - P3 | best_seen_rank | −2.750 | [−5.052, −0.448] | 0.079 | 1/11/8 | inconclusive |
| C100 ResNet-8 | P2 - P4 | best_seen_rank | −1.900 | [−3.784, −0.016] | 0.11 | 2/10/8 | inconclusive |
| C10 SimpleCNN | P2 - P3 | final_rank_vrl | −95.200 | [−141.890, −48.510] | 0.0038 | 2/15/3 | supported |
| C10 SimpleCNN | P2 - P4 | final_rank_vrl | −60.950 | [−106.551, −15.349] | 0.073 | 3/10/7 | inconclusive |
| C10 LeNet-5 | P2 - P3 | final_rank_vrl | −70.600 | [−119.798, −21.402] | 0.061 | 2/15/3 | inconclusive |
| C10 LeNet-5 | P2 - P4 | final_rank_vrl | −59.250 | [−100.244, −18.256] | 0.043 | 2/13/5 | supported |
| C10 ResNet-8 | P2 - P3 | final_rank_vrl | −54.700 | [−104.544, −4.856] | 0.11 | 4/8/8 | inconclusive |
| C10 ResNet-8 | P2 - P4 | final_rank_vrl | −56.400 | [−93.413, −19.387] | 0.015 | 1/11/8 | supported |
| C100 SimpleCNN | P2 - P3 | final_rank_vrl | −57.100 | [−100.545, −13.655] | 0.061 | 4/11/5 | inconclusive |
| C100 SimpleCNN | P2 - P4 | final_rank_vrl | −16.750 | [−43.614, 10.114] | 0.65 | 8/8/4 | inconclusive |
| C100 LeNet-5 | P2 - P3 | final_rank_vrl | −85.650 | [−163.739, −7.561] | 0.13 | 6/10/4 | inconclusive |
| C100 LeNet-5 | P2 - P4 | final_rank_vrl | −66.900 | [−100.953, −32.847] | 0.0053 | 6/13/1 | supported |
| C100 ResNet-8 | P2 - P3 | final_rank_vrl | −5.200 | [−22.501, 12.101] | 0.66 | 3/6/11 | inconclusive |
| C100 ResNet-8 | P2 - P4 | final_rank_vrl | −11.650 | [−33.847, 10.547] | 0.66 | 3/6/11 | inconclusive |

*Source: `results/phase4/phase4_statistics.csv` (families B-*).*

## Appendix D — Archive and Reward-Alignment Diagnostics

### D.1 Phase 4 search quality by variant (VAL-RL (I) ranks among 1,296 policies; medians over 20 runs)

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

*Source: `results/phase4/phase4_search_quality.csv`.*

### D.2 Phase 4 Stage-A rewards (P0, terminal policy)

| Setting | Reward | Sparsity mean | Sparsity SD | VAL-RL retention | Test mean | Reward optimum sparsity | Optimum retention |
|---|---|---|---|---|---|---|---|
| C100 LeNet-5 | R0 | 31.97 | 4.45 | 0.975 | 31.21 | 47.70 | 0.958 |
| C100 LeNet-5 | R1a | 49.18 | 4.96 | 0.886 | 28.51 | 48.46 | 0.945 |
| C100 LeNet-5 | R1b | 31.01 | 4.73 | 0.975 | 31.31 | 47.70 | 0.958 |
| C100 LeNet-5 | R2 | 37.56 | 6.60 | 0.951 | 30.53 | 47.70 | 0.958 |
| C100 ResNet-8 | R0 | 8.28 | 2.83 | 0.988 | 43.11 | 23.53 | 0.934 |
| C100 ResNet-8 | R1a | 42.31 | 4.44 | 0.675 | 28.96 | 45.40 | 0.643 |
| C100 ResNet-8 | R1b | 7.38 | 3.79 | 0.989 | 43.19 | 22.42 | 0.974 |
| C100 ResNet-8 | R2 | 22.97 | 6.99 | 0.928 | 40.20 | 44.23 | 0.663 |
| C100 SimpleCNN | R0 | 51.11 | 6.60 | 0.982 | 49.75 | 53.80 | 0.998 |
| C100 SimpleCNN | R1a | 49.33 | 2.91 | 0.992 | 50.25 | 49.80 | 0.982 |
| C100 SimpleCNN | R1b | 50.16 | 6.95 | 0.984 | 49.86 | 53.80 | 0.998 |
| C100 SimpleCNN | R2 | 52.40 | 7.25 | 0.977 | 49.48 | 53.80 | 0.998 |
| C10 LeNet-5 | R0 | 51.32 | 8.64 | 0.961 | 62.28 | 53.93 | 0.966 |
| C10 LeNet-5 | R1a | 54.03 | 5.84 | 0.946 | 61.50 | 50.28 | 0.970 |
| C10 LeNet-5 | R1b | 49.19 | 10.02 | 0.962 | 62.42 | 53.93 | 0.966 |
| C10 LeNet-5 | R2 | 47.76 | 9.55 | 0.965 | 62.40 | 57.58 | 0.956 |
| C10 ResNet-8 | R0 | 29.03 | 4.65 | 0.973 | 76.82 | 48.78 | 0.897 |
| C10 ResNet-8 | R1a | 40.19 | 5.20 | 0.890 | 70.30 | 48.78 | 0.897 |
| C10 ResNet-8 | R1b | 21.40 | 6.67 | 0.973 | 76.62 | 32.70 | 0.972 |
| C10 ResNet-8 | R2 | 37.30 | 7.10 | 0.929 | 73.35 | 48.78 | 0.897 |
| C10 SimpleCNN | R0 | 54.31 | 7.82 | 0.986 | 76.37 | 56.40 | 0.991 |
| C10 SimpleCNN | R1a | 49.78 | 5.82 | 0.994 | 76.60 | 51.07 | 0.990 |
| C10 SimpleCNN | R1b | 54.49 | 6.08 | 0.987 | 76.48 | 56.40 | 0.991 |
| C10 SimpleCNN | R2 | 54.39 | 7.92 | 0.982 | 76.20 | 58.78 | 0.988 |

*Source: `results/phase4/phase4_reward_comparison.csv`.*

### D.3 Phase 4 final pipeline F and control K against matched baselines (test, pp)

| Pipeline | Setting | Baseline | PPO test | Baseline test | PPO − baseline | PPO wins / ties / baseline wins | Sparsity |
|---|---|---|---|---|---|---|---|
| F | C10 SimpleCNN | lamp | 76.76 | 76.87 | −0.11 | 0/0/20 | 56.06 |
| F | C10 SimpleCNN | global | 76.76 | 76.75 | 0.01 | 3/17/0 | 56.06 |
| F | C10 SimpleCNN | uniform | 76.76 | 40.72 | 36.04 | 20/0/0 | 56.06 |
| F | C10 SimpleCNN | erk | 76.76 | 75.54 | 1.23 | 20/0/0 | 56.06 |
| F | C10 SimpleCNN | random | 76.76 | 15.36 | 61.40 | 20/0/0 | 56.06 |
| F | C10 LeNet-5 | lamp | 63.22 | 63.33 | −0.11 | 12/0/8 | 52.79 |
| F | C10 LeNet-5 | global | 63.22 | 59.17 | 4.05 | 20/0/0 | 52.79 |
| F | C10 LeNet-5 | uniform | 63.22 | 46.43 | 16.79 | 20/0/0 | 52.79 |
| F | C10 LeNet-5 | erk | 63.22 | 61.04 | 2.18 | 20/0/0 | 52.79 |
| F | C10 LeNet-5 | random | 63.22 | 11.97 | 51.25 | 20/0/0 | 52.79 |
| F | C10 ResNet-8 | lamp | 70.83 | 71.51 | −0.68 | 1/0/19 | 47.78 |
| F | C10 ResNet-8 | global | 70.83 | 52.26 | 18.57 | 20/0/0 | 47.78 |
| F | C10 ResNet-8 | uniform | 70.83 | 40.94 | 29.88 | 20/0/0 | 47.78 |
| F | C10 ResNet-8 | erk | 70.83 | 70.92 | −0.09 | 1/0/19 | 47.78 |
| F | C10 ResNet-8 | random | 70.83 | 10.86 | 59.96 | 20/0/0 | 47.78 |
| F | C100 SimpleCNN | lamp | 49.84 | 50.32 | −0.48 | 0/0/20 | 54.20 |
| F | C100 SimpleCNN | global | 49.84 | 49.71 | 0.13 | 20/0/0 | 54.20 |
| F | C100 SimpleCNN | uniform | 49.84 | 18.06 | 31.79 | 20/0/0 | 54.20 |
| F | C100 SimpleCNN | erk | 49.84 | 45.36 | 4.49 | 20/0/0 | 54.20 |
| F | C100 SimpleCNN | random | 49.84 | 2.17 | 47.68 | 20/0/0 | 54.20 |
| F | C100 LeNet-5 | lamp | 30.72 | 30.81 | −0.08 | 3/0/17 | 46.30 |
| F | C100 LeNet-5 | global | 30.72 | 28.08 | 2.65 | 20/0/0 | 46.30 |
| F | C100 LeNet-5 | uniform | 30.72 | 18.67 | 12.05 | 20/0/0 | 46.30 |
| F | C100 LeNet-5 | erk | 30.72 | 25.85 | 4.87 | 20/0/0 | 46.30 |
| F | C100 LeNet-5 | random | 30.72 | 1.74 | 28.99 | 20/0/0 | 46.30 |
| F | C100 ResNet-8 | lamp | 39.22 | 40.84 | −1.62 | 0/0/20 | 27.71 |
| F | C100 ResNet-8 | global | 39.22 | 28.25 | 10.97 | 20/0/0 | 27.71 |
| F | C100 ResNet-8 | uniform | 39.22 | 17.10 | 22.12 | 20/0/0 | 27.71 |
| F | C100 ResNet-8 | erk | 39.22 | 41.30 | −2.08 | 0/0/20 | 27.71 |
| F | C100 ResNet-8 | random | 39.22 | 1.60 | 37.62 | 20/0/0 | 27.71 |
| K | C10 SimpleCNN | lamp | 76.37 | 76.99 | −0.61 | 0/0/20 | 54.31 |
| K | C10 SimpleCNN | global | 76.37 | 76.52 | −0.15 | 16/0/4 | 54.31 |
| K | C10 SimpleCNN | uniform | 76.37 | 46.80 | 29.57 | 20/0/0 | 54.31 |
| K | C10 SimpleCNN | erk | 76.37 | 75.76 | 0.61 | 17/0/3 | 54.31 |
| K | C10 SimpleCNN | random | 76.37 | 17.25 | 59.13 | 20/0/0 | 54.31 |
| K | C10 LeNet-5 | lamp | 62.28 | 63.35 | −1.06 | 0/0/20 | 51.32 |
| K | C10 LeNet-5 | global | 62.28 | 59.16 | 3.12 | 17/0/3 | 51.32 |
| K | C10 LeNet-5 | uniform | 62.28 | 45.78 | 16.51 | 20/0/0 | 51.32 |
| K | C10 LeNet-5 | erk | 62.28 | 59.93 | 2.35 | 20/0/0 | 51.32 |
| K | C10 LeNet-5 | random | 62.28 | 13.01 | 49.27 | 20/0/0 | 51.32 |
| K | C10 ResNet-8 | lamp | 76.82 | 76.55 | 0.27 | 10/0/10 | 29.03 |
| K | C10 ResNet-8 | global | 76.82 | 69.42 | 7.40 | 20/0/0 | 29.03 |
| K | C10 ResNet-8 | uniform | 76.82 | 62.92 | 13.90 | 20/0/0 | 29.03 |
| K | C10 ResNet-8 | erk | 76.82 | 76.87 | −0.05 | 13/0/7 | 29.03 |
| K | C10 ResNet-8 | random | 76.82 | 12.30 | 64.52 | 20/0/0 | 29.03 |
| K | C100 SimpleCNN | lamp | 49.75 | 50.34 | −0.60 | 2/0/18 | 51.11 |
| K | C100 SimpleCNN | global | 49.75 | 49.77 | −0.02 | 7/0/13 | 51.11 |
| K | C100 SimpleCNN | uniform | 49.75 | 21.56 | 28.18 | 20/0/0 | 51.11 |
| K | C100 SimpleCNN | erk | 49.75 | 45.78 | 3.97 | 20/0/0 | 51.11 |
| K | C100 SimpleCNN | random | 49.75 | 2.69 | 47.05 | 20/0/0 | 51.11 |
| K | C100 LeNet-5 | lamp | 31.21 | 31.97 | −0.76 | 0/0/20 | 31.97 |
| K | C100 LeNet-5 | global | 31.21 | 31.52 | −0.31 | 1/0/19 | 31.97 |
| K | C100 LeNet-5 | uniform | 31.21 | 28.65 | 2.57 | 17/3/0 | 31.97 |
| K | C100 LeNet-5 | erk | 31.21 | 31.29 | −0.07 | 7/0/13 | 31.97 |
| K | C100 LeNet-5 | random | 31.21 | 4.09 | 27.12 | 20/0/0 | 31.97 |
| K | C100 ResNet-8 | lamp | 43.11 | 43.80 | −0.70 | 0/2/18 | 8.28 |
| K | C100 ResNet-8 | global | 43.11 | 42.89 | 0.22 | 18/2/0 | 8.28 |
| K | C100 ResNet-8 | uniform | 43.11 | 43.10 | 0.01 | 1/19/0 | 8.28 |
| K | C100 ResNet-8 | erk | 43.11 | 43.86 | −0.75 | 0/2/18 | 8.28 |
| K | C100 ResNet-8 | random | 43.11 | 8.32 | 34.79 | 18/2/0 | 8.28 |

*Source: `results/phase4/phase4_baseline_comparison.csv`.*

### D.4 Phase 4 search, final-model and LAMP families

| Setting | Comparison | Endpoint | Mean diff. | 95% t-CI | Holm p | +/−/tie | Outcome |
|---|---|---|---|---|---|---|---|
| C10 SimpleCNN | R1a - R0 | final_regret_vrl | 0.152 | [−0.236, 0.539] | 1 | 8/7/5 | inconclusive |
| C10 SimpleCNN | R1b - R0 | final_regret_vrl | 0.150 | [−0.022, 0.322] | 1 | 7/4/9 | inconclusive |
| C10 SimpleCNN | R2 - R0 | final_regret_vrl | 0.202 | [−0.047, 0.450] | 1 | 9/2/9 | inconclusive |
| C10 LeNet-5 | R1a - R0 | final_regret_vrl | 0.358 | [−0.344, 1.061] | 1 | 12/5/3 | inconclusive |
| C10 LeNet-5 | R1b - R0 | final_regret_vrl | 0.112 | [−0.247, 0.470] | 1 | 4/7/9 | inconclusive |
| C10 LeNet-5 | R2 - R0 | final_regret_vrl | 0.085 | [−0.147, 0.317] | 1 | 8/4/8 | inconclusive |
| C10 ResNet-8 | R1a - R0 | final_regret_vrl | 1.230 | [0.762, 1.698] | 0.0014 | 15/4/1 | opposite |
| C10 ResNet-8 | R1b - R0 | final_regret_vrl | 0.808 | [0.176, 1.441] | 0.02 | 12/3/5 | opposite |
| C10 ResNet-8 | R2 - R0 | final_regret_vrl | 0.937 | [0.131, 1.742] | 0.42 | 10/7/3 | inconclusive |
| C100 SimpleCNN | R1a - R0 | final_regret_vrl | 0.198 | [0.094, 0.303] | 0.016 | 13/3/4 | opposite |
| C100 SimpleCNN | R1b - R0 | final_regret_vrl | 0.028 | [−0.062, 0.119] | 1 | 4/3/13 | inconclusive |
| C100 SimpleCNN | R2 - R0 | final_regret_vrl | −0.095 | [−0.239, 0.049] | 1 | 5/9/6 | inconclusive |
| C100 LeNet-5 | R1a - R0 | final_regret_vrl | −0.048 | [−0.498, 0.401] | 1 | 3/11/6 | inconclusive |
| C100 LeNet-5 | R1b - R0 | final_regret_vrl | 0.030 | [−0.094, 0.154] | 1 | 4/3/13 | inconclusive |
| C100 LeNet-5 | R2 - R0 | final_regret_vrl | −0.045 | [−0.298, 0.208] | 1 | 6/7/7 | inconclusive |
| C100 ResNet-8 | R1a - R0 | final_regret_vrl | −0.407 | [−0.616, −0.197] | 0.016 | 1/18/1 | supported |
| C100 ResNet-8 | R1b - R0 | final_regret_vrl | 0.005 | [−0.074, 0.084] | 1 | 2/3/15 | inconclusive |
| C100 ResNet-8 | R2 - R0 | final_regret_vrl | 0.040 | [−0.720, 0.800] | 1 | 3/16/1 | inconclusive |
| C10 SimpleCNN | S1 - S0 | sel_rank_vrl | −85.550 | [−95.496, −75.604] | 3.4e-05 | 0/20/0 | supported |
| C10 SimpleCNN | S2 - S0 | sel_rank_vrl | −27.750 | [−42.432, −13.068] | 0.007 | 2/14/4 | supported |
| C10 SimpleCNN | S2A - S0 | sel_rank_vrl | −85.400 | [−95.311, −75.489] | 3.4e-05 | 0/20/0 | supported |
| C10 LeNet-5 | S1 - S0 | sel_rank_vrl | −74.000 | [−95.809, −52.191] | 3.4e-05 | 0/20/0 | supported |
| C10 LeNet-5 | S2 - S0 | sel_rank_vrl | −19.750 | [−41.037, 1.537] | 0.1 | 2/6/12 | inconclusive |
| C10 LeNet-5 | S2A - S0 | sel_rank_vrl | −73.750 | [−95.345, −52.155] | 3.4e-05 | 0/20/0 | supported |
| C10 ResNet-8 | S1 - S0 | sel_rank_vrl | −86.350 | [−99.142, −73.558] | 3.4e-05 | 0/20/0 | supported |
| C10 ResNet-8 | S2 - S0 | sel_rank_vrl | −33.900 | [−62.382, −5.418] | 0.071 | 8/11/1 | inconclusive |
| C10 ResNet-8 | S2A - S0 | sel_rank_vrl | −89.050 | [−101.965, −76.135] | 3.4e-05 | 0/20/0 | supported |
| C100 SimpleCNN | S1 - S0 | sel_rank_vrl | −66.650 | [−85.541, −47.759] | 3.4e-05 | 0/20/0 | supported |
| C100 SimpleCNN | S2 - S0 | sel_rank_vrl | −16.750 | [−35.846, 2.346] | 0.1 | 3/10/7 | inconclusive |
| C100 SimpleCNN | S2A - S0 | sel_rank_vrl | −66.300 | [−84.652, −47.948] | 3.4e-05 | 0/20/0 | supported |
| C100 LeNet-5 | S1 - S0 | sel_rank_vrl | −149.750 | [−176.684, −122.816] | 3.4e-05 | 1/19/0 | supported |
| C100 LeNet-5 | S2 - S0 | sel_rank_vrl | −84.900 | [−120.478, −49.322] | 0.0016 | 3/17/0 | supported |
| C100 LeNet-5 | S2A - S0 | sel_rank_vrl | −149.550 | [−175.962, −123.138] | 3.4e-05 | 1/19/0 | supported |
| C100 ResNet-8 | S1 - S0 | sel_rank_vrl | −97.650 | [−113.231, −82.069] | 3.4e-05 | 0/20/0 | supported |
| C100 ResNet-8 | S2 - S0 | sel_rank_vrl | −37.900 | [−61.703, −14.097] | 0.013 | 2/13/5 | supported |
| C100 ResNet-8 | S2A - S0 | sel_rank_vrl | −97.850 | [−113.834, −81.866] | 3.4e-05 | 0/20/0 | supported |
| C10 SimpleCNN | S2 - S0 | best_seen_rank | −0.200 | [−0.392, −0.008] | 0.5 | 0/4/16 | inconclusive |
| C10 LeNet-5 | S2 - S0 | best_seen_rank | −0.200 | [−0.559, 0.159] | 1 | 1/3/16 | inconclusive |
| C10 ResNet-8 | S2 - S0 | best_seen_rank | −1.950 | [−3.234, −0.666] | 0.0059 | 0/11/9 | supported |
| C100 SimpleCNN | S2 - S0 | best_seen_rank | −0.100 | [−0.244, 0.044] | 1 | 0/2/18 | inconclusive |
| C100 LeNet-5 | S2 - S0 | best_seen_rank | −0.700 | [−1.328, −0.072] | 0.16 | 1/7/12 | inconclusive |
| C100 ResNet-8 | S2 - S0 | best_seen_rank | 0.100 | [−1.364, 1.564] | 1 | 3/5/12 | inconclusive |
| C10 SimpleCNN | S1 - S0 | sel_rank_vselect | −74.600 | [−83.765, −65.435] | 3.4e-05 | 0/20/0 | supported |
| C10 SimpleCNN | S2 - S0 | sel_rank_vselect | −23.700 | [−36.958, −10.442] | 0.01 | 2/14/4 | supported |
| C10 SimpleCNN | S2A - S0 | sel_rank_vselect | −74.750 | [−83.946, −65.554] | 3.4e-05 | 0/20/0 | supported |
| C10 LeNet-5 | S1 - S0 | sel_rank_vselect | −78.350 | [−100.921, −55.779] | 3.4e-05 | 0/20/0 | supported |
| C10 LeNet-5 | S2 - S0 | sel_rank_vselect | −18.850 | [−40.354, 2.654] | 0.17 | 2/6/12 | inconclusive |
| C10 LeNet-5 | S2A - S0 | sel_rank_vselect | −78.500 | [−101.198, −55.802] | 3.4e-05 | 0/20/0 | supported |
| C10 ResNet-8 | S1 - S0 | sel_rank_vselect | −76.850 | [−89.017, −64.683] | 3.4e-05 | 0/20/0 | supported |
| C10 ResNet-8 | S2 - S0 | sel_rank_vselect | −27.300 | [−51.944, −2.656] | 0.1 | 7/12/1 | inconclusive |
| C10 ResNet-8 | S2A - S0 | sel_rank_vselect | −79.000 | [−91.284, −66.716] | 3.4e-05 | 0/20/0 | supported |
| C100 SimpleCNN | S1 - S0 | sel_rank_vselect | −74.200 | [−94.637, −53.763] | 3.4e-05 | 0/20/0 | supported |
| C100 SimpleCNN | S2 - S0 | sel_rank_vselect | −15.750 | [−36.929, 5.429] | 0.17 | 3/10/7 | inconclusive |
| C100 SimpleCNN | S2A - S0 | sel_rank_vselect | −74.500 | [−95.126, −53.874] | 3.4e-05 | 0/20/0 | supported |
| C100 LeNet-5 | S1 - S0 | sel_rank_vselect | −125.550 | [−153.789, −97.311] | 3.4e-05 | 0/20/0 | supported |
| C100 LeNet-5 | S2 - S0 | sel_rank_vselect | −77.050 | [−107.296, −46.804] | 9.4e-04 | 2/18/0 | supported |
| C100 LeNet-5 | S2A - S0 | sel_rank_vselect | −126.700 | [−155.071, −98.329] | 3.4e-05 | 0/20/0 | supported |
| C100 ResNet-8 | S1 - S0 | sel_rank_vselect | −123.500 | [−140.214, −106.786] | 3.4e-05 | 0/20/0 | supported |
| C100 ResNet-8 | S2 - S0 | sel_rank_vselect | −43.150 | [−68.794, −17.506] | 0.012 | 3/12/5 | supported |
| C100 ResNet-8 | S2A - S0 | sel_rank_vselect | −123.750 | [−140.557, −106.943] | 3.4e-05 | 0/20/0 | supported |
| C10 SimpleCNN | F - K | excess_lamp | 0.504 | [0.106, 0.901] | 4.3e-04 | 16/4/0 | supported |
| C10 LeNet-5 | F - K | excess_lamp | 0.951 | [0.652, 1.249] | 6.7e-05 | 18/2/0 | supported |
| C10 ResNet-8 | F - K | excess_lamp | −0.952 | [−1.641, −0.263] | 0.02 | 7/13/0 | opposite |
| C100 SimpleCNN | F - K | excess_lamp | 0.115 | [−0.092, 0.321] | 0.26 | 11/9/0 | inconclusive |
| C100 LeNet-5 | F - K | excess_lamp | 0.673 | [0.494, 0.852] | 2.3e-05 | 19/1/0 | supported |
| C100 ResNet-8 | F - K | excess_lamp | −0.925 | [−1.280, −0.569] | 3.8e-04 | 5/15/0 | opposite |
| C10 SimpleCNN | F - K | excess_global | 0.166 | [−0.206, 0.537] | 0.46 | 4/16/0 | inconclusive |
| C10 LeNet-5 | F - K | excess_global | 0.923 | [0.088, 1.758] | 0.094 | 11/9/0 | inconclusive |
| C10 ResNet-8 | F - K | excess_global | 11.169 | [9.405, 12.933] | 1.5e-05 | 19/1/0 | supported |
| C100 SimpleCNN | F - K | excess_global | 0.155 | [0.015, 0.294] | 0.094 | 13/7/0 | inconclusive |
| C100 LeNet-5 | F - K | excess_global | 2.953 | [2.665, 3.240] | 1.1e-05 | 20/0/0 | supported |
| C100 ResNet-8 | F - K | excess_global | 10.753 | [8.519, 12.987] | 1.1e-05 | 20/0/0 | supported |
| C10 SimpleCNN | F - LAMP | val | −0.111 | [−0.121, −0.101] | 1.1e-05 | 0/20/0 | opposite |
| C10 LeNet-5 | F - LAMP | val | −0.112 | [−0.201, −0.023] | 0.012 | 12/8/0 | opposite |
| C10 ResNet-8 | F - LAMP | val | −0.682 | [−1.157, −0.207] | 0.004 | 1/19/0 | opposite |
| C100 SimpleCNN | F - LAMP | val | −0.481 | [−0.519, −0.443] | 1.1e-05 | 0/20/0 | opposite |
| C100 LeNet-5 | F - LAMP | val | −0.084 | [−0.151, −0.016] | 0.004 | 3/17/0 | opposite |
| C100 ResNet-8 | F - LAMP | val | −1.620 | [−1.941, −1.298] | 1.1e-05 | 0/20/0 | opposite |
| C10 SimpleCNN | F - LAMP > -0.5 | test F - LAMP (+0.5 margin) | −0.111 | [−0.121, −0.101] | 5.7e-06 | 0/20/ | supported |
| C10 LeNet-5 | F - LAMP > -0.5 | test F - LAMP (+0.5 margin) | −0.112 | [−0.201, −0.023] | 5.7e-06 | 12/8/ | supported |
| C10 ResNet-8 | F - LAMP > -0.5 | test F - LAMP (+0.5 margin) | −0.682 | [−1.157, −0.207] | 1 | 1/19/ | inconclusive |
| C100 SimpleCNN | F - LAMP > -0.5 | test F - LAMP (+0.5 margin) | −0.481 | [−0.519, −0.443] | 0.64 | 0/20/ | inconclusive |
| C100 LeNet-5 | F - LAMP > -0.5 | test F - LAMP (+0.5 margin) | −0.084 | [−0.151, −0.016] | 5.7e-06 | 3/17/ | supported |
| C100 ResNet-8 | F - LAMP > -0.5 | test F - LAMP (+0.5 margin) | −1.620 | [−1.941, −1.298] | 1 | 0/20/ | inconclusive |

*Source: `results/phase4/phase4_statistics.csv`. The E family's endpoint is labelled `val` in the source file, but its values are test accuracies (Chapter 5 QA §19.6).*

## Appendix E — Phase 5 Detailed Results

### E.1 Selected policies by setting and condition (20 runs each)

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

*Source: `results/phase5/phase5_constraint_results.csv`.*

### E.2 All pre-registered and descriptive Phase 5 comparisons

| Setting | Comparison | Endpoint | Mean diff. | 95% t-CI | Holm p | +/−/tie | Outcome |
|---|---|---|---|---|---|---|---|
| C10 SimpleCNN | C1 - C0 | test_accuracy | −0.307 | [−0.458, −0.155] | 4.6e-05 | 2/17/1 | opposite |
| C10 LeNet-5 | C1 - C0 | test_accuracy | 0.699 | [0.512, 0.886] | 1.1e-05 | 20/0/0 | supported |
| C10 ResNet-8 | C1 - C0 | test_accuracy | 6.138 | [4.918, 7.357] | 1.1e-05 | 20/0/0 | supported |
| C100 SimpleCNN | C1 - C0 | test_accuracy | −0.208 | [−0.325, −0.091] | 0.0022 | 5/15/0 | opposite |
| C100 LeNet-5 | C1 - C0 | test_accuracy | 1.120 | [0.947, 1.293] | 1.1e-05 | 20/0/0 | supported |
| C100 ResNet-8 | C1 - C0 | test_accuracy | 2.887 | [2.299, 3.475] | 1.1e-05 | 20/0/0 | supported |
| C10 SimpleCNN | C1 - C2 | rank_auc | 0.019 | [0.011, 0.027] | 0.0014 | 18/2/0 | supported |
| C10 LeNet-5 | C1 - C2 | rank_auc | 0.011 | [0.006, 0.016] | 0.0014 | 17/3/0 | supported |
| C10 ResNet-8 | C1 - C2 | rank_auc | 0.004 | [0.001, 0.007] | 0.083 | 16/4/0 | inconclusive |
| C100 SimpleCNN | C1 - C2 | rank_auc | 0.011 | [0.007, 0.016] | 0.0011 | 18/2/0 | supported |
| C100 LeNet-5 | C1 - C2 | rank_auc | 0.009 | [0.004, 0.013] | 0.019 | 16/4/0 | supported |
| C100 ResNet-8 | C1 - C2 | rank_auc | 0.005 | [0.002, 0.008] | 0.016 | 15/5/0 | supported |
| C10 SimpleCNN | C1 - C2 | vrl_accuracy_auc | 0.406 | [0.295, 0.517] | 4.6e-05 | 20/0/0 | supported |
| C10 LeNet-5 | C1 - C2 | vrl_accuracy_auc | 0.152 | [0.097, 0.207] | 9.6e-04 | 19/1/0 | supported |
| C10 ResNet-8 | C1 - C2 | vrl_accuracy_auc | 0.198 | [0.094, 0.301] | 0.011 | 17/3/0 | supported |
| C100 SimpleCNN | C1 - C2 | vrl_accuracy_auc | 0.374 | [0.278, 0.471] | 4.6e-05 | 20/0/0 | supported |
| C100 LeNet-5 | C1 - C2 | vrl_accuracy_auc | 0.095 | [0.065, 0.125] | 3.4e-04 | 18/2/0 | supported |
| C100 ResNet-8 | C1 - C2 | vrl_accuracy_auc | 0.222 | [0.022, 0.421] | 0.27 | 13/7/0 | inconclusive |
| C10 SimpleCNN | C1 - C2 | first_episode_top20 | −123.300 | [−222.999, −23.601] | 0.1 | 0/8/12 | inconclusive |
| C10 LeNet-5 | C1 - C2 | first_episode_top20 | −73.000 | [−153.113, 7.113] | 0.33 | 1/6/13 | inconclusive |
| C10 ResNet-8 | C1 - C2 | first_episode_top20 | −7.950 | [−16.654, 0.754] | 0.33 | 1/6/13 | inconclusive |
| C100 SimpleCNN | C1 - C2 | first_episode_top20 | −36.450 | [−89.448, 16.548] | 0.21 | 2/8/10 | inconclusive |
| C100 LeNet-5 | C1 - C2 | first_episode_top20 | −56.750 | [−174.039, 60.539] | 1 | 2/7/11 | inconclusive |
| C100 ResNet-8 | C1 - C2 | first_episode_top20 | −4.200 | [−11.445, 3.045] | 1 | 1/4/15 | inconclusive |
| C10 SimpleCNN | C1 - C2 | first_episode_top10 | −123.950 | [−222.271, −25.629] | 0.1 | 0/8/12 | inconclusive |
| C10 LeNet-5 | C1 - C2 | first_episode_top10 | −118.350 | [−213.902, −22.798] | 0.17 | 0/7/13 | inconclusive |
| C10 ResNet-8 | C1 - C2 | first_episode_top10 | −5.800 | [−18.766, 7.166] | 1 | 3/5/12 | inconclusive |
| C100 SimpleCNN | C1 - C2 | first_episode_top10 | −66.650 | [−137.451, 4.151] | 0.27 | 0/6/14 | inconclusive |
| C100 LeNet-5 | C1 - C2 | first_episode_top10 | −80.550 | [−163.782, 2.682] | 0.62 | 0/4/16 | inconclusive |
| C100 ResNet-8 | C1 - C2 | first_episode_top10 | −0.400 | [−9.935, 9.135] | 1 | 7/5/8 | inconclusive |
| C10 SimpleCNN | C1 - C2 | destructive_rate | −0.019 | [−0.026, −0.012] | 1.5e-04 | 1/18/1 | supported |
| C10 LeNet-5 | C1 - C2 | destructive_rate | −0.013 | [−0.016, −0.010] | 6.5e-05 | 0/19/1 | supported |
| C10 ResNet-8 | C1 - C2 | destructive_rate | −0.012 | [−0.014, −0.009] | 3.4e-05 | 0/20/0 | supported |
| C100 SimpleCNN | C1 - C2 | destructive_rate | −0.016 | [−0.020, −0.012] | 6.5e-05 | 0/19/1 | supported |
| C100 LeNet-5 | C1 - C2 | destructive_rate | −0.011 | [−0.015, −0.008] | 1.1e-04 | 1/19/0 | supported |
| C100 ResNet-8 | C1 - C2 | destructive_rate | −0.008 | [−0.010, −0.005] | 6.5e-05 | 0/19/1 | supported |
| C10 SimpleCNN | C1 - C2 | best_seen_rank | −7.050 | [−12.803, −1.297] | 0.02 | 0/10/10 | supported |
| C10 LeNet-5 | C1 - C2 | best_seen_rank | −6.250 | [−11.839, −0.661] | 0.094 | 2/8/10 | inconclusive |
| C10 ResNet-8 | C1 - C2 | best_seen_rank | −1.500 | [−2.651, −0.349] | 0.094 | 1/8/11 | inconclusive |
| C100 SimpleCNN | C1 - C2 | best_seen_rank | −2.300 | [−4.203, −0.397] | 0.098 | 1/11/8 | inconclusive |
| C100 LeNet-5 | C1 - C2 | best_seen_rank | −4.400 | [−8.969, 0.169] | 0.23 | 3/7/10 | inconclusive |
| C100 ResNet-8 | C1 - C2 | best_seen_rank | −0.300 | [−0.783, 0.183] | 0.91 | 2/6/12 | inconclusive |
| C10 SimpleCNN | C1 - C2 | selected_rank | −5.800 | [−10.200, −1.400] | 0.02 | 1/11/8 | supported |
| C10 LeNet-5 | C1 - C2 | selected_rank | −7.900 | [−12.394, −3.406] | 0.016 | 1/11/8 | supported |
| C10 ResNet-8 | C1 - C2 | selected_rank | −1.500 | [−2.651, −0.349] | 0.094 | 1/8/11 | inconclusive |
| C100 SimpleCNN | C1 - C2 | selected_rank | −4.950 | [−9.085, −0.815] | 0.0029 | 0/13/7 | supported |
| C100 LeNet-5 | C1 - C2 | selected_rank | −3.150 | [−10.682, 4.382] | 0.91 | 1/7/12 | inconclusive |
| C100 ResNet-8 | C1 - C2 | selected_rank | −0.300 | [−0.783, 0.183] | 0.91 | 2/6/12 | inconclusive |
| C10 SimpleCNN | C1 - LAMP | test_accuracy | −0.494 | [−0.648, −0.340] | 1.1e-05 | 0/20/0 | opposite |
| C10 LeNet-5 | C1 - LAMP | test_accuracy | −0.321 | [−0.439, −0.202] | 1.1e-05 | 1/19/0 | opposite |
| C10 ResNet-8 | C1 - LAMP | test_accuracy | −0.525 | [−0.613, −0.437] | 1.1e-05 | 0/20/0 | opposite |
| C100 SimpleCNN | C1 - LAMP | test_accuracy | −0.289 | [−0.355, −0.223] | 1.1e-05 | 0/20/0 | opposite |
| C100 LeNet-5 | C1 - LAMP | test_accuracy | −0.260 | [−0.362, −0.158] | 5.9e-05 | 4/16/0 | opposite |
| C100 ResNet-8 | C1 - LAMP | test_accuracy | −0.511 | [−0.544, −0.478] | 1.1e-05 | 0/20/0 | opposite |
| C10 SimpleCNN | C1 - global | test_accuracy | −0.094 | [−0.260, 0.073] | 0.26 | 11/9/0 | inconclusive |
| C10 LeNet-5 | C1 - global | test_accuracy | 1.423 | [1.147, 1.699] | 1.1e-05 | 20/0/0 | supported |
| C10 ResNet-8 | C1 - global | test_accuracy | 6.510 | [6.076, 6.944] | 1.1e-05 | 20/0/0 | supported |
| C100 SimpleCNN | C1 - global | test_accuracy | 0.324 | [0.245, 0.403] | 1.1e-05 | 20/0/0 | supported |
| C100 LeNet-5 | C1 - global | test_accuracy | 0.360 | [0.185, 0.535] | 0.0011 | 14/4/2 | supported |
| C100 ResNet-8 | C1 - global | test_accuracy | 3.481 | [2.986, 3.976] | 1.1e-05 | 20/0/0 | supported |
| C10 SimpleCNN | C1 - C0 | minus_lamp | −0.376 | [−0.531, −0.220] | 6.9e-05 | 2/18/0 | opposite |
| C10 LeNet-5 | C1 - C0 | minus_lamp | 0.006 | [−0.133, 0.144] | 1 | 9/10/1 | inconclusive |
| C10 ResNet-8 | C1 - C0 | minus_lamp | 0.108 | [−0.701, 0.917] | 1 | 7/13/0 | inconclusive |
| C100 SimpleCNN | C1 - C0 | minus_lamp | −0.119 | [−0.203, −0.035] | 0.037 | 6/14/0 | opposite |
| C100 LeNet-5 | C1 - C0 | minus_lamp | −0.145 | [−0.276, −0.013] | 0.11 | 6/14/0 | inconclusive |
| C100 ResNet-8 | C1 - C0 | minus_lamp | 0.662 | [0.292, 1.032] | 0.01 | 19/1/0 | supported |
| C10 SimpleCNN | C1 - C0 | selected_rank | −3.050 | [−10.435, 4.335] | 0.81 | 5/15/0 | inconclusive |
| C10 LeNet-5 | C1 - C0 | selected_rank | −506.600 | [−520.349, −492.851] | 1.1e-05 | 0/20/0 | supported |
| C10 ResNet-8 | C1 - C0 | selected_rank | −217.750 | [−253.888, −181.612] | 1.1e-05 | 0/20/0 | supported |
| C100 SimpleCNN | C1 - C0 | selected_rank | −0.850 | [−5.838, 4.138] | 0.81 | 8/12/0 | inconclusive |
| C100 LeNet-5 | C1 - C0 | selected_rank | −369.550 | [−386.738, −352.362] | 1.1e-05 | 0/20/0 | supported |
| C100 ResNet-8 | C1 - C0 | selected_rank | −76.150 | [−87.342, −64.959] | 1.1e-05 | 0/20/0 | supported |
| C10 SimpleCNN | C1 - C0 | selected_sparsity | 0.826 | [−0.020, 1.672] | 0.11 | 15/5/0 | inconclusive |
| C10 LeNet-5 | C1 - C0 | selected_sparsity | −15.323 | [−16.591, −14.055] | 1.1e-05 | 0/20/0 | opposite |
| C10 ResNet-8 | C1 - C0 | selected_sparsity | −21.727 | [−24.570, −18.884] | 1.1e-05 | 0/20/0 | opposite |
| C100 SimpleCNN | C1 - C0 | selected_sparsity | 0.192 | [−0.507, 0.890] | 0.55 | 12/8/0 | inconclusive |
| C100 LeNet-5 | C1 - C0 | selected_sparsity | −15.714 | [−17.385, −14.044] | 1.1e-05 | 0/20/0 | opposite |
| C100 ResNet-8 | C1 - C0 | selected_sparsity | −10.260 | [−11.621, −8.900] | 1.1e-05 | 0/20/0 | opposite |

*Source: `results/phase5/phase5_statistics.csv`.*

## Appendix F — Phase 6 Recovery Results

### F.1 Test accuracy before and after equal recovery, by setting and method (means of 20 runs)

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

*Source: `results/phase6/phase6_pre_post_accuracy.csv`.*

### F.2 PPO vs LAMP and PPO vs global magnitude (H2, H3 and reference families)

| Setting | Family | Endpoint | Mean diff. | 95% t-CI | Holm p | Outcome | CI inside ±0.5 pp |
|---|---|---|---|---|---|---|---|
| C10 SimpleCNN | H2 PPO vs LAMP | test accuracy | −0.219 | [−0.284, −0.154] | 3.8e-05 | opposite | True |
| C10 LeNet-5 | H2 PPO vs LAMP | test accuracy | −0.185 | [−0.276, −0.093] | 0.0026 | opposite | True |
| C10 ResNet-8 | H2 PPO vs LAMP | test accuracy | 0.004 | [−0.024, 0.031] | 0.82 | inconclusive | True |
| C100 SimpleCNN | H2 PPO vs LAMP | test accuracy | −0.050 | [−0.087, −0.013] | 0.055 | inconclusive | True |
| C100 LeNet-5 | H2 PPO vs LAMP | test accuracy | −0.136 | [−0.190, −0.082] | 4.6e-04 | opposite | True |
| C100 ResNet-8 | H2 PPO vs LAMP | test accuracy | −0.039 | [−0.069, −0.009] | 0.055 | inconclusive | True |
| C10 SimpleCNN | H2 PPO vs LAMP | recovery difference | 0.275 | [0.116, 0.434] | 0.0026 | supported | True |
| C10 LeNet-5 | H2 PPO vs LAMP | recovery difference | 0.136 | [0.031, 0.241] | 0.055 | inconclusive | True |
| C10 ResNet-8 | H2 PPO vs LAMP | recovery difference | 0.529 | [0.431, 0.626] | 2.3e-05 | supported | False |
| C100 SimpleCNN | H2 PPO vs LAMP | recovery difference | 0.239 | [0.155, 0.323] | 1.4e-04 | supported | True |
| C100 LeNet-5 | H2 PPO vs LAMP | recovery difference | 0.124 | [0.040, 0.208] | 0.027 | supported | True |
| C100 ResNet-8 | H2 PPO vs LAMP | recovery difference | 0.472 | [0.417, 0.527] | 2.3e-05 | supported | False |
| C10 SimpleCNN | H2 reference: PPO - LAMP before fine-tuning | test accuracy | −0.494 | [−0.648, −0.340] | 1.1e-05 | opposite | False |
| C10 LeNet-5 | H2 reference: PPO - LAMP before fine-tuning | test accuracy | −0.321 | [−0.439, −0.202] | 1.1e-05 | opposite | True |
| C10 ResNet-8 | H2 reference: PPO - LAMP before fine-tuning | test accuracy | −0.525 | [−0.613, −0.437] | 1.1e-05 | opposite | False |
| C100 SimpleCNN | H2 reference: PPO - LAMP before fine-tuning | test accuracy | −0.289 | [−0.355, −0.223] | 1.1e-05 | opposite | True |
| C100 LeNet-5 | H2 reference: PPO - LAMP before fine-tuning | test accuracy | −0.260 | [−0.362, −0.158] | 5.9e-05 | opposite | True |
| C100 ResNet-8 | H2 reference: PPO - LAMP before fine-tuning | test accuracy | −0.511 | [−0.544, −0.478] | 1.1e-05 | opposite | False |
| C10 SimpleCNN | H3 PPO vs global magnitude | test accuracy | −0.161 | [−0.214, −0.107] | 4.6e-05 | opposite | True |
| C10 LeNet-5 | H3 PPO vs global magnitude | test accuracy | 0.051 | [−0.049, 0.151] | 0.92 | inconclusive | True |
| C10 ResNet-8 | H3 PPO vs global magnitude | test accuracy | 0.072 | [0.031, 0.113] | 0.0095 | supported | True |
| C100 SimpleCNN | H3 PPO vs global magnitude | test accuracy | 0.026 | [−0.015, 0.067] | 0.85 | inconclusive | True |
| C100 LeNet-5 | H3 PPO vs global magnitude | test accuracy | 0.119 | [0.057, 0.181] | 0.0066 | supported | True |
| C100 ResNet-8 | H3 PPO vs global magnitude | test accuracy | −0.018 | [−0.065, 0.029] | 0.92 | inconclusive | True |
| C10 SimpleCNN | H3 PPO vs global magnitude | recovery difference | −0.067 | [−0.249, 0.115] | 0.92 | inconclusive | True |
| C10 LeNet-5 | H3 PPO vs global magnitude | recovery difference | −1.372 | [−1.612, −1.132] | 2.3e-05 | opposite | False |
| C10 ResNet-8 | H3 PPO vs global magnitude | recovery difference | −6.438 | [−6.869, −6.007] | 2.3e-05 | opposite | False |
| C100 SimpleCNN | H3 PPO vs global magnitude | recovery difference | −0.298 | [−0.384, −0.212] | 2.3e-05 | opposite | True |
| C100 LeNet-5 | H3 PPO vs global magnitude | recovery difference | −0.241 | [−0.386, −0.096] | 0.016 | opposite | True |
| C100 ResNet-8 | H3 PPO vs global magnitude | recovery difference | −3.499 | [−4.022, −2.976] | 2.3e-05 | opposite | False |
| C10 SimpleCNN | H3 reference: PPO - global magnitude before fine-tuning | test accuracy | −0.094 | [−0.260, 0.073] | 0.26 | inconclusive | True |
| C10 LeNet-5 | H3 reference: PPO - global magnitude before fine-tuning | test accuracy | 1.423 | [1.147, 1.699] | 1.1e-05 | supported | False |
| C10 ResNet-8 | H3 reference: PPO - global magnitude before fine-tuning | test accuracy | 6.510 | [6.076, 6.944] | 1.1e-05 | supported | False |
| C100 SimpleCNN | H3 reference: PPO - global magnitude before fine-tuning | test accuracy | 0.324 | [0.245, 0.403] | 1.1e-05 | supported | True |
| C100 LeNet-5 | H3 reference: PPO - global magnitude before fine-tuning | test accuracy | 0.360 | [0.185, 0.535] | 0.0011 | supported | False |
| C100 ResNet-8 | H3 reference: PPO - global magnitude before fine-tuning | test accuracy | 3.481 | [2.986, 3.976] | 1.1e-05 | supported | False |

*Sources: `results/phase6/phase6_ppo_vs_lamp.csv`, `results/phase6/phase6_ppo_vs_global.csv`. The column 'CI inside ±0.5 pp' is the pre-registered confidence-interval decision rule; it is not a formal equivalence test.*

## Appendix G — Hardware and Latency Protocol

### G.1 H0 calibration: per-configuration decision quantities

| Config. | Signal (ms) | Noise (ms) | SNR | Relative reduction | Positive in all sessions | Usable |
|---|---|---|---|---|---|---|
| A | 0.4127 | 0.0335 | 12.32 | 61.6% | True | True |
| B | 0.1222 | 0.0287 | 4.26 | 31.8% | True | False |
| C | 15.4531 | 1.6070 | 9.62 | 63.2% | True | True |

*Source: `results/paper3/h0_decision.json`. Rule: SNR ≥ 5, reduction ≥ 10%, positive in every session; order A, C, B.*

### G.2 H0 calibration shapes (configuration A)

| Model | Actions | Parameters | Param. reduction | MAC reduction | Median (ms) | Between-session CV | Speed-up vs dense |
|---|---|---|---|---|---|---|---|
| dense | [0, 0, 0, 0] | 620,362 | 0.0% | 0.0% | 0.6699 | 2.14% | 1.000 |
| light | [1, 1, 1, 1] | 501,893 | 19.1% | 17.6% | 0.6015 | 2.22% | 1.116 |
| moderate | [3, 3, 3, 3] | 305,850 | 50.7% | 49.5% | 0.4470 | 1.97% | 1.499 |
| aggressive | [5, 5, 5, 5] | 99,781 | 83.9% | 81.7% | 0.2572 | 2.62% | 2.597 |
| fc_heavy | [0, 0, 0, 5] | 303,276 | 51.1% | 2.9% | 0.6451 | 2.38% | 1.034 |
| conv_heavy | [0, 5, 5, 0] | 232,117 | 62.6% | 65.2% | 0.4566 | 2.49% | 1.466 |

*Source: `results/paper3/h0_latency_summary.csv`.*

### G.3 Latency-table session log (anchor drift; limit 10%)

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

*Source: `results/paper3/h2_latency_freeze.json` (`session_log`).*

### G.4 Proxy validation and ResNet-8 threshold calibration

| Architecture | Validation pairs | Spearman ρ(A₁, A₅) | MAE of q | Top-10 overlap | q₅-feasible pairs | GO |
|---|---|---|---|---|---|---|
| cifar10_simplecnn | 156 | 0.844 | 0.0088 | 3 | 112 | False |
| cifar10_resnet8 | 31 | 0.980 | 0.0679 | 10 | 2 | True |

| Item | Value |
|---|---|
| Calibration policies / truly feasible | 48 / 14 |
| τ_proxy | 0.9601 |
| Recall / precision (TP, FP, FN, TN) | 1.00 / 0.875 (14, 2, 0, 32) |
| Proxy-feasible landscape policies | 52 |
| Best proxy-feasible speed-up | 1.098× |
| Policies ≥ 1.10× | 0 |
| GO to PPO | False |

*Sources: `results/paper3/paper3_proxy_decision.json`, `results/paper3/resnet8_threshold_calibration_decision.json`. The best confirmed five-epoch-feasible calibration policy reached 1.089× ([0, 3, 1, 1]; `resnet8_threshold_calibration_report.md`).*

## Appendix H — Phase 7A Energy Protocol and Session Results

### H.1 Per-session measurements (N = 100,000 batch-1 inferences per block)

| Architecture | Session | Order | Method | J / 1,000 (package) | J / 1,000 (PP0) | Package W | Median latency (ms) |
|---|---|---|---|---|---|---|---|
| simplecnn | 1 | 0 | ppo | 8.781 | 7.179 | 15.55 | 0.5679 |
| simplecnn | 1 | 1 | dense | 10.902 | 8.722 | 14.57 | 0.7472 |
| simplecnn | 1 | 2 | lamp | 9.532 | 7.599 | 14.15 | 0.6690 |
| simplecnn | 1 | 3 | uniform | 7.862 | 6.283 | 14.33 | 0.5444 |
| simplecnn | 2 | 0 | lamp | 10.016 | 8.266 | 15.72 | 0.6384 |
| simplecnn | 2 | 1 | uniform | 7.922 | 6.396 | 14.72 | 0.5341 |
| simplecnn | 2 | 2 | ppo | 8.462 | 6.807 | 14.51 | 0.5831 |
| simplecnn | 2 | 3 | dense | 10.962 | 8.823 | 14.74 | 0.7461 |
| simplecnn | 3 | 0 | dense | 11.496 | 9.422 | 15.97 | 0.7257 |
| simplecnn | 3 | 1 | lamp | 9.760 | 7.875 | 14.71 | 0.6599 |
| simplecnn | 3 | 2 | ppo | 8.541 | 6.830 | 14.51 | 0.5882 |
| simplecnn | 3 | 3 | uniform | 7.906 | 6.374 | 14.53 | 0.5439 |
| simplecnn | 4 | 0 | ppo | 8.806 | 7.254 | 15.80 | 0.5577 |
| simplecnn | 4 | 1 | lamp | 9.716 | 7.847 | 14.70 | 0.6579 |
| simplecnn | 4 | 2 | uniform | 7.859 | 6.381 | 14.64 | 0.5328 |
| simplecnn | 4 | 3 | dense | 11.054 | 8.936 | 15.02 | 0.7313 |
| simplecnn | 5 | 0 | lamp | 10.451 | 8.700 | 16.73 | 0.6296 |
| simplecnn | 5 | 1 | uniform | 8.154 | 6.675 | 15.54 | 0.5242 |
| simplecnn | 5 | 2 | dense | 11.209 | 9.128 | 15.36 | 0.7287 |
| simplecnn | 5 | 3 | ppo | 8.531 | 6.935 | 14.92 | 0.5711 |
| resnet8 | 1 | 0 | dense | 10.876 | 8.853 | 15.64 | 0.6992 |
| resnet8 | 1 | 1 | uniform | 9.588 | 7.665 | 14.72 | 0.6510 |
| resnet8 | 1 | 2 | lamp | 10.076 | 7.978 | 15.09 | 0.6548 |
| resnet8 | 1 | 3 | ppo | 9.303 | 7.488 | 14.55 | 0.6400 |
| resnet8 | 2 | 0 | dense | 11.000 | 9.062 | 15.93 | 0.6929 |
| resnet8 | 2 | 1 | lamp | 9.708 | 7.841 | 14.89 | 0.6490 |
| resnet8 | 2 | 2 | ppo | 9.526 | 7.677 | 14.76 | 0.6425 |
| resnet8 | 2 | 3 | uniform | 9.465 | 7.653 | 14.59 | 0.6459 |
| resnet8 | 3 | 0 | ppo | 10.048 | 8.309 | 16.36 | 0.6176 |
| resnet8 | 3 | 1 | lamp | 9.767 | 7.921 | 15.09 | 0.6452 |
| resnet8 | 3 | 2 | uniform | 9.645 | 7.807 | 14.84 | 0.6457 |
| resnet8 | 3 | 3 | dense | 10.640 | 8.585 | 14.77 | 0.7156 |
| resnet8 | 4 | 0 | dense | 11.192 | 9.238 | 16.11 | 0.6971 |
| resnet8 | 4 | 1 | uniform | 9.801 | 7.943 | 15.03 | 0.6492 |
| resnet8 | 4 | 2 | ppo | 9.592 | 7.767 | 14.88 | 0.6413 |
| resnet8 | 4 | 3 | lamp | 9.819 | 7.903 | 14.96 | 0.6525 |
| resnet8 | 5 | 0 | uniform | 10.372 | 8.651 | 16.91 | 0.6125 |
| resnet8 | 5 | 1 | ppo | 9.709 | 7.977 | 15.59 | 0.6195 |
| resnet8 | 5 | 2 | lamp | 9.909 | 8.090 | 15.52 | 0.6397 |
| resnet8 | 5 | 3 | dense | 10.882 | 8.921 | 15.48 | 0.6984 |

*Source: `results/phase7a_sustainability/energy_raw.csv`.*

### H.2 Session summaries

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

*Source: `results/phase7a_sustainability/energy_summary.csv`.*

## Appendix I — Phase 7B Confirmatory Results

### I.1 Per-seed selected policies (20 confirmatory seeds)

| Seed | Actions | Widths | Proxy q | MAC reduction | Parameter reduction | Table latency (ms) |
|---|---|---|---|---|---|---|
| 700 | [3, 1, 2, 5] | [22, 58, 102, 102] | 0.9900 | 34.3% | 62.4% | 0.5839 |
| 701 | [2, 2, 2, 5] | [26, 51, 102, 102] | 0.9890 | 36.0% | 63.4% | 0.6067 |
| 702 | [2, 2, 2, 5] | [26, 51, 102, 102] | 0.9890 | 36.0% | 63.4% | 0.6067 |
| 703 | [3, 1, 3, 2] | [22, 58, 90, 205] | 0.9890 | 36.8% | 42.5% | 0.5942 |
| 704 | [3, 1, 3, 2] | [22, 58, 90, 205] | 0.9890 | 36.8% | 42.5% | 0.5942 |
| 705 | [2, 2, 2, 4] | [26, 51, 102, 154] | 0.9900 | 35.3% | 49.6% | 0.6157 |
| 706 | [3, 1, 3, 0] | [22, 58, 90, 256] | 0.9895 | 36.2% | 30.6% | 0.6005 |
| 707 | [2, 2, 2, 5] | [26, 51, 102, 102] | 0.9890 | 36.0% | 63.4% | 0.6067 |
| 708 | [3, 1, 3, 0] | [22, 58, 90, 256] | 0.9895 | 36.2% | 30.6% | 0.6005 |
| 709 | [3, 1, 3, 2] | [22, 58, 90, 205] | 0.9890 | 36.8% | 42.5% | 0.5942 |
| 710 | [2, 2, 2, 1] | [26, 51, 102, 230] | 0.9900 | 34.1% | 29.5% | 0.6162 |
| 711 | [2, 2, 2, 1] | [26, 51, 102, 230] | 0.9900 | 34.1% | 29.5% | 0.6162 |
| 712 | [2, 2, 2, 5] | [26, 51, 102, 102] | 0.9890 | 36.0% | 63.4% | 0.6067 |
| 713 | [2, 2, 2, 1] | [26, 51, 102, 230] | 0.9900 | 34.1% | 29.5% | 0.6162 |
| 714 | [2, 2, 2, 4] | [26, 51, 102, 154] | 0.9900 | 35.3% | 49.6% | 0.6157 |
| 715 | [3, 1, 3, 4] | [22, 58, 90, 154] | 0.9890 | 37.5% | 54.4% | 0.5868 |
| 716 | [2, 2, 2, 4] | [26, 51, 102, 154] | 0.9900 | 35.3% | 49.6% | 0.6157 |
| 717 | [3, 1, 3, 4] | [22, 58, 90, 154] | 0.9890 | 37.5% | 54.4% | 0.5868 |
| 718 | [2, 2, 2, 5] | [26, 51, 102, 102] | 0.9890 | 36.0% | 63.4% | 0.6067 |
| 719 | [2, 2, 2, 1] | [26, 51, 102, 230] | 0.9900 | 34.1% | 29.5% | 0.6162 |

*Source: `results/phase7b_green_pruning/selected_policies.json`. Proxy threshold τ̂ = 0.98849. Predicted (surrogate) energies are deliberately omitted: they are search-internal and are not results.*

### I.2 All recovered final models (5-epoch recovery)

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

*Source: `results/phase7b_green_pruning/final_summary.csv`. The last column is the pre-registered joint deployment-efficiency criterion (internal file name GREEN-SUCCESS). Latency and energy were not measured.*

## Appendix J — Energy-Measurement Qualification and Failure Records

### J.1 Dense-only stability pilots and qualification (energy per 1,000 inferences, J)

| Measurement | N per block | Six dense blocks | CV | Limit | Decision |
|---|---|---|---|---|---|
| 7A pilot, simplecnn | 100,000 | 11.518, 11.150, 11.081, 11.253, 10.963, 10.768 | 2.30% | 3% | passed |
| 7A pilot, resnet8 | 100,000 | 10.793, 10.489, 10.555, 10.667, 10.358, 10.312 | 1.74% | 3% | passed |
| 7B surrogate-calibration pilot | 20,000 | 12.200, 11.198, 11.107, 11.729, 11.433, 11.420 | 3.47% | 4% | stable |
| 7B confirmatory pilot (pilot_n100000) | 100,000 | 13.088, 11.414, 11.204, 15.125, 12.265, 12.273 | 11.36% | 3% | unstable |
| 7B confirmatory pilot (pilot_n200000) | 200,000 | 11.814, 12.074, 11.039, 11.539, 11.025, 11.033 | 4.01% | 3% | unstable |
| 7B energy-confirmation qualification | 200,000 | 11.417, 14.278, 12.962, 15.221, 9.514, 9.473 | 19.92% | 3% | not qualified |

*Sources: `results/phase7a_sustainability/energy_pilot/pilot_n100000.json`; `results/phase7b_green_pruning/_energy_calibration/pilot_n20000.json`; `results/phase7b_green_pruning/statistics.json`; `results/phase7b_energy_confirmation/measurement_qualification.json`. A doubled-N 7B pilot attempt stopped by the load gate was declared void by recorded user decision (`_energy_final/void_pilot_n200000_20261027/`, `development_log.md`) and is not listed.*

### J.2 Energy-confirmation qualification blocks (descriptive diagnostics)

| Run | Position | J / 1,000 | Package W | Median latency (ms) | System load (%) | CPU-2 performance (%) | Thermal zone start → end (°C) |
|---|---|---|---|---|---|---|---|
| qual1 | 0 | 11.417 | 11.16 | 1.0456 | 8.4 | 130 | 86 → 76 |
| qual1 | 1 | 14.278 | 13.76 | 1.0513 | 9.2 | 133 | 77 → 83 |
| qual2 | 0 | 12.962 | 12.72 | 1.0479 | 8.9 | 131 | 86 → 84 |
| qual2 | 1 | 15.221 | 14.51 | 1.0517 | 10.2 | 135 | 79 → 92 |
| qual3 | 0 | 9.514 | 9.40 | 1.0420 | 7.6 | 129 | 76 → 81 |
| qual3 | 1 | 9.473 | 9.09 | 1.0430 | 7.9 | 125 | 72 → 73 |

*Source: `results/phase7b_energy_confirmation/measurement_qualification.json` (`blocks`). No cause of the instability is claimed, and no Dense-versus-PPO session was run.*

## Appendix K — Reproducibility Information

### K.1 Frozen inputs (SHA-256 prefixes)

| Item | File | SHA-256 (prefix) |
|---|---|---|
| Dense reference cifar10 simplecnn | cnn_baseline_FIXED.pth | ca28f8345c23365e |
| Dense reference cifar10 lenet5 | lenet5_seed0.pth | 5ca6e0fce0fd4346 |
| Dense reference cifar10 resnet8 | resnet8_seed0.pth | 41fc5ee592b96fa0 |
| Dense reference cifar100 lenet5 | lenet5_seed0.pth | 8e1b1aee6cff5a8e |
| Dense reference cifar100 resnet8 | resnet8_seed0.pth | 8b84bfad252eae41 |
| Dense reference cifar100 simplecnn | simplecnn_seed0.pth | d75dd19ca3178d43 |
| Configuration-I split (CIFAR-10) | archive_validation_split.npz | 55b7ff40a25b9b24 |
| Configuration-I split (CIFAR-100) | cifar100_split_indices.npz | d7351925477304cf |
| Configuration-II split (cifar10) | cifar10_phase5_split.npz | eccd7522d55b4048 |
| Configuration-II split (cifar100) | cifar100_phase5_split.npz | 131cf53457b45ca2 |

*Sources: `results/architecture_generalization/cpu_baselines.csv`; `results/reproducibility/*_split_manifest.json`; `results/phase5/splits/phase5_split_manifest.json`.*

### K.2 Pre-registration and freeze commits

| Experiment | Governing file | Commit that first added it |
|---|---|---|
| Phase 2 dense training | `results/cpu_dense_training_protocol.md` | `366d21ae` |
| Phase 3 | `results/phase3_preregistration.md` | `2dbcc770` |
| Phase 4 | `results/phase4/phase4_preregistration.md` | `f52e87d9` |
| Phase 5 | `results/phase5/phase5_preregistration.md` | `3cbaec1a` |
| Phase 6 | `results/phase6/phase6_preregistration.md` | `b1b5396f` |
| Paper 3 H0 | `results/paper3/h0_protocol.md` | `c2dcb22a` |
| Paper 3 H2 | `results/paper3/paper3_preregistration.md` | `45936846` |
| Paper 3 proxy | `results/paper3/paper3_proxy_protocol.md` | `ec7a7f27` |
| Paper 3 ResNet-8 calibration | `results/paper3/resnet8_threshold_calibration_protocol.md` | `5b127804` |
| Phase 7A | `results/phase7a_sustainability/preregistration.json` | `b8c27665` |
| Phase 7B design | `results/phase7b_green_pruning/phase7b_design.md` | `f5abe039` |
| Phase 7B final | `results/phase7b_green_pruning/phase7b_final_preregistration.json` | `0e101d55` |
| 7B energy confirmation | `results/phase7b_energy_confirmation/protocol.json` | `df233be0` |

*Commits verified with `git log --diff-filter=A` (Chapter 4, Table 4.9).*

### K.3 Software environments

| Item | Core framework (Phases 2–6) | Deployment extension (Paper 3, 7A, 7B, energy confirmation) |
|---|---|---|
| CPU | Intel(R) Core(TM) i5-6400 CPU @ 2.70GHz | 12th Gen Intel(R) Core(TM) i7-1260P |
| OS | Windows-10-10.0.22621-SP0 (Windows 11 build 22621) | Microsoft Windows 11 Home Single Language 10.0.26200 |
| Python | 3.11.9 | 3.11.9 |
| PyTorch | 2.7.1+cpu | 2.7.1+cpu |
| Stable-Baselines3 | 2.8.0 | 2.8.0 (pinned environment) |
| Gymnasium / NumPy | 1.2.3 / 2.2.2 | pinned environment |
| CUDA | False | not used |

*Sources: `results/phase5/phase5_environment.json`; `results/phase7b_energy_confirmation/environment.json`; `requirements_locked.txt`.*

### K.4 Where the full records are

| Phase | Run-level records |
|---|---|
| 3 | `results/architecture_generalization/phase3_all_runs.csv`, `phase3_runs/`, `phase3_predictions/` |
| 4 | `results/phase4/phase4_all_runs.csv`, `runs/`, `predictions/` |
| 5 | `results/phase5/phase5_all_runs.csv`, `phase5_archive.json`, `phase5_final_policies.json`, `runs/` |
| 6 | `results/phase6/phase6_all_runs.csv`, `phase6_final_models.json`, `runs/`, `test_evaluations/` |
| Paper 3 | `results/paper3/h2_latency_table_*.csv`, `h2_latency_raw_*.csv`, `h2_latency_freeze.json` |
| 7A | `results/phase7a_sustainability/energy_raw.csv`, `accuracy_results.csv`, `recovered_models_freeze.json` |
| 7B | `results/phase7b_green_pruning/ppo_all_seeds.csv`, `archive_all_policies.csv`, `recovery_results.csv`, `proxy_landscape.csv` |
| 7B energy confirmation | `results/phase7b_energy_confirmation/energy_raw.csv`, `measurement_qualification.json`, `checkpoint_verification.json` |

