# Phase-2 implementation checks

Generated 2026-09-29T03:39:02+06:00 by `experiments/phase2/phase2_checks.py` (4 threads). **69 passed, 0 failed, 0 skipped.** The test set is not read by these checks. Machine-readable copy: `cpu_phase2_checks.json`.


## 0 reproduction

| Status | Check | Detail |
|---|---|---|
| PASS | src/ reproduces the historical SimpleCNN results | 21/21 static checks, 5/5 complete PPO runs bit-identical (library_verification.json) |

## 1 architectures

| Status | Check | Detail |
|---|---|---|
| PASS | simplecnn C=10: instantiates, output (4,10), deterministic init and eval | init hash seed0 fa038b9d84b0 (repeat equal), seed1 differs |
| PASS | simplecnn C=100: instantiates, output (4,100), deterministic init and eval | init hash seed0 d49a8f1dac7c (repeat equal), seed1 differs |
| PASS | lenet5 C=10: instantiates, output (4,10), deterministic init and eval | init hash seed0 ebec4bbdc601 (repeat equal), seed1 differs |
| PASS | lenet5 C=100: instantiates, output (4,100), deterministic init and eval | init hash seed0 d53232ad5912 (repeat equal), seed1 differs |
| PASS | resnet8 C=10: instantiates, output (4,10), deterministic init and eval | init hash seed0 346e3ad4740f (repeat equal), seed1 differs |
| PASS | resnet8 C=100: instantiates, output (4,100), deterministic init and eval | init hash seed0 9427f2ddf7dd (repeat equal), seed1 differs |
| PASS | smallvgg C=10: instantiates, output (4,10), deterministic init and eval | init hash seed0 f861a6aab7e9 (repeat equal), seed1 differs |
| PASS | smallvgg C=100: instantiates, output (4,100), deterministic init and eval | init hash seed0 34db921ad28e (repeat equal), seed1 differs |

## 2 forward

| Status | Check | Detail |
|---|---|---|
| PASS | cifar10 simplecnn: forward on 128 validation images | (128, 10) |
| PASS | cifar10 lenet5: forward on 128 validation images | (128, 10) |
| PASS | cifar10 resnet8: forward on 128 validation images | (128, 10) |
| PASS | cifar10 smallvgg: forward on 128 validation images | (128, 10) |
| PASS | cifar100 simplecnn: forward on 128 validation images | (128, 100) |
| PASS | cifar100 lenet5: forward on 128 validation images | (128, 100) |
| PASS | cifar100 resnet8: forward on 128 validation images | (128, 100) |
| PASS | cifar100 smallvgg: forward on 128 validation images | (128, 100) |

## 3 parameters

| Status | Check | Detail |
|---|---|---|
| PASS | simplecnn C=10: count = analytic formula | total 620,362, weights 619,872, BN 0 |
| PASS | simplecnn C=100: count = analytic formula | total 643,492, weights 642,912, BN 0 |
| PASS | lenet5 C=10: count = analytic formula | total 62,006, weights 61,770, BN 0 |
| PASS | lenet5 C=100: count = analytic formula | total 69,656, weights 69,330, BN 0 |
| PASS | resnet8 C=10: count = analytic formula | total 78,042, weights 77,360, BN 672 |
| PASS | resnet8 C=100: count = analytic formula | total 83,892, weights 83,120, BN 672 |
| PASS | smallvgg C=10: count = analytic formula | total 815,018, weights 813,408, BN 896 |
| PASS | smallvgg C=100: count = analytic formula | total 838,148, weights 836,448, BN 896 |

## 4 units

| Status | Check | Detail |
|---|---|---|
| PASS | simplecnn: 4 units; pruning unit i changes exactly its tensors by round(r*n) | features.0=864, features.3=18,432, features.6=73,728, classifier.1=524,288; excluded ['classifier.3'] |
| PASS | lenet5: 4 units; pruning unit i changes exactly its tensors by round(r*n) | conv1=450, conv2=2,400, fc1=48,000, fc2=10,080; excluded ['classifier.5'] |
| PASS | resnet8: 4 units; pruning unit i changes exactly its tensors by round(r*n) | stem=432, stage1=4,608, stage2=14,336, stage3=57,344; excluded ['fc'] |
| PASS | smallvgg: 4 units; pruning unit i changes exactly its tensors by round(r*n) | stage1=10,080, stage2=55,296, stage3=221,184, fc1=524,288; excluded ['classifier.3'] |

## 5 zero policy

| Status | Check | Detail |
|---|---|---|
| PASS | simplecnn (cnn_baseline_FIXED.pth): [0,0,0,0] preserves logits and V_RL accuracy; sparsity unchanged | V_RL acc 76.97 == episode acc 76.97 |
| PASS | lenet5 (checkpoints\phase2_dense\cifar10\lenet5_seed0.pth): [0,0,0,0] preserves logits and V_RL accuracy; sparsity unchanged | V_RL acc 64.30 == episode acc 64.30 |
| PASS | resnet8 (checkpoints\phase2_dense\cifar10\resnet8_seed0.pth): [0,0,0,0] preserves logits and V_RL accuracy; sparsity unchanged | V_RL acc 78.53 == episode acc 78.53 |

## 6 checkpoints

| Status | Check | Detail |
|---|---|---|
| PASS | simplecnn: save -> load_checkpoint gives identical logits |  |
| PASS | lenet5: save -> load_checkpoint gives identical logits |  |
| PASS | resnet8: save -> load_checkpoint gives identical logits |  |
| PASS | smallvgg: save -> load_checkpoint gives identical logits |  |
| PASS | cnn_baseline_FIXED.pth unchanged and reloads | ca28f8345c23365e |
| PASS | 15 Phase-2 dense checkpoints: hash matches record, reload reproduces recorded validation accuracy | mismatches [] |

## 7 splits

| Status | Check | Detail |
|---|---|---|
| PASS | train/val permutation deterministic; CIFAR-10 val hash 9d3648af... | 9d3648af9194ccc8 |
| PASS | CIFAR-10 V_RL/V_SELECT regenerated identically from labels | 55b7ff40a25b9b24 |
| PASS | CIFAR-100 split regenerated identically; file hash = manifest; 3,000/2,000 disjoint | sha256 d7351925477304cf; V_RL per class 22-39 |

## 8 guards

| Status | Check | Detail |
|---|---|---|
| PASS | test raises before freeze and inside training(); V_SELECT and freeze raise inside training(); V_RL available in training; freeze unlocks; re-entering training re-locks | TEST requested before the model/policy was frozen / TEST requested during training / V_SELECT requested during training / cannot freeze inside training() |

## 9 baselines

| Status | Check | Detail |
|---|---|---|
| PASS | simplecnn @ 30.0%: uniform/global/random/LAMP/ERK hit K=185,962 zeros exactly; classifier dense; per-method invariants | cnn_baseline_FIXED.pth |
| PASS | simplecnn @ 50.0%: uniform/global/random/LAMP/ERK hit K=309,936 zeros exactly; classifier dense; per-method invariants | cnn_baseline_FIXED.pth |
| PASS | simplecnn @ 58.1957%: uniform/global/random/LAMP/ERK hit K=360,739 zeros exactly; classifier dense; per-method invariants | cnn_baseline_FIXED.pth |
| PASS | lenet5 @ 30.0%: uniform/global/random/LAMP/ERK hit K=18,531 zeros exactly; classifier dense; per-method invariants | checkpoints\phase2_dense\cifar10\lenet5_seed0.pth |
| PASS | lenet5 @ 50.0%: uniform/global/random/LAMP/ERK hit K=30,885 zeros exactly; classifier dense; per-method invariants | checkpoints\phase2_dense\cifar10\lenet5_seed0.pth |
| PASS | lenet5 @ 58.1957%: uniform/global/random/LAMP/ERK hit K=35,947 zeros exactly; classifier dense; per-method invariants | checkpoints\phase2_dense\cifar10\lenet5_seed0.pth |
| PASS | resnet8 @ 30.0%: uniform/global/random/LAMP/ERK hit K=23,208 zeros exactly; classifier dense; per-method invariants | checkpoints\phase2_dense\cifar10\resnet8_seed0.pth |
| PASS | resnet8 @ 50.0%: uniform/global/random/LAMP/ERK hit K=38,680 zeros exactly; classifier dense; per-method invariants | checkpoints\phase2_dense\cifar10\resnet8_seed0.pth |
| PASS | resnet8 @ 58.1957%: uniform/global/random/LAMP/ERK hit K=45,020 zeros exactly; classifier dense; per-method invariants | checkpoints\phase2_dense\cifar10\resnet8_seed0.pth |
| PASS | LAMP score = w_u^2 / sum_{|w_v| >= |w_u|} w_v^2 (brute force, 6 weights) | [0.2808988703845834, 0.009708737892169216, 0.09183673953473791, 0.0, 1.0, 0.039215686389090045] |
| PASS | ERK densities proportional to (sum of dims)/(prod of dims) below 1; total kept exact | [1.0, 0.228, 0.1811] |

## 10 structured

| Status | Check | Detail |
|---|---|---|
| PASS | simplecnn: ratio 0 exact; slimmed == masked-equivalent (max |dlogit| 1.4e-06); MACs 10,848,768 -> 3,745,024 | cnn_baseline_FIXED.pth |
| PASS | resnet8: ratio 0 exact; slimmed == masked-equivalent (max |dlogit| 2.4e-06); MACs 12,501,632 -> 7,783,040 | checkpoints\phase2_dense\cifar10\resnet8_seed0.pth |

## 11 training

| Status | Check | Detail |
|---|---|---|
| PASS | 2-epoch run == interrupted-after-epoch-1 + resumed run (weights, history); resume with another thread count refused; test not read | 9d003f8c43fedafe |
| PASS | same seed + threads -> bit-identical training | LeNet-5, 1,280 images, 2 epochs |

## 12 PPO env

| Status | Check | Detail |
|---|---|---|
| PASS | simplecnn (cnn_baseline_FIXED.pth): memoised == uncached (rewards and global RNG state); unit features over concatenated weights; SB3 PPO trains one rollout | V_RL acc 76.97 |
| PASS | lenet5 (checkpoints\phase2_dense\cifar10\lenet5_seed0.pth): memoised == uncached (rewards and global RNG state); unit features over concatenated weights; SB3 PPO trains one rollout | V_RL acc 64.30 |
| PASS | resnet8 (checkpoints\phase2_dense\cifar10\resnet8_seed0.pth): memoised == uncached (rewards and global RNG state); unit features over concatenated weights; SB3 PPO trains one rollout | V_RL acc 78.53 |
| PASS | SoftPriorPolicy with a zero prior == MlpPolicy (4-unit ResNet-8 env) |  |

## 13 efficiency

| Status | Check | Detail |
|---|---|---|
| PASS | simplecnn: FlopCounterMode = 2 x MACs; ptflops >= MACs; storage raw > gzip > 0 | MACs 10,848,768, ptflops 11,136,266 |
| PASS | lenet5: FlopCounterMode = 2 x MACs; ptflops >= MACs; storage raw > gzip > 0 | MACs 651,720, ptflops 683,862 |
| PASS | resnet8: FlopCounterMode = 2 x MACs; ptflops >= MACs; storage raw > gzip > 0 | MACs 12,501,632, ptflops 12,755,594 |
| PASS | smallvgg: FlopCounterMode = 2 x MACs; ptflops >= MACs; storage raw > gzip > 0 | MACs 39,160,320, ptflops 39,849,226 |
| PASS | nominal sparse MACs of the [1,1,5,5] SimpleCNN < dense MACs (theoretical only) | 7,143,155 vs 10,848,768 |

## 13 sensitivity

| Status | Check | Detail |
|---|---|---|
| PASS | simplecnn: loss sensitivity finite and min-max normalised; accuracy drop runs | [1.0, 0.0475, 0.0, 0.0103] |
| PASS | lenet5: loss sensitivity finite and min-max normalised; accuracy drop runs | [1.0, 0.6264, 0.1024, 0.0] |
| PASS | resnet8: loss sensitivity finite and min-max normalised; accuracy drop runs | [1.0, 0.8471, 0.6974, 0.0] |

## Baseline formulas

K = round(s · W / 100) zeros for target total sparsity s over all W Conv2d/Linear weights; all zeros come from the prunable tensors P (n_t weights, N = Σ n_t); the output classifier stays dense; integer per-tensor counts use largest-remainder rounding.

- **Uniform:** each tensor loses ≈ n_t · K / N of its smallest-|w| weights.
- **Global magnitude:** the K smallest |w| over the union of P.
- **Random:** uniform per-tensor counts; the removed weights inside each tensor are drawn uniformly at random (torch.Generator(seed)); 10 seeds from 1000.
- **LAMP** (Lee et al., 2021): with |w_(1)| ≤ … ≤ |w_(n)| inside a tensor, score(u) = w_(u)² / Σ_{v ≥ u} w_(v)²; remove the K smallest scores over P.
- **ERK** (Evci et al., 2020): density d_t = min(1, ε · Σdims(t) / Πdims(t)), ε solved so that Σ d_t n_t = N − K (saturated tensors fixed dense, ε re-solved); smallest-|w| removal inside each tensor.
