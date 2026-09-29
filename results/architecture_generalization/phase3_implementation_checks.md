# Phase 3 implementation checks

Run 2026-09-29T10:50:05+06:00 before any experimental (seed 300-319) run, with the non-experimental seed 42 only; 1 thread. **57/57 passed.** Timing runs report runtime only.

| Status | Check | Detail |
|---|---|---|
| PASS | thread count fixed at 1 | 1 |
| PASS | frozen inputs content hash | 7a323f87dc172d20 |
| PASS | pre-registration committed file hash | 064c6c0557938466 |
| PASS | cifar10_simplecnn: reference sha256 = Phase-2 record = frozen input | ca28f8345c23365e |
| PASS | cifar10_simplecnn: 4 frozen units, classifier excluded | ['classifier.3'] |
| PASS | cifar10_simplecnn: V_RL base accuracy reproduces the frozen denominator | 76.96666666666667 |
| PASS | cifar10_simplecnn: 14 landscape policies = live V_RL evaluation (accuracy, sparsity, reward) exactly | [] |
| PASS | cifar10_simplecnn: landscape CSV (round-trip parsing) == exact JSON values | 1,296 rows |
| PASS | cifar10_simplecnn: P2 priors are value-changing permutations of S for all 20 seeds; P3 = mean(S) | constant 0.260238 |
| PASS | cifar10_simplecnn: all five baselines reproduce the PPO model's exact zero count | K = 340,463 |
| PASS | cifar10_lenet5: reference sha256 = Phase-2 record = frozen input | 5ca6e0fce0fd4346 |
| PASS | cifar10_lenet5: 4 frozen units, classifier excluded | ['classifier.5'] |
| PASS | cifar10_lenet5: V_RL base accuracy reproduces the frozen denominator | 64.3 |
| PASS | cifar10_lenet5: 14 landscape policies = live V_RL evaluation (accuracy, sparsity, reward) exactly | [] |
| PASS | cifar10_lenet5: landscape CSV (round-trip parsing) == exact JSON values | 1,296 rows |
| PASS | cifar10_lenet5: P2 priors are value-changing permutations of S for all 20 seeds; P3 = mean(S) | constant 0.367088 |
| PASS | cifar10_lenet5: all five baselines reproduce the PPO model's exact zero count | K = 20,973 |
| PASS | cifar10_resnet8: reference sha256 = Phase-2 record = frozen input | 41fc5ee592b96fa0 |
| PASS | cifar10_resnet8: 4 frozen units, classifier excluded | ['fc'] |
| PASS | cifar10_resnet8: V_RL base accuracy reproduces the frozen denominator | 78.53333333333333 |
| PASS | cifar10_resnet8: 14 landscape policies = live V_RL evaluation (accuracy, sparsity, reward) exactly | [] |
| PASS | cifar10_resnet8: landscape CSV (round-trip parsing) == exact JSON values | 1,296 rows |
| PASS | cifar10_resnet8: P2 priors are value-changing permutations of S for all 20 seeds; P3 = mean(S) | constant 0.367124 |
| PASS | cifar10_resnet8: all five baselines reproduce the PPO model's exact zero count | K = 39,672 |
| PASS | cifar100_simplecnn: reference sha256 = Phase-2 record = frozen input | d75dd19ca3178d43 |
| PASS | cifar100_simplecnn: 4 frozen units, classifier excluded | ['classifier.3'] |
| PASS | cifar100_simplecnn: V_RL base accuracy reproduces the frozen denominator | 50.36666666666667 |
| PASS | cifar100_simplecnn: 14 landscape policies = live V_RL evaluation (accuracy, sparsity, reward) exactly | [] |
| PASS | cifar100_simplecnn: landscape CSV (round-trip parsing) == exact JSON values | 1,296 rows |
| PASS | cifar100_simplecnn: P2 priors are value-changing permutations of S for all 20 seeds; P3 = mean(S) | constant 0.279260 |
| PASS | cifar100_simplecnn: all five baselines reproduce the PPO model's exact zero count | K = 340,463 |
| PASS | cifar100_lenet5: reference sha256 = Phase-2 record = frozen input | 8e1b1aee6cff5a8e |
| PASS | cifar100_lenet5: 4 frozen units, classifier excluded | ['classifier.5'] |
| PASS | cifar100_lenet5: V_RL base accuracy reproduces the frozen denominator | 33.3 |
| PASS | cifar100_lenet5: 14 landscape policies = live V_RL evaluation (accuracy, sparsity, reward) exactly | [] |
| PASS | cifar100_lenet5: landscape CSV (round-trip parsing) == exact JSON values | 1,296 rows |
| PASS | cifar100_lenet5: P2 priors are value-changing permutations of S for all 20 seeds; P3 = mean(S) | constant 0.418295 |
| PASS | cifar100_lenet5: all five baselines reproduce the PPO model's exact zero count | K = 20,973 |
| PASS | cifar100_resnet8: reference sha256 = Phase-2 record = frozen input | 8b84bfad252eae41 |
| PASS | cifar100_resnet8: 4 frozen units, classifier excluded | ['fc'] |
| PASS | cifar100_resnet8: V_RL base accuracy reproduces the frozen denominator | 43.766666666666666 |
| PASS | cifar100_resnet8: 14 landscape policies = live V_RL evaluation (accuracy, sparsity, reward) exactly | [] |
| PASS | cifar100_resnet8: landscape CSV (round-trip parsing) == exact JSON values | 1,296 rows |
| PASS | cifar100_resnet8: P2 priors are value-changing permutations of S for all 20 seeds; P3 = mean(S) | constant 0.574284 |
| PASS | cifar100_resnet8: all five baselines reproduce the PPO model's exact zero count | K = 39,672 |
| PASS | cifar10_simplecnn: P1 with the landscape cache == uncached environment (64 episodes, seed 42) | 64 episodes, identical rows and policy parameters |
| PASS | cifar10_lenet5: P1 with the landscape cache == uncached environment (64 episodes, seed 42) | 64 episodes, identical rows and policy parameters |
| PASS | cifar10_resnet8: P1 with the landscape cache == uncached environment (64 episodes, seed 42) | 64 episodes, identical rows and policy parameters |
| PASS | cifar100_simplecnn: P1 with the landscape cache == uncached environment (64 episodes, seed 42) | 64 episodes, identical rows and policy parameters |
| PASS | cifar100_lenet5: P1 with the landscape cache == uncached environment (64 episodes, seed 42) | 64 episodes, identical rows and policy parameters |
| PASS | cifar100_resnet8: P1 with the landscape cache == uncached environment (64 episodes, seed 42) | 64 episodes, identical rows and policy parameters |
| PASS | P0 MlpPolicy == SoftPriorPolicy with a zero prior (C100 ResNet-8, 64 episodes, seed 42) | identical |
| PASS | cifar10_simplecnn: full budget = 2,048 timesteps / 512 episodes; test and V_SELECT locked during learn() | 62 s at 1 thread (runtime only; outcome not inspected) |
| PASS | cifar10_lenet5: full budget = 2,048 timesteps / 512 episodes; test and V_SELECT locked during learn() | 11 s at 1 thread (runtime only; outcome not inspected) |
| PASS | cifar10_resnet8: full budget = 2,048 timesteps / 512 episodes; test and V_SELECT locked during learn() | 11 s at 1 thread (runtime only; outcome not inspected) |
| PASS | PPO run cost within budget R4 (<= 15 machine-minutes at 1 thread) | {'cifar10_simplecnn': 62, 'cifar10_lenet5': 11, 'cifar10_resnet8': 11} |
| PASS | seeds 300-319 unused by earlier PPO experiments (record scan) | scanned results/**/*.json and agent checkpoints; highest earlier seed 239 |
