# Phase 4 implementation checks

Run 2026-09-29T18:38:29+06:00 after prepare.py and before any experimental (seed 400-419) run; seed 42 and existing Phase-3 records only; 1 thread. **21/21 passed.**

| Status | Check | Detail |
|---|---|---|
| PASS | thread count fixed at 1 |  |
| PASS | pre-registration file = committed version (914b0418...) | 914b0418a58cc2b0 |
| PASS | reward tables unchanged since prepare.py |  |
| PASS | Phase-3 frozen inputs unchanged (content hash 7a323f87...) |  |
| PASS | Phase-3 V_RL landscapes and reference checkpoints unchanged |  |
| PASS | kappa calibration | {'R1a': 0.04, 'R1b': 0.01} |
| PASS | reward tables = reward formulas (25 random policies x 4 rewards x 2 splits x 6 settings) | [] |
| PASS | R0 table = the Phase-3 reward (lambda_s 0.01) for all 1,296 policies in every setting |  |
| PASS | R0 + P1 via RewardEnv == Phase-3 PruningEnv + Phase-3 prior (64 episodes, bit-identical) |  |
| PASS | cifar10_lenet5 R0: landscape cache == uncached environment (64 episodes) |  |
| PASS | cifar10_lenet5 R1a: landscape cache == uncached environment (64 episodes) |  |
| PASS | cifar10_lenet5 R1b: landscape cache == uncached environment (64 episodes) |  |
| PASS | cifar10_lenet5 R2: landscape cache == uncached environment (64 episodes) |  |
| PASS | cifar100_resnet8 R1a: landscape cache == uncached environment (64 episodes) |  |
| PASS | cifar100_resnet8 R2: landscape cache == uncached environment (64 episodes) |  |
| PASS | ElitePPO with eta = 0 (archive callback attached) == SB3 PPO (64 episodes, bit-identical) |  |
| PASS | ElitePPO eta = 0.1: behavioural cloning active, changes the policy, elite states == environment observations | 4 BC updates; elites [[0, 3, 5, 4], [2, 3, 5, 4], [2, 0, 5, 4], [2, 0, 5, 2], [1, 0, 4, 4]] |
| PASS | priors: P1 = Phase-3; P2 centred (sum 0, zero net conservatism); P3 value-changing permutations; P4 = mean|S - mean S|; P5 = -P2; ratio 0 never shifted |  |
| PASS | metrics on a Phase-3 run: curve rewards = R0 table; archive rule = independent re-implementation; final rank = Phase-3 recorded rank | final rank 92, best-seen 1, archive rank 3 |
| PASS | guards: TEST and V_SELECT raise inside training(); run.py and select_stage.py contain no test access |  |
| PASS | full budget 2,048 timesteps with ElitePPO; runtime within budget R4 (<= 15 min at 1 thread) | 62 s (outcome not inspected) |
