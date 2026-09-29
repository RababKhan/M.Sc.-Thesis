# Phase 3 integrity checks

Run 2026-09-29T16:31:25+06:00, after all 480 runs and the matched baselines, before any aggregate statistic. **22/22 passed.**

| Status | Check | Detail |
|---|---|---|
| PASS | exactly 480 planned PPO runs, one record each | 480 records |
| PASS | every setting x condition has all 20 seeds 300-319 | no seed dropped or added |
| PASS | every record has status 'complete' |  |
| PASS | training-curve file hashes match their records | [] |
| PASS | reference checkpoint hashes: record == frozen input == file on disk == Phase-2 record | {'cifar10_simplecnn': 'ca28f8345c23', 'cifar10_lenet5': '5ca6e0fce0fd', 'cifar10_resnet8': '41fc5ee592b9', 'cifar100_simplecnn': 'd75dd19ca317', 'cifar100_lenet5': '8e1b1aee6cff', 'cifar100_resnet8': '8b84bfad252e'} |
| PASS | frozen inputs (sensitivity vectors, priors, rules) unchanged since before the first run | 3887ffd15812af47 |
| PASS | sensitivity CSV and destructive rules unchanged in every record |  |
| PASS | pre-registration unchanged in every record | 064c6c0557938466 |
| PASS | landscape hashes unchanged in every record |  |
| PASS | equal PPO budget: 2,048 timesteps and 512 episodes in every run |  |
| PASS | identical PPO hyper-parameters in every run (only the policy class differs P0 vs P1-P3) | {"batch_size": 32, "ent_coef": 0.01, "gamma": 0.99, "learning_rate": 0.0003, "n_epochs": 10, "n_steps": 64} |
| PASS | beta 0 for P0 and 0.50 for P1-P3; lambda_s 0.01; state sensitivity 0 |  |
| PASS | pruning units unchanged (names match the frozen Phase-2 definition) |  |
| PASS | thread count fixed at 1 in every run |  |
| PASS | test guard active during every learn() call (DataBundle.training(); freeze before test) |  |
| PASS | guard mechanism raises for TEST and V_SELECT inside training() (re-verified now) |  |
| PASS | no TEST result can influence training: rewards and curves use V_RL only; test read after learn() and freeze | reward accuracy term recomputed from the V_RL accuracy in every record |
| PASS | corruption / repeated-run log | none |
| PASS | exact float parsing (float_precision='round_trip') for every stored CSV | read_csv helper |
| PASS | reference baselines unchanged (Phase-2 files identical to the committed versions) | git diff empty |
| PASS | baselines: 14 exactly matched evaluations per PPO run (sparsity identical) |  |
| PASS | test predictions stored for every run |  |
