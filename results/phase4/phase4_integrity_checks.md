# Phase 4 integrity checks

Run 2026-09-30T09:27:59+06:00, after all runs, stage selections and the final test evaluation, before any aggregate statistic. **16/16 passed.**

| Status | Check | Detail |
|---|---|---|
| PASS | all expected runs present, no extra runs, no dropped seed | 1200 runs, 1200 expected |
| PASS | all records complete; 2,048 timesteps and 512 episodes each |  |
| PASS | curve files match their hashes |  |
| PASS | TEST never accessed during training, reward, prior or archive selection (record flag; runners and selector contain no test access) |  |
| PASS | stage selections frozen before the test-freeze file (hash recorded in the freeze file) |  |
| PASS | pre-registration, config and reward tables unchanged in every record |  |
| PASS | Phase-3 frozen files unchanged (git diff vs 67f4270 empty for results/architecture_generalization, results/stronger_baselines, checkpoints/phase2_dense, checkpoints/phase3_agents, cnn_baseline_FIXED.pth) |  |
| PASS | reference checkpoints and Phase-3 V_RL landscapes unchanged |  |
| PASS | V_SELECT landscapes unchanged |  |
| PASS | 1 thread in every run |  |
| PASS | identical PPO hyper-parameters in every run (policy class and BC eta aside) |  |
| PASS | BC only in S2 runs (eta 0.1, active), never elsewhere |  |
| PASS | every baseline has exactly the zero count of its PPO policy |  |
| PASS | seeds logged (400-419 in every condition) |  |
| PASS | commit, environment, CPU and package versions recorded | HEAD 75c17a7; Intel(R) Core(TM) i5-6400 CPU @ 2.70GHz; torch 2.7.1+cpu; SB3 2.8.0; threads 1; CUDA False |
| PASS | infrastructure / corruption logs | 2026-09-29T21:55:16+06:00 INFRASTRUCTURE (final test evaluation): worker 0 crashed with PermissionError [WinError 5] in os.replace of the shared file predictions/cifar100_test_labels.npy (two workers created it at the same moment; transient Windows lock). 41 of 1,695 evaluation jobs had been recorde |
