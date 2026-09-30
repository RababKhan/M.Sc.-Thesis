# Phase 5 integrity checks

Run 2026-09-30T18:04:43+06:00, after all 360 runs, the archive freeze, the VAL-SELECT selection and the TEST evaluation, before any aggregate statistic. **14/14 passed.**

| Status | Check | Detail |
|---|---|---|
| PASS | all 360 expected runs completed; no extra run; no seed dropped | 360 records |
| PASS | 2,048 timesteps / 512 episodes in every run; 1 thread |  |
| PASS | curve hashes match |  |
| PASS | TEST and VAL-SELECT never accessed by any training run (record flags; guards raise inside training()) |  |
| PASS | pre-registration, design commitments and config unchanged in every record and on disk |  |
| PASS | archive frozen from VAL-RL curves before the VAL-SELECT selection (archive hash in the final-policy file; file order) |  |
| PASS | final-policy list frozen before any TEST evaluation (freeze file older than every evaluation part) |  |
| PASS | tau, reward constants and seeds unchanged (config vs pre-registration values) |  |
| PASS | Phase-3 and Phase-4 files unchanged (git diff vs 67f4270 / a3653fd empty) |  |
| PASS | dense checkpoints unchanged |  |
| PASS | Phase-5 split files and landscapes unchanged |  |
| PASS | every LAMP / global baseline has exactly the selected policy's zero count (sparsity mismatch 0) |  |
| PASS | commit, CPU/thread settings and package versions recorded | HEAD ddaf72c; Intel(R) Core(TM) i5-6400 CPU @ 2.70GHz; torch 2.7.1+cpu; SB3 2.8.0; threads 1; CUDA False |
| PASS | interruptions logged | no interruption |
