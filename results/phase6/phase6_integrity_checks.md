# Phase 6 integrity checks

Run 2026-10-20T17:54:54+06:00, after all runs and before any aggregate statistic. **20/20 passed.**

| Status | Check | Detail |
|---|---|---|
| PASS | all 480 scheduled runs complete; no run dropped or added | 480 records |
| PASS | identical fine-tuning protocol and epoch count in every run |  |
| PASS | pairing: the four runs of a (setting, policy seed) share the fine-tuning seed and the data stream (order + augmentation) |  |
| PASS | matched sparsity: PPO, LAMP and global have the identical zero count in all 120 triples (mismatch 0) |  |
| PASS | masks equal the frozen matched-mask table |  |
| PASS | frozen list: exactly the 480 final-epoch models of stage 1 (state hash, checkpoint hash, zero count) |  |
| PASS | checkpoint files on disk are the frozen ones |  |
| PASS | all 480 TEST evaluations complete, each of the frozen checkpoint | 480 records |
| PASS | order: every fine-tuning run finished before the list was frozen; every TEST evaluation started after it | last run 2026-10-20T16:57:18+06:00; frozen 2026-10-20T17:20:42+06:00; first TEST 2026-10-20T17:21:35+06:00 |
| PASS | stage-1 records hold no TEST accuracy (validation only) |  |
| PASS | pre-fine-tuning test accuracy equals the Phase-5 record in every run |  |
| PASS | fixed mask: mask hash unchanged and no masked weight non-zero in every run |  |
| PASS | no selection: the evaluated model is the one after the final epoch in every run |  |
| PASS | pre-registration and config unchanged in every record and on disk |  |
| PASS | 1 thread in every run |  |
| PASS | Phase 3-5 files unchanged (git diff vs 67f4270 / a3653fd / da061e6 empty) |  |
| PASS | dense checkpoints and split files unchanged |  |
| PASS | reproduction: the 6 pre-registered repeat runs are bit-identical (weights hash and validation accuracy) [reported; decides criterion F] | 6/6 identical |
| PASS | commit, CPU/threads and package versions recorded | HEAD 3181146; Intel(R) Core(TM) i5-6400 CPU @ 2.70GHz; torch 2.7.1+cpu; threads 1; CUDA False |
| PASS | interruptions logged | ﻿2026-10-19T23:11:23+06:00 resume 1: the four stage-1 workers (launched 2026-10-08T11:20) stopped without an error (empty .err logs) when the session ended, last work-state write 2026-10-08T19:16. State before resuming: 180/480 complete records, all valid (config, protocol, seeds, policies, masks, z |
