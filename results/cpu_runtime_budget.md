# CPU runtime budget (pre-registered)

Part A was written and committed **before** any Phase-2 benchmark or training
run, and is not edited afterwards. Part B, if present, was appended after the
one-epoch benchmark and holds only measurements and the projections computed
from them, as required by rule R2.

## Part A: rules

### Machine and units

- **Machine.** Intel Core i5-6400 (4 cores / 4 threads, 2.7 GHz), 7.9 GB
  RAM, no CUDA GPU; torch 2.7.1+cpu; Windows 11.
- **Machine-hour.** One hour of wall-clock time on this machine. A job at
  4 threads uses the whole machine. Two concurrent 2-thread jobs share it.

### R1: dense training, per run

- **Projection.**
  **T_run = E × (t_epoch + t_val)**, where:
  - E = 30, the protocol's epoch count;
  - t_epoch is the measured wall time of one full training epoch (45,000
    augmented images, batch 128, the protocol's optimiser) at 4 threads;
  - t_val is the time of one 5,000-image validation pass at 4 threads.
- **Limit.** T_run ≤ **2.0 machine-hours** for every (architecture, dataset).

### R2: one-epoch benchmark (before any full run)

- **Procedure.** For each architecture:
  1. Build the seed-0 model.
  2. Run exactly one training epoch under `cpu_dense_training_protocol.md`,
     then one validation pass.
  3. Record t_epoch, t_val, the mean training loss of that epoch,
     non-finite-loss events and the process peak RAM.
  4. Also time one 3,000-image V_RL evaluation (the per-policy cost of an
     exact PPO landscape).
- **Discard.** Benchmark weights are discarded. No accuracy is recorded or
  used.
- **Datasets.** CIFAR-10 for all three architectures. CIFAR-100 is
  benchmarked the same way before its first training run. The architectures
  differ only in the output layer, so its time is expected to be within a few
  percent of CIFAR-10.
- **Report.** After the benchmark, and before any full training run:
  - T_run per (architecture, dataset);
  - the projected Phase-2 total;
  - the projected cost of the exact PPO landscapes.

  These go into Part B of this file and into
  `results/architecture_generalization/cpu_epoch_benchmark.csv`.

### R3: architecture fallback (timing only)

- **Trigger.** If ResNet-8's T_run > 2.0 h on either dataset, it is stopped.
  The pre-registered fallback, SmallVGG (`src/models`), is then benchmarked
  under R2 and used for **both** datasets. One architecture C serves both
  datasets.
- **Fallback also over budget.** If SmallVGG also exceeds the limit, Phase 2
  stops for re-design. No further substitution is made without approval.
- **SimpleCNN or LeNet-5 over budget.** Neither is expected to exceed the
  limit (SimpleCNN measured 0.9 min per epoch in Phase 1). If either did,
  Phase 2 would stop and report.
- **Decision basis.** The decision uses measured time only, **never
  accuracy**. No accuracy of any C candidate is looked at before the decision
  is recorded.

### R4: PPO and landscapes, per run and per setting (applies from Phase 3)

- **PPO run** (512 episodes, exact memoisation): ≤ the cost of the historical
  SimpleCNN PPO run, i.e. **≤ 15 machine-minutes at 1 thread**. For
  reference, the historical non-memoised seed-42 run took 341 s at 4 threads.
- **Exact landscape** (1,296 policies on V_RL, plus V_SELECT where needed):
  **≤ 1.5 machine-hours per setting**.
- **Timing.** These are measured before the Phase-3 pre-registration is
  written.

### R5: phase budgets

- **Phase 2 total.** The benchmarks, 15 dense runs, CIFAR-100 preparation,
  efficiency measurement and checks must total **≤ 30 machine-hours**. It
  must be executable within about 3 calendar days, allowing for power
  interruptions.
- **Later phases.** Each phase from Phase 3 on is budgeted in its own
  pre-registration from measured unit costs. A phase projected above
  **≈ 150 machine-hours** is split or re-designed before it starts.

### R6: no silent reduction

- **Seed counts are fixed.** Seed counts are those of
  `rocksolid_cpu_master_protocol.md` §8. They are never lowered to fit a
  budget.
- **If a budget cannot be met.** The design changes (e.g. the R3 fallback),
  or the problem is reported for a decision, before any run.
