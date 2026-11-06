# Paper 3 — H0 CPU latency calibration protocol

Written 2026-10-23 on branch `paper3-hardware-aware`, **before any H0 timing was run**, and committed
before the measurement. Code: `experiments/paper3/h0_calibrate.py`, `src/evaluation/latency.py`.
Constants: `experiments/paper3/p3_lib.py`.

## Purpose

Decide whether physically compacted structured pruning produces a CPU-latency difference large enough,
relative to measurement noise, to serve as a PPO reward. Decide also which timing configuration is the
primary hardware metric. No accuracy enters this decision.

## Machine

- **CPU:** Intel Core i7-1260P laptop, a hybrid design with 4 P-cores × 2 threads (logical CPUs 0–7)
  and 8 E-cores (logical CPUs 8–15). Windows 11.
- **Software:** torch 2.7.1+cpu (oneDNN 3.7.1, MKL), Python 3.11.9.
- **Not the Phase 1–6 machine.** Those phases ran on an Intel i5-6400 desktop. Every Paper-3 latency
  number (H0, the H2 tables, the final confirmation session) must come from this machine with this
  protocol. Results do not transfer to other CPUs.
- **State checks.** AC power is required: on battery, H0 aborts. The power plan and the AC state are
  recorded. No other experiment runs during H0.

## Models

CIFAR-10 SimpleCNN, built from the frozen reference `cnn_baseline_FIXED.pth` (sha256 asserted) and
physically compacted by `src/pruning/structured_units.slim`. The four units are conv1, conv2, conv3 and
fc1. An action keeps `max(1, round((1 − r)·C))` channels, those with the largest L1 norm.

| Name | Actions | Ratios (conv1, conv2, conv3, fc1) |
|---|---|---|
| dense | [0,0,0,0] | 0, 0, 0, 0 |
| light | [1,1,1,1] | 10%, 10%, 10%, 10% |
| moderate | [3,3,3,3] | 30%, 30%, 30%, 30% |
| aggressive | [5,5,5,5] | 60%, 60%, 60%, 60% |
| fc_heavy | [0,0,0,5] | fc1 60% only |
| conv_heavy | [0,5,5,0] | conv2, conv3 60% only |

## Timing configurations

| Config | Batch | Intra-op threads | Inter-op threads | Pinned logical CPUs | Warm-up / timed passes per model |
|---|---|---|---|---|---|
| A | 1 | 1 | 1 | 2 (one P-core thread, not core 0) | 50 / 300 |
| B | 1 | 4 | 1 | 0, 2, 4, 6 (one thread on each P-core) | 50 / 300 |
| C | 128 | 4 | 1 | 0, 2, 4, 6 | 10 / 50 |

**Pinning.** The CPU affinity and `OMP_NUM_THREADS` / `MKL_NUM_THREADS` are set before torch is
imported; then `torch.set_num_threads` and `torch.set_num_interop_threads(1)` are called. Pinning to
P-cores removes the P-/E-core scheduling noise of the hybrid CPU. It is part of the protocol for every
later Paper-3 latency measurement.

**Measurement.**
- `model.eval()`, `torch.inference_mode()`, float32, NCHW.
- One preallocated input from `torch.randn(B, 3, 32, 32)` with generator seed 0.
- Each pass is timed with `time.perf_counter_ns()`. Garbage collection is disabled during timing.
- Model construction, checkpoint loading and data handling happen before the first warm-up pass and
  are never timed.

**Interleaving and drift.**
- The timed passes of each model are split into 5 rounds.
- Every round starts with an anchor block of the dense model, then times every model once, in an order
  drawn from a recorded seed.
- The anchor blocks track frequency and thermal drift within a session.

**Sessions.**
- 5 independent sessions; each (session, configuration) pair runs in its own fresh Python process.
- The configuration order within a session and the model order within a round are drawn from
  `numpy.random.default_rng` seeded from 20261023 (recorded).
- Before and after each process, the CPU load over 5 s must be at most 15%. A failing process is
  repeated, at most twice, and logged.

## Statistics

For configuration k, session s and model m:
- **Latency of a model in a session:** `L(m, s)` = the median of the model's 300 (or 50) timed passes,
  excluding anchor blocks.
- **Per-session difference:** `Δ(s) = L(dense, s) − L(aggressive, s)`.
- **Signal:** `S_k` = median over sessions of `Δ(s)`.
- **Noise (between-session spread):** `N_k = max(range_s L(dense, s), range_s L(aggressive, s))`.
- **Signal-to-noise:** `SNR_k = |S_k| / N_k`.
- **Relative reduction:** `R_k = S_k / median_s L(dense, s)`.
- **Also reported, for every model and configuration:** median, p95, IQR, throughput, speed-up
  relative to dense, parameter count and reduction, MACs and reduction, and the drift of the anchor
  blocks.

## Decision rule (fixed now; applied mechanically)

1. **Usable configuration.** Configuration k is usable if all three hold:
   - SNR_k ≥ 5;
   - R_k ≥ 0.10, so the aggressive model is at least 10% faster;
   - Δ(s) > 0 in every session, so the direction is reproducible.
2. **Primary metric.** Configurations are checked in the order **A, then C, then B**. The primary
   latency configuration is the first usable one.
   - A is preferred because it is single-image deployment latency.
   - C is the throughput fallback named in the brief, for when batch-1 latency is overhead-dominated or
     too noisy.
   - B is last, as multi-threaded batch-1 latency is mostly dispatch and synchronisation overhead.
3. **H0 GO** requires all four:
   - structured pruning physically reduces parameters (every pruned calibration model has fewer than
     dense);
   - structured pruning physically reduces MACs (likewise);
   - at least one configuration is usable;
   - its signal is larger than the noise, which SNR ≥ 5 already implies.
4. **H0 NO-GO** otherwise. The main experiment stops, and the report explains why the design must be
   reconsidered.

The rule may not be changed after the measurement. A configuration is never chosen because it gives a
more favourable research result.
