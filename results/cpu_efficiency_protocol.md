# CPU efficiency protocol

Replaces `rocksolid_master_protocol.md` §10 for all work from Phase 2 on. The
project has no GPU. **No GPU latency is measured, estimated or claimed**
anywhere in the paper. Implementation: `src/evaluation/efficiency.py`.

## Metrics per model

| Metric | Definition |
|---|---|
| Dense parameters | every parameter (total and trainable), from `src.models.count_parameters` |
| Non-zero parameters | parameters ≠ 0 after pruning masks are folded into the weights |
| Raw storage | bytes of `torch.save(state_dict)` with masks folded (float32, dense layout) |
| Compressed storage | the same bytes, gzip level 9, mtime 0 (deterministic) |
| MACs | conv/linear multiply-accumulates per image (analytic, forward hooks) |
| FLOPs | `torch.utils.flop_counter.FlopCounterMode` (torch 2.7.1): 2 × MACs of convolutions and matrix products; asserted equal to 2 × MACs |
| ptflops MACs | ptflops 0.7.5, pytorch backend: MACs including BatchNorm, activations and pooling |
| Nominal sparse MACs | each conv/linear layer's MACs × its fraction of non-zero weights. **Theoretical only.** Dense CPU kernels do not skip zeros, so this is never presented as a speed-up. |
| CPU latency | ms per forward pass at batch 1, 32 and 128 (below) |
| Throughput | images per second = batch / median latency |
| Peak RAM | process peak working set during inference (below) |

The comparison rules follow.

- **Storage.** Compare raw with raw and gzip with gzip. Sparse storage
  formats (CSR, bitmask) are reported only if actually implemented, and are
  labelled theoretical otherwise.
- **Unstructured pruning.** It changes non-zero parameters, compressed
  storage and nominal MACs. Its latency is measured and reported as it is,
  normally unchanged.
- **Structured pruning.** It is the only source of real MAC and latency
  reductions. Its MACs are counted on the physically slimmed network.

## CPU latency

1. **Machine state.**
   - No other experiment process runs.
   - Mean CPU load is below 10% over 5 s, checked automatically before and
     after the measurement; the measurement is repeated if the check fails.
   - The mains/battery state and Windows power plan are recorded.
2. **Threads.** 4 threads (the whole CPU), primary. Optionally also 1 thread
   (single-core deployment), reported separately.
3. **Mode.** `model.eval()` and `torch.inference_mode()`; float32; a fixed
   random input (generator seed 0) of shape (B, 3, 32, 32).
4. **Batches.** B ∈ {1, 32, 128}.
5. **Warm-up.** 100 untimed passes per model and batch size.
6. **Timed passes.**
   - 500 per model and batch size, in 5 rounds of 100.
   - The model order is rotated every round, so all compared models are
     interleaved in time.
   - Each pass is timed with `time.perf_counter()`.
7. **Reported.**
   - median, mean and SD;
   - bootstrap 95% CI of the median (5,000 resamples);
   - for comparisons, the bootstrap 95% CI of the difference of medians
     against the dense model of the same setting;
   - throughput.
8. **Recorded.** CPU model, logical CPUs, thread count, torch version and
   oneDNN/MKL availability (`torch.backends.mkldnn.is_available()`), OS, and
   date and time.

## Peak RAM

- **Measurement.** Each (model, batch) runs in a fresh Python process:
  1. import torch;
  2. load the checkpoint;
  3. run 20 inference passes.

  The Windows peak working set is read with psutil before and after the
  passes.
- **Reported.**
  - the total process peak (interpreter + torch + model + activations);
  - the increase of the peak caused by the passes;
  - analytically, the parameter bytes and the largest activation tensor.

## Scope of claims

- **CPU only.** All latency and memory claims are for this CPU (Intel Core
  i5-6400, 4 cores) and torch 2.7.1+cpu.
- **Relative claims.** They are stated only where the bootstrap CI of the
  difference excludes zero.
- **Other hardware.** Nothing is extrapolated to GPUs, mobile NPUs or other
  CPUs.
