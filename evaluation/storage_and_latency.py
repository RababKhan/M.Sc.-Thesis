"""What 58.2% unstructured sparsity does and does not buy.

Storage is measured like-for-like (compressed vs compressed, raw vs raw).
Latency is measured under a fixed protocol: pinned thread count, warmup
discarded, many repeats, median reported.

IMPORTANT: run this on an otherwise idle machine. Timings taken while another
job is using the CPU are not comparable.

Run from the repository root:  python evaluation/storage_and_latency.py
"""
import copy
import gzip
import io
import os
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints", "cnn_baseline_FIXED.pth")

THREADS = 4
BATCH = 128
WARMUP = 20
REPEATS = 100

torch.set_num_threads(THREADS)

model = common.load_baseline(CKPT)
pruned = common.prune_layerwise(model, [0.1, 0.1, 0.6, 0.6])   # PPO seed 42

# Bake the masks in so the pruned model is an ordinary module holding zeros.
for _, module in pruned.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight_mask"):
        prune.remove(module, "weight")

sparsity = common.calculate_sparsity(pruned)

# ------------------------------------------------------------------ storage
def sizes(m):
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    raw = buf.getvalue()
    return len(raw), len(gzip.compress(raw, 9))


dense_raw, dense_gz = sizes(model)
pruned_raw, pruned_gz = sizes(pruned)

total = sum(common.layer_param_counts(pruned).values())
nonzero = sum(
    int(torch.count_nonzero(m.weight))
    for m in pruned.modules()
    if isinstance(m, (nn.Conv2d, nn.Linear))
)

print("== Storage ==")
print(f"  sparsity                        : {sparsity:.2f}%")
print(f"  weights                         : {total:,} total, {nonzero:,} non-zero "
      f"({100*nonzero/total:.1f}% density)")
print()
print(f"  dense  .pth                     : {dense_raw/1024:8.1f} KB")
print(f"  pruned .pth                     : {pruned_raw/1024:8.1f} KB   "
      f"({100*(1-pruned_raw/dense_raw):+.1f}% vs dense)")
print(f"  dense  .pth gzipped             : {dense_gz/1024:8.1f} KB")
print(f"  pruned .pth gzipped             : {pruned_gz/1024:8.1f} KB   "
      f"({100*(1-pruned_gz/dense_gz):+.1f}% vs dense gzipped)")
print()
print(f"  theoretical CSR (val+int32 idx) : {nonzero*8/1024:8.1f} KB   "
      f"({100*(1-nonzero*8/(total*4)):+.1f}% vs dense float32)")
print(f"  theoretical bitmask + values    : {(nonzero*4 + total//8)/1024:8.1f} KB   "
      f"({100*(1-(nonzero*4+total//8)/(total*4)):+.1f}% vs dense float32)")
print()
print("  The raw .pth is unchanged: unstructured pruning stores the zeros. The")
print("  saving is realised only by a format that exploits them, and naive CSR")
print("  gives back most of it because int32 indices cost as much as the values.")

# ------------------------------------------------------------------ latency
def time_forward(m, batch):
    m.eval()
    x = torch.randn(batch, 3, 32, 32)
    with torch.no_grad():
        for _ in range(WARMUP):
            m(x)
        samples = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            m(x)
            samples.append((time.perf_counter() - start) * 1000.0)
    return samples


print(f"\n== Inference latency (batch {BATCH}, {THREADS} threads, "
      f"{WARMUP} warmup + {REPEATS} timed) ==")
results = {}
for label, m in (("dense", model), ("pruned (58.2% zeros)", pruned)):
    s = time_forward(m, BATCH)
    results[label] = s
    print(f"  {label:<22} median {statistics.median(s):6.2f} ms   "
          f"mean {statistics.mean(s):6.2f}   min {min(s):6.2f}   "
          f"stdev {statistics.stdev(s):5.2f}")

delta = statistics.median(results["pruned (58.2% zeros)"]) - statistics.median(results["dense"])
print(f"\n  difference: {delta:+.2f} ms "
      f"({100*delta/statistics.median(results['dense']):+.1f}%)")
print("  Dense kernels perform exactly the same multiply-accumulates whether or")
print("  not the operands are zero, so no speedup is expected here.")

# --------------------------------------- does a sparse kernel help this shape?
print("\n== Sparse kernel micro-benchmark (classifier.1, the 524,288-weight layer) ==")
weight = dict(pruned.named_modules())["classifier.1"].weight.detach()
x = torch.randn(BATCH, weight.shape[1])
sparse_weight = weight.to_sparse_csr()

dense_samples, sparse_samples = [], []
with torch.no_grad():
    for _ in range(WARMUP):
        x @ weight.T
        torch.sparse.mm(sparse_weight, x.T)
    for _ in range(REPEATS):
        start = time.perf_counter()
        x @ weight.T
        dense_samples.append((time.perf_counter() - start) * 1000.0)
        start = time.perf_counter()
        torch.sparse.mm(sparse_weight, x.T)
        sparse_samples.append((time.perf_counter() - start) * 1000.0)

print(f"  dense  matmul : median {statistics.median(dense_samples):6.3f} ms")
print(f"  sparse matmul : median {statistics.median(sparse_samples):6.3f} ms   "
      f"({statistics.median(sparse_samples)/statistics.median(dense_samples):.2f}x dense)")
print(f"  layer density : {100*int(torch.count_nonzero(weight))/weight.numel():.1f}%")
