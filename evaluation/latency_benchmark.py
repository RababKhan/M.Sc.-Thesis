"""Inference latency: dense baseline vs PPO seed 42 vs matched global magnitude.

All three models share the hardware, thread count, input shape, batch size,
warm-up and number of timed forward passes. Pruning masks are baked in with
prune.remove(), so each pruned model is an ordinary module holding zeros and no
forward pre-hook adds overhead.

Protocol
  * model.eval() and torch.inference_mode()
  * a fixed input tensor per batch size (seeded)
  * WARMUP untimed passes per model, then ROUNDS x PER_ROUND timed passes,
    with the model order shuffled every round so slow drift (thermal, OS
    activity) is spread across models rather than landing on one
  * CUDA, if present, is synchronised around every timed pass
  * mean, median, SD, 95% CI of the mean (normal approximation) and a
    bootstrap 95% CI of the median; median difference vs dense with its own
    bootstrap CI

Run on an otherwise idle machine. The script measures system CPU load before
starting and refuses to run above IDLE_THRESHOLD unless --force is given; the
load before and after is recorded either way.

Outputs
  results/latency_results.csv   one row per (batch size, model)
  results/latency_samples.csv   every timed pass, for re-analysis

Run from the repository root:  python evaluation/latency_benchmark.py
"""
import argparse
import datetime
import os
import platform
import random
import statistics
import sys
import time

import psutil
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import ppo_experiments  # noqa: E402  (provenance helpers only)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints", "cnn_baseline_FIXED.pth")

THREADS = 4
BATCH_SIZES = [1, 128]
WARMUP = 100
ROUNDS = 10
PER_ROUND = 60            # 600 timed passes per model per batch size
BOOTSTRAP = 2000
IDLE_THRESHOLD = 15.0     # % system CPU, measured over 5 s before starting

SEED42_ACTIONS = [1, 1, 5, 5]
SEED42_SPARSITY = 58.19572427856073


def bake(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight_mask"):
            prune.remove(module, "weight")
    model.eval()
    return model


def build_models():
    dense = common.load_baseline(CKPT)
    ppo = bake(common.prune_layerwise(dense, [common.ACTION_TO_PRUNE[a] for a in SEED42_ACTIONS]))
    gm, _ = common.prune_global_to_total(dense, SEED42_SPARSITY)
    gm = bake(gm)

    sp_ppo, sp_gm = common.calculate_sparsity(ppo), common.calculate_sparsity(gm)
    assert abs(sp_ppo - SEED42_SPARSITY) < 1e-9, f"PPO model sparsity {sp_ppo}"
    assert abs(sp_gm - SEED42_SPARSITY) < 0.01, f"global magnitude not matched: {sp_gm}"
    for m in (ppo, gm):
        assert not any(hasattr(x, "weight_mask") for x in m.modules()), "mask hooks still attached"
    return {"dense": (dense, 0.0),
            "ppo_seed42": (ppo, sp_ppo),
            "global_magnitude_matched": (gm, sp_gm)}


def timed_pass(model, x, cuda):
    if cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    model(x)
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def bootstrap_ci(values, stat, rng):
    n = len(values)
    estimates = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(BOOTSTRAP))
    return estimates[int(0.025 * BOOTSTRAP)], estimates[int(0.975 * BOOTSTRAP) - 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="run even if the machine is busy")
    args = parser.parse_args()

    torch.set_num_threads(THREADS)
    load_before = psutil.cpu_percent(interval=5.0)
    print(f"system CPU load before start: {load_before:.1f}%")
    if load_before > IDLE_THRESHOLD and not args.force:
        raise SystemExit(f"machine is not idle ({load_before:.1f}% > {IDLE_THRESHOLD}%); "
                         f"timings would not be comparable. Re-run when idle or pass --force.")

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    models = {name: (m.to(device), sp) for name, (m, sp) in build_models().items()}
    order = list(models)
    rng = random.Random(0)
    samples, rows = [], []

    with torch.inference_mode():
        for batch in BATCH_SIZES:
            x = torch.randn(batch, 3, 32, 32, generator=torch.Generator().manual_seed(batch)).to(device)
            for name in order:
                for _ in range(WARMUP):
                    models[name][0](x)
            for r in range(ROUNDS):
                rng.shuffle(order)
                for name in order:
                    for _ in range(PER_ROUND):
                        samples.append({"batch_size": batch, "model": name, "round": r,
                                        "ms": timed_pass(models[name][0], x, cuda)})

    load_after = psutil.cpu_percent(interval=2.0)
    print(f"system CPU load after: {load_after:.1f}%")

    env = ppo_experiments.environment()
    env["torch_threads"] = torch.get_num_threads()
    measured_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    boot = random.Random(1)
    for batch in BATCH_SIZES:
        dense_ms = [s["ms"] for s in samples if s["batch_size"] == batch and s["model"] == "dense"]
        for name in models:
            ms = [s["ms"] for s in samples if s["batch_size"] == batch and s["model"] == name]
            mean, sd = statistics.mean(ms), statistics.stdev(ms)
            half = 1.96 * sd / len(ms) ** 0.5
            med_lo, med_hi = bootstrap_ci(ms, statistics.median, boot)
            if name == "dense":
                diff = diff_lo = diff_hi = 0.0
            else:
                diff = statistics.median(ms) - statistics.median(dense_ms)
                diffs = []
                for _ in range(BOOTSTRAP):
                    a = [ms[boot.randrange(len(ms))] for _ in ms]
                    b = [dense_ms[boot.randrange(len(dense_ms))] for _ in dense_ms]
                    diffs.append(statistics.median(a) - statistics.median(b))
                diffs.sort()
                diff_lo, diff_hi = diffs[int(0.025 * BOOTSTRAP)], diffs[int(0.975 * BOOTSTRAP) - 1]
            rows.append({
                "batch_size": batch, "model": name, "sparsity": models[name][1],
                "n_timed": len(ms), "warmup": WARMUP,
                "mean_ms": mean, "median_ms": statistics.median(ms), "sd_ms": sd,
                "min_ms": min(ms), "max_ms": max(ms),
                "mean_ci95_low": mean - half, "mean_ci95_high": mean + half,
                "median_ci95_low": med_lo, "median_ci95_high": med_hi,
                "median_diff_vs_dense_ms": diff,
                "median_diff_ci95_low": diff_lo, "median_diff_ci95_high": diff_hi,
                "median_diff_vs_dense_pct": 100 * diff / statistics.median(dense_ms),
                "device": str(device), "cpu": env["cpu"], "threads": THREADS,
                "cpu_load_before_pct": load_before, "cpu_load_after_pct": load_after,
                "python": env["python"], "torch": env["torch"], "os": env["os"],
                "mode": "eval + inference_mode", "input_shape": f"({batch}, 3, 32, 32)",
                "measured_at": measured_at,
            })

    import pandas as pd
    results = os.path.join(ROOT, "results")
    pd.DataFrame(rows).to_csv(os.path.join(results, "latency_results.csv"), index=False)
    pd.DataFrame(samples).to_csv(os.path.join(results, "latency_samples.csv"), index=False)

    print(f"\n{env['cpu']} | {THREADS} threads | {device} | torch {env['torch']} | "
          f"python {platform.python_version()}")
    for r in rows:
        print(f"  batch {r['batch_size']:>3}  {r['model']:<26} median {r['median_ms']:8.3f} ms "
              f"[{r['median_ci95_low']:.3f}, {r['median_ci95_high']:.3f}]  mean {r['mean_ms']:8.3f}  "
              f"sd {r['sd_ms']:6.3f}  diff vs dense {r['median_diff_vs_dense_ms']:+.3f} ms "
              f"[{r['median_diff_ci95_low']:+.3f}, {r['median_diff_ci95_high']:+.3f}]")


if __name__ == "__main__":
    main()
