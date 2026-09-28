"""Architecture tables for Phase 2.

  units        results/architecture_generalization/cpu_prunable_units.csv
               (weight structure only: identical for trained and untrained models)
  efficiency   results/architecture_generalization/cpu_efficiency_baselines.csv
               for every (dataset, architecture) reference model: parameters, MACs, FLOPs,
               ptflops MACs, storage, CPU latency (batch 1/32/128) and peak RAM.
               Requires the trained references and an idle machine (cpu_efficiency_protocol.md).
  dense        cpu_baseline_runs.csv and cpu_baselines.csv from the dense run records
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

import phase2_common as C
from src import models as M, pruning as P
from src.evaluation import efficiency as E

ARCH_STATUS = {"simplecnn": "A (primary)", "lenet5": "B (primary)", "resnet8": "C (primary)",
               "smallvgg": "C' (pre-registered fallback; used only under budget rule R3)"}


def cmd_units(_args):
    rows = []
    for arch in M.ARCHITECTURES:
        for c in (10, 100):
            m = M.build_model(arch, c, seed=0)
            total = P.weight_total(m)
            units = P.get_units(m, arch)
            prunable = sum(u.param_count for u in units)
            _, macs = E.conv_linear_macs(m)
            for i, u in enumerate(units):
                rows.append({"arch": arch, "arch_status": ARCH_STATUS[arch], "num_classes": c, "unit_index": i,
                             "unit": u.name, "unit_type": u.kind, "role": "prunable (PPO decision step)",
                             "modules": ";".join(n for n, _ in u.modules),
                             "module_types": ";".join(type(mm).__name__ for _, mm in u.modules),
                             "weight_shapes": ";".join("x".join(map(str, mm.weight.shape)) for _, mm in u.modules),
                             "params": u.param_count, "fraction_of_all_weights": u.param_count / total,
                             "fraction_of_prunable_weights": u.param_count / prunable,
                             "macs_per_image": sum(macs[n] for n, _ in u.modules)})
            named = dict(m.named_modules())
            for n in P.EXCLUDED[arch]:
                w = named[n].weight
                rows.append({"arch": arch, "arch_status": ARCH_STATUS[arch], "num_classes": c, "unit_index": None,
                             "unit": n, "unit_type": "output classifier", "role": "excluded (never pruned)",
                             "modules": n, "module_types": type(named[n]).__name__,
                             "weight_shapes": "x".join(map(str, w.shape)), "params": w.numel(),
                             "fraction_of_all_weights": w.numel() / total, "fraction_of_prunable_weights": None,
                             "macs_per_image": macs[n]})
    out = os.path.join(C.AG, "cpu_prunable_units.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows)[["arch", "num_classes", "unit", "unit_type", "params", "fraction_of_all_weights"]]
          .to_string(index=False))


def reference_checkpoint(dataset, arch):
    if dataset == "cifar10" and arch == "simplecnn":
        return C.FIXED
    return os.path.join(C.DENSE_CKPT, dataset, f"{arch}_seed0.pth")


def cmd_efficiency(args):
    torch.set_num_threads(args.threads)
    load_before = E.cpu_load(5.0)
    if load_before > 10.0:
        raise SystemExit(f"machine not idle: CPU load {load_before:.1f}% (protocol requires < 10%)")
    archs = [a for a in ("simplecnn", "lenet5", "resnet8", "smallvgg")
             if any(os.path.exists(reference_checkpoint(d, a)) for d in ("cifar10", "cifar100"))]
    rows, samples = [], []
    for dataset in ("cifar10", "cifar100"):
        nc = 10 if dataset == "cifar10" else 100
        models = {a: M.load_checkpoint(reference_checkpoint(dataset, a), a, nc) for a in archs
                  if os.path.exists(reference_checkpoint(dataset, a))}
        lat = {b: E.latency_samples(models, b, threads=args.threads, warmup=100, iterations=500, rounds=5)
               for b in (1, 32, 128)}
        for a, m in models.items():
            ck = reference_checkpoint(dataset, a)
            pc = M.count_parameters(m)
            macs, _ = E.conv_linear_macs(m)
            pt, _ = E.ptflops_macs(m)
            raw, gz = E.storage_bytes(m)
            row = {"dataset": dataset, "arch": a, "checkpoint": os.path.relpath(ck, C.ROOT).replace("\\", "/"),
                   "params_total": pc["total"], "params_trainable": pc["trainable"], "weights_conv_linear": pc["weights"],
                   "params_nonzero": E.nonzero_parameters(m), "macs_conv_linear": macs, "flops": E.torch_flops(m),
                   "ptflops_macs": pt, "raw_state_dict_bytes": raw, "gzip9_bytes": gz,
                   "checkpoint_file_bytes": os.path.getsize(ck)}
            for b in (1, 32, 128):
                s = E.summarize_latency(lat[b][a])
                row.update({f"latency_b{b}_median_ms": s["median_ms"], f"latency_b{b}_mean_ms": s["mean_ms"],
                            f"latency_b{b}_sd_ms": s["sd_ms"], f"latency_b{b}_median_ci95_low": s["median_ci95_low"],
                            f"latency_b{b}_median_ci95_high": s["median_ci95_high"],
                            f"throughput_b{b}_img_s": b / (s["median_ms"] / 1000.0)})
                samples += [{"dataset": dataset, "arch": a, "batch": b, "i": i, "ms": v} for i, v in enumerate(lat[b][a])]
            for b in (1, 128):
                r = E.peak_inference_ram(C.ROOT, a, nc, ck, b, threads=args.threads)
                row[f"peak_ram_b{b}_mb"] = r["peak"] / 2 ** 20 if r["peak"] else None
                row[f"peak_ram_increase_b{b}_mb"] = (r["inference_peak_increase"] / 2 ** 20
                                                     if r["inference_peak_increase"] is not None else None)
            row["param_bytes_fp32"] = 4 * pc["total"]
            rows.append(row)
            print(dataset, a, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                               if k.startswith(("latency_b1_median", "latency_b128_median", "macs", "params_total"))},
                  flush=True)
    load_after = E.cpu_load(5.0)
    df = pd.DataFrame(rows)
    df["threads"] = args.threads
    df["cpu_load_before_pct"], df["cpu_load_after_pct"] = load_before, load_after
    df["cpu"] = C.environment()["cpu"]
    df["torch"] = torch.__version__
    df["mkldnn"] = torch.backends.mkldnn.is_available()
    df.to_csv(os.path.join(C.AG, "cpu_efficiency_baselines.csv"), index=False)
    pd.DataFrame(samples).to_csv(os.path.join(C.AG, "cpu_efficiency_latency_samples.csv"), index=False)
    print(f"load before {load_before:.1f}%, after {load_after:.1f}%")


def cmd_dense(_args):
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(C.DENSE_RUNS, "*.json")))]
    runs = pd.DataFrame([{
        "dataset": r["dataset"], "arch": r["arch"], "seed": r["seed"], "reference_model": r["reference_model"],
        "lr": r["history"][0]["lr"], "epochs": r["epochs"], "selected_epoch": r["selected_epoch"],
        "val_accuracy": r["val_accuracy"], "val_loss": r["val_loss"], "test_accuracy": r["test_accuracy"],
        "last_epoch_val_accuracy": r["last_epoch_val_accuracy"], "final_train_accuracy": r["final_train_accuracy"],
        "params_total": r["parameters"]["total"], "wall_minutes": r["wall_seconds_total"] / 60.0,
        "threads": r["threads"], "checkpoint": r["checkpoint"], "checkpoint_sha256": r["checkpoint_sha256"],
        "run_id": r["run_id"]} for r in recs]).sort_values(["dataset", "arch", "seed"])
    runs.to_csv(os.path.join(C.AG, "cpu_baseline_runs.csv"), index=False)
    fixed = {"dataset": "cifar10", "arch": "simplecnn", "n_seeds": 1, "reference": "cnn_baseline_FIXED.pth",
             "reference_val_accuracy": 77.32, "reference_test_accuracy": 77.03, "val_mean": 77.32, "val_sd": None,
             "test_mean": 77.03, "test_sd": None, "params_total": 620362,
             "recipe": "historical: Adam 1e-3, 10 epochs, final epoch (not retrained)",
             "reference_sha256": C.FIXED_SHA}
    rows = [fixed]
    for (ds, arch), g in runs.groupby(["dataset", "arch"], sort=False):
        ref = g[g.seed == 0].iloc[0]
        rows.append({"dataset": ds, "arch": arch, "n_seeds": len(g), "reference": ref.checkpoint,
                     "reference_val_accuracy": ref.val_accuracy, "reference_test_accuracy": ref.test_accuracy,
                     "val_mean": g.val_accuracy.mean(), "val_sd": g.val_accuracy.std(ddof=1),
                     "test_mean": g.test_accuracy.mean(), "test_sd": g.test_accuracy.std(ddof=1),
                     "params_total": int(ref.params_total),
                     "recipe": f"Phase-2 protocol: Adam {ref.lr:g} cosine, 30 epochs, best-validation epoch",
                     "reference_sha256": ref.checkpoint_sha256})
    pd.DataFrame(rows).to_csv(os.path.join(C.AG, "cpu_baselines.csv"), index=False)
    print(pd.DataFrame(rows).drop(columns=["reference_sha256", "recipe"]).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("units")
    e = sub.add_parser("efficiency")
    e.add_argument("--threads", type=int, default=4)
    sub.add_parser("dense")
    args = ap.parse_args()
    {"units": cmd_units, "efficiency": cmd_efficiency, "dense": cmd_dense}[args.cmd](args)


if __name__ == "__main__":
    main()
