"""Matched one-shot baselines for Phase 3 (run only after all 480 PPO runs are complete).

  python experiments/phase3/baselines.py --worker K --workers 4     # K = 0..3, resumable per job
  python experiments/phase3/baselines.py --merge

Jobs: for every setting, every distinct zero count K among the 480 final PPO models (and every grid
target), every method (uniform, global, LAMP, ERK; random with seeds 1000-1009). Each baseline model
is built at exactly K zeros (asserted), evaluated once on V_RL and once on TEST (1 thread; test after
the model is fixed), predictions cached. merge -> results/stronger_baselines/phase3_all_runs.csv
(one row per PPO run x method x random seed) and phase3_grid.csv.
"""
import argparse
import glob
import io
import json
import os

import numpy as np
import pandas as pd
import torch

import phase3_common as C
from src import data as D, pruning as P
from src.evaluation import accuracy_from, evaluate_accuracy, predictions

PARTS = os.path.join(C.SB, "_phase3_parts")
METHODS = ["uniform", "global", "lamp", "erk", "random"]


def records():
    recs = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(C.RUNS_DIR, "*.json")))]
    if len(recs) != 480 or any(r["status"] != "complete" for r in recs):
        raise SystemExit(f"{len(recs)} complete PPO records; baselines run only after all 480 runs")
    return recs


def job_list(recs):
    jobs = []
    for setting in C.SETTINGS:
        ks = sorted({r["zero_count"] for r in recs if r["setting"] == setting})
        for kind, values in (("matched", ks), ("grid", C.GRID)):
            for v in values:
                for m in METHODS:
                    for rs in (C.RANDOM_SEEDS if m == "random" else [None]):
                        jobs.append((setting, kind, v, m, rs))
    return jobs


def build(model, arch, method, sparsity, rseed):
    fn = P.BASELINES[method]
    return fn(model, arch, sparsity, rseed) if method == "random" else fn(model, arch, sparsity)


def save_npy(arr, path):
    buf = io.BytesIO()
    np.save(buf, arr)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(buf.getvalue())
    os.replace(tmp, path)


def worker(k, n):
    C.set_threads()
    recs = records()
    jobs = [j for i, j in enumerate(job_list(recs)) if i % n == k]
    os.makedirs(PARTS, exist_ok=True)
    out_path = os.path.join(PARTS, f"part{k}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["setting"], r["kind"], r["target"], r["method"], r["random_seed"]))
            except json.JSONDecodeError:
                pass                                            # a torn last line after a power cut
    bundles, models = {}, {}
    with open(out_path, "a", encoding="utf-8") as out:
        for setting, kind, v, method, rs in jobs:
            if (setting, kind, v, method, rs) in done:
                continue
            d, a = C.split(setting)
            b = bundles.setdefault(d, D.DataBundle(d, train_images=False))
            if setting not in models:
                models[setting] = C.load_reference(setting)[0]
            model = models[setting]
            w = P.weight_total(model)
            sparsity = 100.0 * v / w if kind == "matched" else v
            pm = build(model, a, method, sparsity, rs)
            k_got = P.zero_count(pm)
            if kind == "matched":
                assert k_got == v, f"{setting} {method}: {k_got} zeros != {v}"
            vrl = evaluate_accuracy(pm, b.rl_loader())
            b.freeze({"baseline": method, "zeros": k_got, "random_seed": rs})   # fixed model: test unlocks
            preds, labels = predictions(pm, b.test_loader())
            folder = os.path.join(C.PRED_DIR, setting)
            os.makedirs(folder, exist_ok=True)
            name = f"{method}_K{k_got}" + (f"_seed{rs}" if rs is not None else "") + ".npy"
            save_npy(preds.numpy().astype(np.int16), os.path.join(folder, name))
            row = {"setting": setting, "kind": kind, "target": v, "method": method, "random_seed": rs,
                   "zero_count": k_got, "total_sparsity": P.total_sparsity(pm), "vrl_accuracy": vrl,
                   "test_accuracy": accuracy_from(preds, labels), "predictions_file": f"phase3_predictions/{setting}/{name}"}
            out.write(json.dumps(row) + "\n")
            out.flush()
            os.fsync(out.fileno())
    print(f"worker {k}: finished {len(jobs)} jobs", flush=True)


def merge():
    recs = records()
    rows = []
    for p in sorted(glob.glob(os.path.join(PARTS, "part*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    ev = pd.DataFrame(rows).drop_duplicates(["setting", "kind", "target", "method", "random_seed"])
    n_jobs = len(job_list(recs))
    assert len(ev) == n_jobs, f"{len(ev)} of {n_jobs} baseline jobs"
    matched = ev[ev.kind == "matched"]
    per_run = []
    for r in recs:
        m = matched[(matched.setting == r["setting"]) & (matched.target == r["zero_count"])]
        for _, b in m.iterrows():
            per_run.append({"run_id": r["run_id"], "setting": r["setting"], "dataset": r["dataset"], "arch": r["arch"],
                            "condition": r["condition"], "seed": r["seed"], "zero_count": r["zero_count"],
                            "total_sparsity": r["total_sparsity"], "ppo_test_accuracy": r["test_accuracy"],
                            "ppo_vrl_accuracy": r["val_accuracy_vrl"], "method": b.method,
                            "random_seed": None if pd.isna(b.random_seed) else int(b.random_seed),
                            "baseline_sparsity": b.total_sparsity, "baseline_test_accuracy": b.test_accuracy,
                            "baseline_vrl_accuracy": b.vrl_accuracy,
                            "ppo_minus_baseline_test": r["test_accuracy"] - b.test_accuracy,
                            "predictions_file": b.predictions_file})
    per_run = pd.DataFrame(per_run)
    assert (per_run.total_sparsity == per_run.baseline_sparsity).all(), "unmatched sparsity"
    assert len(per_run) == 480 * 14
    os.makedirs(C.SB, exist_ok=True)
    per_run.to_csv(os.path.join(C.SB, "phase3_all_runs.csv"), index=False, float_format="%.17g")
    ev[ev.kind == "grid"].drop(columns=["kind"]).rename(columns={"target": "target_sparsity"}).to_csv(
        os.path.join(C.SB, "phase3_grid.csv"), index=False, float_format="%.17g")
    ev.to_csv(os.path.join(C.SB, "phase3_baseline_evaluations.csv"), index=False, float_format="%.17g")
    print(f"{len(per_run)} matched rows, {int((ev.kind == 'grid').sum())} grid evaluations")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    merge() if args.merge else worker(args.worker, args.workers)
