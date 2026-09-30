"""Phase 5 TEST evaluation: runs only after phase5_final_policies.json is frozen (and committed).

  python experiments/phase5/final_eval.py --worker K --workers 4      # resumable job log
  python experiments/phase5/final_eval.py --merge
Jobs: the dense reference of each setting; every distinct selected policy; LAMP and global magnitude at exactly the
selected policy's zero count (asserted; sparsity mismatch 0). 1 thread. Shared label files are written lock-safely.
"""
import argparse
import io
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import phase5_lib as L
from src import pruning as P
from src.evaluation import accuracy_from, evaluate_accuracy, predictions

PARTS = os.path.join(L.RESULTS, "_final_eval_parts")


def save_npy(arr, path):
    buf = io.BytesIO()
    np.save(buf, arr)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(buf.getvalue())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_shared_once(arr, path, attempts=20):
    for _ in range(attempts):
        if os.path.exists(path):
            try:
                existing = np.load(path)
            except (OSError, ValueError):
                time.sleep(0.5)
                continue
            if not np.array_equal(existing, arr):
                raise RuntimeError(f"shared file differs: {path}")
            return
        try:
            save_npy(arr, path)
            return
        except PermissionError:
            tmp = f"{path}.{os.getpid()}.tmp"
            if os.path.exists(tmp):
                os.remove(tmp)
            time.sleep(0.5)
    raise RuntimeError(f"could not write {path}")


def job_list():
    fr = json.load(open(L.FREEZE, encoding="utf-8"))
    pols = sorted({(r["setting"], tuple(r["selected_policy"])) for r in fr["runs"]})
    jobs = [("dense", s, [0, 0, 0, 0], None) for s in L.SETTINGS]
    jobs += [("policy", s, list(p), None) for s, p in pols]
    jobs += [("baseline", s, list(p), m) for s, p in pols for m in ("lamp", "global")]
    return jobs


def worker(k, n):
    torch.set_num_threads(L.THREADS)
    assert os.path.exists(L.FREEZE), "freeze the final policies first"
    jobs = [j for i, j in enumerate(job_list()) if i % n == k]
    os.makedirs(PARTS, exist_ok=True)
    out_path = os.path.join(PARTS, f"part{k}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["kind"], r["setting"], tuple(r["policy"]), r["method"]))
            except json.JSONDecodeError:
                pass
    bundles, models = {}, {}
    with open(out_path, "a", encoding="utf-8") as out:
        for kind, s, pol, method in jobs:
            if (kind, s, tuple(pol), method) in done:
                continue
            d, a = L.P3.split(s)
            b = bundles.setdefault(d, L.bundle(d))
            if s not in models:
                models[s] = L.P3.load_reference(s)[0]
            pm = P.prune_actions(models[s], a, pol)
            k0 = P.zero_count(pm)
            if kind == "baseline":
                pm = P.BASELINES[method](models[s], a, P.total_sparsity(pm))
                assert P.zero_count(pm) == k0
            b.freeze({"phase5": kind, "policy": pol, "method": method})        # fixed model: TEST unlocks
            preds, labels = predictions(pm, b.test_loader())
            folder = os.path.join(L.PRED, s)
            os.makedirs(folder, exist_ok=True)
            name = {"dense": "dense", "policy": "ppo_", "baseline": f"{method}_"}[kind] + ("" if kind == "dense" else "".join(map(str, pol))) + ".npy"
            save_npy(preds.numpy().astype(np.int16), os.path.join(folder, name))
            write_shared_once(labels.numpy().astype(np.int16), os.path.join(L.PRED, f"{d}_test_labels.npy"))
            row = {"kind": kind, "setting": s, "policy": pol, "method": method, "zero_count": k0,
                   "total_sparsity": P.total_sparsity(pm), "val_rl_accuracy": evaluate_accuracy(pm, b.rl_loader()),
                   "test_accuracy": accuracy_from(preds, labels),
                   "predictions_file": os.path.relpath(os.path.join(folder, name), L.ROOT).replace("\\", "/")}
            out.write(json.dumps(row) + "\n")
            out.flush()
            os.fsync(out.fileno())
    print(f"worker {k}: finished {len(jobs)} jobs", flush=True)


def merge():
    rows = []
    for p in sorted(os.listdir(PARTS)):
        for line in open(os.path.join(PARTS, p), encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    ev = pd.DataFrame(rows)
    ev["policy"] = ev["policy"].map(lambda v: str(list(v)))
    ev = ev.drop_duplicates(["kind", "setting", "policy", "method"])
    assert len(ev) == len(job_list()), f"{len(ev)} of {len(job_list())}"
    ev.to_csv(os.path.join(L.RESULTS, "phase5_test_evaluations.csv"), index=False, float_format="%.17g")
    print(f"{len(ev)} evaluations merged")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge() if a.merge else worker(a.worker, a.workers)
