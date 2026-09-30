"""Exact 1,296-policy landscapes on VAL-RL and VAL-SELECT (1 thread per worker). Never TEST.

Accuracy is computed with src.evaluation.evaluate_accuracy on the split's DataLoader (batch 128, fixed order):
the same code path as the PPO environment (VAL-RL) and the archive selection (VAL-SELECT).
  python experiments/phase5/landscape.py --worker K --workers 4        # K = 0..3, resumable per (setting, worker)
  python experiments/phase5/landscape.py --merge
"""
import argparse
import json
import os
import time

import torch

import phase5_lib as L
from src import pruning as P
from src.evaluation import evaluate_accuracy

PARTS = os.path.join(L.LAND, "_parts")


def worker(k, n):
    torch.set_num_threads(L.THREADS)
    os.makedirs(PARTS, exist_ok=True)
    bundles = {}
    for s in L.SETTINGS:
        path = os.path.join(PARTS, f"{s}_part{k}.json")
        if os.path.exists(path):
            continue
        d, a = L.P3.split(s)
        b = bundles.setdefault(d, L.bundle(d))
        model, _ = L.P3.load_reference(s)
        rl, sel = b.rl_loader(), b.select_loader()
        t0, rows = time.time(), []
        for i, acts in enumerate(L.POLICIES):
            if i % n != k:
                continue
            pm = P.prune_actions(model, a, acts)
            rows.append({"policy_index": i, "actions": list(acts), "val_rl_accuracy": evaluate_accuracy(pm, rl),
                         "val_select_accuracy": evaluate_accuracy(pm, sel), "total_sparsity": P.total_sparsity(pm),
                         "unit_sparsity": list(P.unit_sparsity(pm, a).values())})
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rows, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        print(f"worker {k}: {s} {len(rows)} policies in {time.time() - t0:.0f}s", flush=True)


def merge():
    for s in L.SETTINGS:
        parts = [json.load(open(os.path.join(PARTS, f), encoding="utf-8"))
                 for f in sorted(os.listdir(PARTS)) if f.startswith(s + "_part") and f.endswith(".json")]
        rows = sorted((r for p in parts for r in p), key=lambda r: r["policy_index"])
        assert [r["policy_index"] for r in rows] == list(range(1296)), s
        with open(os.path.join(L.LAND, f"{s}_landscape.json"), "w", encoding="utf-8") as f:
            json.dump({"setting": s, "threads": L.THREADS, "splits": "VAL-RL 2,500 / VAL-SELECT 2,500", "policies": rows}, f)
        print(s, "merged")
    for fn in os.listdir(PARTS):
        os.remove(os.path.join(PARTS, fn))
    os.rmdir(PARTS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    merge() if args.merge else worker(args.worker, args.workers)
