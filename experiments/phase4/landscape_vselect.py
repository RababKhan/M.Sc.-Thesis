"""Exact 1,296-policy V_SELECT landscape per setting (1 thread per worker). Never TEST.

V_SELECT is held-out validation: in Phase 4 it is used only for the pre-registered archive selection
(S1, S2A), the held-out ranking of final policies and the Stage-C selection rule; never for rewards or
training.
  python experiments/phase4/landscape_vselect.py --worker K --workers 4      # K = 0..3, resumable
  python experiments/phase4/landscape_vselect.py --merge
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn

import phase4_lib as L
from src import data as D, pruning as P

PARTS = os.path.join(L.LAND, "_vselect_parts")


def evaluate(model, x, y):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    correct, loss = 0, 0.0
    with torch.no_grad():
        for i in range(0, len(x), D.EVAL_BATCH):
            out = model(x[i:i + D.EVAL_BATCH])
            _, pred = torch.max(out, 1)
            correct += (pred == y[i:i + D.EVAL_BATCH]).sum().item()
            loss += loss_fn(out, y[i:i + D.EVAL_BATCH]).item()
    return 100 * correct / len(y), loss / len(y)


def worker(k, n):
    torch.set_num_threads(L.THREADS)
    os.makedirs(PARTS, exist_ok=True)
    bundles = {}
    for setting in L.SETTINGS:
        path = os.path.join(PARTS, f"{setting}_part{k}.json")
        if os.path.exists(path):
            continue
        d, a = L.P3.split(setting)
        b = bundles.setdefault(d, D.DataBundle(d, train_images=False))
        model, _ = L.P3.load_reference(setting)
        x, y = b.select_tensors()
        t0, rows = time.time(), []
        for i, actions in enumerate(L.P3.POLICIES):
            if i % n != k:
                continue
            acc, loss = evaluate(P.prune_actions(model, a, actions), x, y)
            rows.append({"policy_index": i, "actions": list(actions), "vselect_accuracy": acc, "vselect_loss": loss})
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rows, f)
        os.replace(tmp, path)
        print(f"worker {k}: {setting} {len(rows)} policies in {time.time() - t0:.0f}s", flush=True)


def merge():
    for setting in L.SETTINGS:
        parts = [json.load(open(os.path.join(PARTS, f), encoding="utf-8"))
                 for f in sorted(os.listdir(PARTS)) if f.startswith(setting + "_part")]
        rows = sorted((r for p in parts for r in p), key=lambda r: r["policy_index"])
        assert [r["policy_index"] for r in rows] == list(range(1296)), setting
        with open(os.path.join(L.LAND, f"{setting}_vselect.json"), "w", encoding="utf-8") as f:
            json.dump({"setting": setting, "data": "V_SELECT (2,000 images)", "threads": L.THREADS, "policies": rows}, f)
        print(setting, "merged")
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
