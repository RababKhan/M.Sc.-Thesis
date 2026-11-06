"""Exact 1,296-policy landscape per setting on V_RL (1 thread per worker). Never on TEST.

  python experiments/phase3/landscape.py --worker K --workers 4     # K = 0..3, resumable per (setting, worker)
  python experiments/phase3/landscape.py --merge

Accuracy is computed exactly as the PPO environment computes it (batches of 128 in V_RL order,
torch.max over the logits, 100 * correct / total); the loss is the mean cross-entropy of the same
logits. The merged file doubles as the environment's exact evaluation cache.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import phase3_common as C
from src import data as D, pruning as P, rl as RL
from src.utils import sha256_file

PARTS = os.path.join(C.LAND_DIR, "_parts")


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
    C.set_threads()
    os.makedirs(PARTS, exist_ok=True)
    bundles = {}
    for setting in C.SETTINGS:
        path = os.path.join(PARTS, f"{setting}_part{k}.json")
        if os.path.exists(path):
            continue
        d, a = C.split(setting)
        b = bundles.setdefault(d, D.DataBundle(d, train_images=False))
        model, _ = C.load_reference(setting)
        x, y = b.rl_tensors()
        t0, rows = time.time(), []
        for i, actions in enumerate(C.POLICIES):
            if i % n != k:
                continue
            pm = P.prune_actions(model, a, actions)
            acc, loss = evaluate(pm, x, y)
            rows.append({"policy_index": i, "actions": list(actions), "val_accuracy": acc, "val_loss": loss,
                         "total_sparsity": P.total_sparsity(pm), "unit_sparsity": list(P.unit_sparsity(pm, a).values())})
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rows, f)
        os.replace(tmp, path)
        print(f"worker {k}: {setting} {len(rows)} policies in {time.time() - t0:.0f}s", flush=True)


def merge():
    inputs = C.frozen_inputs()["settings"]
    frames = []
    for setting in C.SETTINGS:
        parts = [json.load(open(os.path.join(PARTS, f), encoding="utf-8"))
                 for f in sorted(os.listdir(PARTS)) if f.startswith(setting + "_part") and f.endswith(".json")]
        rows = sorted((r for p in parts for r in p), key=lambda r: r["policy_index"])
        assert [r["policy_index"] for r in rows] == list(range(1296)), setting
        base = inputs[setting]["base_vrl_accuracy"]
        units = [u["name"] for u in inputs[setting]["units"]]
        most = inputs[setting]["destructive_rule"]["unit_index"]
        for r in rows:
            parts_ = RL.reward_components(r["val_accuracy"], base, r["total_sparsity"], r["actions"], C.LAMBDA_S)
            r.update({"reward": parts_["total"], "reward_accuracy_term": parts_["accuracy_term"],
                      "reward_sparsity_term": parts_["sparsity_term"], "reward_diversity_term": parts_["diversity_term"],
                      "utility": 1.5 * r["val_accuracy"] / base + C.LAMBDA_S * r["total_sparsity"],
                      "destructive": r["actions"][most] >= C.DESTRUCTIVE_ACTION})
        f = pd.DataFrame(rows)
        f["reward_rank"] = f["reward"].rank(ascending=False, method="min").astype(int)
        f["rank_score"] = (1296 - f["reward_rank"]) / 1295
        f["pareto_val_accuracy"] = C.pareto(f["total_sparsity"].to_numpy(), f["val_accuracy"].to_numpy())
        f["pareto_val_loss"] = C.pareto(f["total_sparsity"].to_numpy(), -f["val_loss"].to_numpy())
        for i, u in enumerate(units):
            f[f"ratio_{i}_{u}"] = f["actions"].map(lambda acts: int(100 * P.ACTION_TO_PRUNE[acts[i]]))
        records = f.to_dict(orient="records")
        for r in records:
            r["pareto_val_accuracy"] = bool(r["pareto_val_accuracy"])
            r["pareto_val_loss"] = bool(r["pareto_val_loss"])
            r["reward_rank"] = int(r["reward_rank"])
            r["destructive"] = bool(r["destructive"])
        with open(C.landscape_json(setting), "w", encoding="utf-8") as fh:
            json.dump({"setting": setting, "units": units, "base_vrl_accuracy": base, "lambda_s": C.LAMBDA_S,
                       "threads": C.THREADS, "data": "V_RL (3,000 images)", "policies": records}, fh)
        csv = f.copy()
        csv.insert(0, "setting", setting)
        csv["actions"] = csv["actions"].map(lambda v: str(list(v)))
        csv["unit_sparsity"] = csv["unit_sparsity"].map(lambda v: str([float(x) for x in v]))
        csv.to_csv(os.path.join(C.LAND_DIR, f"{setting}.csv"), index=False, float_format="%.17g")
        frames.append(csv.rename(columns={c: f"ratio_unit{c.split('_')[1]}" for c in csv.columns if c.startswith("ratio_")}))
        print(f"{setting}: 1296 policies; best reward rank-1 {f.loc[f.reward_rank.idxmin(), 'actions']}; "
              f"{int(f.pareto_val_accuracy.sum())} Pareto-efficient; json sha {sha256_file(C.landscape_json(setting))[:12]}")
    pd.concat(frames).to_csv(os.path.join(C.AG, "phase3_policy_landscapes.csv"), index=False, float_format="%.17g")
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
