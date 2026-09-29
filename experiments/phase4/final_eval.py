"""Phase 4 final evaluation: the only step that reads TEST (after all stage selections are frozen).

  python experiments/phase4/final_eval.py --freeze                      # write + hash the final-policy list
  python experiments/phase4/final_eval.py --worker K --workers 4        # K = 0..3, resumable
  python experiments/phase4/final_eval.py --merge

Frozen list: for every Phase-4 run, its final policy and its archive policy (validation-only rules),
with the pipeline tags F (final pipeline) and K (control R0/P0/S0). Jobs: TEST predictions of every
distinct (setting, policy); LAMP and global magnitude at every distinct zero count; uniform, ERK and random
(seeds 1000-1009) at the zero counts of the F and K final-selected policies. Every baseline is built at
exactly the policy's zero count (asserted).
"""
import argparse
import io
import json
import os

import numpy as np
import pandas as pd
import torch

import phase4_lib as L
import phase4_metrics as M
from src import data as D, pruning as P
from src.evaluation import accuracy_from, evaluate_accuracy, predictions
from src.utils import sha256_file

FREEZE = os.path.join(L.RESULTS, "phase4_final_policies.json")
PARTS = os.path.join(L.RESULTS, "_final_eval_parts")
RANDOM_SEEDS = list(range(1000, 1010))
RULE = {"S0": ("S0", "final"), "S1": ("S0", "archive"), "S2": ("S2", "final"), "S2A": ("S2", "archive")}


def freeze():
    if os.path.exists(FREEZE):
        raise SystemExit("final-policy list already frozen")
    sel = json.load(open(L.SELECTION, encoding="utf-8"))
    assert all(k in sel for k in "ABC"), "all three stage selections must be frozen first"
    cfg = json.load(open(L.CONFIG, encoding="utf-8"))
    F = (sel["A"]["winner"], sel["B"]["winner"], *RULE[sel["C"]["winner"]])
    rows = []
    for r in M.load_runs(cfg):
        m = M.run_metrics(r, cfg)
        tags = []
        if (r["reward"], r["prior"], r["variant"]) == ("R0", "P0", "S0"):
            tags.append("K")
        if (r["reward"], r["prior"], r["variant"]) == F[:3]:
            tags.append(f"F:{F[3]}")
        rows.append({"run_id": r["run_id"], "setting": r["setting"], "reward": r["reward"], "prior": r["prior"],
                     "variant": r["variant"], "seed": r["seed"], "final_policy": json.loads(m["final_policy"]),
                     "archive_policy": json.loads(m["archive_policy"]), "tags": tags})
    out = {"selection_sha256": sha256_file(L.SELECTION), "final_pipeline": {"reward": F[0], "prior": F[1], "variant": F[2],
           "selection_rule": F[3], "search_condition": sel["C"]["winner"]}, "control_pipeline": ["R0", "P0", "S0", "final"],
           "n_runs": len(rows), "runs": rows}
    with open(FREEZE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"frozen {len(rows)} runs; F = {out['final_pipeline']}; sha256 {sha256_file(FREEZE)}")


def job_list():
    fr = json.load(open(FREEZE, encoding="utf-8"))
    pols, full = set(), set()
    for r in fr["runs"]:
        pols.add((r["setting"], tuple(r["final_policy"])))
        pols.add((r["setting"], tuple(r["archive_policy"])))
        for t in r["tags"]:
            rule = "final" if t == "K" else t.split(":")[1]
            full.add((r["setting"], tuple(r[f"{rule}_policy"])))
    jobs = [("policy", s, list(p), None, None) for s, p in sorted(pols)]
    jobs += [("baseline", s, list(p), m, None) for s, p in sorted(pols) for m in ("lamp", "global")]
    jobs += [("baseline", s, list(p), m, None) for s, p in sorted(full) for m in ("uniform", "erk")]
    jobs += [("baseline", s, list(p), "random", rs) for s, p in sorted(full) for rs in RANDOM_SEEDS]
    return jobs


def save_npy(arr, path):
    buf = io.BytesIO()
    np.save(buf, arr)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(buf.getvalue())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def worker(k, n):
    torch.set_num_threads(L.THREADS)
    assert os.path.exists(FREEZE), "freeze first"
    jobs = [j for i, j in enumerate(job_list()) if i % n == k]
    os.makedirs(PARTS, exist_ok=True)
    out_path = os.path.join(PARTS, f"part{k}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["kind"], r["setting"], tuple(r["policy"]), r["method"], r["random_seed"]))
            except json.JSONDecodeError:
                pass
    bundles, models = {}, {}
    with open(out_path, "a", encoding="utf-8") as out:
        for kind, s, pol, method, rs in jobs:
            if (kind, s, tuple(pol), method, rs) in done:
                continue
            d, a = L.P3.split(s)
            b = bundles.setdefault(d, D.DataBundle(d, train_images=False))
            if s not in models:
                models[s] = L.P3.load_reference(s)[0]
            model = models[s]
            pm = P.prune_actions(model, a, pol)
            k0 = P.zero_count(pm)
            if kind == "baseline":
                fn = P.BASELINES[method]
                pm = fn(model, a, P.total_sparsity(pm), rs) if method == "random" else fn(model, a, P.total_sparsity(pm))
                assert P.zero_count(pm) == k0, f"{s} {method}: zero count {P.zero_count(pm)} != {k0}"
            vrl = evaluate_accuracy(pm, b.rl_loader())
            b.freeze({"phase4": kind, "policy": pol, "method": method})          # fixed model: TEST unlocks
            preds, labels = predictions(pm, b.test_loader())
            folder = os.path.join(L.PRED, s)
            os.makedirs(folder, exist_ok=True)
            name = ("ppo_" if kind == "policy" else f"{method}_") + "".join(map(str, pol)) + (f"_seed{rs}" if rs is not None else "") + ".npy"
            save_npy(preds.numpy().astype(np.int16), os.path.join(folder, name))
            lab = os.path.join(L.PRED, f"{d}_test_labels.npy")
            if not os.path.exists(lab):
                save_npy(labels.numpy().astype(np.int16), lab)
            row = {"kind": kind, "setting": s, "policy": pol, "method": method, "random_seed": rs, "zero_count": k0,
                   "total_sparsity": P.total_sparsity(pm), "vrl_accuracy": vrl, "test_accuracy": accuracy_from(preds, labels),
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
    ev = ev.drop_duplicates(["kind", "setting", "policy", "method", "random_seed"])
    n = len(job_list())
    assert len(ev) == n, f"{len(ev)} of {n} jobs"
    ev.to_csv(os.path.join(L.RESULTS, "phase4_test_and_baseline_evaluations.csv"), index=False, float_format="%.17g")
    print(f"{len(ev)} evaluations merged")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--worker", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    freeze() if args.freeze else merge() if args.merge else worker(args.worker, args.workers)
