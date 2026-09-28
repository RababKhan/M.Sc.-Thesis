"""Dense baseline training (results/cpu_dense_training_protocol.md). Resumable; one record per run.

Refuses to start a (dataset, arch) whose one-epoch benchmark decision is missing
from cpu_epoch_benchmark.csv, or whose projected run time exceeds the budget.

  python experiments/phase2/train_dense.py --dataset cifar10 --arch lenet5 --seeds 0 1 2 --threads 4
"""
import argparse
import datetime
import json
import os

import pandas as pd

import phase2_common as C
from src import data as D
from src.evaluation.training import run_dense_training
from src.utils import atomic_json_dump


def lr_decision(dataset, arch):
    if not os.path.exists(C.BENCH_CSV):
        raise SystemExit("no benchmark file: run benchmark_epoch.py first (budget rule R2)")
    b = pd.read_csv(C.BENCH_CSV)
    rows = b[(b.dataset == dataset) & (b.arch == arch)]
    if rows.empty:
        raise SystemExit(f"no benchmark for {dataset}/{arch} (budget rule R2)")
    row = rows.iloc[-1]
    if not bool(row.within_budget_2h):
        raise SystemExit(f"{dataset}/{arch} projected {row.projected_run_h} h > budget (rule R3)")
    return float(row.lr_decision)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(D.DATASETS))
    ap.add_argument("--arch", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    lr = lr_decision(args.dataset, args.arch)
    proto = C.protocol(lr)
    bundle = D.DataBundle(args.dataset)
    for d in (C.DENSE_RUNS, C.WORK, os.path.join(C.DENSE_CKPT, args.dataset)):
        os.makedirs(d, exist_ok=True)
    for seed in args.seeds:
        rid = f"{args.dataset}_{args.arch}_seed{seed}"
        rec_path = os.path.join(C.DENSE_RUNS, f"{rid}.json")
        if os.path.exists(rec_path):
            print(f"{rid}: done, skipping", flush=True)
            continue
        print(f"{rid}: lr {lr}, {args.threads} threads", flush=True)
        started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        work = os.path.join(C.WORK, f"{rid}.state.pt")
        final = os.path.join(C.DENSE_CKPT, args.dataset, f"{args.arch}_seed{seed}.pth")
        rec = run_dense_training(args.arch, bundle, seed, proto, work, final, args.threads,
                                 log=lambda s: print(s, flush=True))
        rec["checkpoint"] = os.path.relpath(final, C.ROOT).replace("\\", "/")
        rec.update({"run_id": rid, "reference_model": seed == 0, "started_at": started,
                    "finished_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                    "val_indices_sha256": __import__("src.utils", fromlist=["sha256_json"]).sha256_json(
                        bundle.val_indices),
                    **C.provenance(args.dataset)})
        atomic_json_dump(rec, rec_path)
        os.remove(work)
        print(f"{rid}: selected epoch {rec['selected_epoch']}  val {rec['val_accuracy']:.2f}%  "
              f"test {rec['test_accuracy']:.2f}%  ({rec['wall_seconds_total'] / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
