"""One-epoch CPU benchmark (results/cpu_runtime_budget.md, rules R1-R3).

For each architecture: seed-0 model, one full training epoch under the dense
protocol, one validation pass (loss + accuracy passes, as in training), one
3,000-image V_RL evaluation, and the cost of a memoised PPO episode (reset +
four unit-pruning steps). Records time and TRAINING loss only; no accuracy of
the benchmark model is computed or stored. Weights are discarded.

  python experiments/phase2/benchmark_epoch.py --dataset cifar10 --archs simplecnn lenet5 resnet8
"""
import argparse
import csv
import datetime
import math
import os
import time

import psutil
import torch

import phase2_common as C
from src import data as D, models as M, rl as RL
from src.evaluation import evaluate_accuracy, mean_loss_accuracy
from src.evaluation.training import (DATA_GENERATOR_OFFSET, make_optimizer, make_scheduler,
                                     train_one_epoch)

FIELDS = ["dataset", "arch", "num_classes", "parameters", "threads", "t_epoch_s", "t_val_s", "t_vrl3000_s",
          "t_episode_memo_s", "projected_run_h", "within_budget_2h", "mean_train_loss", "ln_num_classes",
          "nonfinite_loss", "stability_rule_triggered", "lr_decision", "peak_wset_mb", "measured_at"]


def benchmark(arch, bundle, threads):
    torch.set_num_threads(threads)
    model = M.build_model(arch, bundle.num_classes, seed=0)
    proto = C.protocol()
    opt = make_optimizer(model, proto)
    make_scheduler(opt, proto)
    gen = torch.Generator().manual_seed(0 + DATA_GENERATOR_OFFSET)
    with bundle.training():
        t0 = time.time()
        stats = train_one_epoch(model, bundle, opt, gen, proto["batch_size"])
        t_epoch = time.time() - t0
        vx, vy = bundle.val_tensors()
        t0 = time.time()
        mean_loss_accuracy(model, vx, vy)
        evaluate_accuracy(model, D.eval_loader(vx, vy))
        t_val = time.time() - t0
        rl_loader = bundle.rl_loader()
        t0 = time.time()
        evaluate_accuracy(model, rl_loader)
        t_rl = time.time() - t0
        # memoised PPO episode: every terminal evaluation is a cache hit
        env = RL.PruningEnv(model.eval(), arch, 50.0, rl_loader, cache={})
        acts = [[a, (a + 1) % 6, (a + 2) % 6, (a + 3) % 6] for a in range(6)]
        for a in acts:                                  # fill the cache
            env.reset()
            for x in a:
                env.step(x)
        t0 = time.time()
        for _ in range(5):
            for a in acts:
                env.reset()
                for x in a:
                    env.step(x)
        t_episode = (time.time() - t0) / (5 * len(acts))
    ln_c = math.log(bundle.num_classes)
    triggered = bool(stats["nonfinite"] or not stats["train_loss"] < ln_c)
    projected = C.EPOCHS * (t_epoch + t_val) / 3600.0
    return {"dataset": bundle.dataset, "arch": arch, "num_classes": bundle.num_classes,
            "parameters": M.count_parameters(model)["total"], "threads": threads,
            "t_epoch_s": round(t_epoch, 2), "t_val_s": round(t_val, 2), "t_vrl3000_s": round(t_rl, 3),
            "t_episode_memo_s": round(t_episode, 4), "projected_run_h": round(projected, 3),
            "within_budget_2h": projected <= C.BUDGET_HOURS, "mean_train_loss": round(stats["train_loss"], 4),
            "ln_num_classes": round(ln_c, 4), "nonfinite_loss": stats["nonfinite"],
            "stability_rule_triggered": triggered, "lr_decision": C.FALLBACK_LR if triggered else C.STABLE_LR,
            "peak_wset_mb": round(getattr(psutil.Process().memory_info(), "peak_wset", 0) / 2 ** 20, 1),
            "measured_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(D.DATASETS))
    ap.add_argument("--archs", nargs="+", required=True, choices=list(M.ARCHITECTURES))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    bundle = D.DataBundle(args.dataset)
    os.makedirs(C.AG, exist_ok=True)
    new = not os.path.exists(C.BENCH_CSV)
    for arch in args.archs:
        row = benchmark(arch, bundle, args.threads)
        with open(C.BENCH_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
                new = False
            w.writerow(row)
        print(row, flush=True)


if __name__ == "__main__":
    main()
