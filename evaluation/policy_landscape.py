"""Full validation landscape of all 6^4 = 1,296 pruning policies.

For every policy: per-layer ratios, total sparsity, accuracy on the 1,000-image
RL probe (the reward data), accuracy and cross-entropy loss on the full
5,000-image validation split, the training reward and its three terms for
lambda_s in {0.01, 0.02} (and 0, 0.04, 0.08 for reference), reward ranks, and
validation Pareto status (sparsity up, validation accuracy up). The test set
is never loaded.

The probe accuracy is recomputed here and checked against
results/reward_landscape.csv, which scored the same policies earlier.

Run from the repository root:
  python evaluation/policy_landscape.py --worker K --workers 4   (K = 0..3)
  python evaluation/policy_landscape.py --merge
Output: results/policy_landscape_validation.csv
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import rl_env  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PARTS = os.path.join(RESULTS, "_landscape_parts")
OUT = os.path.join(RESULTS, "policy_landscape_validation.csv")
COEFFICIENTS = [0.00, 0.01, 0.02, 0.04, 0.08]
BASELINE_VAL = 77.32
POLICIES = [list(p) for p in itertools.product(range(len(rl_env.ACTION_TO_PRUNE)), repeat=4)]


def evaluate(model, x, y):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    correct, loss = [], 0.0
    with torch.no_grad():
        for i in range(0, len(x), 128):
            out = model(x[i:i + 128])
            correct.append((out.argmax(1) == y[i:i + 128]))
            loss += loss_fn(out, y[i:i + 128]).item()
    correct = torch.cat(correct)
    return correct, loss / len(x)


def worker(k, n):
    torch.set_num_threads(1)
    model = common.load_baseline(os.path.join(ROOT, "checkpoints", "cnn_baseline_FIXED.pth"))
    splits = rl_env.Splits(os.path.join(ROOT, "data"))
    x, y = splits._val
    rows = []
    for i, actions in enumerate(POLICIES):
        if i % n != k:
            continue
        pruned = common.prune_layerwise(model, [rl_env.ACTION_TO_PRUNE[a] for a in actions])
        correct, loss = evaluate(pruned, x, y)
        probe = 100 * correct[:rl_env.RL_PROBE_SIZE].sum().item() / rl_env.RL_PROBE_SIZE
        rows.append({"policy_index": i, "actions": str(actions),
                     **{f"ratio_{n_}": 100 * rl_env.ACTION_TO_PRUNE[a] for n_, a in zip(rl_env.FILTERED_LAYERS, actions)},
                     "total_sparsity": common.calculate_sparsity(pruned),
                     "probe_accuracy": probe,
                     "val_accuracy": 100 * correct.sum().item() / len(y),
                     "val_loss": loss})
    os.makedirs(PARTS, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(PARTS, f"part{k}.csv"), index=False)
    print(f"worker {k}: {len(rows)} policies")


def pareto(sparsity, accuracy):
    """Non-dominated in (sparsity up, accuracy up)."""
    order = np.lexsort((-accuracy, -sparsity))       # sparsity desc, then accuracy desc
    efficient = np.zeros(len(sparsity), bool)
    best = -np.inf
    for i in order:
        if accuracy[i] > best + 1e-12:
            efficient[i] = True
            best = accuracy[i]
    return efficient


def merge():
    parts = [pd.read_csv(os.path.join(PARTS, f)) for f in sorted(os.listdir(PARTS))]
    frame = pd.concat(parts).sort_values("policy_index").reset_index(drop=True)
    assert len(frame) == len(POLICIES) and frame["policy_index"].tolist() == list(range(len(POLICIES)))

    earlier = pd.read_csv(os.path.join(RESULTS, "reward_landscape.csv")).set_index("actions")
    mismatch = (frame.set_index("actions")["probe_accuracy"] - earlier["probe_accuracy"]).abs().max()
    assert mismatch < 1e-9, f"probe accuracy disagrees with reward_landscape.csv by {mismatch}"
    sp_mismatch = (frame.set_index("actions")["total_sparsity"] - earlier["total_sparsity"]).abs().max()
    assert sp_mismatch < 1e-9

    for coef in COEFFICIENTS:
        parts_ = [rl_env.reward_components(p, BASELINE_VAL, s, eval(a), coef)
                  for p, s, a in zip(frame["probe_accuracy"], frame["total_sparsity"], frame["actions"])]
        frame[f"reward_{coef:.2f}"] = [c["total"] for c in parts_]
        if coef in (0.01, 0.02):
            frame[f"reward_{coef:.2f}_accuracy_term"] = [c["accuracy_term"] for c in parts_]
            frame[f"reward_{coef:.2f}_sparsity_term"] = [c["sparsity_term"] for c in parts_]
            frame[f"reward_{coef:.2f}_diversity_term"] = [c["diversity_term"] for c in parts_]
        frame[f"reward_rank_{coef:.2f}"] = frame[f"reward_{coef:.2f}"].rank(ascending=False, method="min").astype(int)
    frame["pareto_val_accuracy"] = pareto(frame["total_sparsity"].to_numpy(), frame["val_accuracy"].to_numpy())
    frame["pareto_val_loss"] = pareto(frame["total_sparsity"].to_numpy(), -frame["val_loss"].to_numpy())
    frame.to_csv(OUT, index=False)
    for f in os.listdir(PARTS):
        os.remove(os.path.join(PARTS, f))
    os.rmdir(PARTS)
    print(f"{len(frame)} policies; probe accuracy matches reward_landscape.csv; "
          f"{int(frame['pareto_val_accuracy'].sum())} Pareto-efficient (val accuracy)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    merge() if args.merge else worker(args.worker, args.workers)
