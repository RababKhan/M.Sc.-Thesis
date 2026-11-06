"""Exhaustive reward landscape: every one of the 6^4 = 1,296 pruning policies.

Scores each policy exactly as the PPO training reward does -- accuracy on the
1,000-image validation probe, divided by the baseline's full-validation
accuracy (77.32), plus lambda_s * sparsity(%) and 0.05 * distinct actions --
for lambda_s in {0, 0.01, 0.02, 0.04, 0.08}. Validation data only; the test
set is not loaded.

This shows what the reward itself prefers, independent of what PPO learned,
so learned policies can be placed against the reward-optimal ones.

Outputs
  results/reward_landscape.csv            one row per policy
  results/learned_policy_reward_rank.csv  (--rank-learned) every learned policy,
                                          original and refined, placed in it
Run from the repository root:
  python evaluation/reward_landscape.py
  python evaluation/reward_landscape.py --rank-learned
"""
import glob
import itertools
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import rl_env  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COEFFICIENTS = [0.00, 0.01, 0.02, 0.04, 0.08]
BASELINE_VAL = 77.32


def main():
    model = common.load_baseline(os.path.join(ROOT, "checkpoints", "cnn_baseline_FIXED.pth"))
    splits = rl_env.Splits(os.path.join(ROOT, "data"))
    probe = splits.probe_loader()
    assert round(rl_env.evaluate_model(model, splits.val_loader()), 2) == BASELINE_VAL

    rows = []
    for actions in itertools.product(range(len(rl_env.ACTION_TO_PRUNE)), repeat=4):
        actions = list(actions)
        pruned = common.prune_layerwise(model, [rl_env.ACTION_TO_PRUNE[a] for a in actions])
        acc = rl_env.evaluate_model(pruned, probe)
        sparsity = common.calculate_sparsity(pruned)
        row = {"actions": str(actions), "prune_percent": str([100 * rl_env.ACTION_TO_PRUNE[a] for a in actions]),
               "probe_accuracy": acc, "total_sparsity": sparsity, "distinct_actions": len(set(actions)),
               "features3_pruned_60": actions[1] == 5}
        for coef in COEFFICIENTS:
            row[f"reward_{coef:.2f}"] = rl_env.reward_components(acc, BASELINE_VAL, sparsity, actions, coef)["total"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    for coef in COEFFICIENTS:
        frame[f"rank_{coef:.2f}"] = frame[f"reward_{coef:.2f}"].rank(ascending=False, method="min").astype(int)
    frame.to_csv(os.path.join(ROOT, "results", "reward_landscape.csv"), index=False)
    for coef in COEFFICIENTS:
        top = frame.sort_values(f"reward_{coef:.2f}", ascending=False).head(5)
        print(f"lambda {coef:.2f}:")
        print(top[["actions", "probe_accuracy", "total_sparsity", f"reward_{coef:.2f}"]].to_string(index=False))


def rank_learned():
    """Place every learned policy (original and refined runs) in the landscape."""
    landscape = pd.read_csv(os.path.join(ROOT, "results", "reward_landscape.csv")).set_index("actions")
    rows = []
    for folder, experiment in (("runs", "original"), ("refined_runs", "refined")):
        for path in sorted(glob.glob(os.path.join(ROOT, "results", folder, "*.json"))):
            with open(path, encoding="utf-8") as f:
                run = json.load(f)
            coef = f"{run['sparsity_coef']:.2f}"
            actions = str(run["actions"])
            best = landscape[f"reward_{coef}"].max()
            sampled = run["best_training_episode"]["actions"]
            rows.append({
                "experiment": experiment, "run_id": run["run_id"],
                "definition": run.get("sensitivity_definition", "original"),
                "condition": run.get("sensitivity_condition", run.get("experiment_condition")),
                "sparsity_coef": run["sparsity_coef"], "seed": run["seed"], "actions": actions,
                "reward": landscape.loc[actions, f"reward_{coef}"],
                "rank_of_1296": int(landscape.loc[actions, f"rank_{coef}"]),
                "best_reward": best, "gap_to_best": best - landscape.loc[actions, f"reward_{coef}"],
                "reward_optimal_policy": landscape[f"reward_{coef}"].idxmax(),
                "best_sampled_in_training": sampled,
                "best_sampled_rank": int(landscape.loc[sampled, f"rank_{coef}"]),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(ROOT, "results", "learned_policy_reward_rank.csv"), index=False)
    print(frame.groupby(["experiment", "sparsity_coef"])["rank_of_1296"].median().to_string())


if __name__ == "__main__":
    if "--rank-learned" in sys.argv:
        rank_learned()
    else:
        main()
