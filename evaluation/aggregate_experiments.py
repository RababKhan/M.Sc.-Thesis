"""Tables, paired statistics and figure data from the PPO experiment runs.

Reads results/runs/*.json (written by ppo_experiments.py) and the verified
matched-sparsity CSVs. Trains nothing. The only model evaluation it performs is
rebuilding each learned policy's pruned model -- pruning is deterministic --
to get per-sample test predictions for McNemar tests. That is reporting, not
selection: every policy was fixed before its test set was ever touched.

Outputs
  results/sensitivity_ablation.csv            one row per (condition, seed)
  results/sensitivity_ablation_summary.csv    one row per condition
  results/sensitivity_ablation_stats.csv      correct vs zeroed / shuffled, paired
  results/reward_coefficient_sweep.csv        one row per (coefficient, seed)
  results/reward_coefficient_sweep_summary.csv
  results/accuracy_sparsity_tradeoff.csv      figure A
  results/ppo_policy_heatmap.csv              figure B
  results/sensitivity_ablation_plot.csv       figure C
  results/reward_sweep_plot.csv               figure D
  results/ppo_training_curves.csv             figure E, every episode of every run
  results/ppo_training_curves_aggregate.csv   figure E, mean/SD across seeds

Run from the repository root:  python evaluation/aggregate_experiments.py
"""
import ast
import glob
import itertools
import json
import os
import statistics
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import rl_env  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SEEDS = [42, 1, 2, 3]
CONDITIONS = ["correct", "zeroed", "shuffled"]
COEFFICIENTS = [0.00, 0.01, 0.02, 0.04, 0.08]
SMOOTHING_WINDOW = 25  # episodes; a run has 512


# ------------------------------------------------------------------ loading
def load_runs():
    runs = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "runs", "*.json"))):
        with open(path, encoding="utf-8") as f:
            runs.append(json.load(f))
    if not runs:
        raise SystemExit("no runs in results/runs -- run ppo_experiments.py first")

    # Guards: every run must share the baseline, split, PPO settings and budget.
    for key in ("baseline_sha256", "ppo", "timesteps_requested"):
        values = {json.dumps(r[key], sort_keys=True) for r in runs}
        assert len(values) == 1, f"runs disagree on {key}: {values}"
    splits = {r["split"]["val_indices_sha256"] for r in runs}
    assert len(splits) == 1, "runs used different validation splits"
    layer_sets = {tuple(r["per_layer_sparsity"]) for r in runs}
    assert len(layer_sets) == 1, "runs report different prunable layer sets"
    for r in runs:
        assert r["test_set_use"].startswith("final evaluation only"), r["run_id"]
    return runs


def select(runs, condition=None, coef=None):
    out = [r for r in runs
           if (condition is None or r["experiment_condition"] == condition)
           and (coef is None or abs(r["sparsity_coef"] - coef) < 1e-12)]
    return sorted(out, key=lambda r: SEEDS.index(r["seed"]))


def complete(rows, label):
    seeds = [r["seed"] for r in rows]
    if sorted(seeds) != sorted(SEEDS):
        print(f"  WARNING: {label} has seeds {seeds}, expected {SEEDS} -- summary is partial")
        return False
    return True


# ------------------------------------------------------------------- tables
def run_row(r):
    row = {
        "seed": r["seed"],
        "condition": r["experiment_condition"],
        "sparsity_coef": r["sparsity_coef"],
        "actions": str(r["actions"]),
        "prune_percent": str(r["prune_percent"]),
        "total_sparsity": r["total_sparsity"],
    }
    for layer, value in r["per_layer_sparsity"].items():
        row[f"sparsity_{layer}"] = value
    row.update({
        "probe_accuracy": r["probe_accuracy"],
        "val_accuracy": r["val_accuracy"],
        "test_accuracy": r["test_accuracy"],
        "accuracy_retention": r["accuracy_retention"],
        "reward_total": r["training_reward"]["total"],
        "reward_accuracy_term": r["training_reward"]["accuracy_term"],
        "reward_sparsity_term": r["training_reward"]["sparsity_term"],
        "reward_diversity_term": r["training_reward"]["diversity_term"],
        "timesteps_trained": r["timesteps_trained"],
        "episodes": r["episodes"],
        "sensitivity_vector": str([round(v, 6) for v in r["sensitivity_vector"]]),
        "shuffle_mapping": json.dumps(r["shuffle_mapping"]) if r["shuffle_mapping"] else "",
        "policy_reported": r["policy_reported"],
        "best_episode_actions": r["best_training_episode"]["actions"],
        "best_episode_reward": r["best_training_episode"]["reward"],
        "runtime_seconds": r["runtime_seconds"],
        "run_id": r["run_id"],
    })
    return row


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else float("nan")


def summarise(rows, label_key, label):
    test = [r["test_accuracy"] for r in rows]
    val = [r["val_accuracy"] for r in rows]
    sparsity = [r["total_sparsity"] for r in rows]
    retention = [r["accuracy_retention"] for r in rows]
    policies = Counter(str(r["actions"]) for r in rows)
    most_common, count = policies.most_common(1)[0]
    return {
        label_key: label,
        "n_seeds": len(rows),
        "test_accuracy_mean": statistics.mean(test),
        "test_accuracy_sd": sd(test),
        "val_accuracy_mean": statistics.mean(val),
        "val_accuracy_sd": sd(val),
        "sparsity_mean": statistics.mean(sparsity),
        "sparsity_sd": sd(sparsity),
        "accuracy_retention_mean": statistics.mean(retention),
        "distinct_policies": len(policies),
        "policy_frequency": "; ".join(f"{p} x{n}" for p, n in policies.most_common()),
        "most_common_policy": most_common,
        "most_common_policy_count": count,
    }


# -------------------------------------------------------------- statistics
def sign_flip_p(differences):
    """Exact two-sided paired permutation (sign-flip) test on the mean difference.

    With n = 4 there are 16 sign patterns, so the smallest attainable two-sided
    p is 2/16 = 0.125: no n = 4 comparison can reach p < 0.05 with this test.
    """
    observed = abs(statistics.mean(differences))
    if observed == 0:
        return 1.0
    flips = list(itertools.product((1, -1), repeat=len(differences)))
    extreme = sum(1 for signs in flips
                  if abs(statistics.mean(s * d for s, d in zip(signs, differences))) >= observed - 1e-12)
    return extreme / len(flips)


def test_predictions(actions, model, loader, cache):
    key = tuple(actions)
    if key not in cache:
        pruned = common.prune_layerwise(model, [rl_env.ACTION_TO_PRUNE[a] for a in actions])
        cache[key] = common.predictions(pruned, loader)
    return cache[key]


def paired_stats(ablation, model, loader):
    rows = []
    cache = {}
    reference = {r["seed"]: r for r in ablation["correct"]}
    for other in ("zeroed", "shuffled"):
        by_seed = {r["seed"]: r for r in ablation[other]}
        seeds = [s for s in SEEDS if s in reference and s in by_seed]
        acc_diff = [reference[s]["test_accuracy"] - by_seed[s]["test_accuracy"] for s in seeds]
        sp_diff = [reference[s]["total_sparsity"] - by_seed[s]["total_sparsity"] for s in seeds]

        for s in seeds:
            a, b = reference[s], by_seed[s]
            same = a["actions"] == b["actions"]
            preds_a, labels = test_predictions(a["actions"], model, loader, cache)
            preds_b, _ = test_predictions(b["actions"], model, loader, cache)
            only_a, only_b, chi2, p = common.mcnemar(preds_a == labels, preds_b == labels)
            rows.append({
                "comparison": f"correct vs {other}", "level": "per seed", "seed": s,
                "correct_actions": str(a["actions"]), "other_actions": str(b["actions"]),
                "identical_policy": same,
                "correct_test_accuracy": a["test_accuracy"], "other_test_accuracy": b["test_accuracy"],
                "test_accuracy_diff_pp": a["test_accuracy"] - b["test_accuracy"],
                "correct_sparsity": a["total_sparsity"], "other_sparsity": b["total_sparsity"],
                "sparsity_diff_pp": a["total_sparsity"] - b["total_sparsity"],
                "mcnemar_only_correct": only_a, "mcnemar_only_other": only_b,
                "mcnemar_chi2": chi2, "mcnemar_p": p,
                "note": ("identical policy: same pruned model, no difference possible" if same else
                         "McNemar compares these two specific pruned models; sparsities differ"
                         if abs(a["total_sparsity"] - b["total_sparsity"]) > 1e-9 else
                         "McNemar compares these two specific pruned models"),
            })

        if len(seeds) >= 2:
            rows.append({
                "comparison": f"correct vs {other}", "level": f"across seeds (n={len(seeds)})",
                "seed": "",
                "test_accuracy_diff_pp": statistics.mean(acc_diff),
                "sparsity_diff_pp": statistics.mean(sp_diff),
                "identical_policy": sum(reference[s]["actions"] == by_seed[s]["actions"] for s in seeds),
                "mcnemar_p": "",
                "sign_flip_p_accuracy": sign_flip_p(acc_diff),
                "sign_flip_p_sparsity": sign_flip_p(sp_diff),
                "note": (f"exact paired sign-flip test on per-seed differences; with n={len(seeds)} "
                         f"the smallest attainable two-sided p is {2 / 2 ** len(seeds):.3f}. "
                         f"identical_policy counts seeds where both conditions learned the same policy."),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def figure_a():
    """Accuracy-sparsity trade-off from the verified matched-sparsity analysis."""
    methods = pd.read_csv(os.path.join(RESULTS, "matched_sparsity_methods.csv"))
    rows = [{"method": "Dense baseline", "family": "dense", "matched_to_seed": "",
             "sparsity": 0.0, "test_accuracy": 77.03, "actions": "-"}]
    for _, m in methods.iterrows():
        name = m["Method"]
        if name.startswith("PPO seed"):
            family, seed = "ppo", int(name.split()[-1])
        elif name.startswith("Global magnitude @"):
            family, seed = "global_magnitude_matched", int(m["Notes"].split("seed ")[1].split(";")[0])
        elif name.startswith("Uniform per-layer @"):
            family, seed = "uniform_matched", int(m["Notes"].split("seed ")[1].split(";")[0])
        elif name.startswith("Uniform per-layer"):
            family, seed = "uniform_reference_unmatched", ""
        elif name.startswith("Global magnitude"):
            family, seed = "global_magnitude_reference_unmatched_incl_output_layer", ""
        else:
            raise ValueError(name)
        rows.append({"method": name, "family": family, "matched_to_seed": seed,
                     "sparsity": m["Sparsity (%)"], "test_accuracy": m["Accuracy (%)"],
                     "actions": m["Actions"]})

    frame = pd.DataFrame(rows)
    # Seeds 2 and 3 share a policy, so the analysis wrote one matched row per
    # method at that sparsity; restore one row per seed so every seed is plotted.
    for family in ("ppo", "global_magnitude_matched", "uniform_matched"):
        present = set(frame.loc[frame["family"] == family, "matched_to_seed"])
        missing = [s for s in SEEDS if s not in present]
        for s in missing:
            twin = 2 if s == 3 else 3
            source = frame[(frame["family"] == family) & (frame["matched_to_seed"] == twin)]
            if source.empty:
                raise ValueError(f"{family} missing seed {s} and its twin")
            copy = source.iloc[0].copy()
            copy["matched_to_seed"] = s
            copy["method"] = copy["method"] + f" (seed {s}: same policy as seed {twin})"
            frame = pd.concat([frame, copy.to_frame().T], ignore_index=True)

    ppo = pd.read_csv(os.path.join(RESULTS, "ppo_multiseed.csv"))
    for _, p in ppo.iterrows():
        match = frame[(frame["family"] == "ppo") & (frame["matched_to_seed"] == p["Seed"])]
        assert len(match) == 1 and abs(match["test_accuracy"].iloc[0] - p["Accuracy (%)"]) < 1e-9, \
            f"figure A PPO seed {p['Seed']} disagrees with ppo_multiseed.csv"
    for s in SEEDS:
        sp = frame.loc[frame["matched_to_seed"] == s, "sparsity"].astype(float)
        assert sp.max() - sp.min() < 0.01, f"seed {s} rows are not at matched sparsity: {list(sp)}"
    return frame


def figure_b(model):
    ppo = pd.read_csv(os.path.join(RESULTS, "ppo_multiseed.csv"))
    rows = []
    for _, p in ppo.sort_values("Seed", key=lambda s: s.map(SEEDS.index)).iterrows():
        actions = ast.literal_eval(p["Actions"])
        fractions = [rl_env.ACTION_TO_PRUNE[a] for a in actions]
        realised = common.per_layer_sparsity(common.prune_layerwise(model, fractions))
        row = {"seed": int(p["Seed"]), "actions": str(actions)}
        for layer, f in zip(rl_env.FILTERED_LAYERS, fractions):
            row[layer] = round(100 * f, 1)
        for layer in rl_env.FILTERED_LAYERS:
            row[f"{layer}_realised"] = realised[layer]
        row["total_sparsity"] = p["Sparsity (%)"]
        row["test_accuracy"] = p["Accuracy (%)"]
        rows.append(row)
    return pd.DataFrame(rows)


def curves(runs):
    frames = []
    for r in runs:
        path = os.path.join(RESULTS, "training_curves", f"{r['run_id']}.csv")
        c = pd.read_csv(path)
        c.insert(0, "run_id", r["run_id"])
        c.insert(1, "condition", r["experiment_condition"])
        c.insert(2, "sparsity_coef", r["sparsity_coef"])
        c.insert(3, "seed", r["seed"])
        c["episode_reward_smoothed"] = c["episode_reward"].rolling(SMOOTHING_WINDOW, min_periods=1).mean()
        frames.append(c)
    long = pd.concat(frames, ignore_index=True)
    aggregate = (long.groupby(["condition", "sparsity_coef", "episode"])
                 .agg(timestep=("timestep", "mean"),
                      n_seeds=("seed", "nunique"),
                      reward_mean=("episode_reward", "mean"),
                      reward_sd=("episode_reward", "std"),
                      reward_smoothed_mean=("episode_reward_smoothed", "mean"),
                      reward_smoothed_sd=("episode_reward_smoothed", "std"))
                 .reset_index())
    return long, aggregate


# -------------------------------------------------------------------- main
def main():
    runs = load_runs()
    model = common.load_baseline(os.path.join(ROOT, "checkpoints", "cnn_baseline_FIXED.pth"))
    loader = common.test_loader(root=os.path.join(ROOT, "data"))

    # Sensitivity ablation (lambda_s = 0.04)
    ablation = {c: select(runs, c, 0.04) for c in CONDITIONS}
    ablation_rows = [run_row(r) for c in CONDITIONS for r in ablation[c]]
    pd.DataFrame(ablation_rows).to_csv(os.path.join(RESULTS, "sensitivity_ablation.csv"), index=False)
    summary = [summarise(ablation[c], "condition", c) for c in CONDITIONS
               if ablation[c] and (complete(ablation[c], f"ablation {c}") or True)]
    pd.DataFrame(summary).to_csv(os.path.join(RESULTS, "sensitivity_ablation_summary.csv"), index=False)
    if all(ablation[c] for c in CONDITIONS):
        paired_stats(ablation, model, loader).to_csv(
            os.path.join(RESULTS, "sensitivity_ablation_stats.csv"), index=False)
    pd.DataFrame([{k: row[k] for k in ("condition", "seed", "test_accuracy", "val_accuracy",
                                       "total_sparsity", "accuracy_retention", "actions",
                                       "prune_percent")}
                  for row in ablation_rows]).to_csv(
        os.path.join(RESULTS, "sensitivity_ablation_plot.csv"), index=False)

    # Reward coefficient sweep (correct sensitivity)
    sweep = {k: select(runs, "correct", k) for k in COEFFICIENTS}
    sweep_rows = [run_row(r) for k in COEFFICIENTS for r in sweep[k]]
    pd.DataFrame(sweep_rows).to_csv(os.path.join(RESULTS, "reward_coefficient_sweep.csv"), index=False)
    pd.DataFrame([summarise(sweep[k], "sparsity_coef", k) for k in COEFFICIENTS
                  if sweep[k] and (complete(sweep[k], f"sweep {k}") or True)]).to_csv(
        os.path.join(RESULTS, "reward_coefficient_sweep_summary.csv"), index=False)
    pd.DataFrame([{k: row[k] for k in ("sparsity_coef", "seed", "test_accuracy", "val_accuracy",
                                       "total_sparsity", "accuracy_retention", "actions",
                                       "prune_percent")}
                  for row in sweep_rows]).to_csv(os.path.join(RESULTS, "reward_sweep_plot.csv"), index=False)

    # Figures A, B, E
    figure_a().to_csv(os.path.join(RESULTS, "accuracy_sparsity_tradeoff.csv"), index=False)
    figure_b(model).to_csv(os.path.join(RESULTS, "ppo_policy_heatmap.csv"), index=False)
    long, aggregate = curves(runs)
    long.to_csv(os.path.join(RESULTS, "ppo_training_curves.csv"), index=False)
    aggregate.to_csv(os.path.join(RESULTS, "ppo_training_curves_aggregate.csv"), index=False)

    print(f"{len(runs)} runs aggregated")
    print("\n== Sensitivity ablation ==")
    print(pd.DataFrame(summary).to_string(index=False))
    print("\n== Reward coefficient sweep ==")
    print(pd.read_csv(os.path.join(RESULTS, "reward_coefficient_sweep_summary.csv")).to_string(index=False))


if __name__ == "__main__":
    main()
