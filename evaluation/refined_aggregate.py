"""Tables, statistics, policy analysis and figure data for the refined sensitivity analysis.

Analysis plan (fixed in this file before the refined runs finished):

  Unit of analysis: one PPO run. Comparisons are paired by seed (and, when
  pooled, by (lambda_s, seed)). The zeroed run for a (lambda_s, seed) is the
  same run for both definitions, because the zeroed state does not depend on
  the definition.

  Primary: test accuracy, correct minus zeroed and correct minus shuffled, for
  each definition x lambda_s (8 tests). Exact two-sided sign-flip permutation
  test on the mean paired difference; Holm correction across the 8.

  Secondary, per comparison:
    * validation accuracy, total sparsity, and accuracy in excess of global
      magnitude pruning at the run's own exact sparsity (a sparsity-adjusted
      comparison: "accuracy at matched sparsity")
    * median difference, bootstrap 95% CI and t 95% CI of the mean difference,
      exact Wilcoxon signed-rank (zero differences dropped), Cohen's d_z
    * spread across seeds: SD per condition; exact paired permutation test
      (swap within pairs) on the SD difference
    * harmful policy (features.3 pruned at 60%): exact McNemar on discordant seeds
    * pooled over both lambda_s values (n = 20 pairs), same tests, marked secondary
  Model-level: McNemar on the 10,000 test images for every seed whose two
  policies differ (describes those two models, not the feature in general).

The test set is used only to evaluate policies that were already fixed.

Outputs (results/)
  refined_sensitivity_ablation.csv          one row per (definition, condition, lambda_s, seed)
  refined_sensitivity_ablation_summary.csv  one row per (definition, lambda_s, condition)
  refined_sensitivity_policy_stats.csv      policy frequencies and features.3 action counts
  refined_sensitivity_statistics.csv        paired statistics
  refined_sensitivity_mcnemar.csv           model-level tests where policies differ
  refined_sensitivity_state_usage.csv       input importance of the trained agents
  refined_sensitivity_perturbation.csv      sensitivity-only perturbations
  refined_fig_sensitivity_curves.csv        figure 1
  refined_fig_condition_accuracy.csv        figure 2
  refined_fig_policy_stability.csv          figure 3
  refined_fig_accuracy_sparsity.csv         figure 4
"""
import glob
import itertools
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import policy_probe  # noqa: E402
import rl_env  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DEFINITIONS = ["accuracy", "loss"]
COEFFICIENTS = [0.01, 0.02]
CONDITIONS = ["correct", "zeroed", "shuffled"]
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
T975 = [None, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179,
        2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060,
        2.056, 2.052, 2.048, 2.045, 2.042]
BOOT = 10000


# ------------------------------------------------------------------ loading
def load():
    runs = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "refined_runs", "*.json"))):
        with open(path, encoding="utf-8") as f:
            runs.append(json.load(f))
    for key in ("baseline_sha256", "ppo", "timesteps_requested", "refined_values_sha256"):
        assert len({json.dumps(r.get(key), sort_keys=True) for r in runs}) == 1, f"runs disagree on {key}"
    assert len({r["split"]["val_indices_sha256"] for r in runs}) == 1
    for r in runs:
        assert r["test_set_use"].startswith("final evaluation only")
        if r["sensitivity_condition"] == "shuffled":
            m = r["shuffle_mapping"]
            assert all(k != v for k, v in m.items()), f"{r['run_id']}: shuffle has a fixed point"
            assert all(not np.isclose(a, b) for a, b in
                       zip(r["state_sensitivity_vector"], r["normalized_sensitivity_vector"])), \
                f"{r['run_id']}: a layer kept its own value"
    return runs


def rows_for(runs, definition, coef, condition):
    out = []
    for r in runs:
        if abs(r["sparsity_coef"] - coef) > 1e-12 or r["sensitivity_condition"] != condition:
            continue
        if condition != "zeroed" and r["sensitivity_definition"] != definition:
            continue
        out.append(r)
    return sorted(out, key=lambda r: SEEDS.index(r["seed"]))


# --------------------------------------------------------------- statistics
def sign_flip_p(d):
    d = np.asarray(d, dtype=float)
    obs = abs(d.mean())
    if np.allclose(d, 0):
        return 1.0
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(d))))
    stats = np.abs((signs * d).mean(axis=1))
    return float((stats >= obs - 1e-12).mean())


def wilcoxon_exact_p(d):
    d = np.asarray([x for x in d if not np.isclose(x, 0)], dtype=float)
    m = len(d)
    if m == 0:
        return 1.0
    ranks = pd.Series(np.abs(d)).rank(method="average").to_numpy()
    w_obs = ranks[d > 0].sum()
    centre = ranks.sum() / 2
    signs = np.array(list(itertools.product((0, 1), repeat=m)))
    w = (signs * ranks).sum(axis=1)
    return float((np.abs(w - centre) >= abs(w_obs - centre) - 1e-12).mean())


def paired_sd_p(x, y):
    """Exact test of SD(x) = SD(y) by swapping within pairs."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = abs(x.std(ddof=1) - y.std(ddof=1))
    swaps = np.array(list(itertools.product((False, True), repeat=len(x))))
    a = np.where(swaps, y, x)
    b = np.where(swaps, x, y)
    stats = np.abs(a.std(axis=1, ddof=1) - b.std(axis=1, ddof=1))
    return float((stats >= obs - 1e-12).mean())


def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    from math import comb
    probs = [comb(n, i) / 2 ** n for i in range(n + 1)]
    return float(min(1.0, sum(p for p in probs if p <= probs[k] + 1e-15)))


def describe_diff(d, rng):
    d = np.asarray(d, float)
    n = len(d)
    sd = d.std(ddof=1) if n > 1 else float("nan")
    boots = d[rng.integers(0, n, (BOOT, n))].mean(axis=1)
    half = T975[n - 1] * sd / np.sqrt(n) if 1 < n <= 31 else float("nan")
    return {"n_pairs": n, "mean_diff": d.mean(), "median_diff": float(np.median(d)),
            "sd_diff": sd, "boot_ci95_low": float(np.percentile(boots, 2.5)),
            "boot_ci95_high": float(np.percentile(boots, 97.5)),
            "t_ci95_low": d.mean() - half, "t_ci95_high": d.mean() + half,
            "sign_flip_p": sign_flip_p(d), "wilcoxon_p": wilcoxon_exact_p(d),
            "cohens_dz": d.mean() / sd if sd and sd > 0 else float("nan"),
            "n_nonzero": int((~np.isclose(d, 0)).sum())}


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


# ---------------------------------------------------------- matched sparsity
class GlobalMagnitude:
    """Global magnitude over the four agent layers at any exact total sparsity."""

    def __init__(self, model, splits):
        self.model, self.splits, self.cache = model, splits, {}

    def at(self, sparsity):
        if sparsity not in self.cache:
            if sparsity == 0:
                pruned = self.model
            else:
                pruned, _ = common.prune_global_to_total(self.model, sparsity)
                assert abs(common.calculate_sparsity(pruned) - sparsity) < 0.01
            self.cache[sparsity] = (rl_env.evaluate_model(pruned, self.splits.val_loader()),
                                    rl_env.evaluate_model(pruned, self.splits.test_loader()))
        return self.cache[sparsity]


# ---------------------------------------------------------------- per run
def run_row(r, definition, gm):
    gm_val, gm_test = gm.at(r["total_sparsity"])
    f3 = r["prune_percent"][1]
    f0 = r["prune_percent"][0]
    row = {"sensitivity_definition": definition, "sensitivity_condition": r["sensitivity_condition"],
           "sparsity_coef": r["sparsity_coef"], "seed": r["seed"],
           "shared_zeroed_run": r["sensitivity_condition"] == "zeroed",
           "raw_sensitivity_vector": str(r["raw_sensitivity_vector"]),
           "normalized_sensitivity_vector": str(r["normalized_sensitivity_vector"]),
           "state_sensitivity_vector": str([round(v, 6) for v in r["state_sensitivity_vector"]]),
           "shuffle_mapping": json.dumps(r["shuffle_mapping"]) if r["shuffle_mapping"] else "",
           "actions": str(r["actions"]), "prune_percent": str(r["prune_percent"]),
           "total_sparsity": r["total_sparsity"]}
    for layer, v in r["per_layer_sparsity"].items():
        row[f"sparsity_{layer}"] = v
    row.update({
        "probe_accuracy": r["probe_accuracy"], "val_accuracy": r["val_accuracy"],
        "test_accuracy": r["test_accuracy"], "accuracy_retention": r["accuracy_retention"],
        "reward_total": r["training_reward"]["total"],
        "reward_accuracy_term": r["training_reward"]["accuracy_term"],
        "reward_sparsity_term": r["training_reward"]["sparsity_term"],
        "reward_diversity_term": r["training_reward"]["diversity_term"],
        "gm_matched_val_accuracy": gm_val, "gm_matched_test_accuracy": gm_test,
        "test_excess_over_gm": r["test_accuracy"] - gm_test,
        "val_excess_over_gm": r["val_accuracy"] - gm_val,
        "features3_pruned_60": f3 == 60.0, "features0_at_0": f0 == 0.0, "features0_at_most_10": f0 <= 10.0,
        "best_episode_actions": r["best_training_episode"]["actions"],
        "runtime_seconds": r["runtime_seconds"], "train_seconds": r["train_seconds"],
        "torch_threads": r["environment"]["torch_threads"], "run_id": r["run_id"],
    })
    return row


def summary_row(rows, definition, coef, condition):
    test = [r["test_accuracy"] for r in rows]
    sparsity = [r["total_sparsity"] for r in rows]
    policies = Counter(r["actions"] for r in rows)
    return {"sensitivity_definition": definition, "sparsity_coef": coef, "sensitivity_condition": condition,
            "n_seeds": len(rows),
            "test_accuracy_mean": np.mean(test), "test_accuracy_sd": np.std(test, ddof=1),
            "test_accuracy_worst": min(test), "test_accuracy_best": max(test),
            "val_accuracy_mean": np.mean([r["val_accuracy"] for r in rows]),
            "val_accuracy_sd": np.std([r["val_accuracy"] for r in rows], ddof=1),
            "sparsity_mean": np.mean(sparsity), "sparsity_sd": np.std(sparsity, ddof=1),
            "accuracy_retention_mean": np.mean([r["accuracy_retention"] for r in rows]),
            "test_excess_over_gm_mean": np.mean([r["test_excess_over_gm"] for r in rows]),
            "test_excess_over_gm_sd": np.std([r["test_excess_over_gm"] for r in rows], ddof=1),
            "n_features3_pruned_60": sum(r["features3_pruned_60"] for r in rows),
            "n_features0_at_0": sum(r["features0_at_0"] for r in rows),
            "n_features0_at_most_10": sum(r["features0_at_most_10"] for r in rows),
            "unique_policies": len(policies),
            "policy_frequency": "; ".join(f"{p} x{n}" for p, n in policies.most_common()),
            "most_common_policy": policies.most_common(1)[0][0]}


# -------------------------------------------------------------------- main
def main():
    runs = load()
    model = common.load_baseline(os.path.join(ROOT, "checkpoints", "cnn_baseline_FIXED.pth"))
    all_layers, filtered = rl_env.prunable_layers(model)
    max_param_count = max(m.weight.numel() for _, m in all_layers)
    splits = rl_env.Splits(os.path.join(ROOT, "data"))
    gm = GlobalMagnitude(model, splits)
    rng = np.random.default_rng(0)

    # ------------------------------------------------------------ tables
    table, summaries, cells = [], [], {}
    for definition in DEFINITIONS:
        for coef in COEFFICIENTS:
            for condition in CONDITIONS:
                rs = rows_for(runs, definition, coef, condition)
                if [r["seed"] for r in rs] != SEEDS:
                    print(f"  WARNING: {definition}/{coef}/{condition} has seeds "
                          f"{[r['seed'] for r in rs]} -- incomplete")
                rows = [run_row(r, definition, gm) for r in rs]
                cells[(definition, coef, condition)] = rows
                table += rows
                if rows:
                    summaries.append(summary_row(rows, definition, coef, condition))
    pd.DataFrame(table).to_csv(os.path.join(RESULTS, "refined_sensitivity_ablation.csv"), index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(os.path.join(RESULTS, "refined_sensitivity_ablation_summary.csv"), index=False)

    # ------------------------------------------------------ policy stats
    policy_rows = []
    for (definition, coef, condition), rows in cells.items():
        for policy, n in Counter(r["actions"] for r in rows).most_common():
            match = [r for r in rows if r["actions"] == policy]
            policy_rows.append({"table": "policy_frequency", "sensitivity_definition": definition,
                                "sparsity_coef": coef, "sensitivity_condition": condition,
                                "policy": policy, "count": n, "share": n / len(rows),
                                "test_accuracy": match[0]["test_accuracy"],
                                "total_sparsity": match[0]["total_sparsity"],
                                "seeds": str([r["seed"] for r in match])})
        f3 = Counter(r["prune_percent"].strip("[]").split(", ")[1] for r in rows)
        for level in ("0.0", "10.0", "20.0", "30.0", "40.0", "60.0"):
            policy_rows.append({"table": "features3_action", "sensitivity_definition": definition,
                                "sparsity_coef": coef, "sensitivity_condition": condition,
                                "policy": f"features.3 at {level}%", "count": f3.get(level, 0),
                                "share": f3.get(level, 0) / max(1, len(rows))})
    pd.DataFrame(policy_rows).to_csv(os.path.join(RESULTS, "refined_sensitivity_policy_stats.csv"),
                                     index=False)

    # -------------------------------------------------------- statistics
    stats_rows, mcnemar_rows = [], []
    metrics = ["test_accuracy", "val_accuracy", "total_sparsity", "test_excess_over_gm", "val_excess_over_gm"]
    loader = splits.test_loader()
    pred_cache = {}

    def preds(actions):
        if actions not in pred_cache:
            fr = [rl_env.ACTION_TO_PRUNE[int(a)] for a in actions.strip("[]").split(", ")]
            pred_cache[actions] = common.predictions(common.prune_layerwise(model, fr), loader)
        return pred_cache[actions]

    for definition in DEFINITIONS:
        for other in ("zeroed", "shuffled"):
            pooled = {m: [] for m in metrics}
            pooled_pairs = []
            for coef in COEFFICIENTS:
                a = {r["seed"]: r for r in cells[(definition, coef, "correct")]}
                b = {r["seed"]: r for r in cells[(definition, coef, other)]}
                seeds = [s for s in SEEDS if s in a and s in b]
                pairs = [(a[s], b[s]) for s in seeds]
                pooled_pairs += pairs
                label = dict(sensitivity_definition=definition, sparsity_coef=coef,
                             comparison=f"correct vs {other}", level="per lambda_s")
                stats_rows += compare(pairs, metrics, label, rng)
                for m in metrics:
                    pooled[m] += [x[m] - y[m] for x, y in pairs]
                for x, y in pairs:
                    if x["actions"] == y["actions"]:
                        continue
                    pa, labels = preds(x["actions"])
                    pb, _ = preds(y["actions"])
                    only_a, only_b, chi2, p = common.mcnemar(pa == labels, pb == labels)
                    mcnemar_rows.append({**label, "seed": x["seed"], "correct_actions": x["actions"],
                                         "other_actions": y["actions"],
                                         "correct_test": x["test_accuracy"], "other_test": y["test_accuracy"],
                                         "sparsity_diff": x["total_sparsity"] - y["total_sparsity"],
                                         "only_correct_right": only_a, "only_other_right": only_b,
                                         "chi2": chi2, "p": p})
            label = dict(sensitivity_definition=definition, sparsity_coef="pooled 0.01+0.02",
                         comparison=f"correct vs {other}", level="pooled (secondary)")
            stats_rows += compare(pooled_pairs, metrics, label, rng)

    stats = pd.DataFrame(stats_rows)
    if not stats.empty:
        primary = (stats["level"] == "per lambda_s") & (stats["metric"] == "test_accuracy")
        stats["primary"] = primary
        stats["holm_p_primary"] = np.nan
        if primary.any():
            stats.loc[primary, "holm_p_primary"] = holm(stats.loc[primary, "sign_flip_p"].to_numpy())
    stats.to_csv(os.path.join(RESULTS, "refined_sensitivity_statistics.csv"), index=False)
    pd.DataFrame(mcnemar_rows).to_csv(os.path.join(RESULTS, "refined_sensitivity_mcnemar.csv"), index=False)

    # ------------------------------------------------------- state usage
    usage_rows, perturb_rows = [], []
    for r in runs:
        agent = PPO.load(os.path.join(ROOT, "checkpoints", "refined_experiments", r["run_id"]), device="cpu")
        sens = dict(zip(rl_env.FILTERED_LAYERS, r["state_sensitivity_vector"]))
        observations, _ = policy_probe.rollout_observations(agent, filtered, sens, max_param_count,
                                                            expected_actions=r["actions"])
        meta = {"run_id": r["run_id"], "sensitivity_definition": r["sensitivity_definition"],
                "sensitivity_condition": r["sensitivity_condition"], "sparsity_coef": r["sparsity_coef"],
                "seed": r["seed"]}
        for row in policy_probe.permutation_importance(agent, observations):
            usage_rows.append({**meta, **row})
        if r["sensitivity_condition"] != "zeroed":
            for row in policy_probe.sensitivity_perturbation(agent, observations, r["state_sensitivity_vector"],
                                                             extra_values=(0.0, 1.0)):
                perturb_rows.append({**meta, **row})
    pd.DataFrame(usage_rows).to_csv(os.path.join(RESULTS, "refined_sensitivity_state_usage.csv"), index=False)
    pd.DataFrame(perturb_rows).to_csv(os.path.join(RESULTS, "refined_sensitivity_perturbation.csv"), index=False)

    # ----------------------------------------------------------- figures
    curves = pd.read_csv(os.path.join(RESULTS, "sensitivity_probe_curves.csv"))
    curves.to_csv(os.path.join(RESULTS, "refined_fig_sensitivity_curves.csv"), index=False)
    pd.DataFrame(table)[["sensitivity_definition", "sparsity_coef", "sensitivity_condition", "seed",
                         "test_accuracy", "val_accuracy", "total_sparsity", "test_excess_over_gm",
                         "actions", "shared_zeroed_run"]].to_csv(
        os.path.join(RESULTS, "refined_fig_condition_accuracy.csv"), index=False)
    pd.DataFrame([p for p in policy_rows]).to_csv(
        os.path.join(RESULTS, "refined_fig_policy_stability.csv"), index=False)
    points = [{"series": "dense baseline", "sensitivity_definition": "", "sparsity_coef": "",
               "sensitivity_condition": "", "seed": "", "total_sparsity": 0.0, "test_accuracy": 77.03}]
    for row in table:
        points.append({"series": "refined PPO run", **{k: row[k] for k in (
            "sensitivity_definition", "sparsity_coef", "sensitivity_condition", "seed",
            "total_sparsity", "test_accuracy")}})
    for sparsity, (val, test) in sorted(gm.cache.items()):
        points.append({"series": "global magnitude at matched sparsity", "total_sparsity": sparsity,
                       "test_accuracy": test})
    pd.DataFrame(points).to_csv(os.path.join(RESULTS, "refined_fig_accuracy_sparsity.csv"), index=False)

    pd.set_option("display.width", 250)
    print(summary[["sensitivity_definition", "sparsity_coef", "sensitivity_condition", "test_accuracy_mean",
                   "test_accuracy_sd", "test_accuracy_worst", "sparsity_mean", "n_features3_pruned_60",
                   "unique_policies", "policy_frequency"]].to_string(index=False))


def compare(pairs, metrics, label, rng):
    rows = []
    if len(pairs) < 2:
        print(f"  skipping {label}: only {len(pairs)} paired seed(s) so far")
        return rows
    for m in metrics:
        d = [x[m] - y[m] for x, y in pairs]
        row = {**label, "metric": m, **describe_diff(d, rng),
               "identical_policies": sum(x["actions"] == y["actions"] for x, y in pairs)}
        xs, ys = [x[m] for x, _ in pairs], [y[m] for _, y in pairs]
        row.update({"correct_sd": np.std(xs, ddof=1), "other_sd": np.std(ys, ddof=1),
                    "sd_diff_p": paired_sd_p(xs, ys)})
        rows.append(row)
    only_correct = sum(x["features3_pruned_60"] and not y["features3_pruned_60"] for x, y in pairs)
    only_other = sum(y["features3_pruned_60"] and not x["features3_pruned_60"] for x, y in pairs)
    rows.append({**label, "metric": "features3_pruned_60 (count)", "n_pairs": len(pairs),
                 "correct_count": sum(x["features3_pruned_60"] for x, _ in pairs),
                 "other_count": sum(y["features3_pruned_60"] for _, y in pairs),
                 "discordant_correct_only": only_correct, "discordant_other_only": only_other,
                 "exact_mcnemar_p": binom_two_sided(min(only_correct, only_other), only_correct + only_other)})
    return rows


if __name__ == "__main__":
    main()
