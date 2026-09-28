"""Sensitivity-Aware Constrained PPO: tables, statistics, matched sparsity, landscape, figures.

Implements the analysis fixed in results/constrained_preregistration.md.
Reads results/constrained_runs/*.json. The only test-set use is evaluating
already-frozen policies (heuristic C5, matched global-magnitude and uniform
baselines) for reporting and McNemar tests.

Outputs (results/)
  constrained_ppo_all_runs.csv         every run (+ C5 rows); C0 listed under both definitions
  constrained_ppo_summary.csv          per (definition, lambda_s, condition)
  constrained_ppo_policy_stats.csv     policy and layer x action frequencies
  constrained_ppo_statistics.csv       paired comparisons, Holm over the primary family
  constrained_matched_sparsity.csv     per run: GM / uniform at exact sparsity, same-seed C0, McNemar
  constrained_policy_landscape.csv     1,296 policies + masks + where each condition's final policies land
  constrained_predictions/seed_<s>.csv per-example test predictions of every run for seed s
  constrained_predictions/heuristic.csv
  constrained_fig_A_sensitivity_curves.csv ... constrained_fig_E_training.csv
"""
import glob
import json
import math
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import rl_env  # noqa: E402
from refined_aggregate import binom_two_sided, describe_diff, holm, paired_sd_p  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
LAYERS = rl_env.FILTERED_LAYERS
DEFINITIONS = ["accuracy", "loss"]
COEFFICIENTS = [0.01, 0.02]
CONDITIONS = ["C0", "C1", "C2", "C3", "C4"]
PRIMARY = ("accuracy", 0.01)
PRIMARY_FAMILY = ["C2 vs C0", "C2 vs C3", "C2 vs C4"]
SMOOTH = 25


def load_runs():
    runs = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(RESULTS, "constrained_runs", "*.json")))]
    for key in ("baseline_sha256", "ppo", "masks_sha256", "state_sha256", "preregistration_sha256"):
        assert len({json.dumps(r[key], sort_keys=True) for r in runs}) == 1, f"runs disagree on {key}"
    assert len({r["split"]["val_indices_sha256"] for r in runs}) == 1
    assert all(r["masked_actions_taken_in_training"] == 0 for r in runs)
    assert all(r["algorithm"] == "MaskablePPO" for r in runs)
    return runs


def entropy_bits(counter):
    n = sum(counter.values())
    return -sum(c / n * math.log2(c / n) for c in counter.values())


def main():
    runs = load_runs()
    seeds = sorted({r["seed"] for r in runs}, key=lambda s: (s != 42, s))
    model = common.load_baseline(os.path.join(ROOT, "checkpoints", "cnn_baseline_FIXED.pth"))
    loader = common.test_loader(root=os.path.join(ROOT, "data"))
    L = pd.read_csv(os.path.join(RESULTS, "policy_landscape_validation.csv")).set_index("actions")
    masks = pd.read_csv(os.path.join(RESULTS, "sensitivity_action_masks.csv"))
    heuristic = json.load(open(os.path.join(RESULTS, "constrained_heuristic_policies.json")))
    r_star = {d: [masks[(masks.rule == d) & (masks.layer == n)]["safe_max_ratio"].iloc[0] * 100 for n in LAYERS]
              for d in DEFINITIONS}

    # ------------------------------------------------ frozen-policy evaluations (test, reporting only)
    cache = {}

    def evaluate(kind, key):
        if (kind, key) not in cache:
            if kind == "policy":
                pruned = common.prune_layerwise(model, [rl_env.ACTION_TO_PRUNE[a] for a in key])
            elif kind == "gm":
                pruned, _ = common.prune_global_to_total(model, key)
            else:
                pruned, _ = common.prune_uniform_to_total(model, key)
            preds, labels = common.predictions(pruned, loader)
            cache[(kind, key)] = (preds, labels, common.accuracy_from(preds, labels),
                                  common.calculate_sparsity(pruned))
        return cache[(kind, key)]

    def mcnemar(pa, pb, labels):
        return common.mcnemar(pa == labels, pb == labels)

    # ------------------------------------------------ mask-optimal policies
    def inside(mask):
        return [a for a in L.index if all(mask[i][x] for i, x in enumerate(json.loads(a)))]

    mask_opt = {}

    def mask_optimum(mask, coef):
        key = (json.dumps(mask), coef)
        if key not in mask_opt:
            sub = L.loc[inside(mask)]
            mask_opt[key] = sub[f"reward_{coef:.2f}"].idxmax()
        return mask_opt[key]

    # ------------------------------------------------ per-run rows
    rows, pred_cols = [], {}
    for r in runs:
        coef = r["sparsity_coef"]
        a = str(r["actions"])
        mask = [[bool(b) for b in m] for m in r["layer_masks"]]
        preds = np.array([int(c) for c in r["test_predictions"]])
        _, labels, _, _ = evaluate("policy", tuple(r["actions"]))
        assert common.accuracy_from(__import__("torch").tensor(preds), labels) == r["test_accuracy"]
        gm_p, _, gm_acc, gm_sp = evaluate("gm", r["total_sparsity"])
        try:
            un_p, _, un_acc, _ = evaluate("uniform", r["total_sparsity"])
        except ValueError:
            un_p, un_acc = None, float("nan")
        curve = pd.read_csv(os.path.join(RESULTS, "constrained_training_curves", f"{r['run_id']}.csv"))
        opt = mask_optimum(mask, coef)
        glob_opt = L[f"reward_{coef:.2f}"].idxmax()
        within = L.loc[inside(mask)][f"reward_{coef:.2f}"].rank(ascending=False, method="min")
        b_gm = mcnemar(__import__("torch").tensor(preds), gm_p, labels)
        row = {
            "sensitivity_definition": r["sensitivity_definition"], "condition": r["condition"],
            "sparsity_coef": coef, "seed": r["seed"], "actions": a, "prune_percent": str(r["prune_percent"]),
            "total_sparsity": r["total_sparsity"],
            **{f"sparsity_{k}": v for k, v in r["per_layer_sparsity"].items()},
            "probe_accuracy": r["probe_accuracy"], "val_accuracy": r["val_accuracy"],
            "test_accuracy": r["test_accuracy"], "accuracy_retention": r["accuracy_retention"],
            "reward_total": r["training_reward"]["total"],
            "reward_accuracy_term": r["training_reward"]["accuracy_term"],
            "reward_sparsity_term": r["training_reward"]["sparsity_term"],
            "reward_diversity_term": r["training_reward"]["diversity_term"],
            "n_valid_actions": r["n_valid_actions"], "layer_masks": json.dumps(r["layer_masks"]),
            "mask_shuffle_mapping": json.dumps(r["mask_shuffle_mapping"]) if r["mask_shuffle_mapping"] else "",
            "state_sensitivity_vector": str([round(x, 4) for x in r["state_sensitivity_vector"]]),
            "val_pareto_policy": bool(L.loc[a, "pareto_val_accuracy"]),
            "global_reward_rank": int(L.loc[a, f"reward_rank_{coef:.2f}"]),
            "global_reward_optimal_policy": glob_opt,
            "rank_within_own_mask": int(within.loc[a]), "policies_in_own_mask": len(within),
            "mask_optimal_policy": opt, "final_is_mask_optimal": a == opt,
            "sampled_mask_optimal_in_training": bool((curve["actions"] == opt).any()),
            "best_sampled_policy": r["best_training_episode"]["actions"],
            "features3_at_60": r["prune_percent"][1] == 60.0,
            "features0_at_least_20": r["prune_percent"][0] >= 20.0,
            "any_layer_above_accuracy_rule": any(p > s + 1e-9 for p, s in zip(r["prune_percent"], r_star["accuracy"])),
            "gm_matched_test_accuracy": gm_acc, "excess_over_gm": r["test_accuracy"] - gm_acc,
            "mcnemar_vs_gm_p": b_gm[3],
            "uniform_matched_test_accuracy": un_acc, "excess_over_uniform": r["test_accuracy"] - un_acc,
            "runtime_seconds": r["runtime_seconds"], "run_id": r["run_id"],
        }
        rows.append(row)
        pred_cols.setdefault(r["seed"], {})[r["run_id"]] = preds

    # C0 appears under both definitions (it is one shared run)
    table = []
    for row in rows:
        if row["sensitivity_definition"] == "shared":
            for d in DEFINITIONS:
                table.append({**row, "sensitivity_definition": d, "shared_run": True})
        else:
            table.append({**row, "shared_run": False})
    # C5 heuristic rows (no PPO; one policy per definition)
    heur_rows = []
    for d in DEFINITIONS:
        p = tuple(heuristic[d])
        preds, labels, acc, sp = evaluate("policy", p)
        gm_p, _, gm_acc, _ = evaluate("gm", sp)
        un_p, _, un_acc, _ = evaluate("uniform", sp)
        for coef in COEFFICIENTS:
            heur_rows.append({"sensitivity_definition": d, "condition": "C5", "sparsity_coef": coef, "seed": "",
                              "actions": str(list(p)), "prune_percent": str([100 * rl_env.ACTION_TO_PRUNE[a] for a in p]),
                              "total_sparsity": sp, "val_accuracy": float(L.loc[str(list(p)), "val_accuracy"]),
                              "probe_accuracy": float(L.loc[str(list(p)), "probe_accuracy"]),
                              "test_accuracy": acc, "accuracy_retention": 100 * acc / 77.03,
                              "reward_total": float(L.loc[str(list(p)), f"reward_{coef:.2f}"]),
                              "global_reward_rank": int(L.loc[str(list(p)), f"reward_rank_{coef:.2f}"]),
                              "val_pareto_policy": bool(L.loc[str(list(p)), "pareto_val_accuracy"]),
                              "gm_matched_test_accuracy": gm_acc, "excess_over_gm": acc - gm_acc,
                              "mcnemar_vs_gm_p": mcnemar(preds, gm_p, labels)[3],
                              "uniform_matched_test_accuracy": un_acc, "excess_over_uniform": acc - un_acc,
                              "features3_at_60": p[1] == 5, "features0_at_least_20": p[0] >= 2})
    all_runs = pd.DataFrame(table + heur_rows)
    all_runs.to_csv(os.path.join(RESULTS, "constrained_ppo_all_runs.csv"), index=False)

    # ------------------------------------------------ predictions per seed
    pdir = os.path.join(RESULTS, "constrained_predictions")
    os.makedirs(pdir, exist_ok=True)
    _, labels, _, _ = evaluate("policy", (0, 0, 0, 0))
    for s, cols in pred_cols.items():
        pd.DataFrame({"true_label": labels.numpy(), **cols}).to_csv(os.path.join(pdir, f"seed_{s}.csv"), index=False)
    heur = {"true_label": labels.numpy()}
    for d in DEFINITIONS:
        heur[f"C5_{d}_{heuristic[d]}"] = evaluate("policy", tuple(heuristic[d]))[0].numpy()
    pd.DataFrame(heur).to_csv(os.path.join(pdir, "heuristic.csv"), index=False)

    # ------------------------------------------------ summary + policy stats
    def cell(d, coef, cond):
        sub = all_runs[(all_runs.sensitivity_definition == d) & (all_runs.sparsity_coef == coef) & (all_runs.condition == cond)]
        return sub.sort_values("seed", key=lambda s: s.map(lambda x: (x != 42, x)))

    summary, pstats = [], []
    for d in DEFINITIONS:
        for coef in COEFFICIENTS:
            for cond in CONDITIONS:
                c = cell(d, coef, cond)
                if c.empty:
                    continue
                pol = Counter(c["actions"])
                top, topn = pol.most_common(1)[0]
                summary.append({
                    "sensitivity_definition": d, "sparsity_coef": coef, "condition": cond, "n_seeds": len(c),
                    "test_mean": c.test_accuracy.mean(), "test_sd": c.test_accuracy.std(ddof=1),
                    "test_median": c.test_accuracy.median(), "test_min": c.test_accuracy.min(),
                    "test_max": c.test_accuracy.max(), "val_mean": c.val_accuracy.mean(),
                    "sparsity_mean": c.total_sparsity.mean(), "sparsity_sd": c.total_sparsity.std(ddof=1),
                    "retention_mean": c.accuracy_retention.mean(),
                    "excess_over_gm_mean": c.excess_over_gm.mean(), "excess_over_gm_sd": c.excess_over_gm.std(ddof=1),
                    "excess_over_uniform_mean": c.excess_over_uniform.mean(),
                    "distinct_policies": len(pol), "policy_entropy_bits": entropy_bits(pol),
                    "most_common_policy": top, "most_common_share": topn / len(c),
                    "policy_frequency": "; ".join(f"{p} x{n}" for p, n in pol.most_common()),
                    "n_features3_at_60": int(c.features3_at_60.sum()), "n_features0_at_least_20": int(c.features0_at_least_20.sum()),
                    "n_any_layer_above_accuracy_rule": int(c.any_layer_above_accuracy_rule.sum()),
                    "global_rank_median": c.global_reward_rank.median(),
                    "n_final_is_mask_optimal": int(c.final_is_mask_optimal.sum()),
                    "n_sampled_mask_optimal": int(c.sampled_mask_optimal_in_training.sum()),
                    "n_val_pareto_policies": int(c.val_pareto_policy.sum()),
                })
                for p, n in pol.most_common():
                    pstats.append({"table": "policy", "sensitivity_definition": d, "sparsity_coef": coef,
                                   "condition": cond, "item": p, "count": n, "share": n / len(c)})
                for i, layer in enumerate(LAYERS):
                    lv = Counter(json.loads(x)[i] for x in c["actions"])
                    for a in range(6):
                        pstats.append({"table": "layer_action", "sensitivity_definition": d, "sparsity_coef": coef,
                                       "condition": cond, "item": f"{layer}={int(100 * rl_env.ACTION_TO_PRUNE[a])}%",
                                       "count": lv.get(a, 0), "share": lv.get(a, 0) / len(c)})
        for coef in COEFFICIENTS:
            h = cell(d, coef, "C5").iloc[0]
            summary.append({"sensitivity_definition": d, "sparsity_coef": coef, "condition": "C5", "n_seeds": 0,
                            "test_mean": h.test_accuracy, "sparsity_mean": h.total_sparsity,
                            "excess_over_gm_mean": h.excess_over_gm, "excess_over_uniform_mean": h.excess_over_uniform,
                            "most_common_policy": h.actions, "global_rank_median": h.global_reward_rank})
    pd.DataFrame(summary).to_csv(os.path.join(RESULTS, "constrained_ppo_summary.csv"), index=False)
    pd.DataFrame(pstats).to_csv(os.path.join(RESULTS, "constrained_ppo_policy_stats.csv"), index=False)

    # ------------------------------------------------ paired statistics
    rng = np.random.default_rng(0)
    metrics = ["test_accuracy", "val_accuracy", "total_sparsity", "excess_over_gm", "excess_over_uniform"]
    comparisons = [("C2", "C0"), ("C2", "C3"), ("C2", "C4"), ("C2", "C5"), ("C1", "C0"), ("C4", "C0")]
    stats = []
    for d in DEFINITIONS:
        for coef in COEFFICIENTS:
            for a_, b_ in comparisons:
                A = cell(d, coef, a_).set_index("seed")
                if b_ == "C5":
                    h = cell(d, coef, "C5").iloc[0]
                    B = A.copy()
                    for m in metrics:
                        B[m] = h[m]
                    B["actions"] = h.actions
                    B["features3_at_60"] = h.features3_at_60
                else:
                    B = cell(d, coef, b_).set_index("seed")
                common_seeds = [s for s in seeds if s in A.index and s in B.index]
                if len(common_seeds) < 2:
                    continue
                A, B = A.loc[common_seeds], B.loc[common_seeds]
                label = {"sensitivity_definition": d, "sparsity_coef": coef, "comparison": f"{a_} vs {b_}",
                         "primary_family": (d, coef) == PRIMARY and f"{a_} vs {b_}" in PRIMARY_FAMILY}
                for m in metrics:
                    diff = (A[m] - B[m]).astype(float).to_numpy()
                    row = {**label, "metric": m, **describe_diff(diff, rng),
                           "identical_policies": int((A["actions"] == B["actions"]).sum()),
                           "worst_seed_a": A[m].min(), "worst_seed_b": B[m].min()}
                    if b_ != "C5":
                        row.update({"sd_a": A[m].std(ddof=1), "sd_b": B[m].std(ddof=1),
                                    "sd_diff_p": paired_sd_p(A[m].to_numpy(float), B[m].to_numpy(float))})
                    stats.append(row)
                oa = int((A.features3_at_60 & ~B.features3_at_60.astype(bool)).sum())
                ob = int((B.features3_at_60.astype(bool) & ~A.features3_at_60).sum())
                stats.append({**label, "metric": "features3_at_60 (count)", "n_pairs": len(A),
                              "count_a": int(A.features3_at_60.sum()), "count_b": int(B.features3_at_60.astype(bool).sum()),
                              "discordant_a_only": oa, "discordant_b_only": ob,
                              "exact_mcnemar_p": binom_two_sided(min(oa, ob), oa + ob)})
    st = pd.DataFrame(stats)
    fam = st["primary_family"] & (st["metric"] == "test_accuracy")
    st["holm_p_primary"] = np.nan
    st.loc[fam, "holm_p_primary"] = holm(st.loc[fam, "sign_flip_p"].to_numpy())
    st.to_csv(os.path.join(RESULTS, "constrained_ppo_statistics.csv"), index=False)

    # ------------------------------------------------ matched sparsity (per run)
    ms = []
    run_by = {(r["sensitivity_definition"], r["condition"], r["sparsity_coef"], r["seed"]): r for r in runs}
    for r in runs:
        if r["condition"] == "C0":
            continue
        preds = __import__("torch").tensor([int(c) for c in r["test_predictions"]])
        gm_p, labels, gm_acc, gm_sp = evaluate("gm", r["total_sparsity"])
        row = {"sensitivity_definition": r["sensitivity_definition"], "condition": r["condition"],
               "sparsity_coef": r["sparsity_coef"], "seed": r["seed"], "actions": str(r["actions"]),
               "sparsity": r["total_sparsity"], "test_accuracy": r["test_accuracy"],
               "gm_sparsity": gm_sp, "gm_test_accuracy": gm_acc, "diff_vs_gm": r["test_accuracy"] - gm_acc}
        g = mcnemar(preds, gm_p, labels)
        row.update({"gm_only_run_right": g[0], "gm_only_gm_right": g[1], "gm_mcnemar_p": g[3]})
        try:
            un_p, _, un_acc, un_sp = evaluate("uniform", r["total_sparsity"])
            u = mcnemar(preds, un_p, labels)
            row.update({"uniform_sparsity": un_sp, "uniform_test_accuracy": un_acc,
                        "diff_vs_uniform": r["test_accuracy"] - un_acc, "uniform_mcnemar_p": u[3]})
        except ValueError:
            row.update({"uniform_test_accuracy": float("nan")})
        c0 = run_by.get(("shared", "C0", r["sparsity_coef"], r["seed"]))
        if c0:
            p0 = __import__("torch").tensor([int(c) for c in c0["test_predictions"]])
            m0 = mcnemar(preds, p0, labels)
            row.update({"c0_same_seed_actions": str(c0["actions"]), "c0_sparsity": c0["total_sparsity"],
                        "c0_test_accuracy": c0["test_accuracy"],
                        "sparsity_diff_vs_c0": r["total_sparsity"] - c0["total_sparsity"],
                        "diff_vs_c0": r["test_accuracy"] - c0["test_accuracy"],
                        "c0_comparable_sparsity_within_1pp": abs(r["total_sparsity"] - c0["total_sparsity"]) <= 1.0,
                        "c0_mcnemar_p": m0[3]})
        ms.append(row)
    pd.DataFrame(ms).to_csv(os.path.join(RESULTS, "constrained_matched_sparsity.csv"), index=False)

    # ------------------------------------------------ landscape overlay
    land = L.reset_index().copy()
    for d in DEFINITIONS:
        allowed = [set(json.loads(x)) for x in masks[masks.rule == d].set_index("layer").loc[LAYERS]["allowed_actions"]]
        land[f"in_correct_{d}_mask"] = [all(a in allowed[i] for i, a in enumerate(json.loads(x))) for x in land["actions"]]
        land[f"heuristic_{d}"] = land["actions"] == str(heuristic[d])
    for coef in COEFFICIENTS:
        land[f"reward_optimal_{coef:.2f}"] = land[f"reward_rank_{coef:.2f}"] == 1
        for d in DEFINITIONS:
            for cond in CONDITIONS:
                c = Counter(cell(d, coef, cond)["actions"])
                land[f"final_{cond}_{d}_{coef:.2f}"] = land["actions"].map(lambda x: c.get(x, 0))
    land.to_csv(os.path.join(RESULTS, "constrained_policy_landscape.csv"), index=False)

    # ------------------------------------------------ figure data
    curves = pd.read_csv(os.path.join(RESULTS, "constrained_sensitivity_curves.csv"))
    curves.to_csv(os.path.join(RESULTS, "constrained_fig_A_sensitivity_curves.csv"), index=False)
    land.to_csv(os.path.join(RESULTS, "constrained_fig_B_landscape.csv"), index=False)
    fig_c = [{"rule": m.rule, "layer": m.layer, "action": a, "ratio_pct": int(100 * rl_env.ACTION_TO_PRUNE[a]),
              "allowed": a in json.loads(m.allowed_actions)} for m in masks.itertuples() for a in range(6)]
    pd.DataFrame(fig_c).to_csv(os.path.join(RESULTS, "constrained_fig_C_masks.csv"), index=False)
    fig_d = all_runs[all_runs.condition.isin(CONDITIONS)][["sensitivity_definition", "sparsity_coef", "condition",
                                                          "seed", "prune_percent", "actions", "test_accuracy",
                                                          "total_sparsity", "shared_run"]].copy()
    for i, layer in enumerate(LAYERS):
        fig_d[layer] = fig_d["prune_percent"].map(lambda s: json.loads(s)[i])
    fig_d.to_csv(os.path.join(RESULTS, "constrained_fig_D_policy_distribution.csv"), index=False)
    frames = []
    for r in runs:
        c = pd.read_csv(os.path.join(RESULTS, "constrained_training_curves", f"{r['run_id']}.csv"))
        c["reward_smoothed"] = c["episode_reward"].rolling(SMOOTH, min_periods=1).mean()
        c["sampled_policy_global_rank"] = c["actions"].map(lambda a: int(L.loc[a, f"reward_rank_{r['sparsity_coef']:.2f}"]))
        for k in ("sensitivity_definition", "condition", "sparsity_coef", "seed"):
            c[k] = r[k]
        frames.append(c)
    tr = pd.concat(frames)
    agg = (tr.groupby(["sensitivity_definition", "condition", "sparsity_coef", "episode"])
           .agg(n_seeds=("seed", "nunique"), reward_mean=("episode_reward", "mean"), reward_sd=("episode_reward", "std"),
                reward_smoothed_mean=("reward_smoothed", "mean"), reward_smoothed_sd=("reward_smoothed", "std"),
                sampled_rank_median=("sampled_policy_global_rank", "median")).reset_index())
    agg.to_csv(os.path.join(RESULTS, "constrained_fig_E_training.csv"), index=False)

    pd.set_option("display.width", 250)
    print(pd.DataFrame(summary)[["sensitivity_definition", "sparsity_coef", "condition", "test_mean", "test_sd",
                                 "test_min", "sparsity_mean", "excess_over_gm_mean", "distinct_policies",
                                 "n_features3_at_60", "most_common_policy"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
