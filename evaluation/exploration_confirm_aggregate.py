"""Confirmatory exploration study: final analysis (run only after all 120 runs exist).

Applies results/exploration_levelA_preregistration.md exactly: endpoints from
the per-episode checkpoint trajectories (validation landscape), seed-level
paired exact tests, Holm families, and the Level A/B/C rule computed
mechanically. The test set is used only to evaluate frozen final policies
and matched baselines, which cannot affect the classification.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import exploration_confirm as ec  # noqa: E402

RESULTS = ec.RESULTS
OUT = lambda name: os.path.join(RESULTS, name)  # noqa: E731
ENDPOINTS = {  # name: (column, higher_is_better)
    "primary: validation accuracy AUC": ("val_accuracy_auc", True),
    "S1: reward-rank score AUC": ("rank_score_auc", True),
    "S2: utility AUC": ("utility_auc", True),
    "S3: destructive features.0 frequency": ("destructive_f0_frequency", False),
    "S4: worst trajectory validation accuracy": ("worst_trajectory_val_accuracy", True),
    "S5: final policy validation accuracy": ("final_val_accuracy", True),
}


def main():
    runs = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(ec.RUNS_DIR, "*.json")))]
    assert len(runs) == 120, f"{len(runs)} runs; the analysis requires all 120"
    keys = {(r["condition"], r["seed"]) for r in runs}
    assert keys == {(c, s) for c in ec.CONDITIONS for s in ec.SEEDS}, "runs do not match the locked design"
    for k in ("preregistration_sha256", "landscape_sha256", "baseline_sha256", "sensitivity_values_sha256"):
        assert len({r[k] for r in runs}) == 1, f"runs disagree on {k}"
    assert runs[0]["preregistration_sha256"] == ec.base.sha256(ec.PREREG), "pre-registration changed"
    assert all(r["episodes"] == 512 and r["timesteps_trained"] == 2048 for r in runs)

    L = ec.load_landscape()
    model = common.load_baseline(os.path.join(ec.ROOT, "checkpoints", "cnn_baseline_FIXED.pth"))
    loader = common.test_loader(root=os.path.join(ec.ROOT, "data"))
    frozen = {}

    def baseline(kind, sparsity):
        if (kind, sparsity) not in frozen:
            pruned, _ = (common.prune_global_to_total if kind == "gm" else common.prune_uniform_to_total)(model, sparsity)
            p, labels = common.predictions(pruned, loader)
            frozen[(kind, sparsity)] = (p, labels, common.accuracy_from(p, labels))
        return frozen[(kind, sparsity)]

    # ---------------------------------------------------------- per run
    rows, traj_all, safety, preds = [], [], [], {}
    for r in runs:
        curve = pd.read_csv(os.path.join(ec.CURVES_DIR, f"{r['run_id']}.csv"))
        t = ec.trajectory(curve, L)
        m = ec.run_metrics(t)
        a = str(r["actions"])
        pr = torch.tensor([int(c) for c in r["test_predictions"]])
        gm_p, labels, gm_acc = baseline("gm", r["total_sparsity"])
        un_p, _, un_acc = baseline("uniform", r["total_sparsity"])
        base_cols = {"condition": r["condition"], "seed": r["seed"], "run_id": r["run_id"]}
        rows.append({**base_cols, "prior_sensitivity": str([round(x, 6) for x in r["prior_sensitivity"]]),
                     "prior_mapping": json.dumps(r["prior_mapping"]) if r["prior_mapping"] else "", **m,
                     "final_actions": a, "final_prune_percent": str(r["prune_percent"]),
                     "final_sparsity": r["total_sparsity"], "final_val_accuracy": r["val_accuracy"],
                     "final_reward_rank": int(L.loc[a, "reward_rank_0.01"]),
                     "final_val_pareto": bool(L.loc[a, "pareto_val_accuracy"]),
                     "test_accuracy": r["test_accuracy"], "gm_matched_test_accuracy": gm_acc,
                     "excess_over_gm": r["test_accuracy"] - gm_acc,
                     "mcnemar_vs_gm_p": common.mcnemar(pr == labels, gm_p == labels)[3],
                     "uniform_matched_test_accuracy": un_acc, "runtime_seconds": r["runtime_seconds"]})
        t.insert(0, "seed", r["seed"])
        t.insert(0, "condition", r["condition"])
        traj_all.append(t)
        srow = {**base_cols, "destructive_f0_count": m["destructive_f0_count"],
                "destructive_f0_frequency": m["destructive_f0_frequency"], "mean_f0_pruning": m["mean_f0_pruning"],
                "max_f0_pruning": m["max_f0_pruning"]}
        for layer in ec.sp.LAYERS:
            counts = t[f"ratio_{layer}"].value_counts()
            for ratio in (0, 10, 20, 30, 40, 60):
                srow[f"{layer}_{ratio}"] = int(counts.get(ratio, 0))
        safety.append(srow)
        preds.setdefault(r["seed"], {"true_label": labels.numpy()})[r["run_id"]] = pr.numpy()

    table = pd.DataFrame(rows).sort_values(["condition", "seed"]).reset_index(drop=True)
    table.to_csv(OUT("exploration_levelA_all_runs.csv"), index=False)
    traj = pd.concat(traj_all)
    traj.to_csv(OUT("exploration_levelA_trajectory.csv"), index=False)
    table[["condition", "seed", "val_accuracy_auc", "rank_score_auc", "utility_auc", "destructive_f0_frequency",
           "worst_trajectory_val_accuracy", "mean_val_accuracy_plain", "run_id"]].to_csv(
        OUT("exploration_levelA_auc.csv"), index=False)
    pd.DataFrame(safety).sort_values(["condition", "seed"]).to_csv(OUT("exploration_levelA_safety.csv"), index=False)
    table[["condition", "seed", "target_reached", "target_first_episode", "target_first_timestep", "run_id"]].to_csv(
        OUT("exploration_levelA_sample_efficiency.csv"), index=False)

    # McNemar vs same-seed E0 for the final-model table
    by = table.set_index(["condition", "seed"])
    final = table[["condition", "seed", "final_actions", "final_sparsity", "final_val_accuracy", "final_reward_rank",
                   "final_val_pareto", "test_accuracy", "gm_matched_test_accuracy", "excess_over_gm",
                   "mcnemar_vs_gm_p", "uniform_matched_test_accuracy", "run_id"]].copy()
    mc0 = []
    for _, row in final.iterrows():
        e0 = by.loc[("E0", row.seed)]
        p1 = torch.tensor(preds[row.seed][row.run_id])
        p0 = torch.tensor(preds[row.seed][e0.run_id])
        labels = torch.tensor(preds[row.seed]["true_label"])
        mc0.append(np.nan if row.condition == "E0" else common.mcnemar(p1 == labels, p0 == labels)[3])
    final["mcnemar_vs_E0_p"] = mc0
    final.to_csv(OUT("exploration_levelA_final_models.csv"), index=False)
    pdir = OUT("exploration_levelA_predictions")
    os.makedirs(pdir, exist_ok=True)
    for s, cols in preds.items():
        pd.DataFrame(cols).to_csv(os.path.join(pdir, f"seed_{s}.csv"), index=False)

    # ---------------------------------------------------------- statistics
    rng = np.random.default_rng(0)
    stats = []
    for endpoint, (col, higher) in ENDPOINTS.items():
        family = []
        for other in ("E0", "E2", "E3"):
            res = ec.paired(table, col, "E1", other, rng)
            favours = res["mean_diff"] > 0 if higher else res["mean_diff"] < 0
            family.append({"endpoint": endpoint, "column": col, "comparison": f"E1 vs {other}",
                           "higher_is_better": higher, **res, "favours_E1": bool(favours)})
        adj = ec.holm([f["sign_flip_p"] for f in family])
        for f, h in zip(family, adj):
            f["holm_p"] = float(h)
        stats.extend(family)
    # descriptive sample-efficiency sign tests
    for other in ("E0", "E2", "E3"):
        earlier = later = 0
        for s in ec.SEEDS:
            a1, a0 = by.loc[("E1", s), "target_first_episode"], by.loc[(other, s), "target_first_episode"]
            a1 = np.nan if a1 is None else a1
            a0 = np.nan if a0 is None else a0
            if np.isnan(a1) and np.isnan(a0):
                continue
            if np.isnan(a0) or (not np.isnan(a1) and a1 < a0):
                earlier += 1
            elif np.isnan(a1) or a0 < a1:
                later += 1
        from math import comb
        n = earlier + later
        probs = [comb(n, i) / 2 ** n for i in range(n + 1)] if n else [1.0]
        p = min(1.0, sum(q for q in probs if q <= probs[min(earlier, later)] + 1e-15)) if n else 1.0
        stats.append({"endpoint": "descriptive: time to validation target", "comparison": f"E1 vs {other}",
                      "n_seeds": 30, "n_positive": earlier, "n_negative": later, "sign_flip_p": p,
                      "favours_E1": earlier > later})
    st = pd.DataFrame(stats)
    st.to_csv(OUT("exploration_levelA_statistics.csv"), index=False)

    # ---------------------------------------------------------- classification (pre-registered rule)
    def get(endpoint, comp):
        return st[(st.endpoint == endpoint) & (st.comparison == comp)].iloc[0]

    prim = [get("primary: validation accuracy AUC", f"E1 vs {o}") for o in ("E0", "E2", "E3")]
    prim_ok = [bool(p.mean_diff > 0 and p.holm_p < 0.05 and p.t_ci95_low > 0) for p in prim]
    sec_ok = {e: bool(get(e, "E1 vs E0").favours_E1 and get(e, "E1 vs E0").holm_p < 0.05)
              for e in ("S1: reward-rank score AUC", "S2: utility AUC", "S3: destructive features.0 frequency")}
    level_a = all(prim_ok) and sum(sec_ok.values()) >= 2
    b1 = prim_ok[0]
    b2 = all(p.mean_diff > 0 for p in prim)
    b3 = any(sec_ok.values())
    level = "A" if level_a else ("B" if (b1 or b2 or b3) else "C")
    classification = {
        "level": level,
        "primary_criteria_met": dict(zip(["E1 vs E0", "E1 vs E2", "E1 vs E3"], prim_ok)),
        "secondary_exploration_E1_vs_E0_met": sec_ok,
        "level_B_conditions": {"B1_E1_vs_E0_primary_replicates": b1, "B2_all_primary_means_positive": b2,
                               "B3_a_secondary_exploration_outcome_replicates": b3},
        "preregistration": {"commit": "c3a9484", "sha256": runs[0]["preregistration_sha256"]},
    }
    with open(OUT("exploration_levelA_classification.json"), "w", encoding="utf-8") as f:
        json.dump(classification, f, indent=1)

    # ---------------------------------------------------------- summary + figures
    summary = []
    for c in ec.CONDITIONS:
        sub = table[table.condition == c]
        row = {"condition": c, "n_seeds": len(sub)}
        for col in ["val_accuracy_auc", "rank_score_auc", "utility_auc", "destructive_f0_frequency",
                    "worst_trajectory_val_accuracy", "final_val_accuracy", "final_sparsity", "test_accuracy",
                    "excess_over_gm", "mean_f0_pruning"]:
            row[f"{col}_mean"] = sub[col].mean()
            row[f"{col}_sd"] = sub[col].std(ddof=1)
        row["target_reached"] = int(sub.target_reached.sum())
        row["target_first_episode_median_reached"] = sub.target_first_episode.dropna().median()
        row["final_val_pareto"] = int(sub.final_val_pareto.sum())
        row["final_reward_rank_median"] = sub.final_reward_rank.median()
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT("exploration_levelA_summary.csv"), index=False)

    g = traj.groupby(["condition", "episode"])
    fig_a = g.agg(timestep=("timestep", "first"), val_accuracy_mean=("val_accuracy", "mean"),
                  val_accuracy_sd=("val_accuracy", "std"),
                  val_accuracy_q25=("val_accuracy", lambda x: x.quantile(0.25)),
                  val_accuracy_q75=("val_accuracy", lambda x: x.quantile(0.75))).reset_index()
    fig_a.to_csv(OUT("exploration_levelA_fig_A_trajectory.csv"), index=False)
    fig_b = g.agg(timestep=("timestep", "first"), rank_score_mean=("rank_score", "mean"),
                  rank_score_sd=("rank_score", "std"), reward_rank_median=("reward_rank", "median")).reset_index()
    fig_b.to_csv(OUT("exploration_levelA_fig_B_rank_trajectory.csv"), index=False)
    table[["condition", "seed", "destructive_f0_frequency", "destructive_f0_count"]].to_csv(
        OUT("exploration_levelA_fig_C_destructive.csv"), index=False)
    table.pivot(index="seed", columns="condition", values="val_accuracy_auc").reset_index().to_csv(
        OUT("exploration_levelA_fig_D_auc_paired.csv"), index=False)

    pd.set_option("display.width", 250)
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    print(st[["endpoint", "comparison", "mean_diff", "median_diff", "sd_diff", "t_ci95_low", "t_ci95_high",
              "sign_flip_p", "holm_p", "cohens_dz", "n_positive", "n_negative"]].round(5).to_string(index=False))
    print(json.dumps(classification, indent=1))


if __name__ == "__main__":
    main()
