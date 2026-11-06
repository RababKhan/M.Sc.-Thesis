"""Soft Sensitivity-Prior PPO: tables, statistics, sample efficiency, landscape, figures.

Implements the analysis in results/soft_prior_preregistration.md, including
the Level A/B/C rule, computed mechanically. Validation data (the 1,296-policy
landscape) for trajectories and landscape analysis; the test set only to
evaluate frozen policies and matched baselines, for reporting.
"""
import glob
import json
import math
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import rl_env  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
LAYERS = rl_env.FILTERED_LAYERS
BOOT = 10000
T975 = [None, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060]
TARGET_VAL, TARGET_SPARSITY = 76.0, 55.0
SMOOTH = 25


# ------------------------------------------------------------------ exact tests (vectorised)
def _signs(n):
    return 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.int8)


def sign_flip_p(d):
    d = np.asarray(d, float)
    if np.allclose(d, 0):
        return 1.0
    stats = np.abs((_signs(len(d)) * d).mean(axis=1))
    return float((stats >= abs(d.mean()) - 1e-12).mean())


def wilcoxon_p(d):
    d = np.asarray([x for x in d if not np.isclose(x, 0)], float)
    if len(d) == 0:
        return 1.0
    r = pd.Series(np.abs(d)).rank().to_numpy()
    w = ((_signs(len(d)) > 0) * r).sum(axis=1)
    c = r.sum() / 2
    return float((np.abs(w - c) >= abs(r[d > 0].sum() - c) - 1e-12).mean())


def paired_sd_p(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    sw = _signs(len(x)) < 0
    a, b = np.where(sw, y, x), np.where(sw, x, y)
    stats = np.abs(a.std(axis=1, ddof=1) - b.std(axis=1, ddof=1))
    return float((stats >= abs(x.std(ddof=1) - y.std(ddof=1)) - 1e-12).mean())


def binom_p(k, n):
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return float(min(1.0, sum(p for p in probs if p <= probs[k] + 1e-15)))


def describe(d, rng):
    d = np.asarray(d, float)
    n = len(d)
    sd = d.std(ddof=1)
    boots = d[rng.integers(0, n, (BOOT, n))].mean(axis=1)
    half = T975[n - 1] * sd / np.sqrt(n)
    return {"n_pairs": n, "mean_diff": d.mean(), "median_diff": float(np.median(d)), "sd_diff": sd,
            "boot_ci95_low": float(np.percentile(boots, 2.5)), "boot_ci95_high": float(np.percentile(boots, 97.5)),
            "t_ci95_low": d.mean() - half, "t_ci95_high": d.mean() + half,
            "sign_flip_p": sign_flip_p(d), "wilcoxon_p": wilcoxon_p(d),
            "cohens_dz": d.mean() / sd if sd > 0 else float("nan"),
            "n_positive": int((d > 1e-12).sum()), "n_negative": int((d < -1e-12).sum()), "n_zero": int(np.isclose(d, 0).sum())}


def holm(p):
    p = np.asarray(p, float)
    out, run = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        out[i] = run
    return out


def entropy_bits(counter):
    n = sum(counter.values())
    return -sum(c / n * math.log2(c / n) for c in counter.values())


# ------------------------------------------------------------------ main
def main():
    runs = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(RESULTS, "soft_prior_runs", "*.json")))]
    for key in ("baseline_sha256", "preregistration_sha256", "sensitivity_values_sha256"):
        assert len({r[key] for r in runs}) == 1, f"runs disagree on {key}"
    assert all(r["timesteps_trained"] == 2048 and r["episodes"] == 512 for r in runs)
    seeds = sorted({r["seed"] for r in runs}, key=lambda s: (s != 42, s))
    L = pd.read_csv(os.path.join(RESULTS, "policy_landscape_validation.csv")).set_index("actions")
    model = common.load_baseline(os.path.join(ROOT, "checkpoints", "cnn_baseline_FIXED.pth"))
    loader = common.test_loader(root=os.path.join(ROOT, "data"))
    rng = np.random.default_rng(0)

    # validation Pareto regret: best val accuracy at >= this sparsity, minus this policy's
    order = L.sort_values("total_sparsity", ascending=False)
    best_at_or_above = order["val_accuracy"].cummax()
    regret = (best_at_or_above - order["val_accuracy"]).reindex(L.index)

    cache = {}

    def frozen(kind, key):
        if (kind, key) not in cache:
            if kind == "gm":
                pruned, _ = common.prune_global_to_total(model, key)
            elif kind == "uniform":
                pruned, _ = common.prune_uniform_to_total(model, key)
            else:
                pruned = common.prune_layerwise(model, [rl_env.ACTION_TO_PRUNE[a] for a in key])
            p, labels = common.predictions(pruned, loader)
            cache[(kind, key)] = (p, labels, common.accuracy_from(p, labels))
        return cache[(kind, key)]

    # ------------------------------------------------ per-run rows + trajectories
    rows, traj, preds_by_seed = [], [], {}
    for r in runs:
        a = str(r["actions"])
        preds = torch.tensor([int(c) for c in r["test_predictions"]])
        gm_p, labels, gm_acc = frozen("gm", r["total_sparsity"])
        un_p, _, un_acc = frozen("uniform", r["total_sparsity"])
        cur = pd.read_csv(os.path.join(RESULTS, "soft_prior_training_curves_raw", f"{r['run_id']}.csv"))
        cur["val_accuracy"] = cur["actions"].map(L["val_accuracy"])
        cur["val_loss"] = cur["actions"].map(L["val_loss"])
        cur["reward_rank"] = cur["actions"].map(L["reward_rank_0.01"])
        cur["utility"] = 1.5 * cur["val_accuracy"] / 77.32 + 0.01 * cur["sparsity"]
        hit = cur[(cur["val_accuracy"] >= TARGET_VAL) & (cur["sparsity"] >= TARGET_SPARSITY)]
        for k in ("condition", "definition", "beta", "seed"):
            cur[k] = r[k]
        cur["run_id"] = r["run_id"]
        traj.append(cur)
        row = {
            "condition": r["condition"], "definition": r["definition"], "beta": r["beta"], "seed": r["seed"],
            "prior_sensitivity": str([round(x, 6) for x in r["prior_sensitivity"]]),
            "state_sensitivity": str([round(x, 6) for x in r["state_sensitivity"]]),
            "prior_mapping": json.dumps(r["prior_mapping"]) if r["prior_mapping"] else "",
            "actions": a, "prune_percent": str(r["prune_percent"]), "total_sparsity": r["total_sparsity"],
            **{f"sparsity_{k}": v for k, v in r["per_layer_sparsity"].items()},
            "probe_accuracy": r["probe_accuracy"], "val_accuracy": r["val_accuracy"],
            "val_loss": float(L.loc[a, "val_loss"]), "test_accuracy": r["test_accuracy"],
            "accuracy_retention": r["accuracy_retention"],
            "reward_total": r["training_reward"]["total"], "reward_accuracy_term": r["training_reward"]["accuracy_term"],
            "reward_sparsity_term": r["training_reward"]["sparsity_term"],
            "reward_diversity_term": r["training_reward"]["diversity_term"],
            "global_reward_rank": int(L.loc[a, "reward_rank_0.01"]),
            "reward_gap_to_optimum": float(L["reward_0.01"].max() - L.loc[a, "reward_0.01"]),
            "val_pareto_efficient": bool(L.loc[a, "pareto_val_accuracy"]), "val_pareto_regret": float(regret.loc[a]),
            "gm_test_accuracy": gm_acc, "excess_over_gm": r["test_accuracy"] - gm_acc,
            "mcnemar_vs_gm_p": common.mcnemar(preds == labels, gm_p == labels)[3],
            "uniform_test_accuracy": un_acc, "excess_over_uniform": r["test_accuracy"] - un_acc,
            "features0_at_least_20": r["prune_percent"][0] >= 20, "features3_at_60": r["prune_percent"][1] == 60,
            "features6_at_60": r["prune_percent"][2] == 60,
            "harmful": r["prune_percent"][0] >= 20 or r["prune_percent"][1] == 60,
            "reached_target": len(hit) > 0,
            "episodes_to_target": int(hit["episode"].iloc[0]) if len(hit) else np.nan,
            "timestep_to_target": int(hit["timestep"].iloc[0]) if len(hit) else np.nan,
            "auc_val_accuracy": cur["val_accuracy"].mean(), "auc_reward_rank": cur["reward_rank"].mean(),
            "auc_utility": cur["utility"].mean(),
            "runtime_seconds": r["runtime_seconds"], "run_id": r["run_id"],
        }
        rows.append(row)
        preds_by_seed.setdefault(r["seed"], {"true_label": labels.numpy()})[r["run_id"]] = preds.numpy()
    runs_df = pd.DataFrame(rows)
    runs_df.to_csv(os.path.join(RESULTS, "soft_prior_all_runs.csv"), index=False)
    traj_df = pd.concat(traj)
    traj_df.to_csv(os.path.join(RESULTS, "soft_prior_training_curves.csv"), index=False)
    runs_df[["condition", "definition", "beta", "seed", "reached_target", "episodes_to_target", "timestep_to_target",
             "auc_val_accuracy", "auc_reward_rank", "auc_utility", "run_id"]].to_csv(
        os.path.join(RESULTS, "soft_prior_sample_efficiency.csv"), index=False)
    pdir = os.path.join(RESULTS, "soft_prior_predictions")
    os.makedirs(pdir, exist_ok=True)
    for s, cols in preds_by_seed.items():
        pd.DataFrame(cols).to_csv(os.path.join(pdir, f"seed_{s}.csv"), index=False)

    def cell(cond, d, beta):
        if cond == "P0":
            sub = runs_df[runs_df.condition == "P0"]
        else:
            sub = runs_df[(runs_df.condition == cond) & (runs_df.definition == d) & np.isclose(runs_df.beta, beta)]
        return sub.set_index("seed").loc[[s for s in seeds if s in set(sub.seed)]]

    cells = [("P0", "none", 0.0), ("P1", "loss", 0.5), ("P2", "loss", 0.5), ("P3", "loss", 0.5), ("P4", "loss", 0.0),
             ("P5", "loss", 0.5), ("P1", "loss", 1.0), ("P2", "loss", 1.0), ("P3", "loss", 1.0),
             ("P1", "accuracy", 0.5), ("P2", "accuracy", 0.5), ("P3", "accuracy", 0.5)]

    # ------------------------------------------------ summary + policy stats
    summary, pstats = [], []
    for cond, d, beta in cells:
        c = cell(cond, d, beta)
        pol = Counter(c["actions"])
        top, topn = pol.most_common(1)[0]
        reached = c[c.reached_target]
        s = {"condition": cond, "definition": d, "beta": beta, "n_seeds": len(c),
             "test_mean": c.test_accuracy.mean(), "test_sd": c.test_accuracy.std(ddof=1),
             "test_median": c.test_accuracy.median(), "test_worst": c.test_accuracy.min(), "test_best": c.test_accuracy.max(),
             "val_mean": c.val_accuracy.mean(), "sparsity_mean": c.total_sparsity.mean(),
             "sparsity_sd": c.total_sparsity.std(ddof=1), "excess_over_gm_mean": c.excess_over_gm.mean(),
             "excess_over_gm_sd": c.excess_over_gm.std(ddof=1), "excess_over_uniform_mean": c.excess_over_uniform.mean(),
             "unique_policies": len(pol), "most_common_policy": top, "most_common_share": topn / len(c),
             "policy_entropy_bits": entropy_bits(pol),
             "policy_frequency": "; ".join(f"{p} x{n}" for p, n in pol.most_common()),
             "n_features3_at_60": int(c.features3_at_60.sum()), "n_features6_at_60": int(c.features6_at_60.sum()),
             "n_features0_at_least_20": int(c.features0_at_least_20.sum()), "n_harmful": int(c.harmful.sum()),
             "global_rank_mean": c.global_reward_rank.mean(), "global_rank_median": c.global_reward_rank.median(),
             "n_val_pareto": int(c.val_pareto_efficient.sum()), "val_pareto_regret_mean": c.val_pareto_regret.mean(),
             "n_reached_target": len(reached),
             "episodes_to_target_median_reached": reached.episodes_to_target.median() if len(reached) else np.nan,
             "auc_val_accuracy_mean": c.auc_val_accuracy.mean(), "auc_reward_rank_mean": c.auc_reward_rank.mean(),
             "auc_utility_mean": c.auc_utility.mean()}
        for i, layer in enumerate(LAYERS):
            cnt = Counter(json.loads(x)[i] for x in c["prune_percent"])
            for ratio in (0, 10, 20, 30, 40, 60):
                s[f"freq_{layer}_{ratio}"] = cnt.get(float(ratio), 0)
                pstats.append({"table": "layer_action", "condition": cond, "definition": d, "beta": beta,
                               "item": f"{layer}={ratio}%", "count": cnt.get(float(ratio), 0), "share": cnt.get(float(ratio), 0) / len(c)})
        for p, n in pol.most_common():
            pstats.append({"table": "policy", "condition": cond, "definition": d, "beta": beta, "item": p,
                           "count": n, "share": n / len(c)})
        summary.append(s)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(RESULTS, "soft_prior_summary.csv"), index=False)
    pd.DataFrame(pstats).to_csv(os.path.join(RESULTS, "soft_prior_policy_stats.csv"), index=False)

    # ------------------------------------------------ statistics
    stats = []
    metrics = ["excess_over_gm", "test_accuracy", "total_sparsity", "val_accuracy", "excess_over_uniform"]

    def compare(a_cell, b_cell, family):
        A, B = cell(*a_cell), cell(*b_cell)
        common_seeds = [s for s in A.index if s in B.index]
        A, B = A.loc[common_seeds], B.loc[common_seeds]
        label = {"comparison": f"{a_cell[0]} vs {b_cell[0]}", "definition": a_cell[1], "beta": a_cell[2],
                 "family": family, "identical_policies": int((A.actions == B.actions).sum())}
        for m in metrics:
            stats.append({**label, "metric": m, **describe((A[m] - B[m]).to_numpy(float), rng),
                          "sd_a": A[m].std(ddof=1), "sd_b": B[m].std(ddof=1), "worst_a": A[m].min(), "worst_b": B[m].min()})
        return A, B

    for other in ("P0", "P2", "P3"):
        compare(("P1", "loss", 0.5), (other, "loss" if other != "P0" else "none", 0.5 if other != "P0" else 0.0),
                "primary")
    for other in ("P4", "P5"):
        compare(("P1", "loss", 0.5), (other, "loss", 0.0 if other == "P4" else 0.5), "other: state vs prior")
    for d, beta in (("loss", 1.0), ("accuracy", 0.5)):
        for other in ("P0", "P2", "P3"):
            compare(("P1", d, beta), (other, "none" if other == "P0" else d, 0.0 if other == "P0" else beta),
                    f"robustness ({d}, beta {beta})")

    # secondary family: P1 vs P0 (loss, beta 0.5)
    A, B = cell("P1", "loss", 0.5), cell("P0", "none", 0.0)
    B = B.loc[A.index]
    sec = []
    sec.append(("a: test accuracy SD (lower better)", A.test_accuracy.std(ddof=1) - B.test_accuracy.std(ddof=1),
                paired_sd_p(A.test_accuracy, B.test_accuracy), A.test_accuracy.std(ddof=1) <= B.test_accuracy.std(ddof=1)))
    oa, ob = int((A.harmful & ~B.harmful).sum()), int((B.harmful & ~A.harmful).sum())
    sec.append(("b: harmful final policies (fewer better)", int(A.harmful.sum()) - int(B.harmful.sum()),
                binom_p(min(oa, ob), oa + ob), A.harmful.sum() < B.harmful.sum()))
    d_rank = (A.global_reward_rank - B.global_reward_rank).to_numpy(float)
    sec.append(("c: final global reward rank (lower better)", d_rank.mean(), sign_flip_p(d_rank), d_rank.mean() < 0))
    earlier = later = 0
    for s in A.index:
        ea, eb = A.loc[s, "episodes_to_target"], B.loc[s, "episodes_to_target"]
        if np.isnan(ea) and np.isnan(eb):
            continue
        if np.isnan(eb) or (not np.isnan(ea) and ea < eb):
            earlier += 1
        elif np.isnan(ea) or eb < ea:
            later += 1
    sec.append(("d: time to validation target (earlier better; censoring handled)", earlier - later,
                binom_p(min(earlier, later), earlier + later), earlier > later))
    for label, col, better_high in (("e: AUC validation accuracy (higher better)", "auc_val_accuracy", True),
                                    ("f: AUC reward rank (lower better)", "auc_reward_rank", False),
                                    ("g: AUC utility (higher better)", "auc_utility", True)):
        dd = (A[col] - B[col]).to_numpy(float)
        sec.append((label, dd.mean(), sign_flip_p(dd), dd.mean() > 0 if better_high else dd.mean() < 0))
    sec_holm = holm([p for _, _, p, _ in sec])
    for (label, eff, p, fav), hp in zip(sec, sec_holm):
        stats.append({"comparison": "P1 vs P0", "definition": "loss", "beta": 0.5, "family": "secondary",
                      "metric": label, "mean_diff": eff, "sign_flip_p": p, "holm_p_secondary": hp,
                      "favours_p1": bool(fav)})
    st = pd.DataFrame(stats)
    fam = (st.family == "primary") & (st.metric == "excess_over_gm")
    st["holm_p_primary"] = np.nan
    st.loc[fam, "holm_p_primary"] = holm(st.loc[fam, "sign_flip_p"].to_numpy())
    st.to_csv(os.path.join(RESULTS, "soft_prior_statistics.csv"), index=False)

    # ------------------------------------------------ pre-registered classification
    prim = st[fam].set_index("comparison")
    crit = {c: bool(prim.loc[c, "mean_diff"] > 0 and prim.loc[c, "holm_p_primary"] < 0.05)
            for c in ("P1 vs P0", "P1 vs P2", "P1 vs P3")}
    consistent = all(prim.loc[c, "n_positive"] > prim.loc[c, "n_negative"] for c in crit)
    level_b_outcomes = []
    for (label, eff, p, fav), hp in zip(sec, sec_holm):
        if fav and hp < 0.05:
            key = label.split(":")[0]
            col = {"a": None, "b": "harmful", "c": "global_reward_rank", "d": "reached_target",
                   "e": "auc_val_accuracy", "f": "auc_reward_rank", "g": "auc_utility"}[key]
            p1 = cell("P1", "loss", 0.5)
            ok = True
            for other in ("P2", "P3"):
                o = cell(other, "loss", 0.5)
                if key == "a":
                    ok &= p1.test_accuracy.std(ddof=1) <= o.test_accuracy.std(ddof=1)
                elif key in ("b", "c", "f"):
                    ok &= p1[col].mean() <= o[col].mean()
                elif key == "d":
                    ok &= p1[col].sum() >= o[col].sum()
                else:
                    ok &= p1[col].mean() >= o[col].mean()
            if ok:
                level_b_outcomes.append(label)
    level = "A" if all(crit.values()) and consistent else ("B" if level_b_outcomes else "C")
    classification = {"criteria": crit, "consistency": consistent, "level_b_outcomes": level_b_outcomes, "level": level}
    with open(os.path.join(RESULTS, "soft_prior_classification.json"), "w", encoding="utf-8") as f:
        json.dump(classification, f, indent=1)

    # ------------------------------------------------ matched sparsity (every P1 run)
    ms = []
    for d, beta in (("loss", 0.5), ("loss", 1.0), ("accuracy", 0.5)):
        p1 = cell("P1", d, beta)
        for s in p1.index:
            r1 = p1.loc[s]
            pr1 = torch.tensor(preds_by_seed[s][r1.run_id])
            labels = torch.tensor(preds_by_seed[s]["true_label"])
            row = {"definition": d, "beta": beta, "seed": s, "p1_actions": r1.actions, "p1_sparsity": r1.total_sparsity,
                   "p1_test": r1.test_accuracy, "gm_test": r1.gm_test_accuracy, "diff_vs_gm": r1.excess_over_gm,
                   "mcnemar_vs_gm_p": r1.mcnemar_vs_gm_p, "uniform_test": r1.uniform_test_accuracy,
                   "diff_vs_uniform": r1.excess_over_uniform}
            un_p, _, _ = frozen("uniform", r1.total_sparsity)
            row["mcnemar_vs_uniform_p"] = common.mcnemar(pr1 == labels, un_p == labels)[3]
            for other in ("P0", "P2", "P3"):
                o = cell(other, "none" if other == "P0" else d, 0.0 if other == "P0" else beta).loc[s]
                po = torch.tensor(preds_by_seed[s][o.run_id])
                mc = common.mcnemar(pr1 == labels, po == labels)
                row.update({f"{other}_actions": o.actions, f"{other}_sparsity": o.total_sparsity,
                            f"{other}_test": o.test_accuracy, f"sparsity_diff_vs_{other}": r1.total_sparsity - o.total_sparsity,
                            f"test_diff_vs_{other}": r1.test_accuracy - o.test_accuracy,
                            f"comparable_sparsity_{other}": abs(r1.total_sparsity - o.total_sparsity) <= 1.0,
                            f"mcnemar_vs_{other}_p": mc[3]})
            ms.append(row)
    pd.DataFrame(ms).to_csv(os.path.join(RESULTS, "soft_prior_matched_sparsity.csv"), index=False)

    # ------------------------------------------------ landscape + figures
    land = L.reset_index()
    land["val_pareto_regret"] = land["actions"].map(regret)
    land["reward_optimal_0.01"] = land["reward_rank_0.01"] == 1
    for cond, d, beta in cells:
        cnt = Counter(cell(cond, d, beta)["actions"])
        land[f"final_{cond}_{d}_{beta:.2f}"] = land["actions"].map(lambda x: cnt.get(x, 0))
    land.to_csv(os.path.join(RESULTS, "soft_prior_policy_landscape.csv"), index=False)

    fig_a = runs_df[runs_df.condition.isin(["P0", "P1", "P2", "P3"])][
        ["condition", "definition", "beta", "seed", "total_sparsity", "test_accuracy", "gm_test_accuracy", "actions"]]
    fig_a.to_csv(os.path.join(RESULTS, "soft_prior_fig_A_accuracy_sparsity.csv"), index=False)
    traj_df["utility_smoothed"] = traj_df.groupby("run_id")["utility"].transform(lambda x: x.rolling(SMOOTH, min_periods=1).mean())
    traj_df["reward_smoothed"] = traj_df.groupby("run_id")["episode_reward"].transform(lambda x: x.rolling(SMOOTH, min_periods=1).mean())
    agg = (traj_df.groupby(["condition", "definition", "beta", "episode"])
           .agg(timestep=("timestep", "mean"), n_seeds=("seed", "nunique"),
                reward_mean=("episode_reward", "mean"), reward_sd=("episode_reward", "std"),
                reward_smoothed_mean=("reward_smoothed", "mean"), reward_smoothed_sd=("reward_smoothed", "std"),
                val_accuracy_mean=("val_accuracy", "mean"), utility_smoothed_mean=("utility_smoothed", "mean"),
                utility_smoothed_sd=("utility_smoothed", "std"), reward_rank_median=("reward_rank", "median"),
                reward_rank_q25=("reward_rank", lambda x: x.quantile(0.25)),
                reward_rank_q75=("reward_rank", lambda x: x.quantile(0.75))).reset_index())
    agg.drop(columns=["reward_rank_median", "reward_rank_q25", "reward_rank_q75"]).to_csv(
        os.path.join(RESULTS, "soft_prior_fig_B_trajectory.csv"), index=False)
    agg[["condition", "definition", "beta", "episode", "timestep", "reward_rank_median", "reward_rank_q25",
         "reward_rank_q75"]].to_csv(os.path.join(RESULTS, "soft_prior_fig_C_rank_vs_timestep.csv"), index=False)
    fig_d = runs_df[["condition", "definition", "beta", "seed", "actions", "prune_percent", "test_accuracy", "total_sparsity"]].copy()
    for i, layer in enumerate(LAYERS):
        fig_d[layer] = fig_d["prune_percent"].map(lambda x: json.loads(x)[i])
    fig_d.to_csv(os.path.join(RESULTS, "soft_prior_fig_D_policies.csv"), index=False)
    runs_df[runs_df.condition.isin(["P1", "P2", "P3"])][
        ["condition", "definition", "beta", "seed", "test_accuracy", "total_sparsity", "excess_over_gm", "actions",
         "prior_sensitivity", "prior_mapping"]].to_csv(os.path.join(RESULTS, "soft_prior_fig_E_prior_comparison.csv"), index=False)

    pd.set_option("display.width", 250)
    print(summary_df[["condition", "definition", "beta", "test_mean", "test_sd", "test_worst", "sparsity_mean",
                      "excess_over_gm_mean", "unique_policies", "most_common_policy", "n_features3_at_60",
                      "n_features6_at_60", "n_harmful", "n_reached_target"]].round(3).to_string(index=False))
    print(json.dumps(classification, indent=1))


if __name__ == "__main__":
    main()
