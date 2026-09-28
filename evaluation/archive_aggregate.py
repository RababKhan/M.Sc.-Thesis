"""Archive-confirmatory experiment: final analysis (only after all 160 runs exist).

Implements results/archive_confirmatory_preregistration.md, including the
Level A/B/C-FA rule, computed mechanically. The seed is the unit of analysis.
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import archive_confirm as ac  # noqa: E402
import common  # noqa: E402
import exploration_confirm as ec  # noqa: E402  (exact sign-flip and Wilcoxon, validated earlier)

R = ac.RESULTS
OUT = lambda n: os.path.join(R, n)  # noqa: E731
T975 = {**ec.T975, 31: 2.040, 32: 2.037, 33: 2.035, 34: 2.032, 35: 2.030, 36: 2.028, 37: 2.026, 38: 2.024, 39: 2.023}
CONTROLS = ["A0", "A2", "A3"]


def describe(d, rng):
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return {"n_seeds": n}
    sd = d.std(ddof=1)
    half = T975[n - 1] * sd / np.sqrt(n)
    boots = d[rng.integers(0, n, (10000, n))].mean(axis=1)
    return {"n_seeds": n, "mean_diff": d.mean(), "median_diff": float(np.median(d)), "sd_diff": sd,
            "t_ci95_low": d.mean() - half, "t_ci95_high": d.mean() + half,
            "boot_ci95_low": float(np.percentile(boots, 2.5)), "boot_ci95_high": float(np.percentile(boots, 97.5)),
            "cohens_dz": d.mean() / sd if sd > 0 else float("nan"),
            "sign_flip_p": ec.sign_flip_p(d), "wilcoxon_p": ec.wilcoxon_p(d),
            "wins": int((d > 1e-12).sum()), "ties": int((np.abs(d) <= 1e-12).sum()), "losses": int((d < -1e-12).sum())}


def main():
    recs = {}
    for c in ac.CONDITIONS:
        for s in ac.SEEDS:
            p = ac.run_paths(c, s)["record"]
            assert os.path.exists(p), f"missing run {c} seed {s}: the analysis requires all 160 runs"
            recs[(c, s)] = json.load(open(p, encoding="utf-8"))
    for k in ("preregistration_sha256", "split_sha256", "sensitivity_sha256", "baseline_sha256"):
        assert len({r[k] for r in recs.values()}) == 1, f"runs disagree on {k}"
    assert next(iter(recs.values()))["preregistration_sha256"] == ac.sha256(ac.PREREG)
    assert all(r["episodes"] == 512 and r["timesteps_trained"] == 2048 for r in recs.values())
    for (c, s), r in recs.items():
        assert r["frozen_policy_sha256"] == ac.sha256(ac.run_paths(c, s)["selected"]), f"frozen file changed {c} {s}"

    model = common.load_baseline(os.path.join(ac.ROOT, "checkpoints", "cnn_baseline_FIXED.pth"))
    loader = common.test_loader(root=os.path.join(ac.ROOT, "data"))
    frozen_eval = {}

    def baseline(kind, sparsity):
        if (kind, sparsity) not in frozen_eval:
            fn = common.prune_global_to_total if kind == "gm" else common.prune_uniform_to_total
            pruned, _ = fn(model, sparsity)
            p, labels = common.predictions(pruned, loader)
            frozen_eval[(kind, sparsity)] = (p, common.accuracy_from(p, labels), common.calculate_sparsity(pruned))
        return frozen_eval[(kind, sparsity)]

    preds = {k: pd.read_csv(ac.run_paths(*k)["pred"]) for k in recs}
    archives = {k: pd.read_csv(ac.run_paths(*k)["archive"]) for k in recs}
    for k, a in archives.items():
        a.insert(0, "seed", k[1])
        a.insert(0, "condition", k[0])
        a["in_band"] = (a["sparsity"] >= ac.BAND[0]) & (a["sparsity"] <= ac.BAND[1])

    # ------------------------------------------------ per-run tables
    rows, quality, conv, disc, gm_rows = [], [], [], [], []
    for (c, s), r in recs.items():
        a = archives[(c, s)]
        band = a[a.in_band].sort_values("vselect_accuracy", ascending=False)
        pr = preds[(c, s)]
        labels = torch.tensor(pr["true_label"].to_numpy())
        sel = torch.tensor(pr["selected_pred"].to_numpy())
        gm_p, gm_acc, gm_sp = baseline("gm", r["selected_sparsity"])
        un_p, un_acc, _ = baseline("uniform", r["selected_sparsity"])
        rows.append({"condition": c, "seed": s, "selected_actions": str(r["selected_actions"]),
                     "selected_sparsity": r["selected_sparsity"], "selected_vselect_accuracy": r["selected_vselect_accuracy"],
                     "selected_test_accuracy": r["selected_test_accuracy"], "target_band_reached": r["target_band_reached"],
                     "n_band_candidates": r["n_band_candidates"], "n_unique_policies": r["n_unique_policies"],
                     "terminal_actions": str(r["terminal_actions"]), "terminal_sparsity": r["terminal_sparsity"],
                     "terminal_test_accuracy": r["terminal_test_accuracy"],
                     "terminal_vselect_accuracy": r["terminal_vselect_accuracy"],
                     "gm_test_accuracy": gm_acc, "diff_vs_gm": r["selected_test_accuracy"] - gm_acc,
                     "uniform_test_accuracy": un_acc, "frozen_policy_sha256": r["frozen_policy_sha256"],
                     "prior_sensitivity": str([round(x, 6) for x in r["prior_sensitivity"]]),
                     "prior_mapping": json.dumps(r["prior_mapping"]) if r["prior_mapping"] else "",
                     "runtime_seconds": r["runtime_seconds"]})
        top = band["vselect_accuracy"].to_numpy()
        quality.append({"condition": c, "seed": s, "n_unique": len(a), "n_band": len(band),
                        "band_max_vselect": top.max() if len(top) else np.nan,
                        "band_mean_vselect": top.mean() if len(top) else np.nan,
                        "band_median_vselect": float(np.median(top)) if len(top) else np.nan,
                        "band_top5_mean_vselect": top[:5].mean() if len(top) else np.nan,
                        "band_share_ge_76_85": float((top >= 76.85).mean()) if len(top) else np.nan})
        conv.append({"condition": c, "seed": s, "terminal_actions": str(r["terminal_actions"]),
                     "terminal_sparsity": r["terminal_sparsity"], "terminal_test_accuracy": r["terminal_test_accuracy"],
                     "archive_actions": str(r["selected_actions"]), "archive_sparsity": r["selected_sparsity"],
                     "archive_test_accuracy": r["selected_test_accuracy"],
                     "archive_gain": r["selected_test_accuracy"] - r["terminal_test_accuracy"],
                     "same_policy": r["selected_actions"] == r["terminal_actions"]})
        disc.append({"condition": c, "seed": s, "selected_actions": str(r["selected_actions"]),
                     "first_episode": r["selected_first_episode"], "first_timestep": r["selected_first_timestep"],
                     "times_sampled": r["selected_times_sampled"]})
        gm_rows.append({"condition": c, "seed": s, "selected_sparsity": r["selected_sparsity"], "gm_sparsity": gm_sp,
                        "test_accuracy": r["selected_test_accuracy"], "gm_test_accuracy": gm_acc,
                        "diff": r["selected_test_accuracy"] - gm_acc,
                        "mcnemar_p": common.mcnemar(sel == labels, gm_p == labels)[3],
                        "uniform_test_accuracy": un_acc,
                        "mcnemar_vs_uniform_p": common.mcnemar(sel == labels, un_p == labels)[3]})
    runs = pd.DataFrame(rows).sort_values(["condition", "seed"]).reset_index(drop=True)
    runs.to_csv(OUT("archive_all_runs.csv"), index=False)
    runs.drop(columns=["prior_sensitivity", "prior_mapping", "runtime_seconds"]).to_csv(
        OUT("archive_selected_models.csv"), index=False)
    allarch = pd.concat(archives.values(), ignore_index=True)
    allarch.to_csv(OUT("archive_unique_policies.csv"), index=False)
    allarch.drop_duplicates("actions")[["actions", "prune_percent", "sparsity", "vselect_accuracy", "vselect_loss", "in_band"]]\
        .sort_values("vselect_accuracy", ascending=False).to_csv(OUT("archive_selection_scores.csv"), index=False)
    pd.DataFrame(quality).to_csv(OUT("archive_quality.csv"), index=False)
    pd.DataFrame(conv).to_csv(OUT("archive_conversion_analysis.csv"), index=False)
    pd.DataFrame(disc).to_csv(OUT("archive_discovery_time.csv"), index=False)
    gm = pd.DataFrame(gm_rows)
    gm_agg = pd.DataFrame([{
        "condition": c, "n": int((gm.condition == c).sum()), "mean_diff": gm[gm.condition == c]["diff"].mean(),
        "n_sig_run_better": int(((gm.condition == c) & (gm.mcnemar_p < 0.05) & (gm["diff"] > 0)).sum()),
        "n_sig_gm_better": int(((gm.condition == c) & (gm.mcnemar_p < 0.05) & (gm["diff"] < 0)).sum())}
        for c in ac.CONDITIONS])
    pd.concat([gm, gm_agg.assign(seed="aggregate")], ignore_index=True).to_csv(OUT("archive_global_magnitude.csv"), index=False)

    # ------------------------------------------------ primary + matched sparsity + McNemar
    rng = np.random.default_rng(0)
    by = runs.set_index(["condition", "seed"])
    stats, ms_rows, mc_rows, validity = [], [], [], {}
    for ctrl in CONTROLS:
        matched = [s for s in ac.SEEDS if by.loc[("A1", s), "target_band_reached"] and by.loc[(ctrl, s), "target_band_reached"]]
        dsp_all = []
        n_a1, n_ctrl, n_ns = 0, 0, 0
        for s in ac.SEEDS:
            a, b = by.loc[("A1", s)], by.loc[(ctrl, s)]
            pa, pb = preds[("A1", s)], preds[(ctrl, s)]
            lab = torch.tensor(pa["true_label"].to_numpy())
            mc = common.mcnemar(torch.tensor(pa["selected_pred"].to_numpy()) == lab,
                                torch.tensor(pb["selected_pred"].to_numpy()) == lab)
            diff = a.selected_test_accuracy - b.selected_test_accuracy
            if mc[3] < 0.05 and diff > 0:
                n_a1 += 1
            elif mc[3] < 0.05 and diff < 0:
                n_ctrl += 1
            else:
                n_ns += 1
            ms_rows.append({"comparison": f"A1 vs {ctrl}", "seed": s, "matched": s in matched,
                            "a1_sparsity": a.selected_sparsity, "control_sparsity": b.selected_sparsity,
                            "sparsity_diff": a.selected_sparsity - b.selected_sparsity,
                            "a1_test": a.selected_test_accuracy, "control_test": b.selected_test_accuracy,
                            "test_diff": diff, "a1_actions": a.selected_actions, "control_actions": b.selected_actions,
                            "identical_policy": a.selected_actions == b.selected_actions, "mcnemar_p": mc[3]})
            if s in matched:
                dsp_all.append(a.selected_sparsity - b.selected_sparsity)
        dsp = np.abs(np.array(dsp_all))
        v = {"n_matched": len(matched), "mean_abs_sparsity_diff": float(dsp.mean()) if len(dsp) else np.nan,
             "max_abs_sparsity_diff": float(dsp.max()) if len(dsp) else np.nan,
             "share_within_0_50": float((dsp <= 0.5 + 1e-12).mean()) if len(dsp) else np.nan,
             "mean_sparsity_diff": float(np.mean(dsp_all)) if dsp_all else np.nan}
        v["valid"] = bool(len(dsp) and v["mean_abs_sparsity_diff"] <= 0.25 and v["share_within_0_50"] >= 0.90)
        v["explained_by_pruning_less"] = bool(dsp_all and v["mean_sparsity_diff"] < -0.25)
        validity[ctrl] = v
        mc_rows.append({"comparison": f"A1 vs {ctrl}", "seeds_significantly_favouring_A1": n_a1,
                        "seeds_significantly_favouring_control": n_ctrl, "non_significant": n_ns})
        for label, seeds_ in (("primary (matched set)", matched), ("sensitivity (all 40 seeds)", ac.SEEDS)):
            d = [by.loc[("A1", s), "selected_test_accuracy"] - by.loc[(ctrl, s), "selected_test_accuracy"] for s in seeds_]
            row = {"analysis": label, "comparison": f"A1 vs {ctrl}", "metric": "selected test accuracy",
                   "a1_mean": np.mean([by.loc[("A1", s), "selected_test_accuracy"] for s in seeds_]),
                   "a1_sd": np.std([by.loc[("A1", s), "selected_test_accuracy"] for s in seeds_], ddof=1),
                   "control_mean": np.mean([by.loc[(ctrl, s), "selected_test_accuracy"] for s in seeds_]),
                   "control_sd": np.std([by.loc[(ctrl, s), "selected_test_accuracy"] for s in seeds_], ddof=1),
                   **describe(d, rng)}
            stats.append(row)
        dv = [by.loc[("A1", s), "selected_vselect_accuracy"] - by.loc[(ctrl, s), "selected_vselect_accuracy"] for s in matched]
        stats.append({"analysis": "V_SELECT (matched set)", "comparison": f"A1 vs {ctrl}",
                      "metric": "selected V_SELECT accuracy", **describe(dv, rng)})
        for metric, col, frame in (("archive gain (secondary)", "archive_gain", pd.DataFrame(conv)),
                                   ("selected policy first episode (secondary)", "first_episode", pd.DataFrame(disc))):
            f = frame.set_index(["condition", "seed"])
            d = [f.loc[("A1", s), col] - f.loc[(ctrl, s), col] for s in ac.SEEDS]
            stats.append({"analysis": "secondary (all 40 seeds)", "comparison": f"A1 vs {ctrl}", "metric": metric,
                          **describe(d, rng)})
    st = pd.DataFrame(stats)
    prim = st.analysis == "primary (matched set)"
    st["holm_p"] = np.nan
    st.loc[prim, "holm_p"] = ec.holm(st.loc[prim, "sign_flip_p"].fillna(1.0).to_numpy())
    st.to_csv(OUT("archive_statistics.csv"), index=False)
    ms = pd.DataFrame(ms_rows)
    ms_summary = pd.DataFrame([{"comparison": f"A1 vs {c}", **validity[c]} for c in CONTROLS])
    pd.concat([ms, ms_summary.assign(seed="summary")], ignore_index=True).to_csv(OUT("archive_matched_sparsity.csv"), index=False)
    pd.DataFrame(mc_rows).to_csv(OUT("archive_mcnemar_summary.csv"), index=False)

    # ------------------------------------------------ classification (pre-registered)
    P = st[prim].set_index("comparison")
    crit = {}
    for i, ctrl in enumerate(CONTROLS, 1):
        p = P.loc[f"A1 vs {ctrl}"]
        crit[f"C{i}"] = bool(p.get("n_seeds", 0) >= 2 and p.mean_diff > 0 and p.holm_p < 0.05 and p.t_ci95_low > 0)
    crit["C4"] = all(validity[c]["valid"] for c in CONTROLS)
    crit["C5"] = all(validity[c]["n_matched"] >= 30 for c in CONTROLS)
    less = {c: validity[c]["explained_by_pruning_less"] for c in CONTROLS}
    level_a = all(crit.values())
    b1 = crit["C1"] and not less["A0"]
    b2 = all(P.loc[f"A1 vs {c}", "mean_diff"] > 0 and P.loc[f"A1 vs {c}", "wins"] > P.loc[f"A1 vs {c}", "losses"]
             for c in CONTROLS) and not any(less.values())
    b3 = crit["C1"] and crit["C2"] and crit["C3"] and not (crit["C4"] and crit["C5"]) and not any(less.values())
    vs = st[(st.analysis == "V_SELECT (matched set)") & (st.comparison == "A1 vs A0")].iloc[0]
    b4 = (not crit["C1"] and vs.mean_diff > 0 and vs.sign_flip_p < 0.05 and P.loc["A1 vs A0", "mean_diff"] > 0
          and P.loc["A1 vs A0", "wins"] > P.loc["A1 vs A0", "losses"] and not less["A0"])
    level = "A-FA" if level_a else ("B-FA" if (b1 or b2 or b3 or b4) else "C-FA")
    classification = {"level": level, "criteria": crit, "explained_by_pruning_less": less,
                      "level_B_conditions": {"B1": bool(b1), "B2": bool(b2), "B3": bool(b3), "B4": bool(b4)},
                      "matched_sparsity_validity": validity,
                      "preregistration": {"commit": "1bda844", "sha256": next(iter(recs.values()))["preregistration_sha256"]}}
    with open(OUT("archive_classification.json"), "w", encoding="utf-8") as f:
        json.dump(classification, f, indent=1, default=float)

    # ------------------------------------------------ summary + figures
    q, cv, dt = pd.DataFrame(quality), pd.DataFrame(conv), pd.DataFrame(disc)
    summary = []
    for c in ac.CONDITIONS:
        r_ = runs[runs.condition == c]
        summary.append({"condition": c, "n_runs": len(r_), "band_reached": int(r_.target_band_reached.sum()),
                        "selected_test_mean": r_.selected_test_accuracy.mean(), "selected_test_sd": r_.selected_test_accuracy.std(ddof=1),
                        "selected_sparsity_mean": r_.selected_sparsity.mean(), "selected_sparsity_sd": r_.selected_sparsity.std(ddof=1),
                        "selected_vselect_mean": r_.selected_vselect_accuracy.mean(),
                        "terminal_test_mean": r_.terminal_test_accuracy.mean(), "terminal_sparsity_mean": r_.terminal_sparsity.mean(),
                        "archive_gain_mean": cv[cv.condition == c].archive_gain.mean(),
                        "unique_policies_mean": r_.n_unique_policies.mean(), "band_candidates_mean": r_.n_band_candidates.mean(),
                        "band_max_vselect_mean": q[q.condition == c].band_max_vselect.mean(),
                        "selected_first_episode_median": dt[dt.condition == c].first_episode.median(),
                        "diff_vs_gm_mean": r_.diff_vs_gm.mean(),
                        "selected_policy_frequency": "; ".join(f"{p} x{n}" for p, n in Counter(r_.selected_actions).most_common())})
    pd.DataFrame(summary).to_csv(OUT("archive_summary.csv"), index=False)

    cv.to_csv(OUT("archive_fig_A_terminal_vs_archive.csv"), index=False)
    fig_b = runs[["condition", "seed", "selected_sparsity", "selected_test_accuracy"]].rename(
        columns={"selected_sparsity": "sparsity", "selected_test_accuracy": "test_accuracy"}).assign(series="PPO + archive")
    gmpts = gm[["condition", "seed", "gm_sparsity", "gm_test_accuracy"]].rename(
        columns={"gm_sparsity": "sparsity", "gm_test_accuracy": "test_accuracy"}).assign(series="global magnitude at matched sparsity")
    pd.concat([fig_b, gmpts]).to_csv(OUT("archive_fig_B_accuracy_vs_sparsity.csv"), index=False)
    q[["condition", "seed", "band_max_vselect"]].to_csv(OUT("archive_fig_C_best_band_vselect.csv"), index=False)
    curves = []
    for (c, s), a in archives.items():
        b = a[a.in_band]
        best = np.full(512, np.nan)
        for _, row in b.iterrows():
            e = int(row.episode_first_seen)
            best[e - 1:] = np.fmax(best[e - 1:], row.vselect_accuracy)
        curves.append(pd.DataFrame({"condition": c, "seed": s, "episode": np.arange(1, 513), "best_band_vselect": best}))
    cur = pd.concat(curves)
    cur.groupby(["condition", "episode"]).agg(mean_best=("best_band_vselect", "mean"),
                                              share_with_band=("best_band_vselect", lambda x: x.notna().mean())).reset_index()\
        .to_csv(OUT("archive_fig_D_discovery.csv"), index=False)
    ms[ms.comparison == "A1 vs A0"][["seed", "test_diff", "sparsity_diff", "matched", "identical_policy", "mcnemar_p"]]\
        .to_csv(OUT("archive_fig_E_a1_vs_a0.csv"), index=False)

    pd.set_option("display.width", 250)
    print(pd.DataFrame(summary).drop(columns=["selected_policy_frequency"]).round(4).to_string(index=False))
    print(st.round(5).to_string(index=False))
    print(json.dumps(classification, indent=1, default=float))


if __name__ == "__main__":
    main()
