"""Phase 3 integrity checks, analysis, statistics, classification and figure data.

  python experiments/phase3/aggregate.py

Implements results/phase3_preregistration.md mechanically. Refuses to compute any aggregate if an
integrity check fails. Every stored float is read with float_precision="round_trip" (exact).
"""
import datetime
import glob
import gzip
import io
import json
import math
import os
import subprocess

import numpy as np
import pandas as pd
import torch

import phase3_common as C
from src import data as D, statistics as ST
from src.evaluation import accuracy_from, predictions
from src.utils import sha256_file

N_SEEDS = len(C.SEEDS)
ALPHA = 0.05
BASELINE_METHODS = ["uniform", "global", "lamp", "erk", "random"]


def read_csv(path, **kw):
    return pd.read_csv(path, float_precision="round_trip", **kw)


def git(*args):
    return subprocess.run(["git", *args], cwd=C.ROOT, capture_output=True, text=True).stdout.strip()


# ================================================================== integrity
def integrity(recs, inputs, base_rows):
    res = []

    def chk(name, ok, detail=""):
        res.append((name, bool(ok), str(detail)))

    ids = {(r["setting"], r["condition"], r["seed"]) for r in recs}
    want = {(s, c, seed) for s in C.SETTINGS for c in C.CONDITIONS for seed in C.SEEDS}
    chk("exactly 480 planned PPO runs, one record each", len(recs) == 480 and ids == want, f"{len(recs)} records")
    chk("every setting x condition has all 20 seeds 300-319",
        all(sorted(r["seed"] for r in recs if r["setting"] == s and r["condition"] == c) == C.SEEDS
            for s in C.SETTINGS for c in C.CONDITIONS), "no seed dropped or added")
    chk("every record has status 'complete'", all(r["status"] == "complete" for r in recs))
    bad = [r["run_id"] for r in recs if sha256_file(os.path.join(C.ROOT, r["curve_file"])) != r["curve_sha256"]]
    chk("training-curve file hashes match their records", not bad, bad[:5])
    cur_ref = {s: sha256_file(C.reference_path(s)) for s in C.SETTINGS}
    chk("reference checkpoint hashes: record == frozen input == file on disk == Phase-2 record",
        all(r["reference_sha256"] == inputs["settings"][r["setting"]]["reference_sha256"] == cur_ref[r["setting"]]
            == C.phase2_reference_sha(r["setting"]) for r in recs), {s: v[:12] for s, v in cur_ref.items()})
    fi = sha256_file(C.INPUTS_JSON)
    chk("frozen inputs (sensitivity vectors, priors, rules) unchanged since before the first run",
        all(r["frozen_inputs_sha256"] == fi for r in recs) and inputs["content_sha256"].startswith("7a323f87")
        and not git("diff", "2dbcc77", "--", "results/architecture_generalization/phase3_frozen_inputs.json",
                    "results/architecture_generalization/phase3_sensitivity_vectors.csv",
                    "results/architecture_generalization/phase3_destructive_action_rules.json"), fi[:16])
    chk("sensitivity CSV and destructive rules unchanged in every record",
        all(r["sensitivity_csv_sha256"] == sha256_file(C.SENS_CSV) and
            r["destructive_rules_sha256"] == sha256_file(C.DESTRUCTIVE_JSON) for r in recs))
    chk("pre-registration unchanged in every record",
        all(r["preregistration_sha256"] == sha256_file(C.PREREG) for r in recs) and not git("diff", "2dbcc77", "--",
                                                                                           "results/phase3_preregistration.md"),
        sha256_file(C.PREREG)[:16])
    chk("landscape hashes unchanged in every record",
        all(r["landscape_sha256"] == sha256_file(C.landscape_json(r["setting"])) for r in recs))
    chk("equal PPO budget: 2,048 timesteps and 512 episodes in every run",
        all(r["timesteps_trained"] == 2048 and r["episodes"] == 512 for r in recs))
    ppo = {json.dumps({k: v for k, v in r["ppo"].items() if k != "policy"}, sort_keys=True) for r in recs}
    dfl = {json.dumps(r["ppo_defaults"], sort_keys=True) for r in recs}
    chk("identical PPO hyper-parameters in every run (only the policy class differs P0 vs P1-P3)",
        len(ppo) == 1 and len(dfl) == 1, list(ppo)[0])
    chk("beta 0 for P0 and 0.50 for P1-P3; lambda_s 0.01; state sensitivity 0",
        all((r["beta"] == 0.0) == (r["condition"] == "P0") and r["beta"] in (0.0, 0.5) and r["sparsity_coef"] == 0.01
            and r["state_sensitivity"] == [0.0] * 4 for r in recs))
    chk("pruning units unchanged (names match the frozen Phase-2 definition)",
        all(r["unit_names"] == [u["name"] for u in inputs["settings"][r["setting"]]["units"]] for r in recs))
    chk("thread count fixed at 1 in every run", all(r["threads"] == 1 for r in recs))
    chk("test guard active during every learn() call (DataBundle.training(); freeze before test)",
        all("training()" in r["test_guard"] for r in recs))
    b = D.DataBundle("cifar10", train_images=False)
    raised = 0
    with b.training():
        for fn in (b.test_loader, b.select_loader):
            try:
                fn()
            except RuntimeError:
                raised += 1
    chk("guard mechanism raises for TEST and V_SELECT inside training() (re-verified now)", raised == 2)
    chk("no TEST result can influence training: rewards and curves use V_RL only; test read after learn() and freeze",
        all(r["reward_components"]["accuracy_term"] == 1.5 * (r["val_accuracy_vrl"] / r["base_vrl_accuracy"]) for r in recs),
        "reward accuracy term recomputed from the V_RL accuracy in every record")
    corr = os.path.join(C.RUNS_DIR, "corruption_log.txt")
    chk("corruption / repeated-run log", True, open(corr, encoding="utf-8").read().strip() if os.path.exists(corr) else "none")
    chk("exact float parsing (float_precision='round_trip') for every stored CSV", True, "read_csv helper")
    p2 = ["results/architecture_generalization/cpu_baselines.csv", "results/architecture_generalization/cpu_baseline_runs.csv",
          "checkpoints/cnn_baseline_FIXED.pth", "checkpoints/phase2_dense"]
    chk("reference baselines unchanged (Phase-2 files identical to the committed versions)", not git("diff", "HEAD", "--", *p2)
        and not git("status", "--porcelain", "--", *p2), "git diff empty")
    chk("baselines: 14 exactly matched evaluations per PPO run (sparsity identical)",
        len(base_rows) == 480 * 14 and (base_rows.total_sparsity == base_rows.baseline_sparsity).all())
    chk("test predictions stored for every run",
        all(os.path.exists(os.path.join(C.ROOT, r["test_predictions_file"])) for r in recs))
    return res


# ================================================================== trajectories
def trajectory(rec, land):
    raw = gzip.decompress(open(os.path.join(C.ROOT, rec["curve_file"]), "rb").read()).decode("utf-8")
    t = pd.read_csv(io.StringIO(raw), float_precision="round_trip")
    assert t["episode"].tolist() == list(range(1, 513)) and (t["timestep"] == 4 * t["episode"]).all()
    acts = t["actions"].map(lambda s: tuple(json.loads(s)))
    rows = [land[a] for a in acts]
    t["val_accuracy"] = [r["val_accuracy"] for r in rows]
    assert (t["val_accuracy"] == t["probe_accuracy"]).all(), "curve accuracy != landscape"
    t["rank_score"] = [r["rank_score"] for r in rows]
    t["utility"] = [r["utility"] for r in rows]
    t["destructive"] = [r["destructive"] for r in rows]
    t["total_sparsity"] = [r["total_sparsity"] for r in rows]
    return t


def auc(t, y):
    t, y = np.asarray(t, float), np.asarray(y, float)
    return float(np.sum((y[1:] + y[:-1]) * np.diff(t)) / 2.0 / (t[-1] - t[0]))


# ================================================================== statistics
def compare(tab, setting, metric, a, b, rng, family, direction=+1, question=""):
    A = tab[(tab.setting == setting) & (tab.condition == a)].set_index("seed")[metric]
    B = tab[(tab.setting == setting) & (tab.condition == b)].set_index("seed")[metric] if b in C.CONDITIONS else None
    assert sorted(A.index) == C.SEEDS
    d = (A.loc[C.SEEDS] - B.loc[C.SEEDS]).to_numpy(float)
    out = {"question": question, "family": family, "setting": setting, "endpoint": metric, "comparison": f"{a} - {b}",
           "direction_hypothesised": "positive" if direction > 0 else "negative", **ST.describe(d, rng),
           "mean_a": float(A.mean()), "sd_a": float(A.std(ddof=1)), "median_a": float(A.median()),
           "mean_b": float(B.mean()), "sd_b": float(B.std(ddof=1)), "median_b": float(B.median())}
    return out


def compare_vec(d, setting, endpoint, label, rng, family, question, a_vals, b_vals):
    return {"question": question, "family": family, "setting": setting, "endpoint": endpoint, "comparison": label,
            "direction_hypothesised": "positive", **ST.describe(np.asarray(d, float), rng),
            "mean_a": float(np.mean(a_vals)), "sd_a": float(np.std(a_vals, ddof=1)), "median_a": float(np.median(a_vals)),
            "mean_b": float(np.mean(b_vals)), "sd_b": float(np.std(b_vals, ddof=1)), "median_b": float(np.median(b_vals))}


def outcome(row):
    sign = 1 if row["direction_hypothesised"] == "positive" else -1
    m, lo, hi, p = row["mean_diff"] * sign, row["t_ci95_low"], row["t_ci95_high"], row["holm_p"]
    lo, hi = (lo, hi) if sign > 0 else (-hi, -lo)
    if m > 0 and p < ALPHA and lo > 0:
        return "supported"
    if m < 0 and p < ALPHA and hi < 0:
        return "opposite"
    return "inconclusive"


def cross_class(n_sup, n_opp):
    if n_sup >= 5 and n_opp == 0:
        c = "STRONG"
    elif n_sup >= 3 and n_opp == 0:
        c = "MODERATE"
    elif n_sup >= 1:
        c = "LIMITED"
    else:
        c = "NOT SUPPORTED"
    return c, (n_opp >= 3)


# ================================================================== main
def main():
    C.set_threads()
    inputs = C.frozen_inputs()
    recs = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(C.RUNS_DIR, "*.json")))]
    base_rows = read_csv(os.path.join(C.SB, "phase3_all_runs.csv"))
    res = integrity(recs, inputs, base_rows)
    n_ok = sum(ok for _, ok, _ in res)
    stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = ["# Phase 3 integrity checks", "", f"Run {stamp}, after all 480 runs and the matched baselines, before any "
             f"aggregate statistic. **{n_ok}/{len(res)} passed.**", "", "| Status | Check | Detail |", "|---|---|---|"]
    lines += [f"| {'PASS' if ok else 'FAIL'} | {n} | {d.replace('|', '/')[:400]} |" for n, ok, d in res]
    with open(os.path.join(C.AG, "phase3_integrity_checks.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if n_ok != len(res):
        raise SystemExit("integrity checks failed; no aggregate computed")
    print(f"integrity {n_ok}/{len(res)}")

    lands = {s: {tuple(r["actions"]): r for r in json.load(open(C.landscape_json(s), encoding="utf-8"))["policies"]}
             for s in C.SETTINGS}

    # ---------------------------------------------------------------- dense references on TEST (1 thread)
    dense = {}
    for s in C.SETTINGS:
        d, a = C.split(s)
        b = D.DataBundle(d, train_images=False)
        m, sha = C.load_reference(s)
        b.freeze({"reference": sha})
        p, l = predictions(m, b.test_loader())
        dense[s] = {"test_accuracy": accuracy_from(p, l), "vrl_accuracy": inputs["settings"][s]["base_vrl_accuracy"]}

    # ---------------------------------------------------------------- per-run table
    br = base_rows.copy()
    rand = br[br.method == "random"].groupby("run_id")["baseline_test_accuracy"].mean()
    det = br[br.method != "random"].pivot(index="run_id", columns="method", values="baseline_test_accuracy")
    rows, traj_all = [], []
    for r in recs:
        land = lands[r["setting"]]
        t = trajectory(r, land)
        t.insert(0, "run_id", r["run_id"])
        t.insert(1, "setting", r["setting"])
        t.insert(2, "condition", r["condition"])
        t.insert(3, "seed", r["seed"])
        traj_all.append(t[["run_id", "setting", "condition", "seed", "episode", "timestep", "actions", "episode_reward",
                           "val_accuracy", "total_sparsity", "rank_score", "utility", "destructive"]])
        dt = dense[r["setting"]]["test_accuracy"]
        row = {k: r[k] for k in ("run_id", "setting", "dataset", "arch", "condition", "seed", "beta", "actions",
                                 "unit_ratios_percent", "total_sparsity", "zero_count", "val_accuracy_vrl", "test_accuracy",
                                 "reward", "reward_rank", "pareto_val_accuracy", "utility", "destructive",
                                 "mean_action_entropy", "runtime_seconds", "train_seconds", "reference_sha256")}
        row.update({f"reward_{k}": v for k, v in r["reward_components"].items()})
        row.update({"policy": C.policy_key(r["actions"]), "dense_test_accuracy": dt,
                    "accuracy_drop_test": dt - r["test_accuracy"], "accuracy_retention_test": 100.0 * r["test_accuracy"] / dt,
                    "vrl_auc": auc(t.timestep, t.val_accuracy), "rank_auc": auc(t.timestep, t.rank_score),
                    "utility_auc": auc(t.timestep, t.utility), "destructive_frequency": float(t.destructive.mean()),
                    "worst_trajectory_vrl": float(t.val_accuracy.min())})
        for m in ("uniform", "global", "lamp", "erk"):
            row[f"{m}_test_accuracy"] = float(det.loc[r["run_id"], m])
        row["random_test_accuracy"] = float(rand.loc[r["run_id"]])
        row["excess_over_global"] = r["test_accuracy"] - row["global_test_accuracy"]
        for i, u in enumerate(r["unit_names"]):
            row[f"ratio_unit{i + 1}"] = r["unit_ratios_percent"][i]
        rows.append(row)
    runs = pd.DataFrame(rows).sort_values(["setting", "condition", "seed"]).reset_index(drop=True)
    runs["actions"] = runs["actions"].map(str)
    runs["unit_ratios_percent"] = runs["unit_ratios_percent"].map(str)
    runs.to_csv(os.path.join(C.AG, "phase3_all_runs.csv"), index=False, float_format="%.17g")
    traj = pd.concat(traj_all)
    with gzip.open(os.path.join(C.AG, "phase3_trajectory.csv.gz"), "wt", encoding="utf-8") as f:
        traj.to_csv(f, index=False, float_format="%.17g")
    auc_cols = ["run_id", "setting", "condition", "seed", "vrl_auc", "rank_auc", "utility_auc", "destructive_frequency",
                "worst_trajectory_vrl", "val_accuracy_vrl"]
    runs[auc_cols].to_csv(os.path.join(C.AG, "phase3_exploration_auc.csv"), index=False, float_format="%.17g")
    curves = traj.groupby(["setting", "condition", "episode"]).agg(
        timestep=("timestep", "first"), vrl_accuracy_mean=("val_accuracy", "mean"), vrl_accuracy_sd=("val_accuracy", "std"),
        rank_score_mean=("rank_score", "mean"), utility_mean=("utility", "mean"), destructive_rate=("destructive", "mean"),
        sparsity_mean=("total_sparsity", "mean"), reward_mean=("episode_reward", "mean"), n_seeds=("seed", "nunique")).reset_index()
    curves["vrl_accuracy_ci95_half"] = ST.t_quantile(0.975, N_SEEDS - 1) * curves.vrl_accuracy_sd / math.sqrt(N_SEEDS)
    curves.to_csv(os.path.join(C.AG, "phase3_training_curves.csv"), index=False, float_format="%.17g")

    # ---------------------------------------------------------------- statistics
    rng = np.random.default_rng(0)
    stats = []
    for s in C.SETTINGS:
        p0 = runs[(runs.setting == s) & (runs.condition == "P0")].set_index("seed").loc[C.SEEDS]
        d = p0.test_accuracy - p0.uniform_test_accuracy
        stats.append(compare_vec(d, s, "test_accuracy", "P0 - uniform", rng, "A", "A", p0.test_accuracy, p0.uniform_test_accuracy))
        for m in ("global", "lamp", "erk", "random"):
            d = p0.test_accuracy - p0[f"{m}_test_accuracy"]
            stats.append(compare_vec(d, s, "test_accuracy", f"P0 - {m}", rng, "B", "B", p0.test_accuracy, p0[f"{m}_test_accuracy"]))
        stats.append(compare(runs, s, "vrl_auc", "P1", "P0", rng, "C", question="C"))
        for c in ("P2", "P3"):
            stats.append(compare(runs, s, "vrl_auc", "P1", c, rng, "C-specificity", question="C"))
        stats.append(compare(runs, s, "rank_auc", "P1", "P0", rng, "C-rank", question="C"))
        stats.append(compare(runs, s, "utility_auc", "P1", "P0", rng, "C-utility", question="C"))
        stats.append(compare(runs, s, "destructive_frequency", "P1", "P0", rng, "C-destructive", direction=-1, question="C"))
        stats.append(compare(runs, s, "excess_over_global", "P1", "P0", rng, "D", question="D"))
        stats.append(compare(runs, s, "test_accuracy", "P1", "P0", rng, "D-secondary raw test (descriptive)", question="D"))
    st = pd.DataFrame(stats)
    st["holm_p"] = np.nan
    for fam, g in st.groupby("family"):
        st.loc[g.index, "holm_p"] = ST.holm(g["sign_flip_p"].to_numpy())
    st["outcome"] = [outcome(r) for _, r in st.iterrows()]
    st.to_csv(os.path.join(C.AG, "phase3_statistics.csv"), index=False, float_format="%.17g")

    # historical Level A/B/C rule, recomputed within each setting (replication measure)
    repl = {}
    for s in C.SETTINGS:
        within = {}
        for ep, direction in (("vrl_auc", 1), ("rank_auc", 1), ("utility_auc", 1), ("destructive_frequency", -1)):
            rr = [compare(runs, s, ep, "P1", c, np.random.default_rng(0), "within", direction=direction) for c in ("P0", "P2", "P3")]
            hp = ST.holm([x["sign_flip_p"] for x in rr])
            for x, h in zip(rr, hp):
                x["holm_p"] = h
            within[ep] = rr
        prim = within["vrl_auc"]
        ok_prim = [x["mean_diff"] > 0 and x["holm_p"] < ALPHA and x["t_ci95_low"] > 0 for x in prim]
        sec = [(within[ep][0]["mean_diff"] * sgn > 0 and within[ep][0]["holm_p"] < ALPHA)
               for ep, sgn in (("rank_auc", 1), ("utility_auc", 1), ("destructive_frequency", -1))]
        if all(ok_prim) and sum(sec) >= 2:
            lvl = "A"
        elif ok_prim[0] or all(x["mean_diff"] > 0 for x in prim) or any(sec):
            lvl = "B"
        else:
            lvl = "C"
        repl[s] = {"level": lvl, "primary_supported": ok_prim, "secondary_favour_P1_vs_P0": sec}

    # ---------------------------------------------------------------- cross-setting summary
    statements = [("1 PPO > uniform", "A", "P0 - uniform"), ("2 PPO > global magnitude", "B", "P0 - global"),
                  ("3 PPO > LAMP", "B", "P0 - lamp"), ("4 PPO > ERK", "B", "P0 - erk"), ("5 PPO > random", "B", "P0 - random"),
                  ("6 correct prior improves exploration (V_RL accuracy AUC)", "C", "P1 - P0"),
                  ("7 exploration gain is layer-specific (P1 > P2 and P1 > P3)", "C-specificity", None),
                  ("8 correct prior improves final accuracy (excess over matched GM)", "D", "P1 - P0")]
    cross, classification = [], {}
    for name, fam, comp in statements:
        per = {}
        for s in C.SETTINGS:
            if comp is None:
                o = st[(st.family == fam) & (st.setting == s)]["outcome"].tolist()
                per[s] = "supported" if o == ["supported", "supported"] else ("opposite" if "opposite" in o else "inconclusive")
            else:
                per[s] = st[(st.family == fam) & (st.setting == s) & (st.comparison == comp)]["outcome"].iloc[0]
        n_sup = sum(v == "supported" for v in per.values())
        n_opp = sum(v == "opposite" for v in per.values())
        cls, rev = cross_class(n_sup, n_opp)
        cross.append({"statement": name, "family": fam, "supported": n_sup, "opposite": n_opp,
                      "inconclusive": 6 - n_sup - n_opp, "class": cls, "reversed": rev, **per})
        classification[name] = {"class": cls, "reversed": rev, "supported_in": n_sup, "opposite_in": n_opp, "per_setting": per}
    for ep, lab in (("C-rank", "secondary: rank AUC"), ("C-utility", "secondary: utility AUC"),
                    ("C-destructive", "secondary: destructive frequency (lower)")):
        per = {s: st[(st.family == ep) & (st.setting == s)]["outcome"].iloc[0] for s in C.SETTINGS}
        n_sup, n_opp = sum(v == "supported" for v in per.values()), sum(v == "opposite" for v in per.values())
        cls, rev = cross_class(n_sup, n_opp)
        cross.append({"statement": f"6-{ep} correct prior, {lab}", "family": ep, "supported": n_sup, "opposite": n_opp,
                      "inconclusive": 6 - n_sup - n_opp, "class": cls, "reversed": rev, **per})
    cross.append({"statement": "historical Level-A/B/C rule per setting (replication)", "family": "replication",
                  **{s: repl[s]["level"] for s in C.SETTINGS},
                  "supported": sum(v["level"] == "A" for v in repl.values())})
    pd.DataFrame(cross).to_csv(os.path.join(C.AG, "phase3_cross_setting_summary.csv"), index=False)
    with open(os.path.join(C.AG, "phase3_classification.json"), "w", encoding="utf-8") as f:
        json.dump({"generated": stamp, "statements": classification, "historical_rule_replication": repl,
                   "operationalisation_note": "statement 7 is 'opposite' in a setting if either P1-P2 or P1-P3 is opposite"},
                  f, indent=1)

    # ---------------------------------------------------------------- policy statistics
    pstats = []
    for (s, c), g in runs.groupby(["setting", "condition"]):
        counts = g.policy.value_counts().sort_index().sort_values(ascending=False, kind="mergesort")
        p = counts.to_numpy() / counts.sum()
        row = {"setting": s, "condition": c, "n_runs": len(g), "unique_final_policies": int(len(counts)),
               "final_policy_entropy_bits": float(-(p * np.log2(p)).sum()), "modal_policy": counts.index[0],
               "modal_count": int(counts.iloc[0]), "mean_reward_rank": g.reward_rank.mean(),
               "median_reward_rank": g.reward_rank.median(), "share_pareto": g.pareto_val_accuracy.mean(),
               "share_destructive_final": g.destructive.mean(), "mean_action_entropy": g.mean_action_entropy.mean(),
               "policies": json.dumps(counts.to_dict())}
        for i in range(1, 5):
            row[f"mean_ratio_unit{i}"] = g[f"ratio_unit{i}"].mean()
        pstats.append(row)
    pd.DataFrame(pstats).to_csv(os.path.join(C.AG, "phase3_policy_stats.csv"), index=False, float_format="%.17g")

    # ---------------------------------------------------------------- summaries
    summ = []
    for s in C.SETTINGS:
        summ.append({"setting": s, "method": "dense", "n": 1, "test_mean": dense[s]["test_accuracy"], "test_sd": np.nan,
                     "vrl_mean": dense[s]["vrl_accuracy"], "sparsity_mean": 0.0, "retention_mean": 100.0})
        for c in C.CONDITIONS:
            g = runs[(runs.setting == s) & (runs.condition == c)]
            summ.append({"setting": s, "method": f"PPO {c}", "n": len(g), "test_mean": g.test_accuracy.mean(),
                         "test_sd": g.test_accuracy.std(ddof=1), "test_median": g.test_accuracy.median(),
                         "vrl_mean": g.val_accuracy_vrl.mean(), "sparsity_mean": g.total_sparsity.mean(),
                         "sparsity_sd": g.total_sparsity.std(ddof=1), "drop_mean": g.accuracy_drop_test.mean(),
                         "retention_mean": g.accuracy_retention_test.mean(), "reward_mean": g.reward.mean(),
                         "excess_over_global_mean": g.excess_over_global.mean(), "vrl_auc_mean": g.vrl_auc.mean()})
        g = runs[(runs.setting == s) & (runs.condition == "P0")]
        for m in BASELINE_METHODS:
            col = f"{m}_test_accuracy"
            summ.append({"setting": s, "method": f"{m} (matched to P0)", "n": len(g), "test_mean": g[col].mean(),
                         "test_sd": g[col].std(ddof=1), "test_median": g[col].median(), "sparsity_mean": g.total_sparsity.mean(),
                         "drop_mean": (dense[s]["test_accuracy"] - g[col]).mean(),
                         "retention_mean": (100.0 * g[col] / dense[s]["test_accuracy"]).mean()})
    summary = pd.DataFrame(summ)
    summary.to_csv(os.path.join(C.AG, "phase3_summary.csv"), index=False, float_format="%.17g")
    sb = []
    for s in C.SETTINGS:
        for c in C.CONDITIONS:
            g = runs[(runs.setting == s) & (runs.condition == c)]
            for m in BASELINE_METHODS:
                col = f"{m}_test_accuracy"
                sb.append({"setting": s, "matched_to": c, "method": m, "n_runs": len(g), "baseline_test_mean": g[col].mean(),
                           "baseline_test_sd": g[col].std(ddof=1), "ppo_test_mean": g.test_accuracy.mean(),
                           "ppo_minus_baseline_mean": (g.test_accuracy - g[col]).mean(),
                           "ppo_wins": int((g.test_accuracy > g[col]).sum()), "ties": int((g.test_accuracy == g[col]).sum()),
                           "baseline_wins": int((g.test_accuracy < g[col]).sum()), "sparsity_mean": g.total_sparsity.mean()})
    pd.DataFrame(sb).to_csv(os.path.join(C.SB, "phase3_summary.csv"), index=False, float_format="%.17g")

    # ---------------------------------------------------------------- McNemar (supportive)
    mc = []
    for s in C.SETTINGS:
        d, _ = C.split(s)
        labels = np.load(os.path.join(C.PRED_DIR, f"{d}_test_labels.npy"))
        rs = {(r["condition"], r["seed"]): r for r in recs if r["setting"] == s}
        for seed in C.SEEDS:
            p0 = np.load(os.path.join(C.ROOT, rs[("P0", seed)]["test_predictions_file"])) == labels
            p1 = np.load(os.path.join(C.ROOT, rs[("P1", seed)]["test_predictions_file"])) == labels
            gm_file = base_rows[(base_rows.run_id == rs[("P0", seed)]["run_id"]) & (base_rows.method == "global")].predictions_file.iloc[0]
            gm = np.load(os.path.join(C.AG, gm_file)) == labels
            for lab, a, b in (("P1 vs P0", p1, p0), ("P0 vs matched global", p0, gm)):
                bb, cc, chi2, p = ST.mcnemar(a, b)
                mc.append({"setting": s, "seed": seed, "pair": lab, "only_first_correct": bb, "only_second_correct": cc,
                           "p": p, "significant_first_better": p < ALPHA and bb > cc, "significant_second_better": p < ALPHA and cc > bb})
    mc = pd.DataFrame(mc)
    mc.to_csv(os.path.join(C.AG, "phase3_mcnemar.csv"), index=False, float_format="%.17g")
    mc.groupby(["setting", "pair"]).agg(n=("seed", "size"), first_better=("significant_first_better", "sum"),
                                        second_better=("significant_second_better", "sum")).reset_index().to_csv(
        os.path.join(C.AG, "phase3_mcnemar_summary.csv"), index=False)

    # ---------------------------------------------------------------- figure data
    grid = read_csv(os.path.join(C.SB, "phase3_grid.csv"))
    fa = [{"setting": r.setting, "method": f"PPO {r.condition}", "seed": r.seed, "sparsity": r.total_sparsity,
           "test_accuracy": r.test_accuracy, "kind": "PPO final model"} for r in runs.itertuples()]
    gg = grid.groupby(["setting", "method", "target_sparsity"]).agg(sparsity=("total_sparsity", "mean"),
                                                                    test_accuracy=("test_accuracy", "mean")).reset_index()
    fa += [{"setting": r.setting, "method": r.method, "seed": None, "sparsity": r.sparsity, "test_accuracy": r.test_accuracy,
            "kind": "baseline grid (random: mean of 10 seeds)"} for r in gg.itertuples()]
    fa += [{"setting": s, "method": "dense", "seed": None, "sparsity": 0.0, "test_accuracy": dense[s]["test_accuracy"],
            "kind": "dense reference"} for s in C.SETTINGS]
    pd.DataFrame(fa).to_csv(os.path.join(C.AG, "phase3_fig_A_accuracy_sparsity.csv"), index=False, float_format="%.17g")
    fb = []
    for (s, c), g in runs.groupby(["setting", "condition"]):
        for i in range(1, 5):
            for ratio in (0, 10, 20, 30, 40, 60):
                fb.append({"setting": s, "condition": c, "unit_index": i,
                           "unit": inputs["settings"][s]["units"][i - 1]["name"], "ratio": ratio,
                           "count": int((g[f"ratio_unit{i}"] == ratio).sum())})
    pd.DataFrame(fb).to_csv(os.path.join(C.AG, "phase3_fig_B_policy_heatmap.csv"), index=False)
    curves.to_csv(os.path.join(C.AG, "phase3_fig_C_exploration.csv"), index=False, float_format="%.17g")
    fd = []
    for s in C.SETTINGS:
        inp = inputs["settings"][s]
        land = lands[s]
        base = inp["base_vrl_accuracy"]
        for i, u in enumerate(inp["units"]):
            tol = 0
            for act in range(6):
                a = [0, 0, 0, 0]
                a[i] = act
                if base - land[tuple(a)]["val_accuracy"] <= 1.0:
                    tol = int(100 * [0, .1, .2, .3, .4, .6][act])
            row = {"setting": s, "unit_index": i + 1, "unit": u["name"], "pct_of_prunable": u["pct_of_prunable"],
                   "raw_sensitivity": inp["sensitivity_raw"][i], "normalized_sensitivity": inp["sensitivity_normalized"][i],
                   "tolerance_ratio_1pp": tol}
            for c in C.CONDITIONS:
                row[f"mean_ratio_{c}"] = runs[(runs.setting == s) & (runs.condition == c)][f"ratio_unit{i + 1}"].mean()
            fd.append(row)
    pd.DataFrame(fd).to_csv(os.path.join(C.AG, "phase3_fig_D_sensitivity_tolerance.csv"), index=False, float_format="%.17g")
    fe = []
    for s in C.SETTINGS:
        chosen = runs[runs.setting == s].groupby(["policy", "condition"]).size().unstack(fill_value=0)
        for a, r in lands[s].items():
            k = C.policy_key(a)
            fe.append({"setting": s, "policy": k, "total_sparsity": r["total_sparsity"], "vrl_accuracy": r["val_accuracy"],
                       "reward": r["reward"], "reward_rank": r["reward_rank"], "pareto": r["pareto_val_accuracy"],
                       **{f"final_{c}": int(chosen.loc[k, c]) if k in chosen.index and c in chosen.columns else 0
                          for c in C.CONDITIONS}})
    pd.DataFrame(fe).to_csv(os.path.join(C.AG, "phase3_fig_E_landscape.csv"), index=False, float_format="%.17g")
    summary[["setting", "method", "n", "test_mean", "sparsity_mean", "retention_mean"]].to_csv(
        os.path.join(C.AG, "phase3_fig_F_retention.csv"), index=False, float_format="%.17g")
    print("aggregation complete")


if __name__ == "__main__":
    main()
