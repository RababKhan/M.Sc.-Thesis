"""Phase 5 integrity checks, metrics, statistics, criteria A-F and figure data (phase5_preregistration.md).

  python experiments/phase5/aggregate.py
Refuses to compute any aggregate if an integrity check fails. Stored floats are read with round-trip parsing.
"""
import datetime
import gzip
import io
import json
import os
import subprocess

import numpy as np
import pandas as pd

import phase5_lib as L
from src import statistics as ST
from src.utils import environment, sha256_file

OUT = L.RESULTS
ARCHIVE = os.path.join(OUT, "phase5_archive.json")
EVALS = os.path.join(OUT, "phase5_test_evaluations.csv")
RESNETS = ["cifar10_resnet8", "cifar100_resnet8"]


def git(*a):
    return subprocess.run(["git", *a], cwd=L.ROOT, capture_output=True, text=True).stdout.strip()


def auc(t, y):
    t, y = np.asarray(t, float), np.asarray(y, float)
    return float(np.sum((y[1:] + y[:-1]) * np.diff(t)) / 2.0 / (t[-1] - t[0]))


def outcome(row, alpha=0.05):
    sign = 1 if row["direction"] == "positive" else -1
    m = row["mean_diff"] * sign
    lo, hi = (row["t_ci95_low"], row["t_ci95_high"]) if sign > 0 else (-row["t_ci95_high"], -row["t_ci95_low"])
    if m > 0 and row["holm_p"] < alpha and lo > 0:
        return "supported"
    if m < 0 and row["holm_p"] < alpha and hi < 0:
        return "opposite"
    return "inconclusive"


def cross_class(ns, no):
    if ns >= 5 and no == 0:
        return "STRONG"
    if ns >= 3 and no == 0:
        return "MODERATE"
    return "LIMITED" if ns >= 1 else "NOT SUPPORTED"


# ------------------------------------------------------------------ integrity
def integrity(cfg, recs, fr, arc, ev):
    res = []

    def chk(n, ok, d=""):
        res.append((n, bool(ok), str(d)))
    want = {(s, c, x) for s in L.SETTINGS for c in L.CONDITIONS for x in L.SEEDS}
    got = {(r["setting"], r["condition"], r["seed"]) for r in recs}
    chk("all 360 expected runs completed; no extra run; no seed dropped", got == want and len(recs) == 360
        and all(r["status"] == "complete" for r in recs), f"{len(recs)} records")
    chk("2,048 timesteps / 512 episodes in every run; 1 thread", all(r["timesteps_trained"] == 2048 and r["episodes"] == 512
                                                                       and r["threads"] == 1 for r in recs))
    chk("curve hashes match", all(sha256_file(os.path.join(L.ROOT, r["curve_file"])) == r["curve_sha256"] for r in recs))
    chk("TEST and VAL-SELECT never accessed by any training run (record flags; guards raise inside training())",
        all(not r["test_set_accessed"] and not r["val_select_accessed"] for r in recs))
    chk("pre-registration, design commitments and config unchanged in every record and on disk",
        all(r["preregistration_sha256"] == sha256_file(L.PREREG) and r["config_sha256"] == sha256_file(L.CONFIG)
            and r["design_commitments_sha256"] == sha256_file(L.COMMITMENTS) for r in recs))
    chk("archive frozen from VAL-RL curves before the VAL-SELECT selection (archive hash in the final-policy file; "
        "file order)", fr["archive_sha256"] == sha256_file(ARCHIVE) and os.path.getmtime(ARCHIVE) <= os.path.getmtime(L.FREEZE))
    chk("final-policy list frozen before any TEST evaluation (freeze file older than every evaluation part)",
        all(os.path.getmtime(L.FREEZE) <= os.path.getmtime(os.path.join(OUT, "_final_eval_parts", p))
            for p in os.listdir(os.path.join(OUT, "_final_eval_parts"))))
    chk("tau, reward constants and seeds unchanged (config vs pre-registration values)",
        cfg["tau"] == arc["tau"] == fr["tau"] and cfg["constants"]["EPS"] == L.EPS and cfg["constants"]["KAPPA"] == L.KAPPA
        and cfg["seeds"] == L.SEEDS)
    chk("Phase-3 and Phase-4 files unchanged (git diff vs 67f4270 / a3653fd empty)",
        not git("diff", "67f4270", "--", "results/architecture_generalization", "results/stronger_baselines",
                "checkpoints/phase3_agents", "experiments/phase3") and
        not git("diff", "a3653fd", "--", "results/phase4", "checkpoints/phase4_agents", "experiments/phase4"))
    chk("dense checkpoints unchanged", all(L.P3.load_reference(s)[1] == cfg["settings"][s]["reference_sha256"] for s in L.SETTINGS))
    chk("Phase-5 split files and landscapes unchanged",
        cfg["split_manifest_sha256"] == sha256_file(os.path.join(L.SPLITS, "phase5_split_manifest.json"))
        and all(sha256_file(os.path.join(L.LAND, f"{s}_landscape.json")) == cfg["settings"][s]["landscape_sha256"] for s in L.SETTINGS))
    b = ev[ev.kind == "baseline"]
    pol = ev[ev.kind == "policy"].set_index(["setting", "policy"])
    chk("every LAMP / global baseline has exactly the selected policy's zero count (sparsity mismatch 0)",
        all(b.zero_count.to_numpy() == pol.loc[list(zip(b.setting, b.policy)), "zero_count"].to_numpy())
        and all(b.total_sparsity.to_numpy() == pol.loc[list(zip(b.setting, b.policy)), "total_sparsity"].to_numpy()))
    env = environment()
    chk("commit, CPU/thread settings and package versions recorded", bool(git("rev-parse", "HEAD")),
        f"HEAD {git('rev-parse', '--short', 'HEAD')}; {env['cpu']}; torch {env['torch']}; SB3 {env.get('stable-baselines3')}; "
        f"threads 1; CUDA {env['cuda_available']}")
    infra = os.path.join(L.RUNS, "infrastructure_log.txt")
    chk("interruptions logged", True, open(infra, encoding="utf-8").read().strip()[:300] if os.path.exists(infra) else "no interruption")
    return res


def family(rows, name):
    df = pd.DataFrame(rows)
    df["family"] = name
    df["holm_p"] = ST.holm(df["sign_flip_p"].to_numpy())
    df["outcome"] = [outcome(r) for _, r in df.iterrows()]
    return df


def paired(A, B, direction, rng):
    d = np.array([A[x] - B[x] for x in L.SEEDS], float)
    out = ST.describe(d, rng)
    out["direction"] = "positive" if direction > 0 else "negative"
    out["n"] = len(d)
    return out


def main():
    cfg = json.load(open(L.CONFIG, encoding="utf-8"))
    fr = json.load(open(L.FREEZE, encoding="utf-8"))
    arc = json.load(open(ARCHIVE, encoding="utf-8"))
    ev = pd.read_csv(EVALS, float_precision="round_trip")
    recs = [json.load(open(os.path.join(L.RUNS, f), encoding="utf-8")) for f in sorted(os.listdir(L.RUNS)) if f.endswith(".json")]
    res = integrity(cfg, recs, fr, arc, ev)
    n_ok = sum(ok for _, ok, _ in res)
    stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = ["# Phase 5 integrity checks", "", f"Run {stamp}, after all 360 runs, the archive freeze, the VAL-SELECT selection "
             f"and the TEST evaluation, before any aggregate statistic. **{n_ok}/{len(res)} passed.**", "",
             "| Status | Check | Detail |", "|---|---|---|"]
    lines += [f"| {'PASS' if ok else 'FAIL'} | {n} | {d.replace('|', '/')[:400]} |" for n, ok, d in res]
    open(os.path.join(OUT, "phase5_integrity_checks.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    if n_ok != len(res):
        raise SystemExit("integrity checks failed; no aggregate computed")
    print(f"integrity {n_ok}/{len(res)}")

    tau = cfg["tau"]
    T = {s: json.load(open(os.path.join(L.LAND, f"{s}_tables.json"), encoding="utf-8")) for s in L.SETTINGS}
    pol = ev[ev.kind == "policy"].set_index(["setting", "policy"])
    base = ev[ev.kind == "baseline"].set_index(["setting", "policy", "method"])
    dense = ev[ev.kind == "dense"].set_index("setting")
    sel = {r["run_id"]: r for r in fr["runs"]}

    # ---------------------------------------------------------------- per-run metrics
    rows, traj = [], []
    for r in recs:
        s, c = r["setting"], r["condition"]
        Ts = T[s]
        raw = gzip.decompress(open(os.path.join(L.ROOT, r["curve_file"]), "rb").read()).decode("utf-8")
        cv = pd.read_csv(io.StringIO(raw), float_precision="round_trip")
        idx = np.array([L.INDEX[tuple(json.loads(a))] for a in cv["actions"]])
        own = np.array(Ts["R0_reward_val_rl" if c == "C0" else "RC_reward_val_rl"])[idx]
        assert np.array_equal(own, cv["episode_reward"].to_numpy()), "curve reward != frozen table"
        rank = np.array(Ts["RC_rank_val_rl"])[idx]
        acc = np.array(Ts["val_rl_accuracy"])[idx]
        feas = np.array(Ts["feasible_val_rl"])[idx]
        destr = np.array(Ts["destructive"])[idx]
        t = cv["timestep"].to_numpy()
        score = (1296 - rank) / 1295
        m = {"run_id": r["run_id"], "setting": s, "condition": c, "seed": r["seed"], "rank_auc": auc(t, score),
             "vrl_accuracy_auc": auc(t, acc), "destructive_rate": float(destr.mean()), "feasible_sample_rate": float(feas.mean()),
             "best_seen_rank": int(rank.min()), "median_rank_training": float(np.median(rank)),
             "unique_policies_sampled": int(len(set(idx.tolist())))}
        if c == "C0":
            m["own_reward_rank_auc"] = auc(t, (1296 - np.array(Ts["R0_rank_val_rl"])[idx]) / 1295)
        for k in (100, 50, 20, 10, 5, 1):
            hit = np.flatnonzero(rank <= k)
            m[f"best_seen_top{k}"] = bool(len(hit))
            if k in (20, 10):
                m[f"first_episode_top{k}"] = int(hit[0] + 1) if len(hit) else L.CENSOR
        fi = L.INDEX[tuple(r["final_actions"])]
        m["deterministic_final_rank"] = int(Ts["RC_rank_val_rl"][fi])
        p = sel[r["run_id"]]["selected_policy"]
        i = L.INDEX[tuple(p)]
        key = (s, str(list(p)))
        te = float(pol.loc[key, "test_accuracy"])
        dt = float(dense.loc[s, "test_accuracy"])
        o = cfg["oracle"][s]
        m.update({"selected_policy": str(list(p)), "selection_fallback": sel[r["run_id"]]["fallback"],
                  "selected_rank": int(Ts["RC_rank_val_rl"][i]), "selected_sparsity": Ts["total_sparsity"][i],
                  "selected_unit_sparsity": str(Ts["unit_sparsity"][i]), "selected_q_val_rl": Ts["q_val_rl"][i],
                  "selected_q_val_select": Ts["q_val_select"][i], "selected_feasible_val_rl": bool(Ts["feasible_val_rl"][i]),
                  "selected_feasible_val_select": bool(Ts["feasible_val_select"][i]), "test_accuracy": te, "dense_test_accuracy": dt,
                  "test_drop": dt - te, "q_test": te / dt, "selected_feasible_test": te / dt >= tau,
                  "lamp_test_accuracy": float(base.loc[(s, key[1], "lamp"), "test_accuracy"]),
                  "global_test_accuracy": float(base.loc[(s, key[1], "global"), "test_accuracy"]),
                  "oracle_sparsity": o["sparsity"], "max_feasible_sparsity": o["max_feasible_sparsity"],
                  "sparsity_gap_to_oracle": o["sparsity"] - Ts["total_sparsity"][i],
                  "vrl_accuracy_gap_to_oracle": o["val_rl_accuracy"] - Ts["val_rl_accuracy"][i]})
        for k in (100, 50, 20, 10, 5, 1):
            m[f"selected_top{k}"] = m["selected_rank"] <= k
        m["minus_lamp"] = te - m["lamp_test_accuracy"]
        m["minus_global"] = te - m["global_test_accuracy"]
        rows.append(m)
        traj.append(pd.DataFrame({"run_id": r["run_id"], "setting": s, "condition": c, "seed": r["seed"], "episode": cv["episode"],
                                  "timestep": t, "constrained_rank": rank, "rank_score": score, "val_rl_accuracy": acc,
                                  "feasible": feas, "destructive": destr}))
    runs = pd.DataFrame(rows).sort_values(["setting", "condition", "seed"]).reset_index(drop=True)
    runs.to_csv(os.path.join(OUT, "phase5_all_runs.csv"), index=False, float_format="%.17g")
    tr = pd.concat(traj)
    tr["best_so_far_rank"] = tr.groupby("run_id")["constrained_rank"].cummin()
    with gzip.open(os.path.join(OUT, "phase5_policy_rank_trajectory.csv.gz"), "wt", encoding="utf-8") as f:
        tr.to_csv(f, index=False, float_format="%.17g")
    curves = tr.groupby(["setting", "condition", "episode"]).agg(
        timestep=("timestep", "first"), rank_median=("constrained_rank", "median"), best_so_far_rank_median=("best_so_far_rank", "median"),
        rank_score_mean=("rank_score", "mean"), val_rl_accuracy_mean=("val_rl_accuracy", "mean"),
        feasible_rate=("feasible", "mean"), destructive_rate=("destructive", "mean")).reset_index()
    curves.to_csv(os.path.join(OUT, "phase5_training_curves.csv"), index=False, float_format="%.17g")
    runs[["run_id", "setting", "condition", "seed", "best_seen_rank", "selected_rank", "deterministic_final_rank", "median_rank_training"]
         + [f"best_seen_top{k}" for k in (100, 50, 20, 10, 5, 1)] + [f"selected_top{k}" for k in (100, 50, 20, 10, 5, 1)]
         + ["first_episode_top20", "first_episode_top10", "rank_auc"]].to_csv(os.path.join(OUT, "phase5_policy_rank.csv"), index=False)
    runs[["run_id", "setting", "condition", "seed", "selected_policy", "selected_sparsity", "selected_unit_sparsity", "max_feasible_sparsity",
          "oracle_sparsity", "sparsity_gap_to_oracle", "selected_q_val_rl", "selected_q_val_select", "q_test", "selected_feasible_val_rl",
          "selected_feasible_val_select", "selected_feasible_test", "test_accuracy", "dense_test_accuracy", "test_drop",
          "selection_fallback"]].to_csv(os.path.join(OUT, "phase5_constraint_results.csv"), index=False, float_format="%.17g")
    pd.DataFrame([{"run_id": x["run_id"], "setting": x["setting"], "condition": x["condition"], "seed": x["seed"],
                   "unique_policies_sampled": x["unique_policies_sampled"], "archive_rank": j + 1, "actions": str(cd["actions"]),
                   "val_rl_reward": cd["val_rl_reward"], "first_episode": cd["first_episode"]}
                  for x in arc["runs"] for j, cd in enumerate(x["candidates"])]).to_csv(os.path.join(OUT, "phase5_archive.csv"), index=False,
                                                                                         float_format="%.17g")

    # ---------------------------------------------------------------- statistics
    rng = np.random.default_rng(0)
    V = {(s, c): runs[(runs.setting == s) & (runs.condition == c)].set_index("seed") for s in L.SETTINGS for c in L.CONDITIONS}

    def rows_for(a, b, metric, direction):
        return [{"setting": s, "comparison": f"{a} - {b}", "endpoint": metric,
                 "mean_a": V[(s, a)][metric].mean(), "sd_a": V[(s, a)][metric].std(ddof=1),
                 "median_a": V[(s, a)][metric].median(), "mean_b": V[(s, b)][metric].mean(), "sd_b": V[(s, b)][metric].std(ddof=1),
                 "median_b": V[(s, b)][metric].median(),
                 **paired(V[(s, a)][metric].to_dict(), V[(s, b)][metric].to_dict(), direction, rng)} for s in L.SETTINGS]

    fams = [family(rows_for("C1", "C0", "test_accuracy", +1), "H1/H4 test accuracy C1 - C0")]
    d_rows = []
    for metric, dirn in (("rank_auc", 1), ("vrl_accuracy_auc", 1), ("first_episode_top20", -1), ("first_episode_top10", -1)):
        d_rows += rows_for("C1", "C2", metric, dirn)
    fams.append(family(d_rows, "H3 exploration C1 - C2 (primary)"))
    d2 = []
    for metric, dirn in (("destructive_rate", -1), ("best_seen_rank", -1), ("selected_rank", -1)):
        d2 += rows_for("C1", "C2", metric, dirn)
    fams.append(family(d2, "H3 secondary C1 - C2"))
    lamp_rows, glob_rows, gap_rows = [], [], []
    for s in L.SETTINGS:
        for c in ("C1", "C0"):
            g = V[(s, c)]
            lamp_rows.append({"setting": s, "comparison": f"{c} - LAMP", "endpoint": "test_accuracy", "condition": c,
                              "mean_a": g.test_accuracy.mean(), "mean_b": g.lamp_test_accuracy.mean(),
                              "mean_sparsity": g.selected_sparsity.mean(), "abs_sparsity_mismatch": 0.0,
                              **paired(g.test_accuracy.to_dict(), g.lamp_test_accuracy.to_dict(), +1, rng)})
            glob_rows.append({"setting": s, "comparison": f"{c} - global", "endpoint": "test_accuracy", "condition": c,
                              "mean_a": g.test_accuracy.mean(), "mean_b": g.global_test_accuracy.mean(),
                              "mean_sparsity": g.selected_sparsity.mean(), "abs_sparsity_mismatch": 0.0,
                              **paired(g.test_accuracy.to_dict(), g.global_test_accuracy.to_dict(), +1, rng)})
    lamp = pd.DataFrame(lamp_rows)
    fams.append(family([r for r in lamp_rows if r["condition"] == "C1"], "H5 C1 - LAMP"))
    fams.append(family([r for r in glob_rows if r["condition"] == "C1"], "C1 - global (descriptive)"))
    fams.append(family(rows_for("C1", "C0", "minus_lamp", +1), "H5 gap to LAMP C1 - C0"))
    fams.append(family(rows_for("C1", "C0", "selected_rank", -1), "H2 selected rank C1 - C0 (descriptive)"))
    fams.append(family(rows_for("C1", "C0", "selected_sparsity", +1), "sparsity C1 - C0 (descriptive)"))
    st = pd.concat(fams, ignore_index=True)
    st.to_csv(os.path.join(OUT, "phase5_statistics.csv"), index=False, float_format="%.17g")
    st[st.comparison.str.contains("C1 - C0")].to_csv(os.path.join(OUT, "phase5_c0_vs_c1.csv"), index=False, float_format="%.17g")
    st[st.comparison.str.contains("C1 - C2")].to_csv(os.path.join(OUT, "phase5_c1_vs_c2.csv"), index=False, float_format="%.17g")
    lamp.to_csv(os.path.join(OUT, "phase5_lamp_comparison.csv"), index=False, float_format="%.17g")
    pd.DataFrame(glob_rows).to_csv(os.path.join(OUT, "phase5_global_comparison.csv"), index=False, float_format="%.17g")

    # ---------------------------------------------------------------- criteria A-F
    def oc(fam, s, comp=None, endpoint=None):
        q = st[(st.family == fam) & (st.setting == s)]
        if comp:
            q = q[q.comparison == comp]
        if endpoint:
            q = q[q.endpoint == endpoint]
        return q.outcome.iloc[0]
    h1 = {s: oc("H1/H4 test accuracy C1 - C0", s) for s in L.SETTINGS}
    others_opp = sum(h1[s] == "opposite" for s in L.SETTINGS if s not in RESNETS)
    a_res = all(h1[s] == "supported" for s in RESNETS)
    crit = {}
    crit["A"] = {"status": "MET" if a_res and others_opp <= 1 else ("PARTIALLY MET" if any(h1[s] == "supported" for s in RESNETS) else "NOT MET"),
                 "per_setting": h1, "resnet_supported": [h1[s] for s in RESNETS], "other_settings_opposite": others_opp}
    c1 = runs[runs.condition == "C1"]
    med_sp = c1.groupby("setting").selected_sparsity.median().to_dict()
    need20 = min(4, sum(cfg["oracle"][s]["feasible_sparsity_ge_20_exists"] for s in L.SETTINGS))
    b1 = all(v >= 10 for v in med_sp.values())
    b2 = sum(v >= 20 for v in med_sp.values()) >= need20
    crit["B"] = {"status": "MET" if b1 and b2 else ("PARTIALLY MET" if b1 or b2 else "NOT MET"), "median_sparsity": med_sp,
                 "settings_ge_20": sum(v >= 20 for v in med_sp.values()), "required_ge_20": need20}
    med_rank = c1.groupby("setting").selected_rank.median().to_dict()
    c_a = sum(v <= 20 for v in med_rank.values()) >= 5
    top20 = float(c1.selected_top20.mean())
    c_b = top20 >= 0.80
    crit["C"] = {"status": "MET" if c_a and c_b else ("PARTIALLY MET" if c_a or c_b else "NOT MET"), "median_selected_rank": med_rank,
                 "share_runs_selected_top20": top20, "share_runs_best_seen_top20": float(c1.best_seen_top20.mean())}
    dfam = st[st.family == "H3 exploration C1 - C2 (primary)"]
    d_sup = {s: int((dfam[dfam.setting == s].outcome == "supported").sum()) for s in L.SETTINGS}
    d_opp = int((dfam.outcome == "opposite").sum())
    n_set = sum(v >= 1 for v in d_sup.values())
    crit["D"] = {"status": "MET" if n_set >= 4 and d_opp <= 1 else ("PARTIALLY MET" if n_set >= 1 else "NOT MET"),
                 "settings_with_supported_metric": n_set, "opposite_outcomes": d_opp, "supported_per_setting": d_sup}
    e_sup = sum(v == "supported" for v in h1.values())
    e_opp = sum(v == "opposite" for v in h1.values())
    crit["E"] = {"status": "MET" if e_sup >= 2 and e_opp == 0 else ("PARTIALLY MET" if e_sup >= 1 else "NOT MET"),
                 "supported": e_sup, "opposite": e_opp}
    ff = st[st.family == "H5 C1 - LAMP"]
    f_sup, f_opp = int((ff.outcome == "supported").sum()), int((ff.outcome == "opposite").sum())
    means = ff.set_index("setting").mean_diff.to_dict()
    weak = sum(v >= -0.5 for v in means.values()) >= 4 and all(v >= -1.0 for v in means.values())
    crit["F_strong"] = {"status": "MET" if f_sup >= 1 and f_opp == 0 else "NOT MET", "supported": f_sup, "opposite": f_opp}
    crit["F_weak"] = {"status": "MET" if weak else ("PARTIALLY MET" if sum(v >= -0.5 for v in means.values()) >= 1 else "NOT MET"),
                      "mean_C1_minus_LAMP": means, "within_0.5": sum(v >= -0.5 for v in means.values()),
                      "worse_than_1.0": sum(v < -1.0 for v in means.values())}
    cross = []
    for fam in st.family.unique():
        for comp in st[st.family == fam].comparison.unique():
            for ep in st[(st.family == fam) & (st.comparison == comp)].endpoint.unique():
                q = st[(st.family == fam) & (st.comparison == comp) & (st.endpoint == ep)].set_index("setting").outcome.to_dict()
                ns, no = sum(v == "supported" for v in q.values()), sum(v == "opposite" for v in q.values())
                cross.append({"family": fam, "comparison": comp, "endpoint": ep, "supported": ns, "opposite": no,
                              "inconclusive": 6 - ns - no, "class": cross_class(ns, no), "reversed": no >= 3, **q})
    pd.DataFrame(cross).to_csv(os.path.join(OUT, "phase5_cross_setting_summary.csv"), index=False)
    json.dump({"generated": stamp, "tau": tau, "criteria": crit}, open(os.path.join(OUT, "phase5_classification.json"), "w",
                                                                       encoding="utf-8"), indent=1, default=str)

    # ---------------------------------------------------------------- figure data
    fa = runs[runs.condition.isin(["C0", "C1"])][["setting", "condition", "seed", "selected_sparsity", "test_accuracy",
                                                   "lamp_test_accuracy", "global_test_accuracy", "dense_test_accuracy"]]
    fa.to_csv(os.path.join(OUT, "phase5_fig_A_accuracy_sparsity.csv"), index=False, float_format="%.17g")
    p4 = pd.read_csv(os.path.join(L.ROOT, "results", "phase4", "phase4_baseline_comparison.csv"), float_precision="round_trip")
    fb = runs[runs.setting.isin(RESNETS)].groupby(["setting", "condition"]).agg(
        sparsity_median=("selected_sparsity", "median"), test_mean=("test_accuracy", "mean"), test_drop_mean=("test_drop", "mean"),
        feasible_test_share=("selected_feasible_test", "mean")).reset_index()
    for s in RESNETS:
        g = p4[(p4.setting == s) & (p4.pipeline == "F") & (p4.method == "lamp")].iloc[0]
        fb = pd.concat([fb, pd.DataFrame([{"setting": s, "condition": "Phase-4 F (reference)", "sparsity_median": g.sparsity_mean,
                                           "test_mean": g.ppo_test_mean, "test_drop_mean": float(dense.loc[s, "test_accuracy"]) - g.ppo_test_mean}])])
    fb.to_csv(os.path.join(OUT, "phase5_fig_B_resnet_reward_alignment.csv"), index=False, float_format="%.17g")
    runs[["setting", "condition", "seed", "selected_rank", "best_seen_rank", "deterministic_final_rank"]].to_csv(
        os.path.join(OUT, "phase5_fig_C_policy_rank.csv"), index=False)
    curves[curves.condition.isin(["C1", "C2"])][["setting", "condition", "episode", "rank_score_mean", "best_so_far_rank_median",
                                                 "val_rl_accuracy_mean"]].to_csv(os.path.join(OUT, "phase5_fig_D_sensitivity_exploration.csv"),
                                                                                  index=False, float_format="%.17g")
    runs.groupby(["setting", "condition"]).agg(sparsity_median=("selected_sparsity", "median"), sparsity_mean=("selected_sparsity", "mean"),
                                               max_feasible_sparsity=("max_feasible_sparsity", "first"),
                                               oracle_sparsity=("oracle_sparsity", "first")).reset_index().to_csv(
        os.path.join(OUT, "phase5_fig_E_feasible_sparsity_gap.csv"), index=False, float_format="%.17g")
    print("criteria:", {k: v["status"] for k, v in crit.items()})


if __name__ == "__main__":
    main()
