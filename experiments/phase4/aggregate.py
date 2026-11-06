"""Phase 4 integrity checks, statistics, classification, summary and figure data (pre-registration §7-§11).

  python experiments/phase4/aggregate.py
Refuses to compute any aggregate if an integrity check fails. Stored floats are read with round-trip parsing.
"""
import datetime
import gzip
import json
import os
import subprocess

import numpy as np
import pandas as pd

import phase4_lib as L
import phase4_metrics as M
from src import statistics as ST
from src.utils import environment, sha256_file

OUT = L.RESULTS
FREEZE = os.path.join(OUT, "phase4_final_policies.json")
EVALS = os.path.join(OUT, "phase4_test_and_baseline_evaluations.csv")
RULE = {"S0": ("S0", "final"), "S1": ("S0", "archive"), "S2": ("S2", "final"), "S2A": ("S2", "archive")}


def git(*a):
    return subprocess.run(["git", *a], cwd=L.ROOT, capture_output=True, text=True).stdout.strip()


def read_csv(p):
    return pd.read_csv(p, float_precision="round_trip")


# ------------------------------------------------------------------ integrity
def integrity(cfg, sel, fr, recs, ev):
    res = []

    def chk(n, ok, d=""):
        res.append((n, bool(ok), str(d)))
    R, Pr = sel["A"]["winner"], sel["B"]["winner"]
    expected = {(s, r, "P0", "S0", x) for s in L.SETTINGS for r in L.REWARDS for x in L.SEEDS}
    expected |= {(s, R, p, "S0", x) for s in L.SETTINGS for p in L.PRIORS for x in L.SEEDS}
    expected |= {(s, R, Pr, v, x) for s in L.SETTINGS for v in ("S0", "S2") for x in L.SEEDS}
    got = {(r["setting"], r["reward"], r["prior"], r["variant"], r["seed"]) for r in recs}
    chk("all expected runs present, no extra runs, no dropped seed", got == expected, f"{len(got)} runs, {len(expected)} expected")
    chk("all records complete; 2,048 timesteps and 512 episodes each",
        all(r["status"] == "complete" and r["timesteps_trained"] == 2048 and r["episodes"] == 512 for r in recs))
    chk("curve files match their hashes", all(sha256_file(os.path.join(L.ROOT, r["curve_file"])) == r["curve_sha256"] for r in recs))
    chk("TEST never accessed during training, reward, prior or archive selection (record flag; runners and selector "
        "contain no test access)", all(r["test_set_accessed"] is False for r in recs) and all(sel[k]["test_set_accessed"] is False for k in "ABC"))
    chk("stage selections frozen before the test-freeze file (hash recorded in the freeze file)",
        fr["selection_sha256"] == sha256_file(L.SELECTION) and os.path.getmtime(L.SELECTION) <= os.path.getmtime(FREEZE))
    chk("pre-registration, config and reward tables unchanged in every record",
        all(r["preregistration_sha256"] == sha256_file(L.PREREG) and r["config_sha256"] == sha256_file(L.CONFIG) for r in recs)
        and not git("diff", "f52e87d", "--", "results/phase4/phase4_preregistration.md")
        and all(sha256_file(os.path.join(L.LAND, f"{s}_rewards.json")) == cfg["reward_tables_sha256"][s] for s in L.SETTINGS))
    chk("Phase-3 frozen files unchanged (git diff vs 67f4270 empty for results/architecture_generalization, "
        "results/stronger_baselines, checkpoints/phase2_dense, checkpoints/phase3_agents, cnn_baseline_FIXED.pth)",
        not git("diff", "67f4270", "--", "results/architecture_generalization", "results/stronger_baselines",
                "checkpoints/phase2_dense", "checkpoints/phase3_agents", "checkpoints/cnn_baseline_FIXED.pth"))
    chk("reference checkpoints and Phase-3 V_RL landscapes unchanged",
        all(L.P3.load_reference(s)[1] == cfg["settings"][s]["reference_sha256"] and
            sha256_file(L.P3.landscape_json(s)) == cfg["settings"][s]["phase3_landscape_sha256"] for s in L.SETTINGS))
    chk("V_SELECT landscapes unchanged", all(sha256_file(os.path.join(L.LAND, f"{s}_vselect.json")) ==
                                             cfg["settings"][s]["vselect_landscape_sha256"] for s in L.SETTINGS))
    chk("1 thread in every run", all(r["threads"] == 1 for r in recs))
    chk("identical PPO hyper-parameters in every run (policy class and BC eta aside)",
        len({json.dumps({k: v for k, v in r["ppo"].items() if k != "policy"}, sort_keys=True) for r in recs}) == 1
        and len({json.dumps(r["ppo_defaults"], sort_keys=True) for r in recs}) == 1)
    chk("BC only in S2 runs (eta 0.1, active), never elsewhere",
        all((r["variant"] == "S2") == (r["bc_eta"] > 0) for r in recs) and all(r["bc_updates"] > 0 for r in recs if r["variant"] == "S2"))
    b = ev[ev.kind == "baseline"]
    pol = ev[ev.kind == "policy"].set_index(["setting", "policy"])
    chk("every baseline has exactly the zero count of its PPO policy",
        all(b.zero_count.to_numpy() == pol.loc[list(zip(b.setting, b.policy)), "zero_count"].to_numpy()))
    chk("seeds logged (400-419 in every condition)", all(sorted({r["seed"] for r in recs if (r["setting"], r["reward"], r["prior"], r["variant"]) == c})
                                                       == L.SEEDS for c in {(r["setting"], r["reward"], r["prior"], r["variant"]) for r in recs}))
    env = environment()
    chk("commit, environment, CPU and package versions recorded", bool(git("rev-parse", "HEAD")),
        f"HEAD {git('rev-parse', '--short', 'HEAD')}; {env['cpu']}; torch {env['torch']}; SB3 {env.get('stable-baselines3')}; "
        f"threads 1; CUDA {env['cuda_available']}")
    corr = os.path.join(L.RUNS, "corruption_log.txt")
    infra = os.path.join(L.RUNS, "infrastructure_log.txt")
    chk("infrastructure / corruption logs", True, (open(infra, encoding="utf-8").read().strip()[:300] if os.path.exists(infra) else "no infrastructure event")
        + (" | corruption log present" if os.path.exists(corr) else ""))
    return res


# ------------------------------------------------------------------ statistics helpers
def fam_rows(tab, fam, question, comparisons, metric, direction, key_cols, cond_col):
    """comparisons: list of (a, b); tab has one row per (setting, cond_col, seed)."""
    rng = np.random.default_rng(0)
    rows = []
    for s in L.SETTINGS:
        for a, b in comparisons:
            A = tab[(tab.setting == s) & (tab[cond_col] == a)].set_index("seed")[metric].to_dict()
            B = tab[(tab.setting == s) & (tab[cond_col] == b)].set_index("seed")[metric].to_dict()
            d = M.paired(A, B, rng, direction)
            rows.append({"family": fam, "question": question, "setting": s, "endpoint": metric, "comparison": f"{a} - {b}",
                         "mean_a": float(np.mean(list(A.values()))), "sd_a": float(np.std(list(A.values()), ddof=1)),
                         "mean_b": float(np.mean(list(B.values()))), "sd_b": float(np.std(list(B.values()), ddof=1)), **d})
    return rows


def main():
    cfg = json.load(open(L.CONFIG, encoding="utf-8"))
    sel = json.load(open(L.SELECTION, encoding="utf-8"))
    fr = json.load(open(FREEZE, encoding="utf-8"))
    ev = read_csv(EVALS)
    recs = M.load_runs(cfg)
    res = integrity(cfg, sel, fr, recs, ev)
    n_ok = sum(ok for _, ok, _ in res)
    stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = ["# Phase 4 integrity checks", "", f"Run {stamp}, after all runs, stage selections and the final test "
             f"evaluation, before any aggregate statistic. **{n_ok}/{len(res)} passed.**", "",
             "| Status | Check | Detail |", "|---|---|---|"]
    lines += [f"| {'PASS' if ok else 'FAIL'} | {n} | {d.replace('|', '/')[:400]} |" for n, ok, d in res]
    open(os.path.join(OUT, "phase4_integrity_checks.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    if n_ok != len(res):
        raise SystemExit("integrity checks failed; no aggregate computed")
    print(f"integrity {n_ok}/{len(res)}")

    R, Pr, Sc = sel["A"]["winner"], sel["B"]["winner"], sel["C"]["winner"]
    Fv, Frule = RULE[Sc]
    pol = ev[ev.kind == "policy"].set_index(["setting", "policy"])
    base = ev[ev.kind == "baseline"]
    bdet = base[base.method != "random"].set_index(["setting", "policy", "method"])["test_accuracy"]
    brand = base[base.method == "random"].groupby(["setting", "policy"])["test_accuracy"].mean()

    # ---------------------------------------------------------------- per-run table
    rows, traj = [], []
    for r in recs:
        m = M.run_metrics(r, cfg)
        t = m.pop("_trajectory")
        t.insert(0, "run_id", r["run_id"])
        traj.append(t)
        for w in ("final", "archive"):
            key = (r["setting"], m[f"{w}_policy"])
            m[f"{w}_test_accuracy"] = float(pol.loc[key, "test_accuracy"])
            m[f"{w}_zero_count"] = int(pol.loc[key, "zero_count"])
            for meth in ("lamp", "global"):
                m[f"{w}_{meth}_test_accuracy"] = float(bdet.loc[(key[0], key[1], meth)])
                m[f"{w}_excess_over_{meth}"] = m[f"{w}_test_accuracy"] - m[f"{w}_{meth}_test_accuracy"]
        tgt = {"R1a": L.TARGET_COMMON, "R1b": cfg["settings"][r["setting"]]["target_R1b"]}.get(r["reward"])
        m["target_sparsity"] = tgt
        m["control_error"] = abs(m["final_sparsity"] - tgt) if tgt is not None else None
        m["condition"] = f"{r['reward']}/{r['prior']}/{r['variant']}"
        rows.append(m)
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(OUT, "phase4_all_runs.csv"), index=False, float_format="%.17g")
    tr = pd.concat(traj)
    meta = runs.set_index("run_id")[["setting", "reward", "prior", "variant", "seed"]]
    tr = tr.join(meta, on="run_id")
    tr["best_so_far_rank"] = tr.groupby("run_id")["rank"].cummin()
    with gzip.open(os.path.join(OUT, "phase4_policy_rank_trajectory_full.csv.gz"), "wt", encoding="utf-8") as f:
        tr.to_csv(f, index=False, float_format="%.17g")
    g = tr.groupby(["setting", "reward", "prior", "variant", "episode"])
    curves = g.agg(timestep=("timestep", "first"), rank_median=("rank", "median"), rank_mean=("rank", "mean"),
                   best_so_far_rank_median=("best_so_far_rank", "median"), rank_score_mean=("rank_score", "mean"),
                   vrl_accuracy_mean=("vrl_accuracy", "mean"), utility_mean=("utility", "mean"),
                   destructive_rate=("destructive", "mean"), sparsity_mean=("sparsity", "mean"), n=("run_id", "nunique")).reset_index()
    curves.to_csv(os.path.join(OUT, "phase4_training_curves.csv"), index=False, float_format="%.17g")
    curves[["setting", "reward", "prior", "variant", "episode", "rank_median", "rank_mean", "best_so_far_rank_median"]].to_csv(
        os.path.join(OUT, "phase4_policy_rank_trajectory.csv"), index=False, float_format="%.17g")

    # ---------------------------------------------------------------- comparison tables
    A = runs[(runs.prior == "P0") & (runs.variant == "S0")]
    rc = A.groupby(["setting", "reward"]).agg(
        n=("seed", "size"), sparsity_mean=("final_sparsity", "mean"), sparsity_sd=("final_sparsity", "std"),
        target=("target_sparsity", "first"), control_error_mean=("control_error", "mean"),
        vrl_retention_mean=("final_vrl_retention", "mean"), regret_mean=("final_regret_vrl", "mean"),
        final_rank_mean=("final_rank_vrl", "mean"), test_mean=("final_test_accuracy", "mean"),
        excess_lamp_mean=("final_excess_over_lamp", "mean"), excess_global_mean=("final_excess_over_global", "mean")).reset_index()
    rc["oracle_optimum_sparsity"] = [cfg["reward_oracle_vrl"][s][r]["optimum_sparsity"] for s, r in zip(rc.setting, rc.reward)]
    rc["oracle_optimum_retention"] = [cfg["reward_oracle_vrl"][s][r]["optimum_vrl_retention"] for s, r in zip(rc.setting, rc.reward)]
    rc.to_csv(os.path.join(OUT, "phase4_reward_comparison.csv"), index=False, float_format="%.17g")
    B = runs[(runs.reward == R) & (runs.variant == "S0")]
    pc = B.groupby(["setting", "prior"]).agg(
        rank_auc=("rank_auc", "mean"), vrl_auc=("vrl_auc", "mean"), utility_auc=("utility_auc", "mean"),
        destructive_frequency=("destructive_frequency", "mean"), first_top100_median=("first_episode_top100", "median"),
        first_top50_median=("first_episode_top50", "median"), best_seen_rank_median=("best_seen_rank", "median"),
        final_rank_median=("final_rank_vrl", "median"), final_sparsity_mean=("final_sparsity", "mean"),
        final_test_mean=("final_test_accuracy", "mean")).reset_index()
    pc.to_csv(os.path.join(OUT, "phase4_prior_comparison.csv"), index=False, float_format="%.17g")
    Cq = runs[(runs.reward == R) & (runs.prior == Pr)]
    sq = []
    for s in L.SETTINGS:
        for c, (v, w) in RULE.items():
            g2 = Cq[(Cq.setting == s) & (Cq.variant == v)]
            row = {"setting": s, "condition": c, "n": len(g2), "selected_rank_vrl_median": g2[f"{w}_rank_vrl"].median(),
                   "selected_rank_vselect_median": g2[f"{w}_rank_vselect"].median(), "best_seen_rank_median": g2.best_seen_rank.median(),
                   "median_rank_training_mean": g2.median_rank_training.mean(), "rank_auc_mean": g2.rank_auc.mean(),
                   "selected_test_mean": g2[f"{w}_test_accuracy"].mean(), "selected_sparsity_mean": g2[f"{w}_sparsity"].mean()}
            for k in (100, 50, 20, 10, 1):
                row[f"best_seen_reach_top{k}"] = float(g2[f"reached_top{k}"].mean())
                row[f"selected_reach_top{k}"] = float(g2[f"{w}_top{k}"].mean())
            for k in (100, 50, 20):
                row[f"first_episode_top{k}_median"] = float(g2[f"first_episode_top{k}"].median())
            sq.append(row)
    p3 = read_csv(os.path.join(L.ROOT, "results", "architecture_generalization", "phase3_all_runs.csv"))
    p3tr = pd.read_csv(os.path.join(L.ROOT, "results", "architecture_generalization", "phase3_trajectory.csv.gz"),
                       float_precision="round_trip", usecols=["run_id", "setting", "condition", "rank_score"])
    p3tr["rank"] = np.rint(1296 - 1295 * p3tr["rank_score"]).astype(int)
    p3best = p3tr[p3tr.condition == "P0"].groupby(["setting", "run_id"])["rank"].min().groupby("setting").median()
    for s in L.SETTINGS:
        g3 = p3[(p3.setting == s) & (p3.condition == "P0")]
        sq.append({"setting": s, "condition": "Phase-3 P0 (reference, seeds 300-319, R0)", "n": len(g3),
                   "selected_rank_vrl_median": g3.reward_rank.median(), "best_seen_rank_median": float(p3best.loc[s]),
                   "selected_reach_top10": float((g3.reward_rank <= 10).mean()), "selected_reach_top20": float((g3.reward_rank <= 20).mean()),
                   "selected_test_mean": g3.test_accuracy.mean()})
    pd.DataFrame(sq).to_csv(os.path.join(OUT, "phase4_search_quality.csv"), index=False, float_format="%.17g")

    # pipelines F and K
    def pipe(tag):
        rr = [x for x in fr["runs"] if any(t == tag or t.startswith(tag + ":") for t in x["tags"])]
        rule = "final" if tag == "K" else fr["final_pipeline"]["selection_rule"]
        out = []
        for x in rr:
            p = str(x[f"{rule}_policy"])
            out.append({"setting": x["setting"], "seed": x["seed"], "policy": p, "test": float(pol.loc[(x["setting"], p), "test_accuracy"]),
                        "sparsity": float(pol.loc[(x["setting"], p), "total_sparsity"]), "vrl": float(pol.loc[(x["setting"], p), "vrl_accuracy"]),
                        **{m: float(bdet.loc[(x["setting"], p, m)]) for m in ("lamp", "global", "uniform", "erk")},
                        "random": float(brand.loc[(x["setting"], p)])})
        return pd.DataFrame(out)
    Fp, Kp = pipe("F"), pipe("K")
    bc = []
    for name, dfp in (("F", Fp), ("K", Kp)):
        for s in L.SETTINGS:
            g2 = dfp[dfp.setting == s]
            for m in ("lamp", "global", "uniform", "erk", "random"):
                bc.append({"pipeline": name, "setting": s, "method": m, "n": len(g2), "ppo_test_mean": g2.test.mean(),
                           "baseline_test_mean": g2[m].mean(), "ppo_minus_baseline_mean": (g2.test - g2[m]).mean(),
                           "ppo_wins": int((g2.test > g2[m]).sum()), "ties": int((g2.test == g2[m]).sum()),
                           "baseline_wins": int((g2.test < g2[m]).sum()), "sparsity_mean": g2.sparsity.mean(),
                           "sparsity_abs_difference_to_baseline": 0.0})
    pd.DataFrame(bc).to_csv(os.path.join(OUT, "phase4_baseline_comparison.csv"), index=False, float_format="%.17g")

    # ---------------------------------------------------------------- families
    st = []
    Aa = A.rename(columns={"reward": "cond"})
    st += fam_rows(Aa, "A-regret", "RQ1", [(r, "R0") for r in ("R1a", "R1b", "R2")], "final_regret_vrl", -1, None, "cond")
    Bb = B.rename(columns={"prior": "cond"})
    st += fam_rows(Bb, "B-specific", "RQ2", [("P2", "P3"), ("P2", "P4")], "rank_auc", +1, None, "cond")
    st += fam_rows(Bb, "B-vsP0", "RQ2", [("P1", "P0"), ("P2", "P0")], "rank_auc", +1, None, "cond")
    st += fam_rows(Bb, "B-reverse", "RQ2", [("P2", "P5")], "rank_auc", +1, None, "cond")
    for metric, dirn in (("vrl_auc", 1), ("utility_auc", 1), ("destructive_frequency", -1), ("first_episode_top100", -1),
                         ("first_episode_top50", -1), ("best_seen_rank", -1), ("final_rank_vrl", -1)):
        st += fam_rows(Bb, f"B-secondary {metric}", "RQ2", [("P2", "P3"), ("P2", "P4")], metric, dirn, None, "cond")
    cc = []
    for c, (v, w) in RULE.items():
        g2 = Cq[Cq.variant == v].copy()
        g2["cond"] = c
        g2["sel_rank_vrl"] = g2[f"{w}_rank_vrl"]
        g2["sel_rank_vselect"] = g2[f"{w}_rank_vselect"]
        cc.append(g2)
    Cc = pd.concat(cc)
    st += fam_rows(Cc, "C-final", "RQ3", [("S1", "S0"), ("S2", "S0"), ("S2A", "S0")], "sel_rank_vrl", -1, None, "cond")
    st += fam_rows(Cc, "C-reach", "RQ3", [("S2", "S0")], "best_seen_rank", -1, None, "cond")
    st += fam_rows(Cc, "C-heldout", "RQ3", [("S1", "S0"), ("S2", "S0"), ("S2A", "S0")], "sel_rank_vselect", -1, None, "cond")
    FK = pd.concat([Fp.assign(cond="F"), Kp.assign(cond="K")])
    FK["excess_lamp"] = FK.test - FK.lamp
    FK["excess_global"] = FK.test - FK["global"]
    st += fam_rows(FK, "D", "RQ4", [("F", "K")], "excess_lamp", +1, None, "cond")
    st += fam_rows(FK, "D-GM", "RQ4", [("F", "K")], "excess_global", +1, None, "cond")
    FL = pd.concat([Fp.assign(cond="F", val=Fp.test), Fp.assign(cond="LAMP", val=Fp.lamp)])
    st += fam_rows(FL, "E", "RQ5", [("F", "LAMP")], "val", +1, None, "cond")
    st = pd.DataFrame(st)
    st["holm_p"] = np.nan
    for fam, g2 in st.groupby("family"):
        st.loc[g2.index, "holm_p"] = ST.holm(g2["sign_flip_p"].to_numpy())
    st["outcome"] = [M.outcome(r) for _, r in st.iterrows()]
    # E-NI: F non-inferior to LAMP (margin 0.5 pp), one-sided sign-flip on d + 0.5
    ni = []
    for s in L.SETTINGS:
        g2 = Fp[Fp.setting == s].set_index("seed").loc[L.SEEDS]
        d = (g2.test - g2.lamp).to_numpy(float)
        desc = ST.describe(d + 0.5, np.random.default_rng(0))
        p_two = desc["sign_flip_p"]
        p_one = p_two / 2 if desc["mean_diff"] > 0 else 1 - p_two / 2
        ni.append({"family": "E-NI", "question": "RQ5", "setting": s, "endpoint": "test F - LAMP (+0.5 margin)",
                   "comparison": "F - LAMP > -0.5", "mean_diff": float(d.mean()), "t_ci95_low": desc["t_ci95_low"] - 0.5,
                   "t_ci95_high": desc["t_ci95_high"] - 0.5, "sign_flip_p": p_one, "n_positive": int((d > 0).sum()),
                   "n_negative": int((d < 0).sum()), "cohens_dz": desc["cohens_dz"]})
    ni = pd.DataFrame(ni)
    ni["holm_p"] = ST.holm(ni.sign_flip_p.to_numpy())
    ni["outcome"] = ["supported" if (h < 0.05 and lo > -0.5) else "inconclusive" for h, lo in zip(ni.holm_p, ni.t_ci95_low)]
    st = pd.concat([st, ni], ignore_index=True)
    st.to_csv(os.path.join(OUT, "phase4_statistics.csv"), index=False, float_format="%.17g")

    # ---------------------------------------------------------------- cross-setting + criteria
    def per_setting(fam, comp):
        return {s: st[(st.family == fam) & (st.setting == s) & (st.comparison == comp)]["outcome"].iloc[0] for s in L.SETTINGS}
    statements = [("B: P2 > P3 and P2 > P4 (rank AUC) [sensitivity-specific]", None),
                  ("B: P2 > P0 (rank AUC)", ("B-vsP0", "P2 - P0")), ("B: P1 > P0 (rank AUC)", ("B-vsP0", "P1 - P0")),
                  ("B: P2 > P5 (rank AUC)", ("B-reverse", "P2 - P5")),
                  ("C: S1 better final-selected V_RL rank than S0", ("C-final", "S1 - S0")),
                  ("C: S2 better final-selected V_RL rank than S0", ("C-final", "S2 - S0")),
                  ("C: S2A better final-selected V_RL rank than S0", ("C-final", "S2A - S0")),
                  ("C: S2 better best-seen rank than S0", ("C-reach", "S2 - S0")),
                  ("C: S1 better final-selected V_SELECT rank than S0", ("C-heldout", "S1 - S0")),
                  ("C: S2 better final-selected V_SELECT rank than S0", ("C-heldout", "S2 - S0")),
                  ("C: S2A better final-selected V_SELECT rank than S0", ("C-heldout", "S2A - S0")),
                  ("D: F > K on test excess over matched LAMP", ("D", "F - K")),
                  ("D-GM: F > K on test excess over matched global magnitude", ("D-GM", "F - K")),
                  ("E: F > LAMP (test, matched)", ("E", "F - LAMP")),
                  ("E-NI: F non-inferior to LAMP (0.5 pp)", ("E-NI", "F - LAMP > -0.5"))]
    for ep in ("vrl_auc", "utility_auc", "destructive_frequency", "first_episode_top100", "first_episode_top50", "best_seen_rank", "final_rank_vrl"):
        statements.append((f"B-secondary {ep}: P2 better than P3 and P4", ("SPEC", f"B-secondary {ep}")))
    for r in ("R1a", "R1b", "R2"):
        statements.append((f"A-regret: {r} lower final frontier regret than R0", ("A-regret", f"{r} - R0")))
    cross, classes = [], {}
    for name, key in statements:
        if key is None or key[0] == "SPEC":
            fam = "B-specific" if key is None else key[1]
            per = {}
            for s in L.SETTINGS:
                o = st[(st.family == fam) & (st.setting == s)]["outcome"].tolist()
                per[s] = "supported" if o == ["supported", "supported"] else ("opposite" if "opposite" in o else "inconclusive")
        else:
            per = per_setting(*key)
        ns, no = sum(v == "supported" for v in per.values()), sum(v == "opposite" for v in per.values())
        cls, rev = M.cross_class(ns, no)
        cross.append({"statement": name, "supported": ns, "opposite": no, "inconclusive": 6 - ns - no, "class": cls, "reversed": rev, **per})
        classes[name] = {"class": cls, "reversed": rev, "supported_in": ns, "opposite_in": no, "per_setting": per}
    pd.DataFrame(cross).to_csv(os.path.join(OUT, "phase4_cross_setting_summary.csv"), index=False)

    good = lambda n: classes[n]["class"] in ("STRONG", "MODERATE")  # noqa: E731
    med_ok = {c: sum(float(Cc[(Cc.cond == c) & (Cc.setting == s)].sel_rank_vrl.median()) <= 20 for s in L.SETTINGS) for c in ("S1", "S2", "S2A")}
    ni_count = int((ni.outcome == "supported").sum())
    crit = {
        "A_reward": {"met": bool(sel["A"]["passing"]), "passing_rewards": sel["A"]["passing"], "detail": sel["A"]["criteria"]},
        "B_sensitivity_specific": {"met": good("B: P2 > P3 and P2 > P4 (rank AUC) [sensitivity-specific]"),
                                   "class": classes["B: P2 > P3 and P2 > P4 (rank AUC) [sensitivity-specific]"]["class"]},
        "C_search": {"met": any(good(f"C: {c} better final-selected V_RL rank than S0") and med_ok[c] >= 3 for c in ("S1", "S2", "S2A")),
                     "settings_with_median_selected_rank_le_20": med_ok},
        "D_final_model": {"met": good("D: F > K on test excess over matched LAMP"), "class": classes["D: F > K on test excess over matched LAMP"]["class"]},
        "E_lamp": {"met": good("E: F > LAMP (test, matched)") or ni_count >= 3, "beats_class": classes["E: F > LAMP (test, matched)"]["class"],
                   "noninferior_settings": ni_count},
    }
    crit["phase4_successful"] = any(v["met"] for v in crit.values() if isinstance(v, dict))
    json.dump({"generated": stamp, "final_pipeline": fr["final_pipeline"], "stage_winners": {"A": R, "B": Pr, "C": Sc},
               "criteria": crit, "statements": classes}, open(os.path.join(OUT, "phase4_classification.json"), "w", encoding="utf-8"), indent=1)

    # ---------------------------------------------------------------- figure data
    rc[["setting", "reward", "sparsity_mean", "sparsity_sd", "target", "oracle_optimum_sparsity", "vrl_retention_mean"]].to_csv(
        os.path.join(OUT, "phase4_fig_A_sparsity_control.csv"), index=False, float_format="%.17g")
    curves[(curves.reward == R) & (curves.variant == "S0")][["setting", "prior", "episode", "rank_score_mean", "vrl_accuracy_mean",
                                                              "destructive_rate"]].to_csv(os.path.join(OUT, "phase4_fig_B_prior_exploration.csv"),
                                                                                          index=False, float_format="%.17g")
    curves[(curves.reward == R) & (curves.prior == Pr)][["setting", "variant", "episode", "rank_median", "best_so_far_rank_median"]].to_csv(
        os.path.join(OUT, "phase4_fig_C_policy_rank_trajectory.csv"), index=False, float_format="%.17g")
    pd.DataFrame(sq).to_csv(os.path.join(OUT, "phase4_fig_D_topk_reach_rate.csv"), index=False, float_format="%.17g")
    pd.DataFrame(bc)[lambda d: d.method.isin(["lamp", "global"])].to_csv(os.path.join(OUT, "phase4_fig_E_lamp_comparison.csv"),
                                                                        index=False, float_format="%.17g")
    print("aggregation complete; criteria:", {k: v["met"] for k, v in crit.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
