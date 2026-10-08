"""Phase 6 integrity checks, statistics, criteria A-F, tables and figure data (phase6_preregistration.md).

  python experiments/phase6/aggregate.py
Refuses to compute any aggregate if an integrity check fails.
"""
import datetime
import json
import os
import subprocess

import numpy as np
import pandas as pd

import phase6_lib as L
from src import statistics as ST
from src.utils import environment, sha256_file

OUT = L.RESULTS
FIG = os.path.join(OUT, "figures")
PRACTICAL = 0.5                       # pp: pre-registered practical-relevance threshold (the Phase-5 margin)
H1, DENSE, SPECIFIC = "H1 recovery (post - pre)", "dense control gain", "pruning-specific recovery (method - dense control)"
H4, H4P = "H4 architecture dependence (PPO recovery)", "H4 sensitivity: paired by fine-tuning seed"
H6 = "H6 retention gap change (PPO: VAL-SELECT - test)"
NAME = {"lamp": "LAMP", "global": "global magnitude"}


def fam_of(other):
    """Confirmatory family of H2 (LAMP) / H3 (global): post-fine-tuning difference and gap change, 12 tests."""
    return f"{'H2' if other == 'lamp' else 'H3'} PPO vs {NAME[other]} (post difference and gap change)"


def ref_of(other):
    return f"{'H2' if other == 'lamp' else 'H3'} reference: PPO - {NAME[other]} before fine-tuning (Phase-5 result)"
ARCH = {"simplecnn": "SimpleCNN", "lenet5": "LeNet-5", "resnet8": "ResNet-8"}


def git(*a):
    return subprocess.run(["git", *a], cwd=L.ROOT, capture_output=True, text=True).stdout.strip()


def outcome(mean, lo, hi, holm, direction=+1):
    m, a, b = (mean, lo, hi) if direction > 0 else (-mean, -hi, -lo)
    if m > 0 and holm < 0.05 and a > 0:
        return "supported"
    if m < 0 and holm < 0.05 and b < 0:
        return "opposite"
    return "inconclusive"


def test_rows(name, question, items, direction, rng):
    """items: list of (setting, label, endpoint, dict seed->a, dict seed->b)."""
    rows = []
    for s, label, endpoint, A, B in items:
        d = np.array([A[x] - B[x] for x in L.POLICY_SEEDS], float)
        ds = ST.describe(d, rng)
        rows.append({"family": name, "question": question, "setting": s, "comparison": label, "endpoint": endpoint, "n": len(d),
                     "mean_a": float(np.mean(list(A.values()))), "sd_a": float(np.std(list(A.values()), ddof=1)),
                     "median_a": float(np.median(list(A.values()))), "mean_b": float(np.mean(list(B.values()))),
                     "sd_b": float(np.std(list(B.values()), ddof=1)), "median_b": float(np.median(list(B.values()))),
                     "direction": "positive" if direction > 0 else "negative", **ds})
    df = pd.DataFrame(rows)
    df["holm_p"] = ST.holm(df.sign_flip_p.to_numpy())
    df["outcome"] = [outcome(r.mean_diff, r.t_ci95_low, r.t_ci95_high, r.holm_p, direction) for r in df.itertuples()]
    df["practically_relevant"] = df.mean_diff.abs() >= PRACTICAL
    df["equivalent_within_margin"] = (df.t_ci95_low > -PRACTICAL) & (df.t_ci95_high < PRACTICAL)
    return df


def two_sample_rows(name, question, items, rng, n_perm=100000):
    """Unpaired comparison (different settings): mean difference, Welch 95% CI, two-sided permutation p (seeded)."""
    rows = []
    for s, label, endpoint, A, B in items:
        a, b = np.array(list(A.values()), float), np.array(list(B.values()), float)
        diff = a.mean() - b.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        df = se ** 4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1)) if se > 0 else len(a) + len(b) - 2
        half = ST.t_quantile(0.975, df) * se if se > 0 else 0.0
        pool = np.concatenate([a, b])
        perm = np.array([rng.permutation(pool) for _ in range(n_perm)]) if n_perm else None
        null = perm[:, :len(a)].mean(1) - perm[:, len(a):].mean(1)
        p = float((np.sum(np.abs(null) >= abs(diff) - 1e-12) + 1) / (n_perm + 1))
        rows.append({"family": name, "question": question, "setting": s, "comparison": label, "endpoint": endpoint, "n": len(a),
                     "mean_a": a.mean(), "sd_a": a.std(ddof=1), "median_a": float(np.median(a)), "mean_b": b.mean(), "sd_b": b.std(ddof=1),
                     "median_b": float(np.median(b)), "direction": "positive", "mean_diff": diff, "t_ci95_low": diff - half,
                     "t_ci95_high": diff + half, "sign_flip_p": p, "cohens_dz": diff / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2),
                     "test": "two-sample permutation (100,000, seed 0) + Welch CI"})
    df_ = pd.DataFrame(rows)
    df_["holm_p"] = ST.holm(df_.sign_flip_p.to_numpy())
    df_["outcome"] = [outcome(r.mean_diff, r.t_ci95_low, r.t_ci95_high, r.holm_p) for r in df_.itertuples()]
    df_["practically_relevant"] = df_.mean_diff.abs() >= PRACTICAL
    df_["equivalent_within_margin"] = (df_.t_ci95_low > -PRACTICAL) & (df_.t_ci95_high < PRACTICAL)
    return df_


def integrity(cfg, recs, repro, masks, tests, final):
    res = []

    def chk(n, ok, d="", blocking=True):
        res.append((n, bool(ok), str(d), blocking))
    want = {(s, x, m) for s in L.SETTINGS for x in L.POLICY_SEEDS for m in L.ALL_METHODS}
    got = {(r["setting"], r["policy_seed"], r["method"]) for r in recs}
    chk("all 480 scheduled runs complete; no run dropped or added", got == want and len(recs) == 480 and all(r["status"] == "complete" for r in recs),
        f"{len(recs)} records")
    chk("identical fine-tuning protocol and epoch count in every run", len({json.dumps(r["protocol"], sort_keys=True) for r in recs}) == 1
        and {r["epochs"] for r in recs} == {cfg["epochs"]} and all(len(r["history"]) == cfg["epochs"] for r in recs))
    by = {}
    for r in recs:
        by.setdefault((r["setting"], r["policy_seed"]), {})[r["method"]] = r
    chk("pairing: the four runs of a (setting, policy seed) share the fine-tuning seed and the data stream (order + augmentation)",
        all(len({v[m]["ft_seed"] for m in L.ALL_METHODS}) == 1 and len({v[m]["history"][-1]["data_stream_sha256"] for m in L.ALL_METHODS}) == 1
            for v in by.values()))
    chk("matched sparsity: PPO, LAMP and global have the identical zero count in all 120 triples (mismatch 0)",
        all(len({v[m]["zero_count_pre"] for m in L.METHODS}) == 1 for v in by.values()))
    mk = masks.set_index(["setting", "policy_seed"])
    chk("masks equal the frozen matched-mask table", all(r["mask_hash_pre"] == mk.loc[(r["setting"], r["policy_seed"]), f"{r['method']}_mask_sha256"]
                                                         for r in recs if r["method"] != "dense"))
    fz = {m["run_id"]: m for m in final["models"]}
    tmap = {t["run_id"]: t for t in tests}
    chk("frozen list: exactly the 480 final-epoch models of stage 1 (state hash, checkpoint hash, zero count)",
        set(fz) == {r["run_id"] for r in recs} and all(fz[r["run_id"]]["final_state_sha256"] == r["final_state_sha256"]
                                                         and fz[r["run_id"]]["checkpoint_sha256"] == r["checkpoint_sha256"]
                                                         and fz[r["run_id"]]["zero_count"] == r["zero_count_post"] for r in recs))
    chk("checkpoint files on disk are the frozen ones", all(sha256_file(os.path.join(L.ROOT, m["checkpoint_file"])) == m["checkpoint_sha256"]
                                                           for m in final["models"]))
    chk("all 480 TEST evaluations complete, each of the frozen checkpoint", set(tmap) == set(fz) and all(
        t["status"] == "complete" and t["final_state_sha256"] == fz[t["run_id"]]["final_state_sha256"]
        and t["final_models_sha256"] == sha256_file(L.FINAL) for t in tests), f"{len(tests)} records")
    when = datetime.datetime.fromisoformat
    chk("order: every fine-tuning run finished before the list was frozen; every TEST evaluation started after it",
        max(when(r["finished_at"]) for r in recs) <= when(final["created"]) <= min(when(t["started_at"]) for t in tests),
        f"last run {max(r['finished_at'] for r in recs)}; frozen {final['created']}; first TEST {min(t['started_at'] for t in tests)}")
    chk("stage-1 records hold no TEST accuracy (validation only)", all("test" not in r["pre"] and "test" not in r["post"]
                                                                       and r["test_use"].startswith("none") for r in recs))
    chk("pre-fine-tuning test accuracy equals the Phase-5 record in every run", all(t["pre_test"] == t["phase5_recorded_test"] for t in tests))
    chk("fixed mask: mask hash unchanged and no masked weight non-zero in every run",
        all(r["mask_hash_pre"] == r["mask_hash_post"] and r["mask_violations_post"] == 0 for r in recs))
    chk("no selection: the evaluated model is the one after the final epoch in every run",
        all(r["post"]["val_rl"] == r["history"][-1]["val_rl_accuracy"] and len(r["history"]) == r["epochs"] for r in recs))
    chk("pre-registration and config unchanged in every record and on disk",
        all(r["preregistration_sha256"] == sha256_file(L.PREREG) and r["config_sha256"] == sha256_file(L.CONFIG) for r in recs))
    chk("1 thread in every run", all(r["threads"] == 1 for r in recs))
    chk("Phase 3-5 files unchanged (git diff vs 67f4270 / a3653fd / da061e6 empty)",
        not git("diff", "67f4270", "--", "results/architecture_generalization", "results/stronger_baselines", "checkpoints/phase3_agents")
        and not git("diff", "a3653fd", "--", "results/phase4", "checkpoints/phase4_agents")
        and not git("diff", "da061e6", "--", "results/phase5", "checkpoints/phase5_agents"))
    chk("dense checkpoints and split files unchanged", all(L.P3.load_reference(s)[1] == cfg["settings"][s]["reference_sha256"] for s in L.SETTINGS)
        and sha256_file(os.path.join(L.P5, "splits", "cifar10_phase5_split.npz")) == cfg["inputs_sha256"]["cifar10_split"]
        and sha256_file(os.path.join(L.P5, "splits", "cifar100_phase5_split.npz")) == cfg["inputs_sha256"]["cifar100_split"])
    rmap = {(r["setting"], r["policy_seed"], r["method"]): r for r in recs}
    same = [x["final_state_sha256"] == rmap[(x["setting"], x["policy_seed"], x["method"])]["final_state_sha256"]
            and x["post"] == rmap[(x["setting"], x["policy_seed"], x["method"])]["post"] for x in repro]
    chk("reproduction: the 6 pre-registered repeat runs are bit-identical (weights hash and validation accuracy) [reported; decides criterion F]",
        len(repro) == 6 and all(same), f"{sum(same)}/{len(repro)} identical", blocking=False)
    env = environment()
    chk("commit, CPU/threads and package versions recorded", bool(git("rev-parse", "HEAD")),
        f"HEAD {git('rev-parse', '--short', 'HEAD')}; {env['cpu']}; torch {env['torch']}; threads 1; CUDA {env['cuda_available']}")
    for folder in (L.RUNS,):
        infra = os.path.join(folder, "infrastructure_log.txt")
        chk("interruptions logged", True, open(infra, encoding="utf-8").read().strip()[:300] if os.path.exists(infra) else "no interruption")
    return res


def main():
    cfg = json.load(open(L.CONFIG, encoding="utf-8"))
    recs = [json.load(open(os.path.join(L.RUNS, f), encoding="utf-8")) for f in sorted(os.listdir(L.RUNS)) if f.endswith(".json")]
    repro = [json.load(open(os.path.join(L.REPRO, f), encoding="utf-8")) for f in sorted(os.listdir(L.REPRO)) if f.endswith(".json")] \
        if os.path.isdir(L.REPRO) else []
    masks = pd.read_csv(os.path.join(OUT, "phase6_matched_masks.csv"), float_precision="round_trip")
    tests = [json.load(open(os.path.join(L.TESTS, f), encoding="utf-8")) for f in sorted(os.listdir(L.TESTS)) if f.endswith(".json")]
    final = json.load(open(L.FINAL, encoding="utf-8"))
    res = integrity(cfg, recs, repro, masks, tests, final)
    n_ok = sum(ok for _, ok, _, _ in res)
    stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = ["# Phase 6 integrity checks", "", f"Run {stamp}, after all runs and before any aggregate statistic. **{n_ok}/{len(res)} passed.**", "",
             "| Status | Check | Detail |", "|---|---|---|"] + [f"| {'PASS' if ok else 'FAIL'} | {n} | {d.replace('|', '/')[:400]} |" for n, ok, d, _ in res]
    open(os.path.join(OUT, "phase6_integrity_checks.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    if any(blocking and not ok for _, ok, _, blocking in res):
        raise SystemExit("integrity checks failed; no aggregate computed")
    print(f"integrity {n_ok}/{len(res)}")
    tmap = {t["run_id"]: t for t in tests}
    for r in recs:                                           # attach the TEST accuracy of stage 3 to the stage-1 record
        r["pre"]["test"], r["post"]["test"] = tmap[r["run_id"]]["pre_test"], tmap[r["run_id"]]["post_test"]
    os.makedirs(FIG, exist_ok=True)

    dense_ref = {}
    for r in recs:
        if r["method"] == "dense":
            dense_ref[r["setting"]] = r["pre"]                      # frozen dense reference accuracy on each split
    rows, curve = [], []
    for r in recs:
        s, d = r["setting"], dense_ref[r["setting"]]
        row = {"run_id": r["run_id"], "setting": s, "arch": r["arch"], "dataset": s.split("_")[0], "method": r["method"],
               "policy_seed": r["policy_seed"], "ft_seed": r["ft_seed"], "policy": str(r["policy"]), "epochs": r["epochs"],
               "zero_count_pre": r["zero_count_pre"], "zero_count_post": r["zero_count_post"], "sparsity_pre": r["sparsity_pre"],
               "sparsity_post": r["sparsity_post"], "test_pre": r["pre"]["test"], "test_post": r["post"]["test"],
               "recovery": r["post"]["test"] - r["pre"]["test"], "val_rl_pre": r["pre"]["val_rl"], "val_rl_post": r["post"]["val_rl"],
               "val_select_pre": r["pre"]["val_select"], "val_select_post": r["post"]["val_select"],
               "dense_test": d["test"], "dense_gap_pre": d["test"] - r["pre"]["test"], "dense_gap_post": d["test"] - r["post"]["test"],
               "retention_test_pre": r["pre"]["test"] / d["test"], "retention_test_post": r["post"]["test"] / d["test"],
               "retention_select_pre": r["pre"]["val_select"] / d["val_select"], "retention_select_post": r["post"]["val_select"] / d["val_select"],
               "finetune_seconds": r["finetune_seconds"], "train_seconds": r["train_seconds"], "runtime_seconds": r["runtime_seconds"],
               "peak_memory_mb_process": r["peak_memory_mb_process"], "checkpoint_bytes_raw": r["checkpoint_bytes_raw"],
               "checkpoint_bytes_gzip": r["checkpoint_bytes_gzip"], "mask_hash": r["mask_hash_pre"],
               "final_state_sha256": r["final_state_sha256"], "final_train_loss": r["history"][-1]["train_loss"]}
        row["retention_gap_pre"] = row["retention_select_pre"] - row["retention_test_pre"]
        row["retention_gap_post"] = row["retention_select_post"] - row["retention_test_post"]
        rows.append(row)
        curve.append({"run_id": r["run_id"], "setting": s, "method": r["method"], "policy_seed": r["policy_seed"], "epoch": 0,
                      "val_rl_accuracy": r["pre"]["val_rl"], "val_select_accuracy": r["pre"]["val_select"], "train_loss": None})
        curve += [{"run_id": r["run_id"], "setting": s, "method": r["method"], "policy_seed": r["policy_seed"], "epoch": h["epoch"],
                   "val_rl_accuracy": h["val_rl_accuracy"], "val_select_accuracy": h["val_select_accuracy"], "train_loss": h["train_loss"]}
                  for h in r["history"]]
    runs = pd.DataFrame(rows).sort_values(["setting", "policy_seed", "method"]).reset_index(drop=True)
    dft = runs[runs.method == "dense"].set_index(["setting", "policy_seed"]).test_post          # dense control of the same pair
    runs["dense_control_test_post"] = [dft[(s, x)] for s, x in zip(runs.setting, runs.policy_seed)]
    runs["retention_vs_dense_control_post"] = runs.test_post / runs.dense_control_test_post
    runs.to_csv(os.path.join(OUT, "phase6_all_runs.csv"), index=False, float_format="%.17g")
    curves = pd.DataFrame(curve)
    curves.to_csv(os.path.join(OUT, "phase6_training_curves.csv"), index=False, float_format="%.17g")
    V = {(s, m): runs[(runs.setting == s) & (runs.method == m)].set_index("policy_seed") for s in L.SETTINGS for m in L.ALL_METHODS}

    def col(s, m, c):
        return V[(s, m)][c].to_dict()
    rng = np.random.default_rng(0)
    fams = []
    fams.append(test_rows(H1, "RQ1", [(s, f"{m}: post - pre", "test accuracy", col(s, m, "test_post"), col(s, m, "test_pre"))
                                      for s in L.SETTINGS for m in L.METHODS], +1, rng))
    fams.append(test_rows(DENSE, "RQ1", [(s, "dense: post - pre", "test accuracy", col(s, "dense", "test_post"), col(s, "dense", "test_pre"))
                                         for s in L.SETTINGS], +1, rng))
    fams.append(test_rows(SPECIFIC, "RQ1", [(s, f"{m} recovery - dense gain", "recovery", col(s, m, "recovery"), col(s, "dense", "recovery"))
                                            for s in L.SETTINGS for m in L.METHODS], +1, rng))
    for other in ("lamp", "global"):
        fams.append(test_rows(fam_of(other), "RQ2/RQ4",
                              [(s, f"ppo - {other} (post)", "test accuracy", col(s, "ppo", "test_post"), col(s, other, "test_post")) for s in L.SETTINGS]
                              + [(s, f"dG ppo - {other}", "recovery difference", col(s, "ppo", "recovery"), col(s, other, "recovery")) for s in L.SETTINGS],
                              +1, rng))
        fams.append(test_rows(ref_of(other), "RQ2/RQ4", [(s, f"ppo - {other} (pre)", "test accuracy", col(s, "ppo", "test_pre"),
                                                           col(s, other, "test_pre")) for s in L.SETTINGS], +1, rng))
    arch_items = []
    for ds in ("cifar10", "cifar100"):
        for a, b in (("simplecnn", "lenet5"), ("simplecnn", "resnet8"), ("lenet5", "resnet8")):
            arch_items.append((ds, f"ppo recovery: {a} - {b}", "recovery", col(f"{ds}_{a}", "ppo", "recovery"), col(f"{ds}_{b}", "ppo", "recovery")))
    fams.append(two_sample_rows(H4, "RQ3", arch_items, np.random.default_rng(0)))
    fams.append(test_rows(H4P, "RQ3", arch_items, +1, rng))
    fams.append(test_rows(H6, "RQ5", [(s, "ppo: gap post - gap pre", "retention gap", col(s, "ppo", "retention_gap_post"),
                                       col(s, "ppo", "retention_gap_pre")) for s in L.SETTINGS], -1, rng))
    st = pd.concat(fams, ignore_index=True)
    st.to_csv(os.path.join(OUT, "phase6_statistics.csv"), index=False, float_format="%.17g")

    # cluster-level sensitivity: one value per distinct PPO policy (shared masks are not independent draws)
    cl = []
    for s in L.SETTINGS:
        for other in ("lamp", "global"):
            g = V[(s, "ppo")][["policy", "test_post", "recovery"]].join(V[(s, other)][["test_post", "recovery"]], rsuffix="_o")
            g["g_post"], g["dg"] = g.test_post - g.test_post_o, g.recovery - g.recovery_o
            c = g.groupby("policy")[["g_post", "dg"]].mean()
            for metric in ("g_post", "dg"):
                v = c[metric].to_numpy()
                cl.append({"setting": s, "comparison": f"ppo - {other}", "metric": metric, "n_distinct_policies": len(v), "cluster_mean": float(v.mean()),
                           "cluster_sign_flip_p": ST.sign_flip_p(v) if len(v) >= 6 else None,
                           "testable": len(v) >= 6, "run_level_mean": float(g[metric].mean())})
    pd.DataFrame(cl).to_csv(os.path.join(OUT, "phase6_cluster_sensitivity.csv"), index=False, float_format="%.17g")

    # ---------------------------------------------------------------- tables
    pp = runs.groupby(["setting", "method"]).agg(n=("policy_seed", "size"), sparsity=("sparsity_pre", "mean"), test_pre_mean=("test_pre", "mean"),
                                                 test_pre_sd=("test_pre", "std"), test_post_mean=("test_post", "mean"), test_post_sd=("test_post", "std"),
                                                 recovery_mean=("recovery", "mean"), recovery_sd=("recovery", "std"), dense_test=("dense_test", "first"),
                                                 dense_gap_pre=("dense_gap_pre", "mean"), dense_gap_post=("dense_gap_post", "mean"),
                                                 retention_pre=("retention_test_pre", "mean"), retention_post=("retention_test_post", "mean"),
                                                 share_retention_ge_98_pre=("retention_test_pre", lambda v: float((v >= 0.98).mean())),
                                                 share_retention_ge_98_post=("retention_test_post", lambda v: float((v >= 0.98).mean())),
                                                 finetune_minutes=("finetune_seconds", lambda v: float(v.mean() / 60)),
                                                 gzip_bytes=("checkpoint_bytes_gzip", "mean"), raw_bytes=("checkpoint_bytes_raw", "mean"),
                                                 zero_count_change=("zero_count_post", lambda v: 0)).reset_index()
    pp["zero_count_change"] = runs.groupby(["setting", "method"]).apply(lambda g: int((g.zero_count_post - g.zero_count_pre).abs().max())).to_numpy()
    pp["dense_gap_closed_fraction"] = [(a - b) / a if a > 0 else float("nan") for a, b in zip(pp.dense_gap_pre, pp.dense_gap_post)]
    pp.to_csv(os.path.join(OUT, "phase6_pre_post_accuracy.csv"), index=False, float_format="%.17g")
    st[st.family.isin([H1, DENSE, SPECIFIC])].to_csv(
        os.path.join(OUT, "phase6_accuracy_recovery.csv"), index=False, float_format="%.17g")
    st[st.family.str.startswith("H2")].to_csv(os.path.join(OUT, "phase6_ppo_vs_lamp.csv"), index=False, float_format="%.17g")
    st[st.family.str.startswith("H3")].to_csv(os.path.join(OUT, "phase6_ppo_vs_global.csv"), index=False, float_format="%.17g")

    # ---------------------------------------------------------------- criteria
    def oc(fam, s, comp=None):
        q = st[(st.family == fam) & (st.setting == s)]
        return (q[q.comparison == comp] if comp else q).outcome.iloc[0]
    rec_sup = {m: [s for s in L.SETTINGS if oc(H1, s, f"{m}: post - pre") == "supported"] for m in L.METHODS}
    rec_opp = {m: [s for s in L.SETTINGS if oc(H1, s, f"{m}: post - pre") == "opposite"] for m in L.METHODS}
    all3 = [s for s in L.SETTINGS if all(s in rec_sup[m] for m in L.METHODS)]
    crit = {"A": {"status": "MET", "basis": "every blocking integrity check passed: identical protocol and epoch count, shared fine-tuning seed and "
                  "data stream within every (setting, policy seed), zero-count mismatch 0 in all 120 triples, masks equal the frozen table"}}
    crit["B"] = {"status": "MET" if len(all3) >= 2 else ("PARTIALLY MET" if any(len(v) >= 2 for v in rec_sup.values()) or len(all3) == 1 else "NOT MET"),
                 "settings_all_three_methods": all3, "supported_per_method": rec_sup, "opposite_per_method": rec_opp,
                 "pruning_specific_outcomes": {m: {s: oc(SPECIFIC, s, f"{m} recovery - dense gain") for s in L.SETTINGS} for m in L.METHODS}}
    dg = {s: oc(fam_of("lamp"), s, "dG ppo - lamp") for s in L.SETTINGS}
    n_sup, n_opp = sum(v == "supported" for v in dg.values()), sum(v == "opposite" for v in dg.values())
    lamp = st[st.family == fam_of("lamp")]
    crit["C"] = {"status": "MET" if n_sup >= 2 and n_opp == 0 else ("PARTIALLY MET" if n_sup >= 1 else "NOT MET"), "dG_outcomes": dg,
                 "dG_practically_relevant": lamp[lamp.comparison == "dG ppo - lamp"].set_index("setting").practically_relevant.astype(bool).to_dict(),
                 "post_outcomes": {s: oc(fam_of("lamp"), s, "ppo - lamp (post)") for s in L.SETTINGS},
                 "post_mean_ppo_minus_lamp": lamp[lamp.comparison == "ppo - lamp (post)"].set_index("setting").mean_diff.to_dict(),
                 "post_equivalent_within_margin": lamp[lamp.comparison == "ppo - lamp (post)"].set_index("setting")
                 .equivalent_within_margin.astype(bool).to_dict()}
    n_set = runs.groupby("setting").size()
    whole = all(int(n_set.get(s, 0)) == 80 for s in L.SETTINGS)
    covered = all(set(st[st.family == f].setting) == set(L.SETTINGS) for f in (H1, DENSE, SPECIFIC, fam_of("lamp"), fam_of("global"), H6))
    crit["D"] = {"status": "MET" if whole and covered else ("PARTIALLY MET" if covered else "NOT MET"),
                 "basis": "all six settings complete (20 pairs x 4 methods) and present in every per-setting family; none excluded",
                 "architecture_dependence": st[st.family == H4].assign(k=lambda d: d.setting + ": " + d.comparison).set_index("k").outcome.to_dict()}
    pruned = [r for r in recs if r["method"] != "dense"]
    e_same = sum(r["zero_count_post"] == r["zero_count_pre"] and r["layer_zeros_post"] == r["layer_zeros_pre"] for r in pruned)
    mask_ok = sum(r["mask_hash_post"] == r["mask_hash_pre"] and r["mask_violations_post"] == 0 for r in pruned)
    crit["E"] = {"status": "MET" if e_same == mask_ok == len(pruned) else ("PARTIALLY MET" if min(e_same, mask_ok) >= 0.95 * len(pruned) else "NOT MET"),
                 "runs_with_identical_zero_count_per_layer": e_same, "runs_with_unchanged_mask": mask_ok, "pruned_runs": len(pruned)}
    rmap = {(r["setting"], r["policy_seed"], r["method"]): r for r in recs}
    n_rep = sum(x["final_state_sha256"] == rmap[(x["setting"], x["policy_seed"], x["method"])]["final_state_sha256"] for x in repro)
    complete = len(recs) == 480
    crit["F"] = {"status": "MET" if complete and len(repro) == 6 and n_rep == 6 else ("PARTIALLY MET" if complete else "NOT MET"),
                 "runs_complete": len(recs), "reproduction_identical": f"{n_rep}/{len(repro)}"}
    cross = []
    for fam in st.family.unique():
        for comp in st[st.family == fam].comparison.unique():
            q = st[(st.family == fam) & (st.comparison == comp)].set_index("setting").outcome.to_dict()
            ns, no = sum(v == "supported" for v in q.values()), sum(v == "opposite" for v in q.values())
            cross.append({"family": fam, "comparison": comp, "supported": ns, "opposite": no, "inconclusive": len(q) - ns - no, **q})
    pd.DataFrame(cross).to_csv(os.path.join(OUT, "phase6_cross_setting_summary.csv"), index=False)
    json.dump({"generated": stamp, "epochs": cfg["epochs"], "criteria": crit}, open(os.path.join(OUT, "phase6_classification.json"), "w",
                                                                                   encoding="utf-8"), indent=1, default=str)

    # ---------------------------------------------------------------- figure data
    pp.to_csv(os.path.join(FIG, "phase6_fig1_recovery.csv"), index=False, float_format="%.17g")
    f2 = []
    for s in L.SETTINGS:
        p, l = V[(s, "ppo")], V[(s, "lamp")]
        f2 += [{"setting": s, "policy_seed": x, "gap_pre": p.test_pre[x] - l.test_pre[x], "gap_post": p.test_post[x] - l.test_post[x]} for x in L.POLICY_SEEDS]
    pd.DataFrame(f2).to_csv(os.path.join(FIG, "phase6_fig2_ppo_vs_lamp.csv"), index=False, float_format="%.17g")
    curves.groupby(["setting", "method", "epoch"]).agg(val_select_mean=("val_select_accuracy", "mean"), val_select_sd=("val_select_accuracy", "std"),
                                                       val_rl_mean=("val_rl_accuracy", "mean"), n=("run_id", "nunique")).reset_index().to_csv(
        os.path.join(FIG, "phase6_fig3_recovery_curves.csv"), index=False, float_format="%.17g")
    runs[runs.method != "dense"][["setting", "method", "policy_seed", "sparsity_pre", "test_pre", "test_post", "dense_test"]].to_csv(
        os.path.join(FIG, "phase6_fig4_accuracy_sparsity.csv"), index=False, float_format="%.17g")
    pp[["setting", "method", "recovery_mean", "recovery_sd", "finetune_minutes"]].to_csv(os.path.join(FIG, "phase6_fig5_cost.csv"), index=False,
                                                                                          float_format="%.17g")
    print("criteria:", {k: v["status"] for k, v in crit.items()})


if __name__ == "__main__":
    main()
