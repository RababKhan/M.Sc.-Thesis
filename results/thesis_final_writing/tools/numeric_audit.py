"""Numerical audit: re-extract every headline thesis number from the frozen result files and compare it with the
value stated in results/thesis_final_integration/result_freeze.md.

  python results/thesis_final_writing/tools/numeric_audit.py   -> results/thesis_final_writing/numerical_audit.md
Read-only on all result folders. A check passes when |frozen - stated| <= tolerance (stated rounding).
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
R = os.path.join(ROOT, "results")
OUT = os.path.join(R, "thesis_final_writing", "numerical_audit.md")
SET = ["cifar10_simplecnn", "cifar10_lenet5", "cifar10_resnet8", "cifar100_simplecnn", "cifar100_lenet5", "cifar100_resnet8"]


def csv(p):
    return pd.read_csv(os.path.join(R, p))


def js(p):
    return json.load(open(os.path.join(R, p), encoding="utf-8"))


checks = []


def add(cid, phase, claim, src, fn, stated, tol=0.005):
    checks.append((cid, phase, claim, src, fn, stated, tol))


# ---------------- Phase 2: dense references
b = "architecture_generalization/cpu_baselines.csv"
for s, v, t in [("cifar10_simplecnn", 77.32, 77.03), ("cifar10_lenet5", 64.32, 64.96), ("cifar10_resnet8", 78.72, 78.69),
                ("cifar100_simplecnn", 49.08, 50.50), ("cifar100_lenet5", 32.58, 32.28), ("cifar100_resnet8", 43.48, 43.99)]:
    d, a = s.split("_")
    add(f"P2-ref-test-{s}", "2", f"dense reference test accuracy {s}", b,
        lambda d=d, a=a: csv(b).query("dataset==@d and arch==@a").reference_test_accuracy.iloc[0], t)
    add(f"P2-ref-val-{s}", "2", f"dense reference validation accuracy {s}", b,
        lambda d=d, a=a: csv(b).query("dataset==@d and arch==@a").reference_val_accuracy.iloc[0], v)

# ---------------- Phase 3
c3 = "architecture_generalization/phase3_cross_setting_summary.csv"
for stmt, sup, opp in [("1 PPO > uniform", 5, 0), ("2 PPO > global magnitude", 3, 0), ("3 PPO > LAMP", 0, 5),
                       ("4 PPO > ERK", 3, 1), ("5 PPO > random", 6, 0)]:
    add(f"P3-{stmt[:1]}-sup", "3", f"{stmt}: supported settings", c3,
        lambda st=stmt: csv(c3).set_index("statement").loc[st, "supported"], sup, 0)
    add(f"P3-{stmt[:1]}-opp", "3", f"{stmt}: opposite settings", c3,
        lambda st=stmt: csv(c3).set_index("statement").loc[st, "opposite"], opp, 0)
add("P3-8-sup", "3", "correct prior improves final accuracy: supported", c3,
    lambda: csv(c3).set_index("statement").loc["8 correct prior improves final accuracy (excess over matched GM)", "supported"], 0, 0)
add("P3-6-sup", "3", "correct prior improves exploration (V_RL acc AUC): supported", c3,
    lambda: csv(c3).set_index("statement").loc["6 correct prior improves exploration (V_RL accuracy AUC)", "supported"], 3, 0)
s3 = "architecture_generalization/phase3_statistics.csv"
st3 = lambda: csv(s3).query("endpoint=='test_accuracy'")  # noqa: E731
for s, v in [("cifar10_simplecnn", 31.36), ("cifar10_lenet5", 16.29), ("cifar10_resnet8", 14.17), ("cifar100_simplecnn", 29.69),
             ("cifar100_lenet5", 3.67), ("cifar100_resnet8", 0.23)]:
    add(f"P3-unif-{s}", "3", f"P0 - uniform mean test diff {s}", s3,
        lambda s=s: st3().query("setting==@s and comparison=='P0 - uniform'").mean_diff.iloc[0], v)
add("P3-random-min", "3", "P0 - random min over settings", s3, lambda: st3().query("comparison=='P0 - random'").mean_diff.min(), 27.89)
add("P3-random-max", "3", "P0 - random max over settings", s3, lambda: st3().query("comparison=='P0 - random'").mean_diff.max(), 64.36)
add("P3-lamp-min", "3", "P0 - LAMP most negative", s3, lambda: st3().query("comparison=='P0 - lamp'").mean_diff.min(), -1.64)
add("P3-lamp-c10r8", "3", "P0 - LAMP C10 ResNet-8", s3,
    lambda: st3().query("comparison=='P0 - lamp' and setting=='cifar10_resnet8'").mean_diff.iloc[0], 0.15)

# ---------------- Phase 4
q4 = "phase4/phase4_search_quality.csv"
sq = lambda: csv(q4)  # noqa: E731
add("P4-S0-top10-max", "4", "terminal (S0) selected top-10 reach, max over settings", q4,
    lambda: sq().query("condition=='S0'").selected_reach_top10.max(), 0.0)
add("P4-S0-best-top10-min", "4", "S0 best-seen top-10 reach, min", q4, lambda: sq().query("condition=='S0'").best_seen_reach_top10.min(), 0.95)
add("P4-S2A-top10-min", "4", "archive (S2A) selected top-10 reach, min", q4, lambda: sq().query("condition=='S2A'").selected_reach_top10.min(), 0.75)
add("P4-S2A-top20-min", "4", "archive (S2A) selected top-20 reach, min", q4, lambda: sq().query("condition=='S2A'").selected_reach_top20.min(), 1.0)
add("P4-S0-rank-min", "4", "S0 median selected V_RL rank, min", q4, lambda: sq().query("condition=='S0'").selected_rank_vrl_median.min(), 65, 0.5)
add("P4-S0-rank-max", "4", "S0 median selected V_RL rank, max", q4, lambda: sq().query("condition=='S0'").selected_rank_vrl_median.max(), 173, 0.5)
for s, v in [("cifar10_simplecnn", 0.14), ("cifar10_lenet5", 0.90), ("cifar10_resnet8", -5.66), ("cifar100_simplecnn", 0.22),
             ("cifar100_lenet5", 0.45), ("cifar100_resnet8", -4.00)]:
    add(f"P4-archiveΔ-{s}", "4", f"S2A - S0 test accuracy {s}", q4,
        lambda s=s: sq().query("setting==@s and condition=='S2A'").selected_test_mean.iloc[0]
        - sq().query("setting==@s and condition=='S0'").selected_test_mean.iloc[0], v)
r4 = "phase4/phase4_reward_comparison.csv"
rc = lambda: csv(r4)  # noqa: E731
add("P4-K-c10r8-sp", "4", "K (R0, Stage A) C10 ResNet-8 sparsity", r4, lambda: rc().query("setting=='cifar10_resnet8' and reward=='R0'").sparsity_mean.iloc[0], 29.0, 0.05)
add("P4-K-c10r8-test", "4", "K C10 ResNet-8 test", r4, lambda: rc().query("setting=='cifar10_resnet8' and reward=='R0'").test_mean.iloc[0], 76.82)
add("P4-K-c100r8-sp", "4", "K C100 ResNet-8 sparsity", r4, lambda: rc().query("setting=='cifar100_resnet8' and reward=='R0'").sparsity_mean.iloc[0], 8.3, 0.05)
add("P4-K-c100r8-test", "4", "K C100 ResNet-8 test", r4, lambda: rc().query("setting=='cifar100_resnet8' and reward=='R0'").test_mean.iloc[0], 43.11)
add("P4-F-c10r8-sp", "4", "F (S2A) C10 ResNet-8 sparsity", q4, lambda: sq().query("setting=='cifar10_resnet8' and condition=='S2A'").selected_sparsity_mean.iloc[0], 47.8, 0.05)
add("P4-F-c10r8-test", "4", "F C10 ResNet-8 test", q4, lambda: sq().query("setting=='cifar10_resnet8' and condition=='S2A'").selected_test_mean.iloc[0], 70.83)
add("P4-F-c100r8-sp", "4", "F C100 ResNet-8 sparsity", q4, lambda: sq().query("setting=='cifar100_resnet8' and condition=='S2A'").selected_sparsity_mean.iloc[0], 27.7, 0.05)
add("P4-F-c100r8-test", "4", "F C100 ResNet-8 test", q4, lambda: sq().query("setting=='cifar100_resnet8' and condition=='S2A'").selected_test_mean.iloc[0], 39.22)
c4 = "phase4/phase4_classification.json"
for k, v in [("A_reward", False), ("D_final_model", False)]:
    add(f"P4-crit-{k}", "4", f"criterion {k} met", c4, lambda k=k: next(x["met"] for kk, x in js(c4)["criteria"].items() if kk.startswith(k[:2])), v, 0)
add("P4-winners", "4", "stage winners A/B/C", c4, lambda: "/".join(js(c4)["stage_winners"][k] for k in "ABC"), "R0/P2/S2A", 0)

# ---------------- Phase 5
cr5 = "phase5/phase5_constraint_results.csv"
c5 = lambda: csv(cr5)  # noqa: E731
add("P5-C1-test-feas", "5", "C1 selected policies feasible on test (share)", cr5, lambda: c5().query("condition=='C1'").selected_feasible_test.mean(), 0.675)
add("P5-C1-vs-feas", "5", "C1 feasible on VAL-SELECT (share)", cr5, lambda: c5().query("condition=='C1'").selected_feasible_val_select.mean(), 0.992)
add("P5-C1-vrl-feas", "5", "C1 feasible on VAL-RL (share)", cr5, lambda: c5().query("condition=='C1'").selected_feasible_val_rl.mean(), 1.0)
add("P5-C1-c100r8-feas", "5", "C1 test feasibility C100 ResNet-8", cr5, lambda: c5().query("condition=='C1' and setting=='cifar100_resnet8'").selected_feasible_test.mean(), 0.0)
add("P5-C1-vs-but-not-test", "5", "C1 runs feasible on VAL-SELECT but not on test", cr5,
    lambda: int((c5().query("condition=='C1'").selected_feasible_val_select & ~c5().query("condition=='C1'").selected_feasible_test).sum()), 39, 0)
for s, cond, v in [("cifar10_resnet8", "C0", 7.32), ("cifar10_resnet8", "C1", 1.18), ("cifar100_resnet8", "C0", 4.05), ("cifar100_resnet8", "C1", 1.16)]:
    add(f"P5-drop-{s}-{cond}", "5", f"{cond} {s} test drop vs dense", cr5, lambda s=s, cond=cond: c5().query("condition==@cond and setting==@s").test_drop.mean(), v)
for s, cond, v in [("cifar10_resnet8", "C0", 48.8), ("cifar10_resnet8", "C1", 24.7), ("cifar100_resnet8", "C0", 23.5), ("cifar100_resnet8", "C1", 16.1)]:
    add(f"P5-sp-{s}-{cond}", "5", f"{cond} {s} sparsity", cr5, lambda s=s, cond=cond: c5().query("condition==@cond and setting==@s").selected_sparsity.mean(), v, 0.05)
l5 = "phase5/phase5_lamp_comparison.csv"
for s, v in zip(SET, [-0.49, -0.32, -0.53, -0.29, -0.26, -0.51]):
    add(f"P5-lamp-{s}", "5", f"C1 - LAMP test {s}", l5, lambda s=s: csv(l5).query("setting==@s and condition=='C1'").mean_diff.iloc[0], v)
cl5 = "phase5/phase5_classification.json"

# ---------------- Phase 6
pp = "phase6/phase6_pre_post_accuracy.csv"
for m, pre, post in [("ppo", 57.08, 58.38), ("lamp", 57.48, 58.48), ("global", 55.08, 58.36), ("dense", 57.91, 58.51)]:
    add(f"P6-pre-{m}", "6", f"mean pre-FT test {m} (unweighted over 6 settings)", pp, lambda m=m: csv(pp).query("method.str.lower().str.contains(@m)", engine="python").test_pre_mean.mean(), pre)
    add(f"P6-post-{m}", "6", f"mean post-FT test {m}", pp, lambda m=m: csv(pp).query("method.str.lower().str.contains(@m)", engine="python").test_post_mean.mean(), post)
pl6 = "phase6/phase6_ppo_vs_lamp.csv"
post = lambda: csv(pl6)[csv(pl6).comparison.str.contains("post")]  # noqa: E731
for s, v, lo, hi in [("cifar10_simplecnn", -0.22, -0.28, -0.15), ("cifar10_lenet5", -0.18, -0.28, -0.09), ("cifar10_resnet8", 0.00, -0.02, 0.03),
                     ("cifar100_simplecnn", -0.05, -0.09, -0.01), ("cifar100_lenet5", -0.14, -0.19, -0.08), ("cifar100_resnet8", -0.04, -0.07, -0.01)]:
    add(f"P6-gap-{s}", "6", f"PPO - LAMP post-FT {s}", pl6, lambda s=s: post().query("setting==@s").mean_diff.iloc[0], v)
    add(f"P6-gaplo-{s}", "6", f"PPO - LAMP post-FT CI low {s}", pl6, lambda s=s: post().query("setting==@s").t_ci95_low.iloc[0], lo)
    add(f"P6-gaphi-{s}", "6", f"PPO - LAMP post-FT CI high {s}", pl6, lambda s=s: post().query("setting==@s").t_ci95_high.iloc[0], hi)
add("P6-equiv-count", "6", "post-FT PPO-LAMP CI inside +-0.5 (count)", pl6, lambda: int((post().t_ci95_low > -0.5).sum() & 0 + ((post().t_ci95_low > -0.5) & (post().t_ci95_high < 0.5)).sum()), 6, 0)
pg6 = "phase6/phase6_ppo_vs_global.csv"
pgp = lambda: csv(pg6)[csv(pg6).comparison.str.contains("post")]  # noqa: E731
add("P6-glob-min", "6", "PPO - global post-FT min", pg6, lambda: pgp().mean_diff.min(), -0.16)
add("P6-glob-max", "6", "PPO - global post-FT max", pg6, lambda: pgp().mean_diff.max(), 0.12)

# ---------------- Paper 3
p3 = "paper3/h2_e2_latency_correlations.csv"
for t, sp, sm, fr in [("simplecnn_c10", 0.32, 0.89, 0.3905), ("resnet8_c10", 0.55, 0.98, 0.3061)]:
    add(f"PP3-rho-param-{t}", "P3", f"Spearman param vs latency {t}", p3, lambda t=t: csv(p3).set_index("table").loc[t, "spearman_param_reduction_vs_latency_gain"], sp)
    add(f"PP3-rho-mac-{t}", "P3", f"Spearman MAC vs latency {t}", p3, lambda t=t: csv(p3).set_index("table").loc[t, "spearman_mac_reduction_vs_latency_gain"], sm)
    add(f"PP3-frac-{t}", "P3", f"sparser-but-slower fraction {t}", p3, lambda t=t: csv(p3).set_index("table").loc[t, "fraction"], fr, 0.0005)
pd3 = "paper3/paper3_proxy_decision.json"
rd3 = "paper3/resnet8_threshold_calibration_decision.json"

# ---------------- Phase 7A
e7 = "phase7a_sustainability/energy_summary.csv"
a7 = "phase7a_sustainability/accuracy_results.csv"
for arch, meth, sv, lower in [("simplecnn", "ppo", 0.228, 5), ("resnet8", "ppo", 0.134, 5), ("simplecnn", "uniform", 0.279, 5),
                              ("simplecnn", "lamp", 0.121, 5), ("resnet8", "uniform", 0.118, 5), ("resnet8", "lamp", 0.089, 5)]:
    add(f"P7A-sav-{arch}-{meth}", "7A", f"median energy saving {arch} {meth}", e7, lambda a=arch, m=meth: csv(e7).query("arch==@a and method==@m").saving_median.iloc[0], sv, 0.0005)
    add(f"P7A-low-{arch}-{meth}", "7A", f"sessions lower than dense {arch} {meth}", e7, lambda a=arch, m=meth: csv(e7).query("arch==@a and method==@m").sessions_lower_than_dense.iloc[0], lower, 0)
add("P7A-dense-e", "7A", "SimpleCNN dense J/1000 (median)", e7, lambda: csv(e7).query("arch=='simplecnn' and method=='dense'").e1000_median_j.iloc[0], 11.054, 0.0005)
for st, meth, rl, sel in [("cifar10_simplecnn", "dense", 80.60, 80.00), ("cifar10_simplecnn", "ppo", 75.52, 76.00), ("cifar10_simplecnn", "uniform", 79.12, 78.48),
                          ("cifar10_simplecnn", "lamp", 78.84, 79.44), ("cifar10_resnet8", "dense", 78.40, 78.52), ("cifar10_resnet8", "ppo", 75.16, 74.72),
                          ("cifar10_resnet8", "uniform", 75.60, 76.00), ("cifar10_resnet8", "lamp", 75.24, 75.72)]:
    add(f"P7A-rl-{st}-{meth}", "7A", f"VAL-RL post {st} {meth}", a7, lambda s=st, m=meth: csv(a7).query("setting==@s and method==@m").val_rl_post.iloc[0], rl)
    add(f"P7A-sel-{st}-{meth}", "7A", f"VAL-SELECT post {st} {meth}", a7, lambda s=st, m=meth: csv(a7).query("setting==@s and method==@m").val_select_post.iloc[0], sel)
ss7 = "phase7a_sustainability/sustainability_summary.json"

# ---------------- Phase 7B
f7 = "phase7b_green_pruning/final_summary.csv"
fb = lambda: csv(f7).set_index("id")  # noqa: E731
for mid, sel, d, mem, mac in [("0000", 80.00, 0.0, 0.0, 0.0), ("2225", 78.88, -1.12, 0.634, 0.360), ("uniform_r0.208", 78.76, -1.24, 0.373, 0.362),
                              ("lamp_s56.2", 79.64, -0.36, 0.633, 0.193), ("phase7a_ppo", 76.00, -4.00, 0.801, 0.426)]:
    add(f"P7B-sel-{mid}", "7B", f"VAL-SELECT {mid}", f7, lambda m=mid: fb().loc[m, "val_select_post"], sel)
    add(f"P7B-d-{mid}", "7B", f"delta VAL-SELECT {mid}", f7, lambda m=mid: fb().loc[m, "delta_val_select_pp"], d)
    add(f"P7B-mem-{mid}", "7B", f"memory reduction {mid}", f7, lambda m=mid: fb().loc[m, "memory_reduction"], mem, 0.0005)
    add(f"P7B-mac-{mid}", "7B", f"MAC reduction {mid}", f7, lambda m=mid: fb().loc[m, "mac_reduction"], mac, 0.0005)
add("P7B-ppo-widths", "7B", "new PPO widths", f7, lambda: fb().loc["2225", "widths"], "[26, 51, 102, 102]", 0)
add("P7B-ppo-rl", "7B", "new PPO VAL-RL", f7, lambda: fb().loc["2225", "val_rl_post"], 80.00)
st7 = "phase7b_green_pruning/statistics.json"
add("P7B-green-seeds", "7B", "GREEN-SUCCESS seeds", st7, lambda: js(st7)["green_success_seeds"]["count"], 0, 0)
add("P7B-seed-min", "7B", "worst seed-selection delta VAL-SELECT", f7, lambda: csv(f7)[~csv(f7).id.isin(["0000", "uniform_r0.208", "lamp_s56.2", "phase7a_ppo"])].delta_val_select_pp.min(), -1.28)
add("P7B-seed-max", "7B", "best seed-selection delta VAL-SELECT", f7, lambda: csv(f7)[~csv(f7).id.isin(["0000", "uniform_r0.208", "lamp_s56.2", "phase7a_ppo"])].delta_val_select_pp.max(), -1.08)
add("P7B-pilot100", "7B", "pilot N=100k CV", st7, lambda: js(st7)["pilots"]["pilot_n100000.json"]["cv"], 0.1136, 0.00005)
add("P7B-pilot200", "7B", "pilot N=200k CV", st7, lambda: js(st7)["pilots"]["pilot_n200000.json"]["cv"], 0.0401, 0.00005)
add("P7B-proxy-rho", "7B", "accuracy proxy Spearman", "phase7b_green_pruning/accuracy_proxy_metrics.json",
    lambda: next(v for k, v in js("phase7b_green_pruning/phase7b_final_preregistration.json")["proxy"]["validation"].items() if k == "rho"), 0.948, 0.0005)
add("P7B-proxy-prec", "7B", "accuracy proxy precision", "phase7b_green_pruning/phase7b_final_preregistration.json",
    lambda: js("phase7b_green_pruning/phase7b_final_preregistration.json")["proxy"]["validation"]["precision"], 0.655, 0.0005)
add("P7B-sel-count", "7B", "seeds selecting [2,2,2,5]", "phase7b_green_pruning/selected_policies.json",
    lambda: js("phase7b_green_pruning/selected_policies.json")["methods"]["new_ppo"]["frequency"], "5/20", 0)

# ---------------- Phase 7B energy confirmation
q7 = "phase7b_energy_confirmation/measurement_qualification.json"
add("EC-cv", "7B-EC", "qualification CV", q7, lambda: js(q7)["cv"], 0.1992, 0.00005)
add("EC-decision", "7B-EC", "qualification decision", q7, lambda: js(q7)["decision"], "not qualified", 0)
add("EC-confirmed", "7B-EC", "ENERGY-CONFIRMED", "phase7b_energy_confirmation/statistics.json",
    lambda: js("phase7b_energy_confirmation/statistics.json")["energy_confirmed"], "NO", 0)
add("EC-ckpt", "7B-EC", "checkpoint hashes all match", "phase7b_energy_confirmation/checkpoint_verification.json",
    lambda: js("phase7b_energy_confirmation/checkpoint_verification.json")["all_match"], True, 0)


# Discrepancies between result_freeze.md and the frozen files, found by this audit. The frozen file governs; the
# thesis uses the frozen value. Each entry: check id -> (value used in the thesis, explanation).
CORRECTIONS = {
    "P4-S0-top10-max": ("0% in 5/6 settings, 5% in C100 LeNet-5 (top-20: 0-5%)",
                        "result_freeze.md said 0% in all 6; that holds for the Phase-3 P0 reference rows, not Phase-4 S0"),
    "P4-S2A-top10-min": ("80-100% (S2A); 75-100% is the S1 archive-only row",
                         "result_freeze.md quoted the S1 range for S2A"),
    "P4-archiveΔ-cifar10_lenet5": ("+0.89 pp", "rounding: 0.8945 -> 0.89, not 0.90"),
    "P5-sp-cifar10_resnet8-C0": ("mean 46.2% (median 48.8%)", "the Phase-5 summary table reports median sparsity beside mean drops"),
    "P5-sp-cifar10_resnet8-C1": ("mean 24.5% (median 24.7%)", "as above"),
    "P5-sp-cifar100_resnet8-C0": ("mean 25.9% (median 23.5%)", "as above"),
    "P5-sp-cifar100_resnet8-C1": ("mean 15.6% (median 16.1%)", "as above"),
}


def main():
    rows, bad, corr = [], 0, []
    for cid, ph, claim, src, fn, stated, tol in checks:
        try:
            got = fn()
            got = got.item() if hasattr(got, "item") else got
            if isinstance(stated, (bool, str)) or isinstance(got, (bool, str)):
                ok = str(got) == str(stated)
            else:
                ok = abs(float(got) - float(stated)) <= tol + 1e-9
            shown = f"{got:.4f}" if isinstance(got, float) else str(got)
        except Exception as e:  # noqa: BLE001
            ok, shown = False, f"ERROR {e}"
        if not ok and cid in CORRECTIONS:
            res = "CORRECTED"
            corr.append(f"| {cid} | `{src}` | {shown} | {stated} | {CORRECTIONS[cid][0]} | {CORRECTIONS[cid][1]} |")
        else:
            res = "PASS" if ok else "**FAIL**"
            bad += not ok
        rows.append(f"| {cid} | {ph} | {claim} | `{src}` | {shown} | {stated} | {res} |")
    head = ["# Numerical audit (frozen source files vs integration package)", "",
            f"Generated by `results/thesis_final_writing/tools/numeric_audit.py`. {len(checks)} checks: "
            f"{len(checks) - bad - len(corr)} pass, {len(corr)} corrected (frozen file governs), {bad} unresolved fail. "
            "Tolerance = stated rounding. Integration files are not edited; corrections apply to the thesis text.", "",
            "## Corrections to result_freeze.md (frozen value is used in the thesis)", "",
            "| Check | Frozen source | Frozen value | result_freeze.md | Thesis uses | Explanation |", "|---|---|---|---|---|---|",
            *corr, "", "## All checks (source map)", "",
            "| Check | Phase | Claim | Frozen source | Frozen value | Stated (result_freeze.md) | Result |", "|---|---|---|---|---|---|---|"]
    open(OUT, "w", encoding="utf-8").write("\n".join(head + rows) + "\n")
    print(f"{len(checks)} checks, {len(corr)} corrected, {bad} fail")
    for r in rows:
        if "FAIL" in r:
            print(r.encode('ascii','replace').decode())


if __name__ == "__main__":
    main()
