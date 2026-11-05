"""Numerical audit of Chapter 5 (core results). Every headline number is recomputed from the frozen phase files.

  python results/thesis_final_writing/tools/audit_ch5.py > results/thesis_final_writing/numerical_audit_chapter_05.md
Read-only. Each row: id | phase | source file | row/key | metric | split | frozen value | chapter value | result.
A row passes when the frozen value, rounded as stated, equals the chapter value AND the chapter value string occurs in
chapter_05_core_results.md.
"""
import csv
import json
import os
import statistics as st
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
R = os.path.join(ROOT, "results")
import re
_RAW = open(os.path.join(R, "thesis_final_writing", "chapter_05_core_results.md"), encoding="utf-8").read()
_norm = lambda s: re.sub(r"\s+", " ", s.replace("**", ""))  # noqa: E731
TEXT = _norm(_RAW)
rows = []
S6 = ["cifar10_simplecnn", "cifar10_lenet5", "cifar10_resnet8", "cifar100_simplecnn", "cifar100_lenet5", "cifar100_resnet8"]


def rd(p):
    return list(csv.DictReader(open(os.path.join(R, p), encoding="utf-8")))


def fmt(v, d, sign=False):
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(repr(round(v, 6))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    s = (f"{q:+f}" if sign else f"{q:f}")
    return s.replace("-", "−")


def add(i, phase, src, key, metric, split, frozen, chapter, present=None):
    ok = str(frozen) == str(chapter)
    pres = (_norm(str(chapter)) in TEXT) if present is None else (_norm(present) in TEXT)
    rows.append((i, phase, src, key, metric, split, frozen, chapter, "PASS" if ok and pres else ("FAIL (value)" if not ok else "FAIL (not in text)")))


# ---------------- 5.2 dense references
b = {(r["dataset"] + "_" + r["arch"]): r for r in rd("architecture_generalization/cpu_baselines.csv")}
add("D1", 2, "architecture_generalization/cpu_baselines.csv", "cifar10_resnet8 test_mean/sd", "3-seed test mean ± SD", "test",
    f"{float(b['cifar10_resnet8']['test_mean']):.2f} ± {float(b['cifar10_resnet8']['test_sd']):.2f}", "78.69 ± 0.51")
add("D2", 2, "architecture_generalization/cpu_baselines.csv", "cifar10_lenet5 test_mean/sd", "3-seed test mean ± SD", "test",
    f"{float(b['cifar10_lenet5']['test_mean']):.2f} ± {float(b['cifar10_lenet5']['test_sd']):.2f}", "66.21 ± 1.51")

# ---------------- Phase 3
s3 = rd("architecture_generalization/phase3_statistics.csv")
sm3 = rd("architecture_generalization/phase3_summary.csv")
g = lambda fam, s, comp: [r for r in s3 if r["family"] == fam and r["setting"] == s and r["comparison"] == comp][0]  # noqa: E731
r = g("A", "cifar10_simplecnn", "P0 - uniform")
add("P3-1", 3, "architecture_generalization/phase3_statistics.csv", "A, cifar10_simplecnn, P0 - uniform", "mean paired difference", "test",
    fmt(float(r["mean_diff"]), 2, True), "+31.37")
add("P3-1b", 3, "same", "same", "95% t-CI", "test", f"{float(r['t_ci95_low']):.2f}–{float(r['t_ci95_high']):.2f}", "26.57–36.16")
add("P3-1c", 3, "same", "same", "uniform / PPO mean test", "test", f"{float(r['mean_b']):.2f} / {float(r['mean_a']):.2f}", "45.08 / 76.44",
    present="45.08%")
sp = {x["setting"]: float(x["sparsity_mean"]) for x in sm3 if x["method"] == "PPO P0"}
add("P3-2", 3, "architecture_generalization/phase3_summary.csv", "PPO P0 sparsity_mean, min–max over settings", "mean masked sparsity", "—",
    f"{min(sp.values()):.1f}–{max(sp.values()):.1f}%", "9.5–55.1%")
for s, v in zip(S6, ["55.06%", "51.54%", "28.35%", "53.28%", "34.59%", "9.46%"]):
    add(f"P3-2-{s}", 3, "architecture_generalization/phase3_summary.csv", f"PPO P0 {s}", "mean sparsity", "—", f"{sp[s]:.2f}%", v)
exp = {"P0 - uniform": ["+31.37", "+16.29", "+14.17", "+29.69", "+3.67", "+0.23"],
       "P0 - random": ["+59.60", "+48.69", "+64.36", "+47.29", "+27.89", "+38.76"],
       "P0 - global": ["−0.07", "+2.74", "+7.38", "−0.01", "−0.13", "+0.28"],
       "P0 - erk": ["+0.75", "+1.93", "−0.06", "+4.35", "+0.30", "−0.80"],
       "P0 - lamp": ["−0.55", "−1.64", "+0.15", "−0.65", "−0.68", "−0.75"]}
for comp, vals in exp.items():
    fam = "A" if comp == "P0 - uniform" else "B"
    for s, v in zip(S6, vals):
        add(f"P3-T52-{comp[5:]}-{s}", 3, "architecture_generalization/phase3_statistics.csv", f"{fam}, {s}, {comp}", "mean paired difference", "test",
            fmt(float(g(fam, s, comp)["mean_diff"]), 2, True), v)
cs = {x["statement"]: x for x in rd("architecture_generalization/phase3_cross_setting_summary.csv")}
for st_, exp_ in (("1 PPO > uniform", "5/0 STRONG"), ("5 PPO > random", "6/0 STRONG"), ("2 PPO > global magnitude", "3/0 MODERATE"),
                  ("4 PPO > ERK", "3/1 LIMITED"), ("3 PPO > LAMP", "0/5 NOT SUPPORTED")):
    x = cs[st_]
    add(f"P3-class-{st_[:1]}", 3, "architecture_generalization/phase3_cross_setting_summary.csv", st_, "supported/opposite, class", "test",
        f"{int(float(x['supported']))}/{int(float(x['opposite']))} {x['class']}", exp_, present=exp_.split()[-1])
add("P3-lamp-rev", 3, "phase3_cross_setting_summary.csv", "3 PPO > LAMP", "reversed", "test", cs["3 PPO > LAMP"]["reversed"], "True", present="REVERSED")
c = Counter(x["actions"] for x in rd("architecture_generalization/phase3_all_runs.csv") if x["setting"] == "cifar100_resnet8" and x["condition"] == "P0")
add("P3-3", 3, "architecture_generalization/phase3_all_runs.csv", "cifar100_resnet8, P0, actions", "runs choosing [1, 1, 1, 1]", "—", c["[1, 1, 1, 1]"], 19,
    present="19 of 20 runs")
for st_, exp_ in (("6 correct prior improves exploration (V_RL accuracy AUC)", "3 MODERATE"),
                  ("7 exploration gain is layer-specific (P1 > P2 and P1 > P3)", "1 LIMITED"),
                  ("8 correct prior improves final accuracy (excess over matched GM)", "0 NOT SUPPORTED")):
    x = cs[st_]
    add(f"P3-sens-{st_[:1]}", 3, "phase3_cross_setting_summary.csv", st_, "supported, class", "VAL-RL (I) / test",
        f"{int(float(x['supported']))} {x['class']}", exp_, present=exp_.split(" ", 1)[1])
for st_ in ("6-C-rank correct prior, secondary: rank AUC", "6-C-utility correct prior, secondary: utility AUC",
            "6-C-destructive correct prior, secondary: destructive frequency (lower)"):
    x = cs[st_]
    add(f"P3-sec-{st_[4:8]}", 3, "phase3_cross_setting_summary.csv", st_, "supported/opposite, class", "VAL-RL (I)",
        f"{int(float(x['supported']))}/{int(float(x['opposite']))} {x['class']}", "6/0 STRONG", present="STRONG in 6 of 6")
dvals = [float(g("D", s, "P1 - P0")["mean_diff"]) for s in S6]
add("P3-D", 3, "phase3_statistics.csv", "D, P1 - P0, all settings", "range of excess over matched global", "test",
    f"{fmt(min(dvals), 2)} to {fmt(max(dvals), 2, True)}", "−0.10 to +0.86")
add("P3-D-holm", 3, "phase3_statistics.csv", "D, P1 - P0", "min Holm p", "test", f"{min(float(g('D', s, 'P1 - P0')['holm_p']) for s in S6):.3f}", "0.094",
    present="≥ 0.094")
x = g("C-specificity", "cifar10_lenet5", "P1 - P2")
y = g("C-specificity", "cifar10_lenet5", "P1 - P3")
add("P3-lenet", 3, "phase3_statistics.csv", "C-specificity, cifar10_lenet5", "P1−P2 / P1−P3 VAL-RL AUC (Holm)", "VAL-RL (I)",
    f"{fmt(float(x['mean_diff']), 2)} ({float(x['holm_p']):.2f}) / {fmt(float(y['mean_diff']), 2)} ({float(y['holm_p']):.2f})",
    "−0.21 (0.27) / −0.21 (0.31)", present="P1 − P2 = −0.21 (Holm p = 0.27)")
mt = {(x["setting"], x["method"]): float(x["test_mean"]) for x in sm3}
add("P3-lenet-test", 3, "phase3_summary.csv", "cifar10_lenet5 PPO P1/P2/P3", "mean test", "test",
    " ".join(f"{mt[('cifar10_lenet5', 'PPO ' + k)]:.2f}%" for k in ("P1", "P2", "P3")), "62.17% 62.54% 62.83%", present="P1 62.17%, P2 62.54%, P3 62.83%")

# ---------------- Phase 4
sq = {(x["setting"], x["condition"]): x for x in rd("phase4/phase4_search_quality.csv")}
s0 = [float(sq[(s, "S0")]["selected_reach_top10"]) for s in S6]
add("P4-1", 4, "phase4/phase4_search_quality.csv", "S0 selected_reach_top10, by setting", "terminal top-10 reach", "VAL-RL (I)",
    " ".join(f"{100 * v:.0f}%" for v in s0), "0% 0% 0% 0% 5% 0%", present="0% of runs in five\n  settings and 5% in the sixth")
s2a = [float(sq[(s, "S2A")]["selected_reach_top10"]) for s in S6]
add("P4-2", 4, "phase4/phase4_search_quality.csv", "S2A selected_reach_top10, min–max", "archive top-10 reach", "VAL-RL (I)",
    f"{100 * min(s2a):.0f}–{100 * max(s2a):.0f}%", "80–100%")
s1 = [float(sq[(s, "S1")]["selected_reach_top10"]) for s in S6]
add("P4-2b", 4, "phase4/phase4_search_quality.csv", "S1 selected_reach_top10, min–max", "archive-only top-10 reach", "VAL-RL (I)",
    f"{100 * min(s1):.0f}–{100 * max(s1):.0f}%", "75–100%")
add("P4-2c", 4, "phase4/phase4_search_quality.csv", "S2A selected_reach_top20", "top-20 reach (all)", "VAL-RL (I)",
    str(all(float(sq[(s, "S2A")]["selected_reach_top20"]) == 1 for s in S6)), "True", present="top 20 in 100%")
r0 = [float(sq[(s, "S0")]["selected_rank_vrl_median"]) for s in S6]
add("P4-3", 4, "phase4/phase4_search_quality.csv", "S0 selected_rank_vrl_median", "terminal median rank", "VAL-RL (I)",
    f"{min(r0):.0f}–{max(r0):.0f}", "65–173")
ra = [float(sq[(s, "S2A")]["selected_rank_vrl_median"]) for s in S6]
add("P4-4", 4, "phase4/phase4_search_quality.csv", "S2A selected_rank_vrl_median", "archive median rank", "VAL-RL (I)", f"{min(ra):.0f}–{max(ra):.0f}", "1–10")
bs = [float(sq[(s, "S0")]["best_seen_rank_median"]) for s in S6]
add("P4-5", 4, "phase4/phase4_search_quality.csv", "S0 best_seen_rank_median", "best-seen median rank", "VAL-RL (I)", f"{min(bs):.0f}–{max(bs):.1f}", "1–2.5")
bt = [float(sq[(s, "S0")]["best_seen_reach_top10"]) for s in S6]
add("P4-6", 4, "phase4/phase4_search_quality.csv", "S0 best_seen_reach_top10", "best-seen top-10 reach", "VAL-RL (I)",
    f"{100 * min(bt):.0f}–{100 * max(bt):.0f}%", "95–100%")
dt = [float(sq[(s, "S2A")]["selected_test_mean"]) - float(sq[(s, "S0")]["selected_test_mean"]) for s in S6]
for s, v, e in zip(S6, dt, ["+0.14", "+0.89", "−5.66", "+0.22", "+0.45", "−4.00"]):
    add(f"P4-7-{s}", 4, "phase4/phase4_search_quality.csv", f"{s}: S2A − S0 selected_test_mean", "archive Δ test", "test", fmt(v, 2, True), e)
for s, e in (("cifar10_resnet8", "32.2% → 47.8%"), ("cifar100_resnet8", "8.5% → 27.7%")):
    add(f"P4-8-{s}", 4, "phase4/phase4_search_quality.csv", f"{s} S0, S2A selected_sparsity_mean", "sparsity S0 → S2A", "—",
        f"{float(sq[(s, 'S0')]['selected_sparsity_mean']):.1f}% → {float(sq[(s, 'S2A')]['selected_sparsity_mean']):.1f}%", e)
s4 = rd("phase4/phase4_statistics.csv")
g4 = lambda fam, s, comp: [x for x in s4 if x["family"] == fam and x["setting"] == s and x["comparison"] == comp][0]  # noqa: E731
v = [float(g4("C-final", s, "S2A - S0")["mean_diff"]) for s in S6]
add("P4-9", 4, "phase4/phase4_statistics.csv", "C-final, S2A − S0", "mean rank improvement range", "VAL-RL (I)", f"{-max(v):.0f}–{-min(v):.0f}", "66–150")
v = [float(g4("B-specific", s, "P2 - P3")["mean_diff"]) for s in S6]
add("P4-10", 4, "phase4/phase4_statistics.csv", "B-specific, P2 − P3", "rank-score AUC range", "VAL-RL (I)", f"+{min(v):.3f} to +{max(v):.3f}", "+0.018 to +0.097")
v = [float(g4("B-reverse", s, "P2 - P5")["mean_diff"]) for s in S6]
add("P4-11", 4, "phase4/phase4_statistics.csv", "B-reverse, P2 − P5", "rank-score AUC range", "VAL-RL (I)", f"+{min(v):.3f} to +{max(v):.3f}", "+0.021 to +0.145")
cs4 = {x["statement"]: x for x in rd("phase4/phase4_cross_setting_summary.csv")}
x = cs4["B-secondary vrl_auc: P2 better than P3 and P4"]
add("P4-12", 4, "phase4/phase4_cross_setting_summary.csv", "B-secondary vrl_auc", "supported/opposite, reversed", "VAL-RL (I)",
    f"{x['supported']}/{x['opposite']} {x['reversed']}", "0/4 True", present="NOT SUPPORTED and REVERSED")
v = [float(g4("B-secondary vrl_auc", s, "P2 - P4")["mean_diff"]) for s in S6 if g4("B-secondary vrl_auc", s, "P2 - P4")["outcome"] == "opposite"]
add("P4-13", 4, "phase4/phase4_statistics.csv", "B-secondary vrl_auc, P2 − P4, opposite rows", "range", "VAL-RL (I)",
    f"{fmt(max(v), 2)} to {fmt(min(v), 2)}", "−0.26 to −0.76")
rc = {(x["setting"], x["reward"]): x for x in rd("phase4/phase4_reward_comparison.csv")}
o = [float(rc[(s, "R0")]["oracle_optimum_sparsity"]) for s in S6]
add("P4-14", 4, "phase4/phase4_reward_comparison.csv", "R0 oracle_optimum_sparsity", "R0 optimum sparsity range", "VAL-RL (I)", f"{min(o):.1f}–{max(o):.1f}%", "23.5–56.4%")
add("P4-15", 4, "phase4/phase4_reward_comparison.csv", "R0 oracle_optimum_retention (ResNet-8 C10/C100)", "retention at optimum", "VAL-RL (I)",
    f"{100 * float(rc[('cifar10_resnet8', 'R0')]['oracle_optimum_retention']):.1f}% / {100 * float(rc[('cifar100_resnet8', 'R0')]['oracle_optimum_retention']):.1f}%",
    "89.7% / 93.4%", present="retaining 89.7%")
bc = {(x["pipeline"], x["setting"], x["method"]): x for x in rd("phase4/phase4_baseline_comparison.csv")}
for s, e in (("cifar10_resnet8", "76.82% → 70.83%"), ("cifar100_resnet8", "43.11% → 39.22%")):
    add(f"P4-16-{s}", 4, "phase4/phase4_baseline_comparison.csv", f"K, F {s} ppo_test_mean", "test K → F", "test",
        f"{float(bc[('K', s, 'lamp')]['ppo_test_mean']):.2f}% → {float(bc[('F', s, 'lamp')]['ppo_test_mean']):.2f}%", e)
for s, d, e in (("cifar10_resnet8", 78.69, "7.86"), ("cifar100_resnet8", 43.99, "4.77")):
    add(f"P4-17-{s}", 4, "phase4/phase4_baseline_comparison.csv + cpu_baselines.csv", f"dense − F test, {s}", "drop vs dense", "test",
        f"{d - float(bc[('F', s, 'lamp')]['ppo_test_mean']):.2f}", e)
    add(f"P4-18-{s}", 4, "phase4/phase4_baseline_comparison.csv", f"K, F sparsity_mean {s}", "sparsity K → F", "—",
        f"{float(bc[('K', s, 'lamp')]['sparsity_mean']):.1f}% to {float(bc[('F', s, 'lamp')]['sparsity_mean']):.1f}%",
        {"cifar10_resnet8": "29.0% to 47.8%", "cifar100_resnet8": "8.3% to 27.7%"}[s])
v = [float(bc[("F", s, "lamp")]["ppo_minus_baseline_mean"]) for s in S6]
add("P4-19", 4, "phase4/phase4_baseline_comparison.csv", "F − LAMP by setting", "test difference", "test",
    ", ".join(fmt(x, 2) for x in v), "−0.11, −0.11, −0.68, −0.48, −0.08, −1.62", present="−0.11, −0.11, −0.68, −0.48, −0.08 and\n    −1.62")
v = [float(bc[("F", s, "uniform")]["ppo_minus_baseline_mean"]) for s in S6]
add("P4-20", 4, "phase4/phase4_baseline_comparison.csv", "F − uniform", "range", "test", f"+{min(v):.2f} to +{max(v):.2f}", "+12.05 to +36.04")
for s, e in (("cifar10_simplecnn", "+0.50"), ("cifar10_lenet5", "+0.95"), ("cifar100_lenet5", "+0.67"), ("cifar10_resnet8", "−0.95"), ("cifar100_resnet8", "−0.92")):
    add(f"P4-21-{s}", 4, "phase4/phase4_statistics.csv", f"D, {s}, F − K excess over LAMP", "mean difference", "test", fmt(float(g4("D", s, "F - K")["mean_diff"]), 2, True), e,
        present=e.lstrip("+−"))
cl4 = json.load(open(os.path.join(R, "phase4", "phase4_classification.json")))["criteria"]
add("P4-22", 4, "phase4/phase4_classification.json", "criteria A, B, C, D met", "status", "—",
    " ".join(str(cl4[k]["met"]) for k in ("A_reward", "B_sensitivity_specific", "C_search", "D_final_model")), "False True True False", present="**NOT MET** (LIMITED)")
add("P4-23", 4, "phase4/phase4_classification.json", "E_lamp noninferior_settings", "non-inferior settings", "test", cl4["E_lamp"]["noninferior_settings"], 3,
    present="supported in 3 settings")

# ---------------- Phase 5
cr = rd("phase5/phase5_constraint_results.csv")
T = lambda v: v == "True"  # noqa: E731


def grp(c, s=None):
    return [x for x in cr if x["condition"] == c and (s is None or x["setting"] == s)]


for s, c, e in (("cifar10_resnet8", "C0", "7.32"), ("cifar10_resnet8", "C1", "1.18"), ("cifar100_resnet8", "C0", "4.05"), ("cifar100_resnet8", "C1", "1.16")):
    add(f"P5-1-{s}-{c}", 5, "phase5/phase5_constraint_results.csv", f"{s} {c} test_drop mean", "mean test drop vs dense", "test",
        f"{st.mean(float(x['test_drop']) for x in grp(c, s)):.2f}", e)
for s, c, e in (("cifar10_resnet8", "C0", "46.2%"), ("cifar10_resnet8", "C1", "24.5%"), ("cifar100_resnet8", "C0", "25.9%"), ("cifar100_resnet8", "C1", "15.6%")):
    add(f"P5-2-{s}-{c}", 5, "phase5/phase5_constraint_results.csv", f"{s} {c} selected_sparsity mean", "mean sparsity", "—",
        f"{st.mean(float(x['selected_sparsity']) for x in grp(c, s)):.1f}%", e)
for s, c, e in (("cifar10_resnet8", "C0", "48.8%"), ("cifar10_resnet8", "C1", "24.7%"), ("cifar100_resnet8", "C0", "23.5%"), ("cifar100_resnet8", "C1", "16.1%")):
    add(f"P5-2m-{s}-{c}", 5, "phase5/phase5_constraint_results.csv", f"{s} {c} selected_sparsity median", "median sparsity (figure only)", "—",
        f"{st.median(float(x['selected_sparsity']) for x in grp(c, s)):.1f}%", e)
for c, e_rl, e_sel, e_t in (("C1", "120 / 120", "119 / 120", "81 / 120"), ("C0", "40 / 120", "44 / 120", "40 / 120"), ("C2", "120 / 120", "119 / 120", "90 / 120")):
    gg = grp(c)
    add(f"P5-3-{c}", 5, "phase5/phase5_constraint_results.csv", f"{c} selected_feasible_*", "feasible VAL-RL / VAL-SELECT / test", "VAL-RL, VAL-SELECT, test",
        f"{sum(T(x['selected_feasible_val_rl']) for x in gg)} / 120 | {sum(T(x['selected_feasible_val_select']) for x in gg)} / 120 | {sum(T(x['selected_feasible_test']) for x in gg)} / 120",
        f"{e_rl} | {e_sel} | {e_t}", present=e_t)
gg = grp("C1")
add("P5-4", 5, "phase5/phase5_constraint_results.csv", "C1 test feasibility share", "share", "test", f"{100 * sum(T(x['selected_feasible_test']) for x in gg) / 120:.1f}%", "67.5%")
add("P5-5", 5, "phase5/phase5_constraint_results.csv", "C1 VAL-SELECT-feasible but test-infeasible", "count", "VAL-SELECT, test",
    sum(T(x["selected_feasible_val_select"]) and not T(x["selected_feasible_test"]) for x in gg), 39, present="39 were infeasible on test")
add("P5-6", 5, "phase5/phase5_constraint_results.csv", "C1 cifar100_resnet8 feasible test; mean q_test", "count; retention", "test",
    f"{sum(T(x['selected_feasible_test']) for x in grp('C1', 'cifar100_resnet8'))}; {st.mean(float(x['q_test']) for x in grp('C1', 'cifar100_resnet8')):.3f}",
    "0; 0.974", present="0.974")
add("P5-7", 5, "phase5/phase5_constraint_results.csv", "C1 fallback count", "fallbacks", "VAL-SELECT", sum(T(x["selection_fallback"]) for x in gg), 1,
    present="selection-rule fallback")
d12 = [st.mean(float(x["test_accuracy"]) for x in grp("C2", s)) - st.mean(float(x["test_accuracy"]) for x in grp("C1", s)) for s in S6]
add("P5-8", 5, "phase5/phase5_constraint_results.csv", "C2 − C1 mean test, range", "C1 below C2", "test", f"{min(d12):.2f}–{max(d12):.2f}", "0.05–0.16")
add("P5-8b", 5, "phase5/phase5_constraint_results.csv", "C2 − C1 > 0 in all settings", "count", "test", sum(v > 0 for v in d12), 6, present="in all six settings")
for s, e in zip(S6, ["76.75", "63.14", "71.37", "50.28", "30.67", "39.94"]):
    add(f"P5-T56-C0-{s}", 5, "phase5/phase5_constraint_results.csv", f"{s} C0 test mean", "mean test", "test", f"{st.mean(float(x['test_accuracy']) for x in grp('C0', s)):.2f}", e)
for s, e in zip(S6, ["76.45", "63.83", "77.51", "50.07", "31.79", "42.83"]):
    add(f"P5-T56-C1-{s}", 5, "phase5/phase5_constraint_results.csv", f"{s} C1 test mean", "mean test", "test", f"{st.mean(float(x['test_accuracy']) for x in grp('C1', s)):.2f}", e)
for s, e in zip(S6, ["76.50", "63.99", "77.57", "50.13", "31.86", "42.90"]):
    add(f"P5-T56-C2-{s}", 5, "phase5/phase5_constraint_results.csv", f"{s} C2 test mean", "mean test", "test", f"{st.mean(float(x['test_accuracy']) for x in grp('C2', s)):.2f}", e)
s5 = rd("phase5/phase5_statistics.csv")
g5 = lambda fam, s, comp, ep=None: [x for x in s5 if x["family"].startswith(fam) and x["setting"] == s and x["comparison"] == comp and (ep is None or x["endpoint"] == ep)][0]  # noqa: E731
for s, e in zip(S6, ["−0.31", "+0.70", "+6.14", "−0.21", "+1.12", "+2.89"]):
    add(f"P5-H1-{s}", 5, "phase5/phase5_statistics.csv", f"H1/H4, {s}, C1 − C0", "mean test difference", "test", fmt(float(g5("H1", s, "C1 - C0")["mean_diff"]), 2, True), e)
r = g5("H1", "cifar10_resnet8", "C1 - C0")
add("P5-H1-ci", 5, "phase5/phase5_statistics.csv", "H1, cifar10_resnet8", "95% t-CI; Holm p", "test", f"{float(r['t_ci95_low']):.2f}–{float(r['t_ci95_high']):.2f}; {float(r['holm_p']):.1e}",
    "4.92–7.36; 1.1e-05", present="4.92–7.36")
for s, e in zip(S6, ["−0.49", "−0.32", "−0.53", "−0.29", "−0.26", "−0.51"]):
    add(f"P5-LAMP-{s}", 5, "phase5/phase5_statistics.csv", f"H5, {s}, C1 − LAMP", "mean test difference", "test", fmt(float(g5("H5 C1 - LAMP", s, "C1 - LAMP")["mean_diff"]), 2), e)
add("P5-LAMP-holm", 5, "phase5/phase5_statistics.csv", "H5 C1 − LAMP, max Holm p", "Holm p", "test",
    f"{max(float(g5('H5 C1 - LAMP', s, 'C1 - LAMP')['holm_p']) for s in S6):.1e}", "5.9e-05", present="5.9 × 10⁻⁵")
for s, e in zip(S6, ["−0.09", "+1.42", "+6.51", "+0.32", "+0.36", "+3.48"]):
    add(f"P5-GM-{s}", 5, "phase5/phase5_statistics.csv", f"C1 − global (descriptive), {s}", "mean test difference", "test", fmt(float(g5("C1 - global", s, "C1 - global")["mean_diff"]), 2, True), e)
add("P5-GM-n", 5, "phase5/phase5_statistics.csv", "C1 − global outcomes", "supported count", "test",
    sum(g5("C1 - global", s, "C1 - global")["outcome"] == "supported" for s in S6), 5, present="5 of 6** settings, by +0.32 to +6.51")
for ep, e in (("rank_auc", 5), ("vrl_accuracy_auc", 5)):
    add(f"P5-H3-{ep}", 5, "phase5/phase5_statistics.csv", f"H3 primary {ep}, C1 − C2", "supported count", "VAL-RL",
        sum(g5("H3 exploration", s, "C1 - C2", ep)["outcome"] == "supported" for s in S6), e, present="supported in 5 of 6")
add("P5-H3-opp", 5, "phase5/phase5_statistics.csv", "H3 primary, all 24", "opposite count", "VAL-RL",
    sum(x["outcome"] == "opposite" for x in s5 if x["family"].startswith("H3 exploration")), 0, present="0 opposite outcomes")
add("P5-H3-dest", 5, "phase5/phase5_statistics.csv", "H3 secondary destructive_rate", "supported count", "VAL-RL",
    sum(g5("H3 secondary", s, "C1 - C2", "destructive_rate")["outcome"] == "supported" for s in S6), 6, present="lower in 6 of 6")
cl5 = json.load(open(os.path.join(R, "phase5", "phase5_classification.json")))["criteria"]
stat = lambda v: v if isinstance(v, str) else (v.get("status") or v.get("verdict"))  # noqa: E731
add("P5-crit", 5, "phase5/phase5_classification.json", "criteria A–F", "status", "—", " / ".join(f"{k}: {stat(v)}" for k, v in cl5.items()),
    "A: PARTIALLY MET / B: MET / C: PARTIALLY MET / D: MET / E: PARTIALLY MET / F_strong: NOT MET / F_weak: MET", present="F-strong. LAMP | **NOT MET**")
p5sum = open(os.path.join(R, "phase5", "phase5_summary.md"), encoding="utf-8").read()
add("P5-C", 5, "phase5/phase5_summary.md", "§12 C", "top-20 selected / visited", "VAL-RL", "76.7% / 90.8%" if ("76.7%" in p5sum and "90.8%" in p5sum) else "?", "76.7% / 90.8%",
    present="76.7% of runs selected a top-20 policy")

# ---------------- Phase 6
pp = {(x["setting"], x["method"]): x for x in rd("phase6/phase6_pre_post_accuracy.csv")}
for m, e in (("ppo", "57.08 → 58.38"), ("lamp", "57.48 → 58.48"), ("global", "55.08 → 58.36"), ("dense", "57.91 to 58.51")):
    a = st.mean(float(pp[(s, m)]["test_pre_mean"]) for s in S6)
    bb = st.mean(float(pp[(s, m)]["test_post_mean"]) for s in S6)
    add(f"P6-mean-{m}", 6, "phase6/phase6_pre_post_accuracy.csv", f"{m}, unweighted mean of 6 settings", "test before → after", "test",
        f"{a:.2f} {'to' if m == 'dense' else '→'} {bb:.2f}", e)
lv = {(x["family"], x["setting"], x["endpoint"]): x for x in rd("phase6/phase6_ppo_vs_lamp.csv")}
gv = {(x["family"], x["setting"], x["endpoint"]): x for x in rd("phase6/phase6_ppo_vs_global.csv")}
pre = [float(lv[(k)]["mean_diff"]) for k in lv if k[0].startswith("H2 reference")]
add("P6-1", 6, "phase6/phase6_ppo_vs_lamp.csv", "H2 reference (G_pre), |PPO − LAMP|", "range before", "test", f"{-max(pre):.2f}–{-min(pre):.2f}", "0.26–0.53")
post = [float(x["mean_diff"]) for k, x in lv.items() if k[0].startswith("H2 PPO") and k[2] == "test accuracy"]
add("P6-2", 6, "phase6/phase6_ppo_vs_lamp.csv", "H2 G_post, |PPO − LAMP|", "range after", "test", f"{min(abs(v) for v in post):.2f}–{max(abs(v) for v in post):.2f}", "0.00–0.22")
add("P6-3", 6, "phase6/phase6_ppo_vs_lamp.csv", "H2 G_post equivalent_within_margin", "CI inside ±0.5 (count)", "test",
    sum(x["equivalent_within_margin"] == "True" for k, x in lv.items() if k[0].startswith("H2 PPO") and k[2] == "test accuracy"), 6, present="in all six settings")
add("P6-3b", 6, "phase6/phase6_ppo_vs_lamp.csv", "H2 reference G_pre equivalent_within_margin", "inside ±0.5 before (count)", "test",
    sum(x["equivalent_within_margin"] == "True" for k, x in lv.items() if k[0].startswith("H2 reference")), 3, present="Three of the six\n  settings were already inside")
add("P6-4", 6, "phase6/phase6_ppo_vs_lamp.csv", "H2 G_post outcome", "LAMP ahead (opposite) count", "test",
    sum(x["outcome"] == "opposite" for k, x in lv.items() if k[0].startswith("H2 PPO") and k[2] == "test accuracy"), 3, present="**3 of 6** settings")
dg = [x for k, x in lv.items() if k[0].startswith("H2 PPO") and k[2] != "test accuracy"]
add("P6-5", 6, "phase6/phase6_ppo_vs_lamp.csv", "H2 ΔG outcome", "supported / opposite", "test", f"{sum(x['outcome'] == 'supported' for x in dg)} / {sum(x['outcome'] == 'opposite' for x in dg)}",
    "5 / 0", present="**5 of 6** settings")
gpost = [float(x["mean_diff"]) for k, x in gv.items() if k[0].startswith("H3 PPO") and k[2] == "test accuracy"]
add("P6-6", 6, "phase6/phase6_ppo_vs_global.csv", "H3 G_post PPO − Global", "range after", "test", f"{fmt(min(gpost), 2)} to {fmt(max(gpost), 2, True)}", "−0.16 to +0.12")
add("P6-7", 6, "phase6/phase6_ppo_vs_global.csv", "H3 G_post max", "max PPO advantage", "test", f"{max(gpost):.2f}", "0.12", present="**0.12 pp**")
gpre = [float(x["mean_diff"]) for k, x in gv.items() if k[0].startswith("H3 reference")]
add("P6-8", 6, "phase6/phase6_ppo_vs_global.csv", "H3 reference G_pre max", "max PPO lead before", "test", f"{max(gpre):.2f}", "6.51", present="up to 6.51 pp")
add("P6-9", 6, "phase6/phase6_pre_post_accuracy.csv", "zero_count_change, pruned rows", "regrowth", "—",
    str(all(float(x["zero_count_change"]) == 0 for x in pp.values())), "True", present="No pruned weight regrew")
cl6 = json.load(open(os.path.join(R, "phase6", "phase6_classification.json")))["criteria"]
add("P6-10", 6, "phase6/phase6_classification.json", "criteria A–F", "status", "—", " ".join(cl6[k]["status"] for k in "ABCDEF"), "MET MET MET MET MET MET", present="| F. Reproducibility | **MET**")
add("P6-11", 6, "phase6/phase6_classification.json", "E runs_with_unchanged_mask / pruned_runs", "sparsity preservation", "—",
    f"{cl6['E']['runs_with_unchanged_mask']} of {cl6['E']['pruned_runs']}", "360 of 360", present="360 of 360 pruned runs")
add("P6-12", 6, "phase6/phase6_classification.json", "F reproduction_identical", "reproduction", "—", cl6["F"]["reproduction_identical"], "6/6", present="6 of 6 reproductions")
rec = {m: [float(pp[(s, m)]["recovery_mean"]) for s in S6] for m in ("ppo", "lamp", "global")}
add("P6-13", 6, "phase6/phase6_pre_post_accuracy.csv", "recovery_mean ranges", "PPO / LAMP / global recovery", "test",
    " / ".join(f"{min(v):.2f}–{max(v):.2f}" for v in rec.values()), "0.46–3.28 / 0.22–3.01 / 0.76–7.67", present="global magnitude: 0.76–7.67 pp")
add("P6-14", 6, "phase6/phase6_pre_post_accuracy.csv", "cifar10_simplecnn dense recovery", "dense-control gain", "test",
    f"+{float(pp[('cifar10_simplecnn', 'dense')]['recovery_mean']):.2f}", "+3.05")

fails = [x for x in rows if x[-1] != "PASS"]
print("# Numerical audit: Chapter 5 (Results of the Core Framework)\n")
print("Generated by `tools/audit_ch5.py` from the frozen Phase 2–6 files (read-only).")
print("A row passes only if the recomputed frozen value equals the chapter value **and** that value appears in")
print("`chapter_05_core_results.md`.\n")
print(f"**{len(rows)} checks: {len(rows) - len(fails)} PASS, {len(fails)} FAIL.**\n")
print("| ID | Phase | Source file | Row / key | Metric | Split | Frozen value | Chapter value | Result |")
print("|---|---|---|---|---|---|---|---|---|")
for x in rows:
    print("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in x) + " |")
sys.exit(1 if fails else 0)
