"""Numerical audit of Chapter 6 (deployment and sustainability). Headline numbers recomputed from frozen files.

  python results/thesis_final_writing/tools/audit_ch6.py > results/thesis_final_writing/numerical_audit_chapter_06.md
Read-only. Rounding: round to 6 decimals (float noise), then round-half-up to the stated precision (thesis convention).
A row passes only if the recomputed value equals the chapter value and that value occurs in the chapter text
(whitespace and ** markers normalised).
"""
import csv
import json
import os
import re
import sys
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
R = os.path.join(ROOT, "results")
_norm = lambda s: re.sub(r"\s+", " ", str(s).replace("**", ""))  # noqa: E731
TEXT = _norm(open(os.path.join(R, "thesis_final_writing", "chapter_06_deployment_sustainability.md"), encoding="utf-8").read())
rows = []


def f(v, d, pct=False, sign=False):
    v = 100 * v if pct else v
    q = Decimal(repr(round(float(v), 6))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    s = f"{q:+f}" if sign else f"{q:f}"
    return s.replace("-", "−")


def add(i, phase, src, key, metric, split, frozen, chapter, present=None):
    ok = str(frozen) == str(chapter)
    pres = _norm(present if present is not None else chapter) in TEXT
    rows.append((i, phase, src, key, metric, split, frozen, chapter, "PASS" if ok and pres else ("FAIL (value)" if not ok else "FAIL (not in text)")))


def j(p):
    return json.load(open(os.path.join(R, p), encoding="utf-8"))


# ---------------- Paper 3 tables (recomputed; the C10 rows are checked against the frozen correlation file)
frozen_corr = {r["table"]: r for r in csv.DictReader(open(os.path.join(R, "paper3/h2_e2_latency_correlations.csv")))}
exp = {"simplecnn_c10": ("0.32", "0.89", "39.0%", "2.61×"), "simplecnn_c100": ("0.31", "0.89", "39.4%", "2.59×"),
       "resnet8_c10": ("0.55", "0.98", "30.6%", "1.67×"), "resnet8_c100": ("0.54", "0.98", "30.8%", "1.72×")}
n_total = 0
for t, e in exp.items():
    d = pd.read_csv(os.path.join(R, f"paper3/h2_latency_table_{t}.csv"))
    n_total += len(d)
    rp = d.param_reduction.rank().corr(d.latency_gain.rank())
    rm = d.mac_reduction.rank().corr(d.latency_gain.rank())
    p, L = d.param_reduction.values, d.latency_ms.values
    inv = int(((p[:, None] - p[None, :] > 0) & (L[:, None] - L[None, :] > 0)).sum())
    frac = inv / (len(d) * (len(d) - 1) // 2)
    src = f"paper3/h2_latency_table_{t}.csv"
    add(f"P3-{t}-n", "Paper 3", src, "rows", "policies", "—", len(d), 1296, present="1,296")
    add(f"P3-{t}-rp", "Paper 3", src, "spearman(param_reduction, latency_gain)", "Spearman", "—", f(rp, 2), e[0])
    add(f"P3-{t}-rm", "Paper 3", src, "spearman(mac_reduction, latency_gain)", "Spearman", "—", f(rm, 2), e[1])
    add(f"P3-{t}-inv", "Paper 3", src, "pairs: more param reduction and higher latency / all pairs", "inversion fraction", "—", f(frac, 1, pct=True) + "%", e[2])
    add(f"P3-{t}-max", "Paper 3", src, "max speedup", "max speed-up", "—", f(d.speedup.max(), 2) + "×", e[3])
    if t in frozen_corr:
        fc = frozen_corr[t]
        add(f"P3-{t}-frozen", "Paper 3", "paper3/h2_e2_latency_correlations.csv", t, "recomputation = frozen file", "—",
            f"{float(fc['spearman_param_reduction_vs_latency_gain']):.4f}/{float(fc['spearman_mac_reduction_vs_latency_gain']):.4f}/{int(fc['pairs_sparser_but_slower'])}",
            f"{rp:.4f}/{rm:.4f}/{inv}", present="0.89")
add("P3-total", "Paper 3", "paper3/h2_latency_table_*.csv", "sum of rows", "structured policies", "—", n_total, 5184, present="5,184")
d = pd.read_csv(os.path.join(R, "paper3/h2_latency_table_simplecnn_c10.csv"))
for pol, e in (("[5, 5, 1, 0]", ("18.7%", "69.3%", "2.08×")), ("[0, 0, 0, 5]", ("51.1%", "2.9%", "1.03×"))):
    r = d[d.policy == pol].iloc[0]
    add(f"P3-ex-{pol}", "Paper 3", "paper3/h2_latency_table_simplecnn_c10.csv", pol, "param ↓ / MAC ↓ / speed-up", "—",
        f"{f(r.param_reduction, 1, True)}% / {f(r.mac_reduction, 1, True)}% / {f(r.speedup, 2)}×", f"{e[0]} / {e[1]} / {e[2]}", present=e[2])
h0 = j("paper3/h0_decision.json")["configs"]["A"]
add("P3-H0", "Paper 3", "paper3/h0_decision.json", "configs.A snr, relative_reduction", "SNR; reduction", "—",
    f"{f(h0['snr'], 1)}; {f(h0['relative_reduction'], 1, True)}%", "12.3; 61.6%", present="12.3")
pdj = j("paper3/paper3_proxy_decision.json")["architectures"]
add("P3-proxy", "Paper 3", "paper3/paper3_proxy_decision.json", "spearman_rho, n_pairs, go", "proxy validity", "VAL-RL",
    f"{f(pdj['cifar10_simplecnn']['spearman_rho'], 3)} ({pdj['cifar10_simplecnn']['n_pairs']}, {pdj['cifar10_simplecnn']['go']}) / "
    f"{f(pdj['cifar10_resnet8']['spearman_rho'], 3)} ({pdj['cifar10_resnet8']['n_pairs']}, {pdj['cifar10_resnet8']['go']})",
    "0.844 (156, False) / 0.980 (31, True)", present="0.980")
c = j("paper3/resnet8_threshold_calibration_decision.json")
lr = c["latency_room"]
add("P3-cal-1", "Paper 3", "paper3/resnet8_threshold_calibration_decision.json", "n_calibration, n_true_feasible", "counts", "VAL-RL", f"{c['n_calibration']} / {c['n_true_feasible']}", "48 / 14", present="| 14 |")
add("P3-cal-2", "Paper 3", "same", "tau_proxy", "threshold", "VAL-RL", f(c["tau_proxy"], 3), "0.960")
add("P3-cal-3", "Paper 3", "same", "recall, precision, tp, fp", "proxy quality", "VAL-RL", f"{f(c['recall'], 2)} / {f(c['precision'], 3)} ({c['tp']}, {c['fp']})", "1.00 / 0.875 (14, 2)",
    present="1.00 / 0.875 (14 true positives, 2 false positives)")
add("P3-cal-4", "Paper 3", "same", "latency_room.proxy_feasible", "proxy-feasible policies", "VAL-RL", lr["proxy_feasible"], 52, present="| 52 |")
add("P3-cal-5", "Paper 3", "same", "latency_room.best_speedup", "best proxy-feasible speed-up", "—", f(lr["best_speedup"], 3) + "×", "1.098×")
add("P3-cal-6", "Paper 3", "same", "latency_room.policies_with_speedup_ge_1.10; go_ppo", "≥1.10× count; GO", "—",
    f"{lr['policies_with_speedup_ge_1.10']}; {c['go_ppo']}", "0; False", present="no hardware-aware PPO search was started")
add("P3-cal-7", "Paper 3", "same", "param/mac reduction range", "ranges", "—",
    f"{f(lr['param_reduction_range'][0], 0, True)}–{f(lr['param_reduction_range'][1], 1, True)}% / {f(lr['mac_reduction_range'][0], 0, True)}–{f(lr['mac_reduction_range'][1], 1, True)}%",
    "0–28.9% / 0–18.4%")
rep = open(os.path.join(R, "paper3/resnet8_threshold_calibration_report.md"), encoding="utf-8").read()
add("P3-cal-8", "Paper 3", "paper3/resnet8_threshold_calibration_report.md", "best truly feasible calibration speed-up", "speed-up", "VAL-RL (5-epoch)",
    "1.089× ([0, 3, 1, 1])" if "1.089× ([0, 3, 1, 1]" in rep else "?", "1.089× ([0, 3, 1, 1])")

# ---------------- Phase 7A
es = {(r["arch"], r["method"]): r for r in csv.DictReader(open(os.path.join(R, "phase7a_sustainability/energy_summary.csv")))}
ls = {(r["arch"], r["method"]): r for r in csv.DictReader(open(os.path.join(R, "phase7a_sustainability/latency_summary.csv")))}
acc = {(r["setting"].split("_")[1], r["method"]): r for r in csv.DictReader(open(os.path.join(R, "phase7a_sustainability/accuracy_results.csv")))}
E = {("simplecnn", "dense"): "11.05", ("simplecnn", "ppo"): "8.54", ("simplecnn", "uniform"): "7.91", ("simplecnn", "lamp"): "9.76",
     ("resnet8", "dense"): "10.88", ("resnet8", "ppo"): "9.59", ("resnet8", "uniform"): "9.64", ("resnet8", "lamp"): "9.82"}
S = {("simplecnn", "ppo"): "22.8%", ("simplecnn", "uniform"): "27.9%", ("simplecnn", "lamp"): "12.1%",
     ("resnet8", "ppo"): "13.4%", ("resnet8", "uniform"): "11.8%", ("resnet8", "lamp"): "8.9%"}
A = {("simplecnn", "dense"): "80.60", ("simplecnn", "ppo"): "75.52", ("simplecnn", "uniform"): "79.12", ("simplecnn", "lamp"): "78.84",
     ("resnet8", "dense"): "78.40", ("resnet8", "ppo"): "75.16", ("resnet8", "uniform"): "75.60", ("resnet8", "lamp"): "75.24"}
D = {("simplecnn", "ppo"): "−5.08", ("simplecnn", "uniform"): "−1.48", ("simplecnn", "lamp"): "−1.76",
     ("resnet8", "ppo"): "−3.24", ("resnet8", "uniform"): "−2.80", ("resnet8", "lamp"): "−3.16"}
EDP = {("simplecnn", "ppo"): "0.603", ("simplecnn", "uniform"): "0.518", ("simplecnn", "lamp"): "0.783",
       ("resnet8", "ppo"): "0.791", ("resnet8", "uniform"): "0.818", ("resnet8", "lamp"): "0.828"}
for k, v in E.items():
    add(f"7A-E-{k}", "7A", "phase7a_sustainability/energy_summary.csv", f"{k} e1000_median_j", "J / 1,000 inferences (median of 5)", "—", f(float(es[k]["e1000_median_j"]), 2), v)
for k, v in S.items():
    add(f"7A-S-{k}", "7A", "phase7a_sustainability/energy_summary.csv", f"{k} saving_median", "energy saving vs same-session dense", "—", f(float(es[k]["saving_median"]), 1, True) + "%", v)
    add(f"7A-5of5-{k}", "7A", "same", f"{k} sessions_lower_than_dense, wilcoxon_one_sided_p, consistent_reduction", "5/5; p; consistent", "—",
        f"{es[k]['sessions_lower_than_dense']}; {f(float(es[k]['wilcoxon_one_sided_p']), 3)}; {es[k]['consistent_reduction']}", "5; 0.031; True", present="p = 0.031")
for k, v in A.items():
    add(f"7A-A-{k}", "7A", "phase7a_sustainability/accuracy_results.csv", f"{k} val_rl_post", "recovered accuracy", "VAL-RL", f(float(acc[k]["val_rl_post"]), 2), v)
for k, v in D.items():
    add(f"7A-D-{k}", "7A", "phase7a_sustainability/accuracy_results.csv", f"{k} delta_val_rl_pp", "Δ vs recovered dense", "VAL-RL", f(float(acc[k]["delta_val_rl_pp"]), 2), v)
add("7A-pres", "7A", "phase7a_sustainability/accuracy_results.csv", "accuracy_preserving (pruned rows)", "any within ±0.5 pp", "VAL-RL",
    str(any(r["accuracy_preserving"] == "True" for k, r in acc.items() if k[1] != "dense")), "False", present="not met by any pruned method")
for k, v in EDP.items():
    add(f"7A-EDP-{k}", "7A", "phase7a_sustainability/latency_summary.csv", f"{k} normalized_edp_median", "normalised EDP", "—", f(float(ls[k]["normalized_edp_median"]), 3), v)
for k, v in ((("simplecnn", "ppo"), "4.00"), (("resnet8", "ppo"), "3.80")):
    add(f"7A-VS-{k}", "7A", "phase7a_sustainability/accuracy_results.csv", f"{k} dense − method val_select_post", "Δ", "VAL-SELECT",
        f(float(acc[(k[0], 'dense')]["val_select_post"]) - float(acc[k]["val_select_post"]), 2), v, present=f"{v} pp")
pw = {(r["arch"], r["comparison"]): r for r in csv.DictReader(open(os.path.join(R, "phase7a_sustainability/pairwise_comparisons.csv")))}
r = pw[("simplecnn", "ppo_vs_uniform")]
add("7A-uni", "7A", "phase7a_sustainability/pairwise_comparisons.csv", "simplecnn ppo_vs_uniform", "Δacc; baseline lower sessions; verdict", "VAL-RL",
    f"{f(-float(r['delta_acc_pp_ppo_minus_baseline']), 2)}; {r['sessions_baseline_lower']}; {r['verdict']}", "3.60; 5; baseline better", present="3.60 pp more accurate")
add("7A-uni-e", "7A", "same", "simplecnn ppo_vs_uniform energy_diff_median_pct", "median per-session difference", "—", f(abs(float(r["energy_diff_median_pct"])), 1) + "%", "8.0%")
r = pw[("simplecnn", "ppo_vs_lamp")]
add("7A-lamp", "7A", "same", "simplecnn ppo_vs_lamp", "Δacc; energy diff; ppo lower sessions", "VAL-RL",
    f"{f(-float(r['delta_acc_pp_ppo_minus_baseline']), 2)}; {f(float(r['energy_diff_median_pct']), 1)}%; {r['sessions_ppo_lower']}", "3.32; 12.5%; 5", present="12.5% median difference")
for comp, e in (("ppo_vs_uniform", "0.44; 3"), ("ppo_vs_lamp", "0.08; 4")):
    r = pw[("resnet8", comp)]
    add(f"7A-r8-{comp}", "7A", "same", f"resnet8 {comp}", "|Δacc|; PPO lower sessions", "VAL-RL",
        f"{f(abs(float(r['delta_acc_pp_ppo_minus_baseline'])), 2)}; {r['sessions_ppo_lower']}", e, present=f"differed by {e.split(';')[0]} pp")
pil = j("phase7a_sustainability/energy_pilot/pilot_n100000.json")
add("7A-pilot", "7A", "phase7a_sustainability/energy_pilot/pilot_n100000.json", "pilot decision", "stable", "—", "stable" in json.dumps(pil).lower(), True, present="CV 2.30%")
add("7A-noise", "7A", "phase7a_sustainability/energy_summary.csv", "noise_threshold simplecnn / resnet8", "2 × dense CV", "—",
    f"{f(float(es[('simplecnn', 'dense')]['noise_threshold']), 1, True)}% / {f(float(es[('resnet8', 'dense')]['noise_threshold']), 1, True)}%", "4.3% / 3.7%", present="4.3% for SimpleCNN; 3.7% for ResNet-8")

# ---------------- Phase 7B
fs = {r["id"]: r for r in csv.DictReader(open(os.path.join(R, "phase7b_green_pruning/final_summary.csv")))}
nw, un, la, op, de = fs["2225"], fs["uniform_r0.208"], fs["lamp_s56.2"], fs["phase7a_ppo"], fs["0000"]
add("7B-roles", "7B", "phase7b_green_pruning/final_summary.csv", "2225 roles (seeds)", "selected in", "—", sum(1 for x in json.loads(nw["roles"]) if x.startswith("seed")), 5, present="5 of 20")
add("7B-widths", "7B", "same", "2225 widths", "widths", "—", nw["widths"], "[26, 51, 102, 102]")
for m, e in ((nw, "78.88"), (de, "80.00"), (un, "78.76"), (la, "79.64"), (op, "76.00")):
    add(f"7B-VS-{m['id']}", "7B", "same", f"{m['id']} val_select_post", "accuracy", "VAL-SELECT", f(float(m["val_select_post"]), 2), e)
for m, e in ((nw, "−1.12"), (un, "−1.24"), (la, "−0.36"), (op, "−4.00")):
    add(f"7B-D-{m['id']}", "7B", "same", f"{m['id']} delta_val_select_pp", "Δ vs recovered dense", "VAL-SELECT", f(float(m["delta_val_select_pp"]), 2), e)
for m, e in ((nw, "63.4%"), (un, "37.3%"), (la, "63.3%"), (op, "80.1%")):
    add(f"7B-M-{m['id']}", "7B", "same", f"{m['id']} memory_reduction", "parameter-memory reduction", "—", f(float(m["memory_reduction"]), 1, True) + "%", e)
for m, e in ((nw, "36.0%"), (un, "36.2%"), (la, "19.3%"), (op, "42.6%")):
    add(f"7B-MAC-{m['id']}", "7B", "same", f"{m['id']} mac_reduction", "MAC reduction", "—", f(float(m["mac_reduction"]), 1, True) + "%", e)
add("7B-bytes", "7B", "same", "memory_bytes dense / new", "bytes", "—", f"{int(de['memory_bytes']):,} / {int(nw['memory_bytes']):,}", "2,481,448 / 908,916", present="908,916 vs 2,481,448")
add("7B-ser", "7B", "phase7b_green_pruning/memory_results.csv", "2225 serialized_reduction", "serialised reduction", "—",
    f(float({r['id']: r for r in csv.DictReader(open(os.path.join(R, 'phase7b_green_pruning/memory_results.csv')))}['2225']['serialized_reduction']), 1, True) + "%", "63.3%",
    present="serialised checkpoint is 63.3% smaller")
add("7B-vsU", "7B", "same", "new − uniform val_select", "Δ", "VAL-SELECT", f(float(nw["val_select_post"]) - float(un["val_select_post"]), 2), "0.12", present="0.12 pp more accurate")
add("7B-vsL", "7B", "same", "lamp − new val_select", "Δ", "VAL-SELECT", f(float(la["val_select_post"]) - float(nw["val_select_post"]), 2), "0.76", present="0.76 pp less accurate")
seeds = [r for r in fs.values() if any(x.startswith("seed") for x in json.loads(r["roles"]))]
dv = [float(r["delta_val_select_pp"]) for r in seeds]
add("7B-range", "7B", "same", "per-seed selections delta_val_select_pp range", "range", "VAL-SELECT", f"{f(-max(dv), 2)} and {f(-min(dv), 2)}", "1.08 and 1.28")
add("7B-nseeds", "7B", "same", "seed roles total", "seeds with a selection", "—", sum(sum(1 for x in json.loads(r["roles"]) if x.startswith("seed")) for r in seeds), 20,
    present="Every one of the 20 seeds selected a policy")
add("7B-success", "7B", "same", "green_success of seed selections", "successes", "VAL-SELECT", sum(r["green_success"].startswith("YES") for r in seeds), 0, present="met by 0 of 20 seeds")
add("7B-other", "7B", "same", "distinct non-primary seed policies", "count", "—", len([r for r in seeds if r["id"] != "2225"]), 6, present="six other policies")
add("7B-VRL", "7B", "same", "2225 / old delta_val_rl_pp", "Δ", "VAL-RL", f"{f(-float(nw['delta_val_rl_pp']), 2)} / {f(-float(op['delta_val_rl_pp']), 2)}", "0.60 / 5.08", present="0.60 pp, against 5.08 pp")
pre = j("phase7b_green_pruning/phase7b_final_preregistration.json")
add("7B-tau", "7B", "phase7b_green_pruning/phase7b_final_preregistration.json", "proxy.tau, rho, precision", "proxy", "VAL-RL",
    f"{f(pre['proxy']['tau'], 5)}; {f(pre['proxy']['validation']['rho'], 3)}; {f(pre['proxy']['validation']['precision'], 3)}", "0.98849; 0.948; 0.655", present="0.98849")
add("7B-sur", "7B", "same", "surrogate coefficients, spearman, mdape", "surrogate", "—",
    f"{f(pre['surrogate']['coefficients'][0], 3)}; {f(pre['surrogate']['coefficients'][1], 3)}; {f(pre['surrogate']['validation']['spearman'], 3)}; {f(pre['surrogate']['validation']['mdape'], 2, True)}%",
    "−0.117; 13.762; 0.996; 1.05%", present="Spearman 0.996, median absolute percentage error 1.05%")
sp = j("phase7b_green_pruning/selected_policies.json")["per_seed"]["701"]
add("7B-q", "7B", "phase7b_green_pruning/selected_policies.json", "per_seed.701 q_proxy (policy 2225)", "proxy retention", "VAL-RL", f(sp["q_proxy"], 4), "0.9890")
st7 = j("phase7b_green_pruning/statistics.json")["pilots"]
add("7B-p1", "7B", "phase7b_green_pruning/statistics.json", "pilots.pilot_n100000.cv", "CV", "—", f(st7["pilot_n100000.json"]["cv"], 2, True) + "%", "11.36%")
add("7B-p2", "7B", "same", "pilots.pilot_n200000.cv", "CV", "—", f(st7["pilot_n200000.json"]["cv"], 2, True) + "%", "4.01%")

# ---------------- Energy confirmation
q = j("phase7b_energy_confirmation/measurement_qualification.json")
e = [float(x) for x in q["e1000_j"]]
add("EC-cv", "7B-EC", "phase7b_energy_confirmation/measurement_qualification.json", "cv", "CV of 6 dense blocks", "—", f(q["cv"], 2, True) + "%", "19.92%")
add("EC-n", "7B-EC", "same", "runs × blocks", "blocks", "—", int(q["runs"]) * int(q["blocks_per_run"]), 6, present="6 blocks")
add("EC-range", "7B-EC", "same", "e1000_j min–max, mean", "J / 1,000", "—", f"{f(min(e), 2)} to {f(max(e), 2)} J (mean {f(float(q['mean_j']), 2)} J)", "9.47 to 15.22 J (mean 12.14 J)")
add("EC-dec", "7B-EC", "same", "decision", "qualification", "—", q["decision"], "not qualified")
lat = [b["latency_median_ms"] if "latency_median_ms" in b else None for b in q.get("blocks", [])]
interp = open(os.path.join(R, "phase7b_energy_confirmation/interpretation.md"), encoding="utf-8").read()
for s_, e_ in (("1.042–1.052 ms", "1.042–1.052 ms"), ("9.1 to 14.5 W", "9.1 to 14.5 W"), ("6.7–9.7 W", "6.7 to 9.7 W"), ("2.4–5.1 W", "2.4 to 5.1 W"), ("72 to 92 °C", "72 to 92 °C")):
    add(f"EC-diag-{s_}", "7B-EC", "phase7b_energy_confirmation/interpretation.md", "qualification diagnostics", "recorded range", "—", e_ if s_ in interp else "?", e_)
st = j("phase7b_energy_confirmation/statistics.json")
add("EC-verdict", "7B-EC", "phase7b_energy_confirmation/statistics.json", "energy_confirmed", "verdict", "—", st["energy_confirmed"], "NO", present="remains unresolved")

# ---------------- additions: stability, 7A latency/power, 7B seed spread, surrogate candidates
fz = j("paper3/h2_latency_freeze.json")["note_on_failed_check"]
for s_ in ("CV median 1.2-1.6%", "Spearman of gains 0.96-0.995", ">= 0.997", "+16%"):
    add(f"P3-stab-{s_}", "Paper 3", "paper3/h2_latency_freeze.json", "note_on_failed_check", "stability statement", "—", s_ in fz, True,
        present={"CV median 1.2-1.6%": "1.2–1.6%", "Spearman of gains 0.96-0.995": "0.96–0.995", ">= 0.997": "≥ 0.997", "+16%": "about 16%"}[s_])
for k, v in ((("simplecnn", "ppo"), "1.28×"), (("simplecnn", "uniform"), "1.37×"), (("simplecnn", "lamp"), "1.12×"),
             (("resnet8", "ppo"), "1.09×"), (("resnet8", "uniform"), "1.07×"), (("resnet8", "lamp"), "1.07×")):
    add(f"7A-SU-{k}", "7A", "phase7a_sustainability/latency_summary.csv", f"{k} speedup_median", "median speed-up", "—", f(float(ls[k]["speedup_median"]), 2) + "×", v)
pw_ = [float(r["power_median_w"]) for r in es.values()]
add("7A-power", "7A", "phase7a_sustainability/energy_summary.csv", "power_median_w, all 8 models", "median package power range", "—", f"{f(min(pw_), 1)}–{f(max(pw_), 1)} W", "14.6–15.6 W")
mr = [float(r["memory_reduction"]) for r in seeds]
mc = [float(r["mac_reduction"]) for r in seeds]
add("7B-spread", "7B", "phase7b_green_pruning/final_summary.csv", "per-seed selections memory_reduction / mac_reduction", "ranges", "—",
    f"{f(min(mc), 1, True)}% and {f(max(mc), 1, True)}% / {f(min(mr), 1, True)}% to {f(max(mr), 1, True)}%", "34.1% and 37.5% / 29.5% to 63.4%", present="between 34.1% and 37.5%")
add("7B-spread-n", "7B", "same", "distinct per-seed policies", "count", "—", len(seeds), 7, present="seven distinct per-seed selections")
sm = j("phase7b_green_pruning/energy_surrogate_metrics.json")["candidates"]
add("7B-cand1", "7B", "phase7b_green_pruning/energy_surrogate_metrics.json", "candidates.1", "passed; spearman; low-quartile MdAPE", "—",
    f"{sm['1']['passed']}; {f(sm['1']['spearman'], 3)}; {f(sm['1']['low_quartile_mdape'], 2, True)}%", "False; 0.946; 8.55%", present="8.55% against 5%")

fails = [x for x in rows if x[-1] != "PASS"]
print("# Numerical audit: Chapter 6 (Deployment and Sustainability Analysis)\n")
print("Generated by `tools/audit_ch6.py` from the frozen Paper 3, Phase 7A, Phase 7B and energy-confirmation files")
print("(read-only). Paper-3 correlations and inversion fractions are recomputed from the four frozen latency tables; the")
print("recomputation reproduces the frozen `h2_e2_latency_correlations.csv` exactly for the two CIFAR-10 tables (rows")
print("P3-*-frozen). Rounding: 6-decimal noise removal, then round-half-up to the stated precision.\n")
print(f"**{len(rows)} checks: {len(rows) - len(fails)} PASS, {len(fails)} FAIL.**\n")
print("| ID | Phase | Source file | Row / key | Metric | Split | Frozen value | Chapter value | Result |")
print("|---|---|---|---|---|---|---|---|---|")
for x in rows:
    print("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in x) + " |")
sys.exit(1 if fails else 0)
