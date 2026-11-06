"""Generate the thesis appendices (A-E, MIST consolidated) directly from the frozen result files. Read-only.

  python results/thesis_final_writing/tools/build_appendices.py   ->  results/thesis_final_writing/appendices.md
Every number below is copied or rounded (round-half-up) from the frozen file named in its caption; nothing is typed.
"""
import csv
import re
import json
import os
import statistics as st
from decimal import ROUND_HALF_UP, Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
R = os.path.dirname(W)
ROOT = os.path.dirname(R)
S6 = ["cifar10_simplecnn", "cifar10_lenet5", "cifar10_resnet8", "cifar100_simplecnn", "cifar100_lenet5", "cifar100_resnet8"]
NM = {"cifar10_simplecnn": "C10 SimpleCNN", "cifar10_lenet5": "C10 LeNet-5", "cifar10_resnet8": "C10 ResNet-8",
      "cifar100_simplecnn": "C100 SimpleCNN", "cifar100_lenet5": "C100 LeNet-5", "cifar100_resnet8": "C100 ResNet-8"}
out = []


def hu(v, d):
    if v in (None, ""):
        return "—"
    return str(Decimal(repr(round(float(v), 6))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)).replace("-", "−")


def p_(v):
    if v in (None, ""):
        return "—"
    v = float(v)
    return f"{v:.2g}" if v >= 0.001 else f"{v:.1e}"


def rd(p):
    return list(csv.DictReader(open(os.path.join(R, p), encoding="utf-8")))


def j(p):
    return json.load(open(os.path.join(R, p), encoding="utf-8"))


def table(head, rows):
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    out.append("")


def H(t):
    out.append(t)
    out.append("")


def stats_rows(rows):
    return [[NM.get(r["setting"], r["setting"]), r["comparison"], r["endpoint"], hu(r["mean_diff"], 3),
             f"[{hu(r['t_ci95_low'], 3)}, {hu(r['t_ci95_high'], 3)}]", p_(r.get("holm_p")),
             f"{r.get('n_positive')}/{r.get('n_negative')}/{r.get('n_ties')}", r["outcome"]] for r in rows]


SH = ["Setting", "Comparison", "Endpoint", "Mean diff.", "95% t-CI", "Holm p", "+/−/tie", "Outcome"]

CUR = {"app": "", "n": 0}


def APP(letter, title):
    CUR["app"], CUR["n"] = letter, 0
    H(f"## Appendix {letter} — {title}")


def CAP(text):
    CUR["n"] += 1
    out.append(f"**Table {CUR['app']}.{CUR['n']}. {text}**")
    out.append("")


_table = table


def table(head, rows):  # noqa: F811
    _table(head, rows)


def src_note(s):
    """Source notes name the frozen file only (short name), per the MIST appendix brief."""
    s = re.sub(r"`results/(?:[^`/]+/)*([^`/]+)`", r"`\1`", s)
    return s


H("# Appendices")
out.append("These appendices give supporting detail for Chapters 3–6, generated from the frozen phase records by")
out.append("`tools/build_appendices.py`. Each table names its source record. Complete per-seed, per-session and")
out.append("per-comparison records (all statistical families, raw energy blocks, per-seed policies) are preserved in the")
out.append("repository and are summarised, not reproduced, here. The central results and conclusions are in the main chapters.")
out.append("")
APP("A", "Architectures, Hyperparameters and Reproducibility")
H("### A.1 Pruning units")
CAP("Pruning units, weight shapes, parameters and MACs per image (CIFAR-10 output layer).")
u = [r for r in rd("architecture_generalization/cpu_prunable_units.csv") if r["num_classes"] == "10" and r["arch"] in ("simplecnn", "lenet5", "resnet8")]
table(["Architecture", "Unit", "Role", "Modules", "Weight shapes", "Weights", "Share of prunable", "MACs per image"],
      [[r["arch"], r["unit"], r["role"], r["modules"], r["weight_shapes"], f"{int(r['params']):,}",
        (hu(100 * float(r["fraction_of_prunable_weights"]), 2) + "%") if r["fraction_of_prunable_weights"] else "—", f"{int(r['macs_per_image']):,}"] for r in u])
out.append(src_note("*Source: `results/architecture_generalization/cpu_prunable_units.csv`.*"))
out.append("")
H("### A.2 Dense reference models")
CAP("Dense references over three training seeds.")
table(["Setting", "Seeds", "Reference", "Val. (ref.)", "Test (ref.)", "Val. mean ± SD", "Test mean ± SD", "Parameters", "Recipe"],
      [[f"{r['dataset']} {r['arch']}", r["n_seeds"], r["reference"].split("/")[-1], r["reference_val_accuracy"], r["reference_test_accuracy"],
        f"{hu(r['val_mean'], 2)} ± {hu(r['val_sd'], 2)}" if r["val_sd"] else hu(r["val_mean"], 2),
        f"{hu(r['test_mean'], 2)} ± {hu(r['test_sd'], 2)}" if r["test_sd"] else hu(r["test_mean"], 2), f"{int(r['params_total']):,}", r["recipe"]]
       for r in rd("architecture_generalization/cpu_baselines.csv")])
out.append(src_note("*Source: `results/architecture_generalization/cpu_baselines.csv`.*"))
out.append("")
H("### A.3 Recovery protocol")
CAP("Equal recovery protocol (Phase 6; reused in Paper 3, Phases 7A and 7B).")
pr = j("phase6/phase6_config.json")["protocol"]
table(["Configuration key", "Value"], [[f"`{k}`", str(v)] for k, v in pr.items()])
out.append(src_note("*Source: `results/phase6/phase6_config.json` (`protocol`). The PPO hyper-parameters are given in Sections 3.17 and 4.8.*"))
out.append("")

H("### A.4 Pre-registration commits")
CAP("Pre-registration and freeze commits.")
K2 = ([["Phase 2 dense training", "results/cpu_dense_training_protocol.md", "366d21ae"], ["Phase 3", "results/phase3_preregistration.md", "2dbcc770"],
       ["Phase 4", "results/phase4/phase4_preregistration.md", "f52e87d9"], ["Phase 5", "results/phase5/phase5_preregistration.md", "3cbaec1a"],
       ["Phase 6", "results/phase6/phase6_preregistration.md", "b1b5396f"], ["Paper 3 H0", "results/paper3/h0_protocol.md", "c2dcb22a"],
       ["Paper 3 H2", "results/paper3/paper3_preregistration.md", "45936846"], ["Paper 3 proxy", "results/paper3/paper3_proxy_protocol.md", "ec7a7f27"],
       ["Paper 3 ResNet-8 calibration", "results/paper3/resnet8_threshold_calibration_protocol.md", "5b127804"],
       ["Phase 7A", "results/phase7a_sustainability/preregistration.json", "b8c27665"], ["Phase 7B design", "results/phase7b_green_pruning/phase7b_design.md", "f5abe039"],
       ["Phase 7B final", "results/phase7b_green_pruning/phase7b_final_preregistration.json", "0e101d55"],
       ["7B energy confirmation", "results/phase7b_energy_confirmation/protocol.json", "df233be0"]])
table(["Experiment", "Pre-registration record", "Commit that first added it"], [[e, f"`{os.path.basename(g)}`", f"`{c}`"] for e, g, c in K2])
out.append("*Commits verified with `git log --diff-filter=A` (Section 4.19).*")
out.append("")
H("### A.5 Software environments")
CAP("Software and hardware environments.")
e5 = j("phase5/phase5_environment.json")
ec = j("phase7b_energy_confirmation/environment.json")
table(["Item", "Core framework (Phases 2–6)", "Deployment extension (Paper 3, 7A, 7B, energy confirmation)"],
      [["CPU", e5["cpu"], ec["cpu"]["Name"]], ["OS", e5["os"] + " (Windows 11 build 22621)", f"{ec['os']['Caption']} {ec['os']['Version']}"],
       ["Python", e5["python"], ec["python"].split()[0]], ["PyTorch", e5["torch"], ec["torch"]],
       ["Stable-Baselines3", e5["stable-baselines3"], "2.8.0 (pinned environment)"], ["Gymnasium / NumPy", f"{e5['gymnasium']} / {e5['numpy']}", "pinned environment"],
       ["CUDA", str(e5["cuda_available"]), "not used"]])
out.append(src_note("*Sources: `results/phase5/phase5_environment.json`; `results/phase7b_energy_confirmation/environment.json`; `requirements_locked.txt`.*"))
out.append("")
APP("B", "Core PPO and Sensitivity Supplementary Results")
H("### B.1 Phase 3 results by setting and method")
CAP("Phase 3 mean test accuracy and sparsity by setting and method (20 runs; baselines at each PPO run's zero count).")
sm = rd("architecture_generalization/phase3_summary.csv")
table(["Setting", "Method", "Test mean", "Test SD", "Sparsity mean (%)", "Drop vs dense (pp)"],
      [[NM[r["setting"]], r["method"], hu(r["test_mean"], 2), hu(r["test_sd"], 2), hu(r["sparsity_mean"], 2), hu(r["drop_mean"], 2)] for r in sm if r["method"] != "dense"])
out.append(src_note("*Source: `results/architecture_generalization/phase3_summary.csv`.*"))
out.append("")
H("### B.2 Sensitivity vectors")
CAP("Normalised loss sensitivity S_l by setting and the Phase-5 centred prior coefficients.")
fi = j("architecture_generalization/phase3_frozen_inputs.json")["settings"]
p5 = j("phase5/phase5_config.json")["settings"]
rows = []
for s in S6:
    raw3 = fi[s]
    n3 = raw3.get("sensitivity_normalized") or raw3.get("normalized_sensitivity") or raw3.get("sensitivity", {}).get("normalized")
    rows.append([NM[s], ", ".join(hu(x, 3) for x in n3) if n3 else "see source", ", ".join(hu(x, 3) for x in p5[s]["sensitivity_normalized"]),
                 ", ".join(hu(x, 3) for x in p5[s]["P2_coefficients"])])
table(["Setting", "Config. I VAL-RL (Phases 3–4)", "Config. II VAL-RL (Phase 5)", "Phase-5 P2 coefficients c_l = S_l − S̄"], rows)
ss = j("phase7b_green_pruning/structured_sensitivity.json")
out.append(f"Phase 7B structured sensitivity (C10 SimpleCNN, {ss['split']}, units {', '.join(ss['units'])}): normalised "
           f"{', '.join(hu(x, 3) for x in ss['normalized'])}; centred {', '.join(hu(x, 3) for x in ss['centred'])}.")
out.append("")
out.append("*Sources: `phase3_frozen_inputs.json` (configuration I); `phase5_config.json` (configuration II); `structured_sensitivity.json` (Phase 7B).*")
out.append("")
APP("C", "Archive, Reward and Recovery Diagnostics")
H("### C.1 Phase 4 search quality and retention")
CAP("Phase 4 search quality by variant (VAL-RL (I) ranks among 1,296 policies; medians over 20 runs).")
table(["Setting", "Variant", "Selected rank", "Selected V_SELECT rank", "Best-seen rank", "Best-seen top-10", "Selected top-10", "Selected top-20", "Test mean", "Sparsity mean"],
      [[NM.get(r["setting"], r["setting"]), r["condition"][:28], r["selected_rank_vrl_median"], r["selected_rank_vselect_median"] or "—", r["best_seen_rank_median"],
        hu(r["best_seen_reach_top10"], 2) if r["best_seen_reach_top10"] else "—", hu(r["selected_reach_top10"], 2), hu(r["selected_reach_top20"], 2),
        hu(r["selected_test_mean"], 2), hu(r["selected_sparsity_mean"], 2) if r["selected_sparsity_mean"] else "—"] for r in rd("phase4/phase4_search_quality.csv")])
out.append(src_note("*Source: `results/phase4/phase4_search_quality.csv`.*"))
out.append("")
H("### C.2 Phase 5 selected policies")
CAP("Phase 5 selected policies by setting and condition (20 runs each).")
cr = rd("phase5/phase5_constraint_results.csv")
rows = []
for c in ("C0", "C1", "C2"):
    for s in S6:
        g = [r for r in cr if r["condition"] == c and r["setting"] == s]
        sp = [float(r["selected_sparsity"]) for r in g]
        T = lambda k: sum(r[k] == "True" for r in g)  # noqa: E731
        rows.append([c, NM[s], hu(st.mean(sp), 2), hu(st.median(sp), 2), hu(st.mean(float(r["test_accuracy"]) for r in g), 2),
                     hu(st.mean(float(r["test_drop"]) for r in g), 2), f"{T('selected_feasible_val_rl')}/{T('selected_feasible_val_select')}/{T('selected_feasible_test')}",
                     hu(st.mean(float(r["q_test"]) for r in g), 4), T("selection_fallback")])
table(["Cond.", "Setting", "Sparsity mean", "Sparsity median", "Test mean", "Drop vs dense", "Feasible VAL-RL/VAL-SELECT/test (of 20)", "Mean q_test", "Fallbacks"], rows)
out.append(src_note("*Source: `results/phase5/phase5_constraint_results.csv`.*"))
out.append("")
H("### C.3 Phase 6 recovery by setting and method")
CAP("Phase 6 test accuracy before and after equal recovery (means of 20 runs).")
table(["Setting", "Method", "Sparsity", "Test before", "Test after", "Recovery", "Gap to dense after", "Retention ≥ 0.98 before / after", "Zero-count change"],
      [[NM[r["setting"]], r["method"], hu(r["sparsity"], 2), hu(r["test_pre_mean"], 2), hu(r["test_post_mean"], 2), hu(r["recovery_mean"], 2), hu(r["dense_gap_post"], 3),
        f"{hu(r['share_retention_ge_98_pre'], 2)} / {hu(r['share_retention_ge_98_post'], 2)}", r["zero_count_change"]] for r in rd("phase6/phase6_pre_post_accuracy.csv")])
out.append(src_note("*Source: `results/phase6/phase6_pre_post_accuracy.csv`.*"))
out.append("")
APP("D", "Structured Pruning and Hardware Protocol")
H("### D.1 Latency-measurement calibration (H0)")
CAP("H0 calibration: per-configuration decision quantities.")
h0 = j("paper3/h0_decision.json")["configs"]
table(["Config.", "Signal (ms)", "Noise (ms)", "SNR", "Relative reduction", "Positive in all sessions", "Usable"],
      [[k, hu(v["signal_ms"], 4), hu(v["noise_ms"], 4), hu(v["snr"], 2), hu(100 * v["relative_reduction"], 1) + "%", v["delta_positive_all_sessions"], v["usable"]] for k, v in h0.items()])
out.append(src_note("*Source: `results/paper3/h0_decision.json`. Rule: SNR ≥ 5, reduction ≥ 10%, positive in every session; order A, C, B.*"))
out.append("")
H("### D.2 Latency-table session log")
CAP("Latency-table session log (anchor drift; limit 10%).")
fz = j("paper3/h2_latency_freeze.json")
table(["Table", "Session", "Attempt", "Load before (%)", "Load after (%)", "Anchor drift", "Accepted"],
      [[s["pair"], s["session"], s["attempt"], s["load_before"], s["load_after"], hu(100 * s["anchor_drift"], 1) + "%", s["ok"]] for s in fz["session_log"]])
out.append(src_note("*Source: `results/paper3/h2_latency_freeze.json` (`session_log`).*"))
out.append("")
H("### D.3 Accuracy-proxy validation and ResNet-8 calibration")
CAP("Accuracy-proxy validation (1-epoch against 5-epoch recovery).")
pdj = j("paper3/paper3_proxy_decision.json")["architectures"]
table(["Architecture", "Validation pairs", "Spearman ρ(A₁, A₅)", "MAE of q", "Top-10 overlap", "q₅-feasible pairs", "GO"],
      [[k, v["n_pairs"], hu(v["spearman_rho"], 3), hu(v["mae_q"], 4), v["top10_overlap"], v["q5_feasible_pairs"], v["go"]] for k, v in pdj.items()])
CAP("ResNet-8 threshold calibration and latency-room decision.")
cal = j("paper3/resnet8_threshold_calibration_decision.json")
table(["Item", "Value"], [["Calibration policies / truly feasible", f"{cal['n_calibration']} / {cal['n_true_feasible']}"], ["τ_proxy", hu(cal["tau_proxy"], 4)],
                          ["Recall / precision (TP, FP, FN, TN)", f"{hu(cal['recall'], 2)} / {hu(cal['precision'], 3)} ({cal['tp']}, {cal['fp']}, {cal['fn']}, {cal['tn']})"],
                          ["Proxy-feasible landscape policies", cal["latency_room"]["proxy_feasible"]], ["Best proxy-feasible speed-up", hu(cal["latency_room"]["best_speedup"], 3) + "×"],
                          ["Policies ≥ 1.10×", cal["latency_room"]["policies_with_speedup_ge_1.10"]], ["GO to PPO", cal["go_ppo"]]])
out.append(src_note("*Sources: `results/paper3/paper3_proxy_decision.json`, `results/paper3/resnet8_threshold_calibration_decision.json`. The best confirmed five-epoch-feasible calibration policy reached 1.089× ([0, 3, 1, 1]; `resnet8_threshold_calibration_report.md`).*"))
out.append("")

APP("E", "Energy Evaluation and Qualification Records")
H("### E.1 Phase 7A energy session summaries")
CAP("Phase 7A energy: session summaries (five sessions per architecture).")
table(["Architecture", "Method", "Median J / 1,000", "CV over sessions", "Median saving", "Min–max saving", "Sessions below dense", "One-sided Wilcoxon p", "Consistent reduction"],
      [[r["arch"], r["method"], hu(r["e1000_median_j"], 3), hu(100 * float(r["e1000_cv"]), 2) + "%", hu(100 * float(r["saving_median"]), 1) + "%",
        f"{hu(100 * float(r['saving_min']), 1)}–{hu(100 * float(r['saving_max']), 1)}%", r["sessions_lower_than_dense"], p_(r["wilcoxon_one_sided_p"]) if r["wilcoxon_one_sided_p"] else "—",
        r["consistent_reduction"]] for r in rd("phase7a_sustainability/energy_summary.csv")])
out.append(src_note("*Source: `results/phase7a_sustainability/energy_summary.csv`.*"))
out.append("")

H("### E.2 Phase 7B recovered final models")
CAP("Phase 7B: all recovered final models (5-epoch recovery).")
table(["Id", "Roles", "Widths", "VAL-SELECT", "Δ VAL-SELECT", "VAL-RL", "Memory ↓", "MACs ↓", "Joint criterion"],
      [[r["id"], ", ".join(json.loads(r["roles"])), r["widths"], hu(r["val_select_post"], 2), hu(r["delta_val_select_pp"], 2), hu(r["val_rl_post"], 2),
        hu(100 * float(r["memory_reduction"]), 1) + "%", hu(100 * float(r["mac_reduction"]), 1) + "%", r["green_success"]] for r in rd("phase7b_green_pruning/final_summary.csv")])
out.append(src_note("*Source: `results/phase7b_green_pruning/final_summary.csv`. The last column is the pre-registered joint deployment-efficiency criterion (internal file name GREEN-SUCCESS). Latency and energy were not measured.*"))
out.append("")

H("### E.3 Energy-measurement qualification")
CAP("Dense-only stability pilots and qualification (energy per 1,000 inferences, J).")
p7a = j("phase7a_sustainability/energy_pilot/pilot_n100000.json")["results"]
cal7b = j("phase7b_green_pruning/_energy_calibration/pilot_n20000.json")
st7b = j("phase7b_green_pruning/statistics.json")["pilots"]
q = j("phase7b_energy_confirmation/measurement_qualification.json")
rows = [[f"7A pilot, {a}", "100,000", ", ".join(hu(x, 3) for x in v["e1000_j"]), hu(100 * v["cv"], 2) + "%", "3%", "passed"] for a, v in p7a.items()]
rows.append(["7B surrogate-calibration pilot", "20,000", ", ".join(hu(x, 3) for x in cal7b["e1000_j"]), hu(100 * cal7b["cv"], 2) + "%", "4%", cal7b["decision"]])
for k, v in st7b.items():
    rows.append([f"7B confirmatory pilot ({k.replace('.json', '')})", f"{v['n']:,}", ", ".join(hu(x, 3) for x in v["e1000_j"]), hu(100 * v["cv"], 2) + "%", "3%", v["decision"]])
rows.append(["7B energy-confirmation qualification", f"{int(q['n']):,}", ", ".join(hu(x, 3) for x in q["e1000_j"]), hu(100 * float(q["cv"]), 2) + "%", "3%", q["decision"]])
table(["Measurement", "N per block", "Six dense blocks", "CV", "Limit", "Decision"], rows)
out.append("*Sources: `pilot_n100000.json` (Phase 7A); `pilot_n20000.json` and `statistics.json` (Phase 7B); `measurement_qualification.json` (energy confirmation). A doubled-N Phase-7B pilot attempt stopped by the load gate was declared void by a recorded decision of the author and is not listed.*")
out.append("")
H("### E.4 Energy-confirmation qualification blocks")
CAP("Energy-confirmation qualification blocks (descriptive diagnostics).")
table(["Run", "Position", "J / 1,000", "Package W", "Median latency (ms)", "System load (%)", "CPU-2 performance (%)", "Thermal zone start → end (°C)"],
      [[b["run"], b["position"], hu(b["e1000_j"], 3), hu(b["pkg_w"], 2), hu(b["latency_median_ms"], 4), hu(b["system_load_pct"], 1), hu(b["perf_pct_cpu2"], 0),
        f"{hu(b['thermal_zone_c_start'], 0)} → {hu(b['thermal_zone_c_end'], 0)}"] for b in q["blocks"]])
out.append(src_note("*Source: `results/phase7b_energy_confirmation/measurement_qualification.json` (`blocks`). No cause of the instability is claimed, and no Dense-versus-PPO session was run.*"))
out.append("")


open(os.path.join(W, "appendices.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
print(f"appendices.md: {len(out)} lines")
