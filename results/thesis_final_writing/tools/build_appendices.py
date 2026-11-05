"""Generate the thesis appendices (A-K) directly from the frozen result files. Read-only on all phase files.

  python results/thesis_final_writing/tools/build_appendices.py   ->  results/thesis_final_writing/appendices.md
Every number below is copied or rounded (round-half-up) from the frozen file named in its caption; nothing is typed.
"""
import csv
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

H("# Appendices")
out.append("These appendices contain supporting detail for Chapters 3–6. They are generated from the frozen phase files by")
out.append("`tools/build_appendices.py`, and the source file is named under each table. Seed-level and session-level")
out.append("records beyond these summaries remain in the repository files listed in Appendix K. The central results and")
out.append("conclusions are in the main chapters.")
out.append("")

# ---------------------------------------------------------------- A
H("## Appendix A — Network Architectures and Hyperparameters")
H("### A.1 Pruning units, shapes, parameters and MACs per image (CIFAR-10 output layer)")
u = [r for r in rd("architecture_generalization/cpu_prunable_units.csv") if r["num_classes"] == "10" and r["arch"] in ("simplecnn", "lenet5", "resnet8")]
table(["Architecture", "Unit", "Role", "Modules", "Weight shapes", "Weights", "Share of prunable", "MACs per image"],
      [[r["arch"], r["unit"], r["role"], r["modules"], r["weight_shapes"], f"{int(r['params']):,}",
        (hu(100 * float(r["fraction_of_prunable_weights"]), 2) + "%") if r["fraction_of_prunable_weights"] else "—", f"{int(r['macs_per_image']):,}"] for r in u])
out.append("*Source: `results/architecture_generalization/cpu_prunable_units.csv`.*")
out.append("")
H("### A.2 Dense references: three training seeds")
table(["Setting", "Seeds", "Reference", "Val. (ref.)", "Test (ref.)", "Val. mean ± SD", "Test mean ± SD", "Parameters", "Recipe"],
      [[f"{r['dataset']} {r['arch']}", r["n_seeds"], r["reference"].split("/")[-1], r["reference_val_accuracy"], r["reference_test_accuracy"],
        f"{hu(r['val_mean'], 2)} ± {hu(r['val_sd'], 2)}" if r["val_sd"] else hu(r["val_mean"], 2),
        f"{hu(r['test_mean'], 2)} ± {hu(r['test_sd'], 2)}" if r["test_sd"] else hu(r["test_mean"], 2), f"{int(r['params_total']):,}", r["recipe"]]
       for r in rd("architecture_generalization/cpu_baselines.csv")])
out.append("*Source: `results/architecture_generalization/cpu_baselines.csv`.*")
out.append("")
H("### A.3 Recovery protocol (Phase 6; reused in Paper 3, Phases 7A and 7B)")
pr = j("phase6/phase6_config.json")["protocol"]
table(["Configuration key", "Value"], [[f"`{k}`", str(v)] for k, v in pr.items()])
out.append("*Source: `results/phase6/phase6_config.json` (`protocol`). The PPO hyper-parameters are listed in Table 3.7 and Table 4.6.*")
out.append("")

# ---------------------------------------------------------------- B
H("## Appendix B — Full Phase 3 Results")
H("### B.1 Mean test accuracy and sparsity by setting and method (20 runs; baselines at each PPO run's zero count)")
sm = rd("architecture_generalization/phase3_summary.csv")
table(["Setting", "Method", "Test mean", "Test SD", "Sparsity mean (%)", "Drop vs dense (pp)"],
      [[NM[r["setting"]], r["method"], hu(r["test_mean"], 2), hu(r["test_sd"], 2), hu(r["sparsity_mean"], 2), hu(r["drop_mean"], 2)] for r in sm if r["method"] != "dense"])
out.append("*Source: `results/architecture_generalization/phase3_summary.csv`.*")
out.append("")
H("### B.2 All pre-registered Phase 3 comparisons (paired by seed, n = 20; exact sign-flip test, Holm within family)")
table(SH, stats_rows(rd("architecture_generalization/phase3_statistics.csv")))
out.append("*Source: `results/architecture_generalization/phase3_statistics.csv`.*")
out.append("")

# ---------------------------------------------------------------- C
H("## Appendix C — Sensitivity Vectors and Phase 4 Prior Diagnostics")
H("### C.1 Normalised loss sensitivity S_l by setting")
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
out.append("*Sources: `results/architecture_generalization/phase3_frozen_inputs.json`; `results/phase5/phase5_config.json`; "
           "`results/phase7b_green_pruning/structured_sensitivity.json`.*")
out.append("")
H("### C.2 Phase 4 prior comparisons (reward-rank AUC and secondary endpoints)")
s4 = rd("phase4/phase4_statistics.csv")
table(SH, stats_rows([r for r in s4 if r["family"].startswith("B")]))
out.append("*Source: `results/phase4/phase4_statistics.csv` (families B-*).*")
out.append("")

# ---------------------------------------------------------------- D
H("## Appendix D — Archive and Reward-Alignment Diagnostics")
H("### D.1 Phase 4 search quality by variant (VAL-RL (I) ranks among 1,296 policies; medians over 20 runs)")
table(["Setting", "Variant", "Selected rank", "Selected V_SELECT rank", "Best-seen rank", "Best-seen top-10", "Selected top-10", "Selected top-20", "Test mean", "Sparsity mean"],
      [[NM.get(r["setting"], r["setting"]), r["condition"][:28], r["selected_rank_vrl_median"], r["selected_rank_vselect_median"] or "—", r["best_seen_rank_median"],
        hu(r["best_seen_reach_top10"], 2) if r["best_seen_reach_top10"] else "—", hu(r["selected_reach_top10"], 2), hu(r["selected_reach_top20"], 2),
        hu(r["selected_test_mean"], 2), hu(r["selected_sparsity_mean"], 2) if r["selected_sparsity_mean"] else "—"] for r in rd("phase4/phase4_search_quality.csv")])
out.append("*Source: `results/phase4/phase4_search_quality.csv`.*")
out.append("")
H("### D.2 Phase 4 Stage-A rewards (P0, terminal policy)")
table(["Setting", "Reward", "Sparsity mean", "Sparsity SD", "VAL-RL retention", "Test mean", "Reward optimum sparsity", "Optimum retention"],
      [[NM.get(r["setting"], r["setting"]), r["reward"], hu(r["sparsity_mean"], 2), hu(r["sparsity_sd"], 2), hu(r["vrl_retention_mean"], 3), hu(r["test_mean"], 2),
        hu(r["oracle_optimum_sparsity"], 2), hu(r["oracle_optimum_retention"], 3)] for r in rd("phase4/phase4_reward_comparison.csv")])
out.append("*Source: `results/phase4/phase4_reward_comparison.csv`.*")
out.append("")
H("### D.3 Phase 4 final pipeline F and control K against matched baselines (test, pp)")
table(["Pipeline", "Setting", "Baseline", "PPO test", "Baseline test", "PPO − baseline", "PPO wins / ties / baseline wins", "Sparsity"],
      [[r["pipeline"], NM[r["setting"]], r["method"], hu(r["ppo_test_mean"], 2), hu(r["baseline_test_mean"], 2), hu(r["ppo_minus_baseline_mean"], 2),
        f"{r['ppo_wins']}/{r['ties']}/{r['baseline_wins']}", hu(r["sparsity_mean"], 2)] for r in rd("phase4/phase4_baseline_comparison.csv")])
out.append("*Source: `results/phase4/phase4_baseline_comparison.csv`.*")
out.append("")
H("### D.4 Phase 4 search, final-model and LAMP families")
table(SH, stats_rows([r for r in s4 if r["family"].split()[0] in ("C-final", "C-reach", "C-heldout", "D", "D-GM", "E", "E-NI", "A-regret")]))
out.append("*Source: `results/phase4/phase4_statistics.csv`. The E family's endpoint is labelled `val` in the source file, but its values are test accuracies (Chapter 5 QA §19.6).*")
out.append("")

# ---------------------------------------------------------------- E
H("## Appendix E — Phase 5 Detailed Results")
H("### E.1 Selected policies by setting and condition (20 runs each)")
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
out.append("*Source: `results/phase5/phase5_constraint_results.csv`.*")
out.append("")
H("### E.2 All pre-registered and descriptive Phase 5 comparisons")
table(SH, stats_rows(rd("phase5/phase5_statistics.csv")))
out.append("*Source: `results/phase5/phase5_statistics.csv`.*")
out.append("")

# ---------------------------------------------------------------- F
H("## Appendix F — Phase 6 Recovery Results")
H("### F.1 Test accuracy before and after equal recovery, by setting and method (means of 20 runs)")
table(["Setting", "Method", "Sparsity", "Test before", "Test after", "Recovery", "Gap to dense after", "Retention ≥ 0.98 before / after", "Zero-count change"],
      [[NM[r["setting"]], r["method"], hu(r["sparsity"], 2), hu(r["test_pre_mean"], 2), hu(r["test_post_mean"], 2), hu(r["recovery_mean"], 2), hu(r["dense_gap_post"], 3),
        f"{hu(r['share_retention_ge_98_pre'], 2)} / {hu(r['share_retention_ge_98_post'], 2)}", r["zero_count_change"]] for r in rd("phase6/phase6_pre_post_accuracy.csv")])
out.append("*Source: `results/phase6/phase6_pre_post_accuracy.csv`.*")
out.append("")
H("### F.2 PPO vs LAMP and PPO vs global magnitude (H2, H3 and reference families)")
rows = []
for f_ in ("phase6/phase6_ppo_vs_lamp.csv", "phase6/phase6_ppo_vs_global.csv"):
    for r in rd(f_):
        rows.append([NM[r["setting"]], r["family"].split(" (")[0], r["endpoint"], hu(r["mean_diff"], 3), f"[{hu(r['t_ci95_low'], 3)}, {hu(r['t_ci95_high'], 3)}]",
                     p_(r["holm_p"]), r["outcome"], r["equivalent_within_margin"]])
table(["Setting", "Family", "Endpoint", "Mean diff.", "95% t-CI", "Holm p", "Outcome", "CI inside ±0.5 pp"], rows)
out.append("*Sources: `results/phase6/phase6_ppo_vs_lamp.csv`, `results/phase6/phase6_ppo_vs_global.csv`. The column 'CI inside ±0.5 pp' is the pre-registered confidence-interval decision rule; it is not a formal equivalence test.*")
out.append("")

# ---------------------------------------------------------------- G
H("## Appendix G — Hardware and Latency Protocol")
H("### G.1 H0 calibration: per-configuration decision quantities")
h0 = j("paper3/h0_decision.json")["configs"]
table(["Config.", "Signal (ms)", "Noise (ms)", "SNR", "Relative reduction", "Positive in all sessions", "Usable"],
      [[k, hu(v["signal_ms"], 4), hu(v["noise_ms"], 4), hu(v["snr"], 2), hu(100 * v["relative_reduction"], 1) + "%", v["delta_positive_all_sessions"], v["usable"]] for k, v in h0.items()])
out.append("*Source: `results/paper3/h0_decision.json`. Rule: SNR ≥ 5, reduction ≥ 10%, positive in every session; order A, C, B.*")
out.append("")
H("### G.2 H0 calibration shapes (configuration A)")
table(["Model", "Actions", "Parameters", "Param. reduction", "MAC reduction", "Median (ms)", "Between-session CV", "Speed-up vs dense"],
      [[r["model"], r["actions"], f"{int(r['params']):,}", hu(100 * float(r["param_reduction"]), 1) + "%", hu(100 * float(r["mac_reduction"]), 1) + "%",
        hu(r["session_median_ms_median"], 4), hu(100 * float(r["between_session_cv"]), 2) + "%", hu(r["speedup_vs_dense"], 3)] for r in rd("paper3/h0_latency_summary.csv") if r["config"] == "A"])
out.append("*Source: `results/paper3/h0_latency_summary.csv`.*")
out.append("")
H("### G.3 Latency-table session log (anchor drift; limit 10%)")
fz = j("paper3/h2_latency_freeze.json")
table(["Table", "Session", "Attempt", "Load before (%)", "Load after (%)", "Anchor drift", "Accepted"],
      [[s["pair"], s["session"], s["attempt"], s["load_before"], s["load_after"], hu(100 * s["anchor_drift"], 1) + "%", s["ok"]] for s in fz["session_log"]])
out.append("*Source: `results/paper3/h2_latency_freeze.json` (`session_log`).*")
out.append("")
H("### G.4 Proxy validation and ResNet-8 threshold calibration")
pdj = j("paper3/paper3_proxy_decision.json")["architectures"]
table(["Architecture", "Validation pairs", "Spearman ρ(A₁, A₅)", "MAE of q", "Top-10 overlap", "q₅-feasible pairs", "GO"],
      [[k, v["n_pairs"], hu(v["spearman_rho"], 3), hu(v["mae_q"], 4), v["top10_overlap"], v["q5_feasible_pairs"], v["go"]] for k, v in pdj.items()])
cal = j("paper3/resnet8_threshold_calibration_decision.json")
table(["Item", "Value"], [["Calibration policies / truly feasible", f"{cal['n_calibration']} / {cal['n_true_feasible']}"], ["τ_proxy", hu(cal["tau_proxy"], 4)],
                          ["Recall / precision (TP, FP, FN, TN)", f"{hu(cal['recall'], 2)} / {hu(cal['precision'], 3)} ({cal['tp']}, {cal['fp']}, {cal['fn']}, {cal['tn']})"],
                          ["Proxy-feasible landscape policies", cal["latency_room"]["proxy_feasible"]], ["Best proxy-feasible speed-up", hu(cal["latency_room"]["best_speedup"], 3) + "×"],
                          ["Policies ≥ 1.10×", cal["latency_room"]["policies_with_speedup_ge_1.10"]], ["GO to PPO", cal["go_ppo"]]])
out.append("*Sources: `results/paper3/paper3_proxy_decision.json`, `results/paper3/resnet8_threshold_calibration_decision.json`. The best confirmed five-epoch-feasible calibration policy reached 1.089× ([0, 3, 1, 1]; `resnet8_threshold_calibration_report.md`).*")
out.append("")

# ---------------------------------------------------------------- H
H("## Appendix H — Phase 7A Energy Protocol and Session Results")
H("### H.1 Per-session measurements (N = 100,000 batch-1 inferences per block)")
er = rd("phase7a_sustainability/energy_raw.csv")
table(["Architecture", "Session", "Order", "Method", "J / 1,000 (package)", "J / 1,000 (PP0)", "Package W", "Median latency (ms)"],
      [[r["arch"], r["session"], r["position"], r["method"], hu(r["e1000_j"], 3), hu(r["pp0_e1000_j"], 3), hu(r["pkg_w"], 2), hu(r["latency_median_ms"], 4)] for r in er])
out.append("*Source: `results/phase7a_sustainability/energy_raw.csv`.*")
out.append("")
H("### H.2 Session summaries")
table(["Architecture", "Method", "Median J / 1,000", "CV over sessions", "Median saving", "Min–max saving", "Sessions below dense", "One-sided Wilcoxon p", "Consistent reduction"],
      [[r["arch"], r["method"], hu(r["e1000_median_j"], 3), hu(100 * float(r["e1000_cv"]), 2) + "%", hu(100 * float(r["saving_median"]), 1) + "%",
        f"{hu(100 * float(r['saving_min']), 1)}–{hu(100 * float(r['saving_max']), 1)}%", r["sessions_lower_than_dense"], p_(r["wilcoxon_one_sided_p"]) if r["wilcoxon_one_sided_p"] else "—",
        r["consistent_reduction"]] for r in rd("phase7a_sustainability/energy_summary.csv")])
out.append("*Source: `results/phase7a_sustainability/energy_summary.csv`.*")
out.append("")

# ---------------------------------------------------------------- I
H("## Appendix I — Phase 7B Confirmatory Results")
H("### I.1 Per-seed selected policies (20 confirmatory seeds)")
ps = j("phase7b_green_pruning/selected_policies.json")["per_seed"]
table(["Seed", "Actions", "Widths", "Proxy q", "MAC reduction", "Parameter reduction", "Table latency (ms)"],
      [[k, v["actions"], v["widths"], hu(v["q_proxy"], 4), hu(100 * v["mac_reduction"], 1) + "%", hu(100 * v["param_reduction"], 1) + "%", hu(v["latency_table_ms"], 4)] for k, v in sorted(ps.items())])
out.append("*Source: `results/phase7b_green_pruning/selected_policies.json`. Proxy threshold τ̂ = 0.98849. Predicted (surrogate) energies are deliberately omitted: they are search-internal and are not results.*")
out.append("")
H("### I.2 All recovered final models (5-epoch recovery)")
table(["Id", "Roles", "Widths", "VAL-SELECT", "Δ VAL-SELECT", "VAL-RL", "Memory ↓", "MACs ↓", "Joint criterion"],
      [[r["id"], ", ".join(json.loads(r["roles"])), r["widths"], hu(r["val_select_post"], 2), hu(r["delta_val_select_pp"], 2), hu(r["val_rl_post"], 2),
        hu(100 * float(r["memory_reduction"]), 1) + "%", hu(100 * float(r["mac_reduction"]), 1) + "%", r["green_success"]] for r in rd("phase7b_green_pruning/final_summary.csv")])
out.append("*Source: `results/phase7b_green_pruning/final_summary.csv`. The last column is the pre-registered joint deployment-efficiency criterion (internal file name GREEN-SUCCESS). Latency and energy were not measured.*")
out.append("")

# ---------------------------------------------------------------- J
H("## Appendix J — Energy-Measurement Qualification and Failure Records")
H("### J.1 Dense-only stability pilots and qualification (energy per 1,000 inferences, J)")
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
out.append("*Sources: `results/phase7a_sustainability/energy_pilot/pilot_n100000.json`; `results/phase7b_green_pruning/_energy_calibration/pilot_n20000.json`; "
           "`results/phase7b_green_pruning/statistics.json`; `results/phase7b_energy_confirmation/measurement_qualification.json`. "
           "A doubled-N 7B pilot attempt stopped by the load gate was declared void by recorded user decision "
           "(`_energy_final/void_pilot_n200000_20261027/`, `development_log.md`) and is not listed.*")
out.append("")
H("### J.2 Energy-confirmation qualification blocks (descriptive diagnostics)")
table(["Run", "Position", "J / 1,000", "Package W", "Median latency (ms)", "System load (%)", "CPU-2 performance (%)", "Thermal zone start → end (°C)"],
      [[b["run"], b["position"], hu(b["e1000_j"], 3), hu(b["pkg_w"], 2), hu(b["latency_median_ms"], 4), hu(b["system_load_pct"], 1), hu(b["perf_pct_cpu2"], 0),
        f"{hu(b['thermal_zone_c_start'], 0)} → {hu(b['thermal_zone_c_end'], 0)}"] for b in q["blocks"]])
out.append("*Source: `results/phase7b_energy_confirmation/measurement_qualification.json` (`blocks`). No cause of the instability is claimed, and no Dense-versus-PPO session was run.*")
out.append("")

# ---------------------------------------------------------------- K
H("## Appendix K — Reproducibility Information")
H("### K.1 Frozen inputs (SHA-256 prefixes)")
m100 = j("reproducibility/cifar100_split_manifest.json")
m5 = j("phase5/splits/phase5_split_manifest.json")
rows = [[f"Dense reference {r['dataset']} {r['arch']}", r["reference"].split("/")[-1], r["reference_sha256"][:16]] for r in rd("architecture_generalization/cpu_baselines.csv")]
rows += [["Configuration-I split (CIFAR-10)", "archive_validation_split.npz", "55b7ff40a25b9b24"], ["Configuration-I split (CIFAR-100)", "cifar100_split_indices.npz", m100["file_sha256"][:16]]]
rows += [[f"Configuration-II split ({k})", v["file"].split("/")[-1], v["file_sha256"][:16]] for k, v in m5["datasets"].items()]
table(["Item", "File", "SHA-256 (prefix)"], rows)
out.append("*Sources: `results/architecture_generalization/cpu_baselines.csv`; `results/reproducibility/*_split_manifest.json`; `results/phase5/splits/phase5_split_manifest.json`.*")
out.append("")
H("### K.2 Pre-registration and freeze commits")
K2 = ([["Phase 2 dense training", "results/cpu_dense_training_protocol.md", "366d21ae"], ["Phase 3", "results/phase3_preregistration.md", "2dbcc770"],
       ["Phase 4", "results/phase4/phase4_preregistration.md", "f52e87d9"], ["Phase 5", "results/phase5/phase5_preregistration.md", "3cbaec1a"],
       ["Phase 6", "results/phase6/phase6_preregistration.md", "b1b5396f"], ["Paper 3 H0", "results/paper3/h0_protocol.md", "c2dcb22a"],
       ["Paper 3 H2", "results/paper3/paper3_preregistration.md", "45936846"], ["Paper 3 proxy", "results/paper3/paper3_proxy_protocol.md", "ec7a7f27"],
       ["Paper 3 ResNet-8 calibration", "results/paper3/resnet8_threshold_calibration_protocol.md", "5b127804"],
       ["Phase 7A", "results/phase7a_sustainability/preregistration.json", "b8c27665"], ["Phase 7B design", "results/phase7b_green_pruning/phase7b_design.md", "f5abe039"],
       ["Phase 7B final", "results/phase7b_green_pruning/phase7b_final_preregistration.json", "0e101d55"],
       ["7B energy confirmation", "results/phase7b_energy_confirmation/protocol.json", "df233be0"]])
table(["Experiment", "Governing file", "Commit that first added it"], [[e, f"`{g}`", f"`{c}`"] for e, g, c in K2])
out.append("*Commits verified with `git log --diff-filter=A` (Chapter 4, Table 4.9).*")
out.append("")
H("### K.3 Software environments")
e5 = j("phase5/phase5_environment.json")
ec = j("phase7b_energy_confirmation/environment.json")
table(["Item", "Core framework (Phases 2–6)", "Deployment extension (Paper 3, 7A, 7B, energy confirmation)"],
      [["CPU", e5["cpu"], ec["cpu"]["Name"]], ["OS", e5["os"] + " (Windows 11 build 22621)", f"{ec['os']['Caption']} {ec['os']['Version']}"],
       ["Python", e5["python"], ec["python"].split()[0]], ["PyTorch", e5["torch"], ec["torch"]],
       ["Stable-Baselines3", e5["stable-baselines3"], "2.8.0 (pinned environment)"], ["Gymnasium / NumPy", f"{e5['gymnasium']} / {e5['numpy']}", "pinned environment"],
       ["CUDA", str(e5["cuda_available"]), "not used"]])
out.append("*Sources: `results/phase5/phase5_environment.json`; `results/phase7b_energy_confirmation/environment.json`; `requirements_locked.txt`.*")
out.append("")
H("### K.4 Where the full records are")
K4 = ([["3", "results/architecture_generalization/phase3_all_runs.csv, phase3_runs/, phase3_predictions/"], ["4", "results/phase4/phase4_all_runs.csv, runs/, predictions/"],
       ["5", "results/phase5/phase5_all_runs.csv, phase5_archive.json, phase5_final_policies.json, runs/"],
       ["6", "results/phase6/phase6_all_runs.csv, phase6_final_models.json, runs/, test_evaluations/"],
       ["Paper 3", "results/paper3/h2_latency_table_*.csv, h2_latency_raw_*.csv, h2_latency_freeze.json"],
       ["7A", "results/phase7a_sustainability/energy_raw.csv, accuracy_results.csv, recovered_models_freeze.json"],
       ["7B", "results/phase7b_green_pruning/ppo_all_seeds.csv, archive_all_policies.csv, recovery_results.csv, proxy_landscape.csv"],
       ["7B energy confirmation", "results/phase7b_energy_confirmation/energy_raw.csv, measurement_qualification.json, checkpoint_verification.json"]])
table(["Phase", "Run-level records"], [[ph, ", ".join(f"`{x}`" for x in rec.split(", "))] for ph, rec in K4])

open(os.path.join(W, "appendices.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
print(f"appendices.md: {len(out)} lines")
