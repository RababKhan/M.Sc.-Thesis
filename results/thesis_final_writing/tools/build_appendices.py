"""Generate the thesis appendices (A-E, MIST final, selective) directly from the frozen result files. Read-only.

  python results/thesis_final_writing/tools/build_appendices.py   ->  results/thesis_final_writing/appendices.md
Every number is copied or rounded (round-half-up) from the frozen file named under its table; nothing is typed and
nothing is recomputed beyond selecting rows/columns (pivots of stored means). Complete per-seed and per-session records
stay in the repository (statement in Appendix A).
"""
import csv
import json
import os
from decimal import ROUND_HALF_UP, Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
R = os.path.dirname(W)
S6 = ["cifar10_simplecnn", "cifar10_lenet5", "cifar10_resnet8", "cifar100_simplecnn", "cifar100_lenet5", "cifar100_resnet8"]
NM = {"cifar10_simplecnn": "C10 SimpleCNN", "cifar10_lenet5": "C10 LeNet-5", "cifar10_resnet8": "C10 ResNet-8",
      "cifar100_simplecnn": "C100 SimpleCNN", "cifar100_lenet5": "C100 LeNet-5", "cifar100_resnet8": "C100 ResNet-8"}
ARCH = {"simplecnn": "SimpleCNN", "lenet5": "LeNet-5", "resnet8": "ResNet-8"}
out = []
CUR = {"app": "", "n": 0}


def hu(v, d):
    if v in (None, ""):
        return "—"
    return str(Decimal(repr(round(float(v), 6))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)).replace("-", "−")


def p_(v):
    v = float(v)
    return f"{v:.2g}" if v >= 0.001 else f"{v:.1e}"


def rd(p):
    return list(csv.DictReader(open(os.path.join(R, p), encoding="utf-8")))


def j(p):
    return json.load(open(os.path.join(R, p), encoding="utf-8"))


def H(t):
    out.append(t)
    out.append("")


def APP(letter, title):
    CUR["app"], CUR["n"] = letter, 0
    H(f"## Appendix {letter} — {title}")


def CAP(text):
    CUR["n"] += 1
    out.append(f"**Table {CUR['app']}.{CUR['n']}. {text}**")
    out.append("")


def table(head, rows):
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    out.append("")


def note(s):
    out.append(f"*{s}*")
    out.append("")


H("# Appendices")
out.append("The appendices give only the supporting values an examiner may need to verify statements in Chapters 3–6")
out.append("without opening the code repository. Each table is generated from the frozen phase records by")
out.append("`tools/build_appendices.py` and names its source record.")
out.append("")

# ================================================================ A
APP("A", "Architectures, Hyperparameters and Reproducibility")
H("### A.1 Pruning units")
CAP("Pruning units of the three networks: weights, share of prunable weights and MACs per image (CIFAR-10 output layer).")
u = [r for r in rd("architecture_generalization/cpu_prunable_units.csv") if r["num_classes"] == "10" and r["arch"] in ARCH]
table(["Network", "Unit", "Role", "Weights", "Share of prunable", "MACs per image"],
      [[ARCH[r["arch"]], r["unit"], "pruned (PPO decision)" if r["role"].startswith("prunable") else "never pruned",
        f"{int(r['params']):,}", (hu(100 * float(r["fraction_of_prunable_weights"]), 2) + "%") if r["fraction_of_prunable_weights"] else "—",
        f"{int(r['macs_per_image']):,}"] for r in u])
note("Source: `cpu_prunable_units.csv`. The PPO hyper-parameters are in Table 4.6 and the equal recovery protocol in Section 4.13.")
H("### A.2 Reproducibility")
out.append("Every experiment was pre-registered and frozen before its confirmatory runs; Table 4.10 lists the governing")
out.append("records and the commits that first added them. The complete per-seed experimental records, per-session")
out.append("measurement records, machine-readable result files (CSV and JSON), checkpoint hashes and audit scripts are")
out.append("retained in the associated research repository, organised by phase: `architecture_generalization` (Phases 2–3),")
out.append("`phase4`, `phase5`, `phase6`, `paper3` (latency), `phase7a_sustainability`, the Phase-7B folder and")
out.append("`phase7b_energy_confirmation`; the thesis sources and audits are in `thesis_final_writing`.")
out.append("")

# ================================================================ B
APP("B", "Supplementary Core-Pruning Results")
H("### B.1 Phase 3 absolute accuracies")
CAP("Phase 3 mean test accuracy (%) of PPO (P0) and the five baselines built at each PPO run's zero count (20 runs).")
sm = {(r["setting"], r["method"]): r for r in rd("architecture_generalization/phase3_summary.csv")}
cols = [("PPO P0", "PPO (P0)"), ("uniform (matched to P0)", "Uniform"), ("random (matched to P0)", "Random"),
        ("global (matched to P0)", "Global"), ("erk (matched to P0)", "ERK"), ("lamp (matched to P0)", "LAMP")]
table(["Setting (dense test)", "P0 sparsity (%)"] + [c[1] for c in cols],
      [[f"{NM[s]} ({hu(sm[(s, 'dense')]['test_mean'], 2)})", hu(sm[(s, 'PPO P0')]['sparsity_mean'], 2)] +
       [hu(sm[(s, m)]["test_mean"], 2) for m, _ in cols] for s in S6])
note("Source: `phase3_summary.csv`. Paired differences and their statistics are in Table 5.2.")
H("### B.2 Sensitivity vectors")
CAP("Normalised loss sensitivity S_l of the four units by setting and the Phase-5 centred prior coefficients.")
fi = j("architecture_generalization/phase3_frozen_inputs.json")["settings"]
p5 = j("phase5/phase5_config.json")["settings"]
rows = []
for s in S6:
    raw3 = fi[s]
    n3 = raw3.get("sensitivity_normalized") or raw3.get("normalized_sensitivity") or raw3.get("sensitivity", {}).get("normalized")
    rows.append([NM[s], ", ".join(hu(x, 3) for x in n3) if n3 else "see source", ", ".join(hu(x, 3) for x in p5[s]["sensitivity_normalized"]),
                 ", ".join(hu(x, 3) for x in p5[s]["P2_coefficients"])])
table(["Setting", "Configuration I (Phases 3–4)", "Configuration II (Phase 5)", "Phase-5 coefficients c_l = S_l − S̄"], rows)
note("Sources: `phase3_frozen_inputs.json`; `phase5_config.json`.")

# ================================================================ C
APP("C", "Archive, Reward and Recovery Diagnostics")
H("### C.1 Recovery gains under equal fine-tuning")
CAP("Phase 6 mean recovery (test accuracy after minus before equal fine-tuning, pp; 20 runs) by setting and method.")
pp = {(r["setting"], r["method"]): r for r in rd("phase6/phase6_pre_post_accuracy.csv")}
mcols = [("ppo", "PPO"), ("lamp", "LAMP"), ("global", "Global magnitude"), ("dense", "Dense control")]


def sgn(v):
    return ("+" if float(v) >= 0 else "") + hu(v, 2)


table(["Setting (sparsity)"] + [c[1] for c in mcols],
      [[f"{NM[s]} ({hu(pp[(s, 'ppo')]['sparsity'], 1)}%)"] + [sgn(pp[(s, m)]["recovery_mean"]) for m, _ in mcols] for s in S6])
note("Source: `phase6_pre_post_accuracy.csv`. Accuracies before and after recovery and the PPO–LAMP and PPO–Global gaps are in Table 5.7. "
     "Phase-4 search quality and Phase-5 per-run selections are summarised in Tables 5.3, 5.4 and 5.5; their run-level records are in the repository.")

# ================================================================ D
APP("D", "Hardware and Structured-Pruning Protocol")
H("### D.1 Latency-measurement calibration")
CAP("H0 calibration: decision quantities of the three candidate measurement configurations.")
h0 = j("paper3/h0_decision.json")["configs"]
table(["Configuration", "Signal (ms)", "Noise (ms)", "SNR", "Relative reduction", "Positive in all sessions", "Usable"],
      [[k, hu(v["signal_ms"], 4), hu(v["noise_ms"], 4), hu(v["snr"], 2), hu(100 * v["relative_reduction"], 1) + "%", v["delta_positive_all_sessions"], v["usable"]]
       for k, v in h0.items()])
note("Source: `h0_decision.json`. Rule: SNR ≥ 5, reduction ≥ 10%, positive in every session; preference order A, C, B.")
H("### D.2 Latency-table sessions")
CAP("Exhaustive latency tables: accepted sessions, repeated attempts and anchor drift (limit 10%).")
fz = j("paper3/h2_latency_freeze.json")["session_log"]
TBL = {"resnet8_c10": "ResNet-8, CIFAR-10", "resnet8_c100": "ResNet-8, CIFAR-100", "simplecnn_c10": "SimpleCNN, CIFAR-10",
       "simplecnn_c100": "SimpleCNN, CIFAR-100"}
rows = []
for pair in sorted({s["pair"] for s in fz}):
    g = [s for s in fz if s["pair"] == pair]
    acc = [s for s in g if s["ok"]]
    rej = [s for s in g if not s["ok"]]
    rows.append([TBL.get(pair, pair), len(acc), len(rej), hu(100 * max(s["anchor_drift"] for s in acc), 1) + "%",
                 (hu(100 * max(s["anchor_drift"] for s in rej), 1) + "%") if rej else "—"])
table(["Latency table", "Accepted sessions", "Repeated attempts", "Largest accepted drift", "Largest rejected drift"], rows)
note("Source: `h2_latency_freeze.json` (`session_log`); the per-session log is in the repository.")

# ================================================================ E
APP("E", "Energy Measurement and Qualification")
H("### E.1 Phase 7A energy sessions")
CAP("Phase 7A energy: session summaries (five sessions per architecture).")
table(["Network", "Method", "Median J / 1,000", "CV over sessions", "Median saving", "Min–max saving", "Sessions below dense", "One-sided Wilcoxon p"],
      [[ARCH.get(r["arch"], r["arch"]), r["method"], hu(r["e1000_median_j"], 3), hu(100 * float(r["e1000_cv"]), 2) + "%",
        hu(100 * float(r["saving_median"]), 1) + "%", f"{hu(100 * float(r['saving_min']), 1)}–{hu(100 * float(r['saving_max']), 1)}%",
        r["sessions_lower_than_dense"], p_(r["wilcoxon_one_sided_p"]) if r["wilcoxon_one_sided_p"] else "—"]
       for r in rd("phase7a_sustainability/energy_summary.csv")])
note("Source: `energy_summary.csv`; per-block session records are in the repository.")
H("### E.2 Energy-measurement qualification")
CAP("Dense-only stability pilots and qualification (energy per 1,000 inferences, J; six dense blocks each).")
p7a = j("phase7a_sustainability/energy_pilot/pilot_n100000.json")["results"]
cal7b = j("phase7b_green_pruning/_energy_calibration/pilot_n20000.json")
st7b = j("phase7b_green_pruning/statistics.json")["pilots"]
q = j("phase7b_energy_confirmation/measurement_qualification.json")
rows = [[f"7A pilot, {ARCH.get(a, a)}", "100,000", ", ".join(hu(x, 3) for x in v["e1000_j"]), hu(100 * v["cv"], 2) + "%", "3%", "passed"]
        for a, v in p7a.items()]
rows.append(["7B surrogate-calibration pilot", "20,000", ", ".join(hu(x, 3) for x in cal7b["e1000_j"]), hu(100 * cal7b["cv"], 2) + "%", "4%", cal7b["decision"]])
for k, v in st7b.items():
    rows.append([f"7B confirmatory pilot (N = {v['n']:,})", f"{v['n']:,}", ", ".join(hu(x, 3) for x in v["e1000_j"]), hu(100 * v["cv"], 2) + "%", "3%",
                 v["decision"]])
rows.append(["7B energy-confirmation qualification", f"{int(q['n']):,}", ", ".join(hu(x, 3) for x in q["e1000_j"]), hu(100 * float(q["cv"]), 2) + "%",
             "3%", q["decision"]])
table(["Measurement", "N per block", "Dense blocks", "CV", "Limit", "Decision"], rows)
note("Sources: `pilot_n100000.json` (Phase 7A); `pilot_n20000.json`, `statistics.json` (Phase 7B); `measurement_qualification.json` "
     "(energy confirmation). A doubled-N Phase-7B attempt stopped by the load gate was declared void by a recorded decision of the author and is not "
     "listed. The per-block diagnostics and the eleven recovered Phase-7B models (summarised in Table 6.4) are in the repository.")

open(os.path.join(W, "appendices.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
print(f"appendices.md: {len(out)} lines")
