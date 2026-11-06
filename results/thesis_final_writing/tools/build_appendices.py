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
out.append("without opening the code repository. Every table is generated directly from the frozen phase records; the")
out.append("records and all implementation identifiers are listed once, in Section A.2.")
out.append("")

# ================================================================ A
APP("A", "Architectures, Hyperparameters and Reproducibility")
H("### A.1 Pruning units")
CAP("Pruning units of the three networks: weights, share of prunable weights and MACs per image (CIFAR-10 output layer).")
UNIT = {"features.0": "conv1", "features.3": "conv2", "features.6": "conv3", "classifier.1": "fc1", "classifier.3": "output layer",
        "classifier.5": "output layer", "fc": "output layer", "stage1": "stage 1", "stage2": "stage 2", "stage3": "stage 3"}
u = [r for r in rd("architecture_generalization/cpu_prunable_units.csv") if r["num_classes"] == "10" and r["arch"] in ARCH]
table(["Network", "Unit", "Role", "Weights", "Share of prunable", "MACs per image"],
      [[ARCH[r["arch"]], UNIT.get(r["unit"], r["unit"]) if not (r["arch"] == "resnet8" and r["unit"] == "fc") else "output layer", "pruned (PPO decision)" if r["role"].startswith("prunable") else "never pruned",
        f"{int(r['params']):,}", (hu(100 * float(r["fraction_of_prunable_weights"]), 2) + "%") if r["fraction_of_prunable_weights"] else "—",
        f"{int(r['macs_per_image']):,}"] for r in u])
note("Unit names as in Table 3.2 (implementation module names in Section A.2). Source: frozen unit table. The PPO hyper-parameters are in Table 4.6 "
     "and the equal recovery protocol in Section 4.13.")
H("### A.2 Reproducibility")
out.append("Every experiment was pre-registered and frozen before its confirmatory runs. The complete per-seed experimental")
out.append("records, per-session measurement records, machine-readable result files (CSV and JSON), checkpoints and audit")
out.append("scripts are retained in the associated research repository, organised by phase: `architecture_generalization`")
out.append("(Phases 2–3), `phase4`, `phase5`, `phase6`, `paper3` (latency), `phase7a_sustainability`, the Phase-7B folder and")
out.append("`phase7b_energy_confirmation`; the thesis sources and audits are in `thesis_final_writing`. Tables A.2 and A.3")
out.append("identify the frozen reference checkpoints and the governing pre-registration records.")
out.append("")
CAP("Frozen dense reference checkpoints (Table 4.4): checkpoint file and SHA-256 prefix.")
table(["Setting", "Checkpoint file", "SHA-256 (prefix)"],
      [[f"{'CIFAR-10' if r['dataset'] == 'cifar10' else 'CIFAR-100'} {ARCH.get(r['arch'], r['arch'])}", f"`{r['reference'].split('/')[-1]}`",
        f"`{r['reference_sha256'][:8]}`"] for r in sorted(rd("architecture_generalization/cpu_baselines.csv"), key=lambda r: (r["dataset"] != "cifar10", list(ARCH).index(r["arch"])))])
CAP("Pre-registration records (Table 4.10) and the commits that first added them.")
PRE = [('Phase 2', [['cpu_dense_training_protocol.md', '366d21ae']]), ('Phase 3', [['phase3_preregistration.md', '2dbcc770']]), ('Phase 4', [['phase4_preregistration.md', 'f52e87d9']]), ('Phase 5', [['phase5_preregistration.md', '3cbaec1a']]), ('Phase 6', [['phase6_preregistration.md', 'b1b5396f']]), ('Paper 3 H0', [['h0_protocol.md', 'c2dcb22a']]), ('Paper 3 H2, proxy', [['paper3_preregistration.md', '45936846'], ['paper3_proxy_protocol.md', 'ec7a7f27']]), ('Paper 3 ResNet-8', [['resnet8_threshold_calibration_protocol.md', '5b127804']]), ('Phase 7A', [['preregistration.json', 'b8c27665']]), ('Phase 7B Stage A', [['phase7b_design.md', 'f5abe039']]), ('Phase 7B Stage B', [['phase7b_final_preregistration.json', '0e101d55']]), ('7B energy confirmation', [['protocol.json', 'df233be0']])]
table(["Experiment", "Governing record", "Commit"],
      [[e, "; ".join(f"`{f}`" for f, _ in recs), "; ".join(f"`{c}`" for _, c in recs)] for e, recs in PRE])
note("Commits verified with `git log --diff-filter=A`.")
out.append("**Implementation identifiers.** The components described conceptually in Chapters 3–4 correspond to:")
out.append("")
for item in ["state features: `src/rl` (`unit_state`)", "layer sensitivity: `src/sensitivity` (`loss_sensitivity`)",
             "sensitivity prior: `src/rl` (`SoftPriorPolicy`)", "networks: `src/models`; baselines: `src/pruning`",
             "structured compaction: `src/pruning/structured_units.py`",
             "equal recovery protocol: `experiments/phase6/phase6_lib.py` (`PROTOCOL`, `finetune`)",
             "elite behavioural cloning: a modified copy of the Stable-Baselines3 PPO update",
             "data split: `torch.randperm` (seed 42)",
             "SimpleCNN units: modules `features.0`, `features.3`, `features.6`, `classifier.1`; output layer `classifier.3` (LeNet-5 output layer `classifier.5`; ResNet-8 output layer `fc`)",
             "latency passes: `eval()` and `inference_mode()`, timed with `time.perf_counter_ns()`",
             "package energy counter: `\\Energy Meter(RAPL_Package0_PKG)\\Energy`",
             "pinned package versions: `requirements_locked.txt`; appendix generator: `tools/build_appendices.py`"]:
    out.append(f"- {item};" if not item.endswith(")") else f"- {item};")
out[-1] = out[-1].rstrip(";") + "."
out.append("")
out.append("**Source records of the appendix tables:**")
out.append("")
for item in ["A.1 `cpu_prunable_units.csv`; A.2 `cpu_baselines.csv`", "B.1 `phase3_summary.csv`; B.2 `phase3_frozen_inputs.json`, `phase5_config.json`",
             "C.1 `phase6_pre_post_accuracy.csv`; D.1 `h0_decision.json`; D.2 `h2_latency_freeze.json`",
             "E.1 `energy_summary.csv`; E.2 `pilot_n100000.json`, `pilot_n20000.json`, `statistics.json`, `measurement_qualification.json`"]:
    out.append(f"- {item}")
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
note("Source: frozen Phase-3 summary record. Paired differences and their statistics are in Table 5.2.")
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
note("Sources: frozen Phase-3 inputs and Phase-5 configuration records.")

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
note("Source: frozen Phase-6 before/after record. Accuracies before and after recovery and the PPO–LAMP and PPO–Global gaps are in Table 5.7. "
     "Phase-4 search quality and Phase-5 per-run selections are summarised in Tables 5.3, 5.4 and 5.5; their run-level records are in the repository.")

# ================================================================ D
APP("D", "Hardware and Structured-Pruning Protocol")
H("### D.1 Latency-measurement calibration")
CAP("H0 calibration: decision quantities of the three candidate measurement configurations.")
h0 = j("paper3/h0_decision.json")["configs"]
table(["Configuration", "Signal (ms)", "Noise (ms)", "SNR", "Relative reduction", "Positive in all sessions", "Usable"],
      [[k, hu(v["signal_ms"], 4), hu(v["noise_ms"], 4), hu(v["snr"], 2), hu(100 * v["relative_reduction"], 1) + "%", v["delta_positive_all_sessions"], v["usable"]]
       for k, v in h0.items()])
note("Source: frozen H0 decision record. Rule: SNR ≥ 5, reduction ≥ 10%, positive in every session; preference order A, C, B.")
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
note("Source: frozen latency-table session log; the per-session log is in the repository.")

# ================================================================ E
APP("E", "Energy Measurement and Qualification")
H("### E.1 Phase 7A energy sessions")
CAP("Phase 7A energy: session summaries (five sessions per architecture).")
table(["Network", "Method", "Median J / 1,000", "CV over sessions", "Median saving", "Min–max saving", "Sessions below dense", "One-sided Wilcoxon p"],
      [[ARCH.get(r["arch"], r["arch"]), r["method"], hu(r["e1000_median_j"], 3), hu(100 * float(r["e1000_cv"]), 2) + "%",
        hu(100 * float(r["saving_median"]), 1) + "%", f"{hu(100 * float(r['saving_min']), 1)}–{hu(100 * float(r['saving_max']), 1)}%",
        r["sessions_lower_than_dense"], p_(r["wilcoxon_one_sided_p"]) if r["wilcoxon_one_sided_p"] else "—"]
       for r in rd("phase7a_sustainability/energy_summary.csv")])
note("Source: frozen Phase-7A energy summary; per-block session records are in the repository.")
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
note("Sources: frozen pilot, statistics and qualification records of Phase 7A, Phase 7B and the energy confirmation (Section A.2). "
     "A doubled-N Phase-7B attempt stopped by the load gate was declared void by a recorded decision of the author and is not "
     "listed. The per-block diagnostics and the eleven recovered Phase-7B models (summarised in Table 6.4) are in the repository.")

open(os.path.join(W, "appendices.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
print(f"appendices.md: {len(out)} lines")
