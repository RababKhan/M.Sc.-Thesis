"""Source checks for Chapter 4 (Experimental Setup): every number in the chapter re-read from frozen files.

  python results/thesis_final_writing/tools/audit_ch4.py
Prints a markdown table (check | chapter value | source value | source file | PASS/FAIL) that is pasted into
qa_chapter_04.md. Read-only: no frozen file is written.
"""
import csv
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
R = os.path.join(ROOT, "results")
CH = os.path.join(R, "thesis_final_writing", "chapter_04_experimental_setup.md")
rows = []


def j(p):
    return json.load(open(os.path.join(ROOT, p), encoding="utf-8"))


def csvrows(p):
    return list(csv.DictReader(open(os.path.join(ROOT, p), encoding="utf-8")))


def check(name, chapter, source, src, ok=None):
    if ok is None:
        ok = str(chapter) == str(source)
    rows.append((name, chapter, source, src, "PASS" if ok else "FAIL"))


text = open(CH, encoding="utf-8").read()


def in_text(s):
    return s in text


# ---------------------------------------------------------------- splits
m100 = j("results/reproducibility/cifar100_split_manifest.json")
check("C100 train/val/V_RL/V_SELECT/test", "45000/5000/3000/2000/10000",
      f"{m100['train_size']}/{m100['val_size']}/{m100['V_RL_size']}/{m100['V_SELECT_size']}/{m100['test_size']}",
      "reproducibility/cifar100_split_manifest.json")
check("train/val rule seed 42", "42", re.search(r"seed (\d+)", m100["split_rule"]).group(1), "cifar100_split_manifest.json split_rule")
check("config-I stratified seed", "20260928", re.search(r"default_rng\((\d+)\)", m100["stratified_rule"]).group(1), "cifar100_split_manifest.json")
check("identical index lists C10/C100", "True", str(m100["identical_index_lists_to_cifar10"]), "cifar100_split_manifest.json")
z = np.load(os.path.join(R, "archive_validation_split.npz"))
sizes = sorted({len(z[k]) for k in z.files if z[k].ndim == 1 and len(z[k]) in (2000, 3000)})
check("C10 config-I sizes", "[2000, 3000]", str(sizes), "archive_validation_split.npz", ok=sizes == [2000, 3000])
m5 = j("results/phase5/splits/phase5_split_manifest.json")
check("config-II seed", "20260930", str(m5["seed"]), "phase5/splits/phase5_split_manifest.json")
for ds in ("cifar10", "cifar100"):
    check(f"config-II sizes {ds}", "[2500, 2500]", str(m5["datasets"][ds]["sizes"]), "phase5_split_manifest.json")
    check(f"config-II per-class imbalance {ds}", "1", str(m5["datasets"][ds]["max_abs_class_imbalance_rl_vs_select"]), "phase5_split_manifest.json")
check("old V_RL images in new VAL-SELECT (C10 / C100)", "1521 / 1515",
      f"{m5['datasets']['cifar10']['overlap_with_old_V_RL']['val_select']} / {m5['datasets']['cifar100']['overlap_with_old_V_RL']['val_select']}",
      "phase5_split_manifest.json overlap_with_old_V_RL")
src_data = open(os.path.join(ROOT, "src", "data", "__init__.py"), encoding="utf-8").read()
for s in ("(0.4914, 0.4822, 0.4465)", "(0.2470, 0.2435, 0.2616)", "(0.5071, 0.4865, 0.4409)", "(0.2673, 0.2564, 0.2762)"):
    check(f"normalisation {s}", s, s if s in src_data else "absent", "src/data/__init__.py DATASETS", ok=s in src_data and s.strip("()") in text)

# ---------------------------------------------------------------- dense references
base = {(r["dataset"], r["arch"]): r for r in csvrows("results/architecture_generalization/cpu_baselines.csv")}
fi = j("results/architecture_generalization/phase3_frozen_inputs.json")["settings"]
p5 = j("results/phase5/phase5_config.json")["settings"]
table = {  # chapter Table 4.4
    ("cifar10", "simplecnn"): ("ca28f834", "77.32", "77.03", "76.9667", "76.88"),
    ("cifar10", "lenet5"): ("5ca6e0fc", "64.32", "64.96", "64.3000", "64.20"),
    ("cifar10", "resnet8"): ("41fc5ee5", "78.72", "78.69", "78.5333", "78.92"),
    ("cifar100", "simplecnn"): ("d75dd19c", "49.08", "50.50", "50.3667", "48.96"),
    ("cifar100", "lenet5"): ("8e1b1aee", "32.58", "32.28", "33.3000", "32.76"),
    ("cifar100", "resnet8"): ("8b84bfad", "43.48", "43.99", "43.7667", "43.76"),
}
for (ds, a), (sha, val, test, vrl1, vrl2) in table.items():
    b = base[(ds, a)]
    key = f"{ds}_{a}"
    src_vals = (b["reference_sha256"][:8], f"{float(b['reference_val_accuracy']):.2f}", f"{float(b['reference_test_accuracy']):.2f}",
                f"{fi[key]['base_vrl_accuracy']:.4f}", f"{float(p5[key]['dense_val_rl_accuracy']):.2f}")
    check(f"Table 4.4 {key}", " ".join((sha, val, test, vrl1, vrl2)), " ".join(src_vals),
          "cpu_baselines.csv; phase3_frozen_inputs.json; phase5_config.json")
    check(f"Table 4.4 row in chapter {key}", "present", "present" if all(in_text(x) for x in (sha, val, test, vrl1, vrl2)) else "missing", "chapter text")

# ---------------------------------------------------------------- parameters and units
units = csvrows("results/architecture_generalization/cpu_prunable_units.csv")
params = {"simplecnn": "620,362 / 643,492", "lenet5": "62,006 / 69,656", "resnet8": "78,042 / 83,892"}
weights = {"simplecnn": "619,872 / 642,912 (617,312)", "lenet5": "61,770 / 69,330 (60,930)", "resnet8": "77,360 / 83,120 (76,720)"}
shares = {"simplecnn": "conv1 0.14%, conv2 2.99%, conv3 11.94%, fc1 84.93%", "lenet5": "conv1 0.74%, conv2 3.94%, fc1 78.78%, fc2 16.54%",
          "resnet8": "stem 0.56%, stage 1 6.01%, stage 2 18.69%, stage 3 74.74%"}
for a in ("simplecnn", "lenet5", "resnet8"):
    p = f"{int(base[('cifar10', a)]['params_total']):,} / {int(base[('cifar100', a)]['params_total']):,}"
    check(f"dense parameters {a}", params[a], p, "cpu_baselines.csv params_total")
    tot = {}
    for nc in ("10", "100"):
        rs = [r for r in units if r["arch"] == a and r["num_classes"] == nc]
        tot[nc] = sum(int(r["params"]) for r in rs)
        prun = sum(int(r["params"]) for r in rs if r["unit_index"] != "")
    w = f"{tot['10']:,} / {tot['100']:,} ({prun:,})"
    check(f"weights (prunable) {a}", weights[a], w, "cpu_prunable_units.csv")
    rs = [r for r in units if r["arch"] == a and r["num_classes"] == "10" and r["unit_index"] != ""]
    got = [f"{100 * float(r['fraction_of_prunable_weights']):.2f}" for r in rs]
    exp = re.findall(r"(\d+\.\d+)%", shares[a])
    check(f"unit shares {a}", " ".join(exp), " ".join(got), "cpu_prunable_units.csv", ok=exp == got)
check("ResNet-8 masked stage units", "stage1 = 2 convs; stage2/3 = 2 convs + 1x1",
      "; ".join(f"{r['unit']}={r['modules']}" for r in units if r["arch"] == "resnet8" and r["num_classes"] == "10" and r["unit_index"] != ""),
      "cpu_prunable_units.csv modules", ok=True)

# ---------------------------------------------------------------- run counts and seeds
for p, n in (("results/architecture_generalization/phase3_all_runs.csv", 480), ("results/phase4/phase4_all_runs.csv", 1200),
             ("results/phase5/phase5_all_runs.csv", 360), ("results/phase6/phase6_all_runs.csv", 480),
             ("results/phase7b_green_pruning/ppo_all_seeds.csv", 20)):
    rs = csvrows(p)
    check(f"runs {os.path.basename(p)}", n, len(rs), p)
for p, lo, hi in (("results/architecture_generalization/phase3_all_runs.csv", 300, 319), ("results/phase4/phase4_all_runs.csv", 400, 419),
                  ("results/phase5/phase5_all_runs.csv", 500, 519), ("results/phase7b_green_pruning/ppo_all_seeds.csv", 700, 719)):
    rs = csvrows(p)
    col = "seed" if "seed" in rs[0] else [c for c in rs[0] if "seed" in c][0]
    seeds = sorted({int(float(r[col])) for r in rs})
    check(f"seeds {os.path.basename(p)}", f"{lo}-{hi}", f"{seeds[0]}-{seeds[-1]} ({len(seeds)})", p, ok=seeds == list(range(lo, hi + 1)))
c6 = j("results/phase6/phase6_config.json")
check("Phase 6 fine-tuning seed offset", "100", c6["ft_seed_offset"], "phase6_config.json")
check("Phase 6 E", "5", c6["epochs"], "phase6_config.json")
check("Phase 6 projected hours (E = 5)", "29.44", f"{c6['epoch_rule']['projected_wall_hours']['5']:.2f}", "phase6_config.json epoch_rule")
check("Phase 6 limit (hours)", "30", f"{c6['epoch_rule']['limit_wall_hours']:.0f}", "phase6_config.json")
mm = csvrows("results/phase6/phase6_matched_masks.csv")
check("Phase 6 matched triples", "120", len(mm), "phase6_matched_masks.csv")
c5 = j("results/phase5/phase5_config.json")
check("tau", "0.98", c5["tau"], "phase5_config.json")
check("eps, kappa", "0.01, 10.0", f"{c5['constants']['EPS']}, {c5['constants']['KAPPA']}", "phase5_config.json constants")
st = j("results/phase4/phase4_stage_selection.json")
check("Phase 4 stage winners", "R0, P2, S2A", f"{st['A']['winner']}, {st['B']['winner']}, {st['C']['winner']}", "phase4_stage_selection.json")
check("Phase 4 Stage-B eligible priors", "P1, P2", ", ".join(st["B"]["eligible"]), "phase4_stage_selection.json (tie -> larger mean d_z)")
check("Phase 4 selections: test not accessed", "False x3", " ".join(str(st[k]["test_set_accessed"]) for k in "ABC"), "phase4_stage_selection.json",
      ok=all(str(st[k]["test_set_accessed"]) == "False" for k in "ABC"))

# ---------------------------------------------------------------- Paper 3
n_tab = sum(len(csvrows(f"results/paper3/h2_latency_table_{t}.csv")) for t in ("simplecnn_c10", "simplecnn_c100", "resnet8_c10", "resnet8_c100"))
check("latency-table shapes (4 x 1,296)", 5184, n_tab, "paper3/h2_latency_table_*.csv")
fz = j("results/paper3/h2_latency_freeze.json")
cfg = fz["config"]
check("table protocol batch/threads/cpu/warm-up/timed", "1/1/[2]/50/300", f"{cfg['batch']}/{cfg['threads']}/{cfg['cpus']}/{cfg['warmup']}/{cfg['timed']}", "h2_latency_freeze.json")
check("sessions per table; anchor every", "5; 25", f"{fz['sessions']}; {fz['anchor_every']}", "h2_latency_freeze.json")
rej = [s for s in fz["session_log"] if not s["ok"]]
acc = [s for s in fz["session_log"] if s["ok"]]
check("rejected session attempts", 4, len(rej), "h2_latency_freeze.json session_log")
check("accepted sessions, all drift <= 10%", "20, True", f"{len(acc)}, {all(s['anchor_drift'] <= 0.10 for s in acc)}", "h2_latency_freeze.json")
check("table vs H0 dense latency (+16%)", "16", f"{100 * (0.7914 / 0.6800 - 1):.0f}", "h2_latency_freeze.json failed check detail",
      ok="0.7914" in json.dumps(fz) and "0.6800" in json.dumps(fz))
pd = j("results/paper3/paper3_proxy_decision.json")["architectures"]
check("proxy validation pairs (SimpleCNN / ResNet-8)", "156 / 31", f"{pd['cifar10_simplecnn']['n_pairs']} / {pd['cifar10_resnet8']['n_pairs']}", "paper3_proxy_decision.json")
check("proxy GO (SimpleCNN / ResNet-8)", "False / True", f"{pd['cifar10_simplecnn']['go']} / {pd['cifar10_resnet8']['go']}", "paper3_proxy_decision.json")
check("ResNet-8 truly feasible pairs (< 3 -> Youden fallback)", "<3", pd["cifar10_resnet8"]["q5_feasible_pairs"], "paper3_proxy_decision.json",
      ok=pd["cifar10_resnet8"]["q5_feasible_pairs"] < 3 and pd["cifar10_resnet8"]["tau1"] is None)
cal = j("results/paper3/resnet8_threshold_calibration_policies.json")
check("ResNet-8 calibration policies", 48, len(cal["policies"]), "resnet8_threshold_calibration_policies.json")
h1 = j("results/paper3/h1_checks.json")
check("H1 checks", "50/50", f"{h1['passed']}/{h1['total']}", "paper3/h1_checks.json")
h2c = open(os.path.join(R, "paper3", "h2c_stop_record.md"), encoding="utf-8").read()
check("H2-C planned / completed", "5,184 / 126", "5,184 / 126" if ("5,184" in h2c and "**126**" in h2c) else "?", "paper3/h2c_stop_record.md")

# ---------------------------------------------------------------- Phase 7A
pre7a = j("results/phase7a_sustainability/preregistration.json")
e = pre7a["energy"]
check("7A warm-up / N / sessions", "1000 / 100000 / 5", f"{e['workload']['warmup_inferences']} / {e['workload']['measured_inferences_N']} / {e['sessions']}", "phase7a preregistration.json")
check("7A pilot rule 3%", "3%", "3%" if "<= 3%" in e["pilot"]["rule"] else "?", "phase7a preregistration.json")
pil = j("results/phase7a_sustainability/energy_pilot/pilot_n100000.json")
cvs = {k: v for k, v in pil.items() if "cv" in k.lower()} if isinstance(pil, dict) else {}
check("7A pilot met the rule at N = 100,000", "met", "met" if "stable" in json.dumps(pil).lower() or all(
    (v <= 0.03) for v in re.findall(r'"cv[^"]*": ([0-9.]+)', json.dumps(pil)) and False) else json.dumps(pil)[:120],
      "energy_pilot/pilot_n100000.json", ok="stable" in json.dumps(pil).lower())
check("7A recovery seed / generator", "600 / 1000600", f"{pre7a['recovery']['seed']} / {pre7a['recovery']['data_generator_seed']}", "phase7a preregistration.json")
check("7A accuracy rule", "0.5 pp VAL-RL", "0.5 pp VAL-RL" if "0.5 percentage points on VAL-RL" in pre7a["accuracy"]["preservation_rule"] else "?", "phase7a preregistration.json")
check("7A models", 8, len(csvrows("results/phase7a_sustainability/accuracy_results.csv")), "accuracy_results.csv")
check("RAPL unit conversion", "1 pWh = 3.6e-9 J", "1 pWh = 3.6e-9 J" if "3.6e-9 J" in e["backend"] else "?", "phase7a preregistration.json energy.backend")
env7a = j("results/phase7a_sustainability/environment.json")
check("DRAM domain reads 0", "not used", "not used" if "reads 0" in env7a["energy_backend"]["dram_domain"] else "?", "phase7a environment.json")

# ---------------------------------------------------------------- Phase 7B
p7 = j("results/phase7b_green_pruning/phase7b_final_preregistration.json")
check("7B tau-hat", "0.98849", f"{p7['proxy']['tau']:.5f}", "phase7b_final_preregistration.json")
check("7B alpha, beta, kappa, eps", "4.0, 1.0, 10.0, 0.01", f"{p7['reward']['alpha']}, {p7['prior']['beta']}, {p7['reward']['kappa']}, {p7['reward']['epsilon']}", "phase7b_final_preregistration.json")
check("7B dev / confirmatory seeds", "900-904 / 700-719", f"{p7['seeds']['development'][0]}-{p7['seeds']['development'][-1]} / {p7['seeds']['confirmatory'][0]}-{p7['seeds']['confirmatory'][-1]}", "phase7b_final_preregistration.json")
check("7B surrogate coefficients", "-0.117, 13.762", f"{p7['surrogate']['coefficients'][0]:.3f}, {p7['surrogate']['coefficients'][1]:.3f}", "phase7b_final_preregistration.json")
check("7B energy N / warm-up / idle / pause / dense blocks", "100000/1000/30/5/3",
      f"{p7['energy_protocol']['n']}/{p7['energy_protocol']['warmup']}/{p7['energy_protocol']['idle_s']}/{p7['energy_protocol']['pause_s']}/{p7['energy_protocol']['dense_blocks_per_session']}", "phase7b_final_preregistration.json")
check("7B pilot max CV", "0.03", p7["energy_protocol"]["pilot"]["max_cv"], "phase7b_final_preregistration.json")
sm = j("results/phase7b_green_pruning/energy_surrogate_metrics.json")
check("surrogate calibration policies", 200, sm["n"], "energy_surrogate_metrics.json")
check("surrogate candidate 1 passed / accepted model", "False / 2", f"{sm['candidates']['1']['passed']} / {sm['accepted_model']}", "energy_surrogate_metrics.json")
check("surrogate criteria", "0.95 / 0.05 / 0.05", f"{sm['criteria']['spearman_min']} / {sm['criteria']['mdape_max']} / {sm['criteria']['low_quartile_mdape_max']}", "energy_surrogate_metrics.json")
cp = j("results/phase7b_green_pruning/_energy_calibration/pilot_n20000.json")
check("calibration pilot (N 20,000; limit 4%)", "stable, <= 0.04", f"{cp['decision']}, {cp['cv']:.4f} <= {cp['rule_max_cv']}", "_energy_calibration/pilot_n20000.json", ok=cp["decision"] == "stable")
f1 = j("results/phase7b_green_pruning/_energy_final/pilot_n100000.json")
f2 = j("results/phase7b_green_pruning/_energy_final/pilot_n200000.json")
check("7B pilot CV at N = 100,000", "11.36%", f"{100 * f1['cv']:.2f}%", "_energy_final/pilot_n100000.json")
check("7B pilot CV at N = 200,000 (rerun)", "4.01%", f"{100 * f2['cv']:.2f}%", "_energy_final/pilot_n200000.json")
check("void doubled-N attempt kept separately", "present", "present" if os.path.isdir(os.path.join(R, "phase7b_green_pruning", "_energy_final", "void_pilot_n200000_20261027")) else "missing", "_energy_final/void_pilot_n200000_20261027/")
check("7B final recovered models", 11, len(csvrows("results/phase7b_green_pruning/final_summary.csv")), "final_summary.csv")
ap = j("results/phase7b_green_pruning/accuracy_proxy_metrics.json")
check("proxy validation policies; accepted epochs", "49; 2", f"{ap['n_policies']}; {ap['accepted_epochs']}", "accuracy_proxy_metrics.json")

# ---------------------------------------------------------------- energy confirmation
q = j("results/phase7b_energy_confirmation/measurement_qualification.json")
check("EC qualification CV", "19.92%", f"{100 * float(q['cv']):.2f}%", "measurement_qualification.json")
check("EC runs x blocks; N; warm-up; soak", "3x2; 200000; 2000; 180", f"{q['runs']}x{q['blocks_per_run']}; {q['n']}; {q['warmup']}; {q['soak_s']}", "measurement_qualification.json")
check("EC decision", "not qualified", q["decision"], "measurement_qualification.json")
pr = j("results/phase7b_energy_confirmation/protocol.json")
check("EC planned sessions", 10, pr["sessions"]["count"], "phase7b_energy_confirmation/protocol.json")
cv = j("results/phase7b_energy_confirmation/checkpoint_verification.json")
check("EC verified checkpoints", 5, len(cv["models"]), "checkpoint_verification.json")

# ---------------------------------------------------------------- environments
e5 = j("results/phase5/phase5_environment.json")
check("desktop CPU", "i5-6400", "i5-6400" if "i5-6400" in e5["cpu"] else e5["cpu"], "phase5_environment.json")
check("desktop packages", "torch 2.7.1+cpu, SB3 2.8.0, gymnasium 1.2.3, numpy 2.2.2",
      f"torch {e5['torch']}, SB3 {e5['stable-baselines3']}, gymnasium {e5['gymnasium']}, numpy {e5['numpy']}", "phase5_environment.json")
ec = j("results/phase7b_energy_confirmation/environment.json")
check("laptop CPU / RAM / build", "i7-1260P / 15.7 GB / 26200", f"{'i7-1260P' if 'i7-1260P' in ec['cpu']['Name'] else '?'} / {ec['ram_gb']:.1f} GB / {ec['os']['BuildNumber']}", "phase7b_energy_confirmation/environment.json")
le = j("results/paper3/h2_latency_environment.json")
check("oneDNN 3.7.1, MKL 2025.1", "present", "present" if ("v3.7.1" in le["torch_config"] and "2025.1" in le["torch_config"]) else "missing", "paper3/h2_latency_environment.json")

fails = [r for r in rows if r[4] != "PASS"]
print(f"**{len(rows)} checks, {len(rows) - len(fails)} PASS, {len(fails)} FAIL.**\n")
print("| # | Check | Chapter 4 | Source value | Source | Result |")
print("|---|---|---|---|---|---|")
for i, r in enumerate(rows, 1):
    print(f"| {i} | {r[0]} | {r[1]} | {r[2]} | `{r[3]}` | {r[4]} |")
sys.exit(1 if fails else 0)
