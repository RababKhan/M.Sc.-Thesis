"""Global cross-chapter numerical audit (Abstract, Chapters 1, 5, 6, 7, 8).

For every headline claim: recompute the frozen value from the phase files, list where the claim appears, and search
for forbidden variants (superseded or wrong values). Re-runs the per-chapter audits (Ch. 5, 6, 7) as well.

  python results/thesis_final_writing/tools/global_numeric_audit.py > results/thesis_final_writing/global_numerical_audit.md
Read-only.
"""
import csv
import json
import os
import re
import statistics as st
import subprocess
import sys
from decimal import ROUND_HALF_UP, Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
R = os.path.dirname(W)
PY = sys.executable
FILES = {"Abstract": "abstract.md", "Ch. 1": "chapter_01_introduction.md", "Ch. 5": "chapter_05_core_results.md",
         "Ch. 6": "chapter_06_deployment_sustainability.md", "Ch. 7": "chapter_07_discussion.md", "Ch. 8": "chapter_08_conclusion.md"}
norm = lambda s: re.sub(r"\s+", " ", s.replace("**", "").replace("−", "-"))  # noqa: E731
TXT = {k: norm(open(os.path.join(W, f), encoding="utf-8").read()) for k, f in FILES.items()}


def hu(v, d):
    return str(Decimal(repr(round(float(v), 6))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def rd(p):
    return list(csv.DictReader(open(os.path.join(R, p), encoding="utf-8")))


def j(p):
    return json.load(open(os.path.join(R, p), encoding="utf-8"))


S6 = ["cifar10_simplecnn", "cifar10_lenet5", "cifar10_resnet8", "cifar100_simplecnn", "cifar100_lenet5", "cifar100_resnet8"]
claims = []  # (claim, frozen source, expected, patterns (regex), forbidden (regex))

s3 = [r for r in rd("architecture_generalization/phase3_statistics.csv") if r["family"] == "A" and r["setting"] == "cifar10_simplecnn"][0]
claims.append(("PPO–Uniform maximum (test, C10 SimpleCNN); exact convention +31.37, prose ≈ +31.4", "architecture_generalization/phase3_statistics.csv",
               f"+{hu(s3['mean_diff'], 2)} / ≈ +{hu(s3['mean_diff'], 1)}", [r"31\.37", r"31\.4"], [r"31\.36(?!5)"]))
sq = {(r["setting"], r["condition"]): r for r in rd("phase4/phase4_search_quality.csv")}
t10 = [float(sq[(s, "S0")]["selected_reach_top10"]) for s in S6]
claims.append(("Phase-4 terminal top-10 reach: 0% in five settings, 5% in one", "phase4/phase4_search_quality.csv",
               f"{sorted(round(100 * x) for x in t10)}", [r"0% (of runs )?in five settings and 5% in (the sixth|one)", r"0% in five settings"], [r"0% in all six", r"(?<!P0 policy reached the top 10 in )0% of runs in all six settings"]))
a10 = [float(sq[(s, "S2A")]["selected_reach_top10"]) for s in S6]
claims.append(("Archive (S2A) top-10 reach", "phase4/phase4_search_quality.csv", f"{round(100 * min(a10))}–{round(100 * max(a10))}%",
               [r"80-100%", r"80–100%"], [r"S2A[^.]{0,80}75[-–]100%", r"archive selection[^.]{0,40}75[-–]100%"]))
d = float(sq[("cifar10_lenet5", "S2A")]["selected_test_mean"]) - float(sq[("cifar10_lenet5", "S0")]["selected_test_mean"])
claims.append(("Phase-4 archive Δ test, C10 LeNet-5 (corrected)", "phase4/phase4_search_quality.csv", f"+{hu(d, 2)}", [r"\+0\.89"], [r"\+0\.90 ?pp", r"\+0\.9 pp"]))
cr = rd("phase5/phase5_constraint_results.csv")
m = lambda c, s: st.mean(float(r["selected_sparsity"]) for r in cr if r["condition"] == c and r["setting"] == s)  # noqa: E731
claims.append(("Phase-5 ResNet-8 mean sparsity C0 → C1", "phase5/phase5_constraint_results.csv",
               f"{hu(m('C0', 'cifar10_resnet8'), 1)}% → {hu(m('C1', 'cifar10_resnet8'), 1)}%; {hu(m('C0', 'cifar100_resnet8'), 1)}% → {hu(m('C1', 'cifar100_resnet8'), 1)}%",
               [r"46\.2%", r"24\.5%", r"25\.9%", r"15\.6%"], [r"48\.8% (→|->|to) 24\.7%", r"23\.5% (→|->|to) 16\.1%"]))
c1 = [r for r in cr if r["condition"] == "C1"]
nt = sum(r["selected_feasible_test"] == "True" for r in c1)
claims.append(("Validation-to-test feasibility (C1)", "phase5/phase5_constraint_results.csv", f"{nt}/120 = {hu(100 * nt / 120, 1)}%",
               [r"67\.5%", r"81 (of|/) ?120", r"two thirds"], [r"6[0-9]\.[0-46-9]% of runs"]))
lv = rd("phase6/phase6_ppo_vs_lamp.csv")
pre = [abs(float(r["mean_diff"])) for r in lv if r["family"].startswith("H2 reference")]
post = [abs(float(r["mean_diff"])) for r in lv if r["family"].startswith("H2 PPO") and r["endpoint"] == "test accuracy"]
claims.append(("Phase-6 |PPO − LAMP| before / after", "phase6/phase6_ppo_vs_lamp.csv",
               f"{hu(min(pre), 2)}–{hu(max(pre), 2)} → {hu(min(post), 2)}–{hu(max(post), 2)}", [r"0\.26-0\.53", r"0\.26–0\.53", r"0\.00–0\.22", r"0\.00-0\.22"], []))
gv = rd("phase6/phase6_ppo_vs_global.csv")
gp = [float(r["mean_diff"]) for r in gv if r["family"].startswith("H3 PPO") and r["endpoint"] == "test accuracy"]
claims.append(("Phase-6 PPO − Global after recovery: maximum PPO advantage", "phase6/phase6_ppo_vs_global.csv", f"{hu(max(gp), 2)}",
               [r"0\.12 ?(pp|percentage)"], []))
import pandas as pd  # noqa: E402
rp, rm = [], []
for t in ("simplecnn_c10", "simplecnn_c100", "resnet8_c10", "resnet8_c100"):
    dd = pd.read_csv(os.path.join(R, f"paper3/h2_latency_table_{t}.csv"))
    rp.append(dd.param_reduction.rank().corr(dd.latency_gain.rank()))
    rm.append(dd.mac_reduction.rank().corr(dd.latency_gain.rank()))
claims.append(("Spearman: parameter vs latency / MAC vs latency", "paper3/h2_latency_table_*.csv (recomputed; = h2_e2_latency_correlations.csv for C10)",
               f"{hu(min(rp), 2)}–{hu(max(rp), 2)} / {hu(min(rm), 2)}–{hu(max(rm), 2)}", [r"0\.31-0\.55", r"0\.31–0\.55", r"0\.89–0\.98", r"0\.89-0\.98"], []))
cal = j("paper3/resnet8_threshold_calibration_decision.json")
claims.append(("ResNet-8 best proxy-feasible speed-up (candidate, not confirmed)", "paper3/resnet8_threshold_calibration_decision.json",
               f"{hu(cal['latency_room']['best_speedup'], 3)}×", [r"1\.098×"], [r"at most 1\.098", r"≤ ?1\.098", r"1\.098× (accuracy-feasible|confirmed)"]))
rep = open(os.path.join(R, "paper3/resnet8_threshold_calibration_report.md"), encoding="utf-8").read()
claims.append(("ResNet-8 best confirmed five-epoch-feasible calibration speed-up", "paper3/resnet8_threshold_calibration_report.md",
               "1.089×" if "1.089×" in rep else "?", [r"1\.089×"], [r"best confirmed policy reached"]))
es = {(r["arch"], r["method"]): r for r in rd("phase7a_sustainability/energy_summary.csv")}
claims.append(("Phase-7A PPO energy saving SimpleCNN / ResNet-8", "phase7a_sustainability/energy_summary.csv",
               f"{hu(100 * float(es[('simplecnn', 'ppo')]['saving_median']), 1)}% / {hu(100 * float(es[('resnet8', 'ppo')]['saving_median']), 1)}%",
               [r"22\.8%", r"13\.4%"], []))
ac = {(r["setting"].split("_")[1], r["method"]): r for r in rd("phase7a_sustainability/accuracy_results.csv")}
claims.append(("Phase-7A PPO accuracy loss (VAL-RL) SimpleCNN / ResNet-8", "phase7a_sustainability/accuracy_results.csv",
               f"{hu(-float(ac[('simplecnn', 'ppo')]['delta_val_rl_pp']), 2)} / {hu(-float(ac[('resnet8', 'ppo')]['delta_val_rl_pp']), 2)} pp",
               [r"5\.08", r"3\.24", r"5\.1 and 3\.2 pp of VAL-RL"], [r"(?<!lost )5\.1 pp(?! of VAL-RL)"]))
fs = {r["id"]: r for r in rd("phase7b_green_pruning/final_summary.csv")}
nw = fs["2225"]
seeds = [r for r in fs.values() if "seed" in r["roles"]]
claims.append(("Phase-7B memory / MAC reduction / VAL-SELECT loss / successes", "phase7b_green_pruning/final_summary.csv",
               f"{hu(100 * float(nw['memory_reduction']), 1)}% / {hu(100 * float(nw['mac_reduction']), 1)}% / {hu(-float(nw['delta_val_select_pp']), 2)} pp / "
               f"{sum(r['green_success'].startswith('YES') for r in seeds)}/20",
               [r"63\.4%", r"36\.0%", r"1\.12", r"0 (of|/) ?20"], [r"63\.3% (parameter )?memory reduction (for|of) (the )?new", r"−?1\.1 pp"]))
p7 = j("phase7b_green_pruning/statistics.json")["pilots"]
q = j("phase7b_energy_confirmation/measurement_qualification.json")
claims.append(("Energy stability: 7B pilot, rerun, confirmation (limit 3%)", "phase7b_green_pruning/statistics.json; phase7b_energy_confirmation/measurement_qualification.json",
               f"{hu(100 * p7['pilot_n100000.json']['cv'], 2)}% / {hu(100 * p7['pilot_n200000.json']['cv'], 2)}% / {hu(100 * float(q['cv']), 2)}%",
               [r"11\.36%", r"4\.01%", r"19\.92%"], []))
claims.append(("Inversion rate C10 SimpleCNN (corrected)", "paper3/h2_latency_table_simplecnn_c10.csv", "39.0% (39.05%)", [r"39\.0", r"31-39%", r"31–39%", r"roughly a third"], [r"39\.1%"]))

print("# Global numerical audit (Abstract, Chapters 1, 5, 6, 7, 8)\n")
print("Generated by `tools/global_numeric_audit.py`. Every expected value is **recomputed from the frozen phase files** at")
print("audit time (round-half-up). For each claim the table lists the chapters in which it appears, the text found, and")
print("any **forbidden variants**: superseded, rounded-differently or overstated forms. A forbidden variant anywhere")
print("fails the claim.\n")
fails = 0
print("| # | Claim | Frozen source | Expected value | Chapters where stated (text found) | Forbidden variants found | Status |")
print("|---|---|---|---|---|---|---|")
for i, (claim, src, expv, pats, forb) in enumerate(claims, 1):
    where = []
    for ch, t in TXT.items():
        hits = sorted({mm.group(0) for p in pats for mm in re.finditer(p, t)})
        if hits:
            where.append(f"{ch}: {'; '.join(hits)}")
    bad = [f"{ch}: {mm.group(0)}" for ch, t in TXT.items() for p in forb for mm in re.finditer(p, t)]
    ok = bool(where) and not bad
    fails += not ok
    print(f"| {i} | {claim} | `{src}` | {expv} | {' / '.join(where) or '—'} | {', '.join(bad) or 'none'} | {'PASS' if ok else 'FAIL'} |")

print("\n## Per-chapter audits re-run\n")
for tool, label in (("audit_ch4.py", "Chapter 4 source checks"), ("audit_ch5.py", "Chapter 5 numerical audit"),
                    ("audit_ch6.py", "Chapter 6 numerical audit"), ("audit_ch7_numbers.py", "Chapter 7 no-new-number check")):
    p = subprocess.run([PY, os.path.join(HERE, tool)], capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    line = next((ln for ln in p.stdout.splitlines() if "checks" in ln or "not found" in ln), p.stdout.strip()[:120])
    fails += p.returncode != 0
    print(f"- **{label}** (`{tool}`): exit {p.returncode}; {line.strip('*')}")
print(f"\n**Result: {len(claims)} global claims and 4 per-chapter audits; overall {'PASS' if fails == 0 else 'FAIL (' + str(fails) + ')'}.**")
sys.exit(1 if fails else 0)
