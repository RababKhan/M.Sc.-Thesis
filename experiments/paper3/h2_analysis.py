"""Paper 3, H2 analysis: merge the recovered landscapes, join the frozen latency tables, apply the pre-registered
tau rule, exhaustive Pareto fronts, correlations, integrity checks (pre-registration §3, §6). VAL-RL only.

  python experiments/paper3/h2_analysis.py
VAL-SELECT accuracies are merged into the landscape files for the H4 selection step and never used here.
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3_lib as L  # noqa: E402

R = L.phase_module("h2_recovery")
PAIR = {"cifar10_simplecnn": "simplecnn_c10", "cifar100_simplecnn": "simplecnn_c100",
        "cifar10_resnet8": "resnet8_c10", "cifar100_resnet8": "resnet8_c100"}
TAUS, TAU0, MIN_FEASIBLE, MIN_GAIN = [0.95, 0.97, 0.98, 0.99], 0.98, 20, 0.10


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def spearman(a, b):
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def discordant(sparsity, latency):
    """Pairs with different sparsity in which the sparser policy is slower: count and fraction."""
    s, lat = np.asarray(sparsity), np.asarray(latency)
    ds, dl = s[:, None] - s[None, :], lat[:, None] - lat[None, :]
    valid = np.triu(ds != 0, 1)
    bad = valid & (np.sign(ds) == np.sign(dl)) & (dl != 0)
    return int(bad.sum()), int(valid.sum())


def pareto(acc, lat):
    """Indices of policies not dominated in (higher accuracy, lower latency)."""
    order = np.lexsort((-acc, lat))                    # by latency, then accuracy descending
    front, best = [], -np.inf
    for i in order:
        if acc[i] > best:
            front.append(i)
            best = acc[i]
    return sorted(front, key=lambda i: lat[i])


def main():
    checks, feas_rows, corr_rows, freeze = [], [], [], {}

    def chk(name, ok, detail=""):
        checks.append({"check": name, "pass": None if ok is None else bool(ok), "detail": str(detail)})
        print(f"[{'N/A ' if ok is None else ('PASS' if ok else 'FAIL')}] {name}: {str(detail)[:160]}", flush=True)
    lat_freeze = json.load(open(os.path.join(L.RESULTS, "h2_latency_freeze.json"), encoding="utf-8"))
    prereg_time = subprocess.run(["git", "log", "-1", "--format=%cI", "4593684"], cwd=L.ROOT, capture_output=True, text=True).stdout.strip()
    first_result = []
    tables = {}
    for setting, pair in PAIR.items():
        recs = []
        for p in R.POLICIES:
            path = R.run_path(setting, p)
            recs.append(json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None)
        complete = [r for r in recs if r is not None and r["status"] == "complete"]
        chk(f"{setting}: 1,296 complete recovery records, one per policy, no duplicate", len(complete) == 1296 and
            len({tuple(r["policy"]) for r in complete}) == 1296, len(complete))
        if len(complete) != 1296:
            continue
        lt = pd.read_csv(os.path.join(L.RESULTS, f"h2_latency_table_{pair}.csv"))
        chk(f"{setting}: latency table unchanged since its freeze (sha256)", sha(os.path.join(L.RESULTS, f"h2_latency_table_{pair}.csv")) ==
            lat_freeze["tables"][pair]["sha256"], lat_freeze["tables"][pair]["sha256"][:16])
        df = pd.DataFrame([{"policy": str(r["policy"]), "widths_rec": str(r["widths"]), "params_rec": r["params"], "macs_rec": r["macs"],
                            "pre_val_rl": r["pre"]["val_rl"], "post_val_rl": r["post"]["val_rl"], "post_val_select": r["post"]["val_select"],
                            "pre_val_select": r["pre"]["val_select"], "loss_epoch1": r["train_loss"][0], "loss_epoch5": r["train_loss"][-1],
                            "final_state_sha256": r["final_state_sha256"], "seconds": r["seconds"]} for r in complete])
        t = lt.merge(df, on="policy", validate="one_to_one")
        ident = t[t.policy == "[0, 0, 0, 0]"].iloc[0]
        a_dense, a_ref = float(ident.post_val_rl), float(ident.pre_val_rl)
        t["q"] = t.post_val_rl / a_dense
        t["q_vs_reference"] = t.post_val_rl / a_ref
        t["recovery_gain"] = t.post_val_rl - t.pre_val_rl
        tables[setting] = t
        ref, ref_sha = L.reference(setting)
        p5 = json.load(open(os.path.join(L.ROOT, "results", "phase5", "phase5_config.json"), encoding="utf-8"))
        chk(f"{setting}: identity policy before recovery = frozen reference (VAL-RL accuracy equals the Phase-5 dense record)",
            a_ref == p5["settings"][setting]["dense_val_rl_accuracy"], f"{a_ref} vs {p5['settings'][setting]['dense_val_rl_accuracy']}")
        chk(f"{setting}: every record: 5 epochs, seed 600, Phase-6 protocol, 1 thread, no test use, same preregistration, frozen reference",
            all(r["epochs"] == 5 and r["recovery"]["seed"] == 600 and r["recovery"]["protocol"] == complete[0]["recovery"]["protocol"]
                and r["threads"] == 1 and r["test_use"] == "none" and r["reference_sha256"] == ref_sha
                and r["preregistration_sha256"] == complete[0]["preregistration_sha256"] for r in complete), ref_sha[:16])
        chk(f"{setting}: identical minibatch exposure: one data-stream hash for all 1,296 policies",
            len({r["data_stream_sha256"] for r in complete}) == 1, "")
        chk(f"{setting}: no topology change or regrowth: shape unchanged in every run; widths, parameters and MACs equal the latency table",
            all(r["shape_unchanged"] for r in complete) and bool((t.widths == t.widths_rec).all() and (t.params == t.params_rec).all()
                                                                  and (t.macs == t.macs_rec).all()), "")
        first_result.append(min(r["started_at"] for r in complete))
        out_csv = os.path.join(L.RESULTS, f"h2_recovered_landscape_{pair}.csv")
        t.drop(columns=["widths_rec", "params_rec", "macs_rec"]).to_csv(out_csv, index=False, float_format="%.9g")
        out_json = os.path.join(L.RESULTS, f"h2_recovered_landscape_{pair}.json")
        json.dump({"setting": setting, "a_dense_recovered_val_rl": a_dense, "a_reference_val_rl": a_ref, "reference_sha256": ref_sha,
                   "recovery": complete[0]["recovery"], "importance_rule": "L1 per filter/neuron on the dense model; stem group: sum of the two writing filters",
                   "code_sha256": complete[0]["code_sha256"], "git": complete[0]["git"], "records": complete},
                  open(out_json, "w", encoding="utf-8"), indent=0)
        freeze[pair] = {"csv": os.path.basename(out_csv), "csv_sha256": sha(out_csv), "json": os.path.basename(out_json), "json_sha256": sha(out_json)}
        # feasibility at every tau of the rule
        for tau in TAUS:
            f = t[t.q >= tau]
            feas_rows.append({"setting": setting, "tau": tau, "feasible": len(f), "fraction": len(f) / 1296,
                              "feasible_with_gain_ge_10pct": int((f.latency_gain >= MIN_GAIN).sum()),
                              "max_latency_gain": float(f.latency_gain.max()) if len(f) else float("nan"),
                              "max_speedup": float(f.speedup.max()) if len(f) else float("nan"),
                              "median_latency_gain": float(f.latency_gain.median()) if len(f) else float("nan"),
                              "max_param_reduction": float(f.param_reduction.max()) if len(f) else float("nan"),
                              "max_mac_reduction": float(f.mac_reduction.max()) if len(f) else float("nan"),
                              "passes": bool(len(f) >= MIN_FEASIBLE and (f.latency_gain >= MIN_GAIN).any())})
        # Pareto front
        front = pareto(t.post_val_rl.to_numpy(), t.latency_ms.to_numpy())
        pf = t.iloc[front][["policy", "widths", "post_val_rl", "q", "latency_ms", "speedup", "latency_gain", "param_reduction", "mac_reduction"]]
        pf.to_csv(os.path.join(L.RESULTS, f"h2_pareto_front_{pair}.csv"), index=False, float_format="%.6g")
        # correlations
        for subset, d in (("all", t), ("feasible_tau0.98", t[t.q >= TAU0])):
            if len(d) < 3:
                continue
            nb, nv = discordant(d.param_reduction, d.latency_ms)
            corr_rows.append({"setting": setting, "subset": subset, "n": len(d),
                              "spearman_param_reduction_vs_latency_gain": spearman(d.param_reduction, d.latency_gain),
                              "spearman_mac_reduction_vs_latency_gain": spearman(d.mac_reduction, d.latency_gain),
                              "pairs_sparser_but_slower": nb, "pairs_compared": nv, "fraction_sparser_but_slower": nb / nv if nv else float("nan")})
    feas = pd.DataFrame(feas_rows)
    feas.to_csv(os.path.join(L.RESULTS, "h2_feasibility_analysis.csv"), index=False, float_format="%.6g")
    pd.DataFrame(corr_rows).to_csv(os.path.join(L.RESULTS, "h2_correlation_analysis.csv"), index=False, float_format="%.6g")
    passes = {tau: bool(feas[feas.tau == tau].passes.all()) and len(feas[feas.tau == tau]) == 4 for tau in TAUS}
    tau_ok = passes[TAU0]
    fallback = next((tau for tau in (0.97, 0.95) if passes[tau]), None)
    literal = max([tau for tau in TAUS if passes[tau]], default=None)
    # regeneration and sanity determinism
    reps = [json.load(open(os.path.join(L.RESULTS, "_h2_sanity", f), encoding="utf-8")) for f in os.listdir(os.path.join(L.RESULTS, "_h2_sanity"))
            if f.startswith("repeat_cpu") and f.endswith(".json") and "cifar" not in f]
    rep = [x for r in reps for x in r]
    chk("deterministic regeneration: repeated recoveries on a different core type are bit-identical (weights and accuracy)",
        bool(rep) and all(x["identical_weights"] and x["identical_accuracy"] for x in rep), len(rep))
    regen = os.path.join(L.RESULTS, "_h2_regenerate.json")
    if os.path.exists(regen):
        rg = json.load(open(regen, encoding="utf-8"))
        chk("deterministic regeneration: landscape policies regenerated after the run are bit-identical to the landscape records",
            all(x["identical"] for x in rg), len(rg))
    chk("latency tables complete and frozen (H2-A integrity all passed)", all(c["pass"] for c in lat_freeze["checks"]), f"{len(lat_freeze['checks'])} checks")
    chk("latency environment = H0 (see H2-A checks: affinity, threads, torch, python)", all(c["pass"] for c in lat_freeze["checks"] if "metadata" in c["check"]), "")
    when = datetime.datetime.fromisoformat
    parts = os.path.join(L.RESULTS, "_h2_latency_parts")
    first_latency = min(os.path.getmtime(os.path.join(parts, f)) for f in os.listdir(parts) if f.endswith(".npz"))
    first_latency = datetime.datetime.fromtimestamp(first_latency).astimezone()
    chk("pre-registration amendment (4593684) committed before every H2 result (first latency session, first recovery run)",
        bool(first_result) and when(prereg_time) < first_latency and when(prereg_time) < min(when(x) for x in first_result),
        f"amendment {prereg_time}; first latency session written {first_latency.isoformat(timespec='seconds')}; "
        f"first recovery {min(first_result) if first_result else None}")
    chk("structured sensitivity vectors: not computed in H2 (variant not frozen; pre-registration §7, left to the PI)", None, "deferred")
    decision = {"tau_start": TAU0, "rule": {"min_feasible": MIN_FEASIBLE, "min_latency_gain": MIN_GAIN, "all_settings": True},
                "passes": passes, "tau_0.98_viable": tau_ok, "fallback_if_0.98_fails": fallback, "literal_phase5_rule_would_choose": literal}
    json.dump({"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "decision": decision, "landscapes": freeze,
               "checks": checks}, open(os.path.join(L.RESULTS, "h2_landscape_freeze.json"), "w", encoding="utf-8"), indent=1, default=str)
    n_ok = sum(c["pass"] is True for c in checks)
    n_na = sum(c["pass"] is None for c in checks)
    lines = ["# Paper 3 — H2 integrity checks", "", f"**{n_ok}/{len(checks) - n_na} passed** ({n_na} not applicable).", "",
             "| Status | Check | Detail |", "|---|---|---|"]
    lines += [f"| {'N/A' if c['pass'] is None else ('PASS' if c['pass'] else 'FAIL')} | {c['check']} | {c['detail'].replace('|', '/')[:200]} |" for c in checks]
    lines += ["", "Latency-table checks (H2-A): `h2_latency_freeze.json`.", ""]
    lines += [f"| {'PASS' if c['pass'] else 'FAIL'} | {c['check']} | {c['detail'].replace('|', '/')[:200]} |" for c in lat_freeze["checks"]]
    open(os.path.join(L.RESULTS, "h2_integrity_checks.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("tau decision:", decision)
    print(f"integrity {n_ok}/{len(checks) - n_na} ({n_na} N/A)")


if __name__ == "__main__":
    main()
