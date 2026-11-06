"""Phase 7B, Stage B close-out after the pre-registered energy-pilot STOP (decision of 2026-11-03, development_log.md).

  python experiments/phase7b/stopped_summary.py
The doubled-N pilot was unstable (CV 4.01% > 3%), so no hardware session was run and final_analysis.py (which needs
the 5 sessions) is NOT used. This script reports only what remains assessable under the frozen rules: VAL accuracy,
parameter memory and MACs of the 5-epoch recovered models, with the same integrity checks as final_analysis.py.
Energy and latency are NOT measured: the surrogate prediction (Paper-3 latency table -> accepted surrogate) is listed
for grid policies, labelled as a prediction, and never used for a GREEN label, a case or a comparison verdict.
Outputs: recovery_results.csv, memory_results.csv, latency_energy_raw.csv (pilot blocks only), final_summary.csv,
pareto_results.csv, statistics.json, figures/fig_accuracy_planes.png.
"""
import csv
import datetime
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7b_lib as L  # noqa: E402
import final_analysis as FA  # noqa: E402  (recovery/memory/integrity/Pareto code, unchanged)

import pandas as pd  # noqa: E402

NA = "not assessable (hardware validation stopped)"


def pilot_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join(FA.EF, "**", "pilot_n*_rep*.json"), recursive=True)):
        r = FA.jload(path)
        status = "void (interrupted by the CPU-load gate)" if "void_" in path else "pilot (dense only)"
        for b in r["blocks"]:
            rows.append({"file": os.path.relpath(path, L.RESULTS).replace("\\", "/"), "status": status, "n": r["n"],
                         "session": r["session"], "started_at": r["started_at"], "idle_pkg_w": r["idle"]["pkg_w"],
                         **{k: b[k] for k in ("model", "appearance", "at", "seconds", "pkg_j", "pp0_j", "e1000_j",
                                              "net_e1000_j", "pkg_w", "latency_median_ms", "latency_mean_ms",
                                              "latency_p95_ms", "latency_iqr_ms")}})
    return rows


def figure(S, method_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(FA.RC)
    os.makedirs(FA.FIG, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, (x, xl) in zip(axes, [("macs_m", "MACs per image (M)"), ("memory_kb", "Parameter memory (KB)")]):
        others = S[~S.id.isin(method_id.values())]
        ax.scatter(others[x], others.val_select_post, s=16, color=FA.MUTED, edgecolor="white", linewidth=0.6, zorder=2,
                   label="Other seed / enumeration policies")
        for m in FA.METHODS:
            r = S[S.id == method_id[m]].iloc[0]
            ax.scatter(r[x], r.val_select_post, s=46, marker=FA.MARK[m], color=FA.COLOR[m], edgecolor="white",
                       linewidth=0.8, zorder=3, label=FA.LABEL[m])
        ax.axhline(S[S.id == FA.DENSE].val_select_post.iloc[0] - FA.TOL, color=FA.INK2, lw=0.8, ls="--", zorder=1)
        ax.set_xlabel(xl)
        ax.set_ylabel("VAL-SELECT accuracy (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    fig.savefig(os.path.join(FA.FIG, "fig_accuracy_planes.png"))
    plt.close(fig)


def main():
    pre = FA.jload(L.FINAL_PREREG)
    sel = FA.jload(os.path.join(L.RESULTS, "selected_policies.json"))
    assert sel["preregistration_sha256"] == L.sha256_file(L.FINAL_PREREG)
    jobs = FA.jload(os.path.join(L.RESULTS, "final_jobs.json"))
    method_id = {m: sel["methods"][m]["id"] for m in FA.METHODS}
    method_id["dense"] = FA.DENSE
    ids = [j["id"] for j in jobs] + [FA.OLD]
    roles = {j["id"]: j["roles"] for j in jobs}
    roles[FA.OLD] = ["old_ppo"]

    acc, integ = FA.recovery(sel, jobs)
    mem = FA.memory(acc)
    pred = {tuple(json.loads(r["policy"])): r for r in
            csv.DictReader(open(os.path.join(L.RESULTS, "energy_surrogate_predictions.csv"), encoding="utf-8"))}
    m0 = L.counts((0, 0, 0, 0))[0]
    summary = []
    for mid in ids:
        a, mm = acc[mid], mem[mid]
        p = pred.get(tuple(a["policy"])) if a["policy"] else None
        summary.append({"id": mid, "roles": roles[mid], "policy": a["policy"], "widths": a["widths"],
                        "val_select_post": a["val_select_post"], "delta_val_select_pp": a["delta_val_select_pp"],
                        "val_rl_post": a["val_rl_post"], "delta_val_rl_pp": a["delta_val_rl_pp"],
                        "accuracy_preserving": a["accuracy_preserving"], "params": a["params"],
                        "memory_bytes": mm["param_bytes"], "memory_kb": mm["param_bytes"] / 1024,
                        "memory_reduction": mm["param_bytes_reduction"], "serialized_bytes": mm["serialized_bytes"],
                        "macs": a["macs"], "macs_m": a["macs"] / 1e6, "mac_reduction": 1 - a["macs"] / m0,
                        "latency_measured": "not measured", "energy_measured": "not measured",
                        "latency_table_ms_paper3": float(p["latency_table_ms"]) if p else None,
                        "e1000_pred_j_surrogate": float(p["e1000_pred_j"]) if p else None,
                        "saving_pred_surrogate": float(p["saving_pred"]) if p else None,
                        "green_success": NA})
    for r in summary:                    # GREEN-SUCCESS needs all five criteria: one failed assessable criterion decides NO
        failed = [k for k, ok in (("accuracy", r["accuracy_preserving"]), ("memory", r["memory_reduction"] > 0),
                                  ("MACs", r["mac_reduction"] > 0)) if not ok]
        if failed:
            r["green_success"] = f"NO ({', '.join(failed)} failed; latency and energy not measured)"
    S = pd.DataFrame(summary)
    pts = {r["id"]: r for r in summary}
    planes = {"accuracy_macs": [("val_select_post", "max"), ("macs", "min")],
              "accuracy_memory": [("val_select_post", "max"), ("memory_bytes", "min")]}
    front = {k: FA.pareto(pts, v) for k, v in planes.items()}
    FA.write_csv("pareto_results.csv", [{"id": i, "roles": roles[i], **{f"pareto_{k}": front[k][i] for k in planes},
                                         "pareto_accuracy_energy": NA, "pareto_energy_macs": NA} for i in ids])
    FA.write_csv("recovery_results.csv", [acc[i] for i in ids])
    FA.write_csv("memory_results.csv", [{"id": i, "roles": roles[i], **mem[i]} for i in ids])
    FA.write_csv("latency_energy_raw.csv", pilot_rows())
    FA.write_csv("final_summary.csv", summary)

    def assessable(i):
        r = pts[i]
        return {"accuracy": r["accuracy_preserving"], "memory": r["memory_reduction"] > 0, "MACs": r["mac_reduction"] > 0}
    per_seed = []
    for s in pre["seeds"]["confirmatory"]:
        ps = sel["per_seed"][str(s)]
        per_seed.append({"seed": s, "selected": ps["actions"] if ps else None, "id": ps["id"] if ps else None,
                         "assessable_criteria_met": assessable(ps["id"]) if ps else None})
    p = pts[method_id["new_ppo"]]
    if p["memory_reduction"] > 0 and p["mac_reduction"] > 0 and not p["accuracy_preserving"]:
        case = "undetermined: GREEN-C if energy were reduced, otherwise none of GREEN-A-D (energy not measured)"
    elif not p["accuracy_preserving"]:
        case = "none of GREEN-A-D (accuracy and memory/MACs criteria fail)"
    else:
        case = NA
    seeds_no = sum(1 for s in per_seed if s["id"] is None or str(pts[s["id"]]["green_success"]).startswith("NO"))
    green_seeds = ({"count": 0, "of": len(per_seed), "note": "every selected policy fails an assessable criterion"}
                   if seeds_no == len(per_seed) else NA)
    comps = {b: {"delta_val_select_pp": p["val_select_post"] - pts[method_id[b]]["val_select_post"],
                 "comparable_accuracy": abs(p["val_select_post"] - pts[method_id[b]]["val_select_post"]) <= FA.TOL + 1e-9,
                 "macs_ratio": p["macs"] / pts[method_id[b]]["macs"],
                 "memory_ratio": p["memory_bytes"] / pts[method_id[b]]["memory_bytes"],
                 "verdict": NA + "; the Phase-7A rule needs measured session energy"} for b in ("uniform", "lamp", "old_ppo")}
    pilots = {os.path.basename(f): FA.jload(f) for f in sorted(glob.glob(os.path.join(FA.EF, "pilot_n*.json")))
              if "_rep" not in f}
    try:
        figure(S, method_id)
        fig_status = "produced (accuracy-MACs and accuracy-memory planes; the energy planes are not assessable)"
    except ImportError as exc:
        fig_status = f"not produced (matplotlib import blocked: {exc})"
    stats = {"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
             "status": "hardware validation STOPPED by the pre-registered energy-pilot rule",
             "stop_rule": pre["energy_protocol"]["pilot"] if "energy_protocol" in pre else "N doubles once; still unstable -> STOP",
             "pilots": {k: {"n": v["n"], "e1000_j": v["e1000_j"], "cv": v["cv"], "decision": v["decision"]} for k, v in pilots.items()},
             "preregistration_sha256": L.sha256_file(L.FINAL_PREREG), "integrity": integ,
             "methods": {m: pts[method_id[m]] for m in FA.METHODS},
             "new_ppo_assessable_criteria_met": assessable(method_id["new_ppo"]),
             "new_ppo_unassessed_criteria": ["latency", "energy"],
             "case": case, "green_success_new_ppo": p["green_success"], "green_success_seeds": green_seeds,
             "green_success_models": {i: pts[i]["green_success"] for i in ids},
             "per_seed_assessable": per_seed,
             "comparisons_new_ppo_vs": comps, "enumeration_reference": sel["enumeration_reference"]["id"],
             "pareto": {**{k: [i for i in ids if front[k][i]] for k in planes},
                        "accuracy_energy": NA, "energy_macs": NA},
             "surrogate_note": "e1000_pred_j_surrogate is a prediction from the Paper-3 latency table and the accepted "
                               "surrogate; it is not a measurement and enters no label, case or verdict",
             "figures": fig_status, "test_use": "none"}
    json.dump(stats, open(os.path.join(L.RESULTS, "statistics.json"), "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps({"integrity": integ, "new_ppo": assessable(method_id["new_ppo"]), "comparisons": comps,
                      "figures": fig_status}, indent=1, default=float))
    print(S[["id", "val_select_post", "delta_val_select_pp", "val_rl_post", "accuracy_preserving", "memory_reduction",
             "mac_reduction", "e1000_pred_j_surrogate"]].to_string())


if __name__ == "__main__":
    main()
