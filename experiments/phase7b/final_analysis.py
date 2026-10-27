"""Phase 7B, Stage B: integrity checks, statistics, GREEN labels, cases, Pareto analysis and figures
(design sections 11-12; every rule as frozen in phase7b_final_preregistration.json, nothing tuned to the data).

  python experiments/phase7b/final_analysis.py
Inputs: selected_policies.json, final_jobs.json, _recovery_runs/final/*.json, _energy_final/session*.json, the
Phase-7A accuracy record of the old PPO model (read only).
Outputs: recovery_results.csv, memory_results.csv, latency_energy_raw.csv, final_summary.csv, pareto_results.csv,
statistics.json, figures/fig_pareto_planes.png, figures/fig_energy_sessions.png.
"""
import csv
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7b_lib as L  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SESSIONS, TOL = 5, L.ACC_TOL_PP
EF = os.path.join(L.RESULTS, "_energy_final")
FIG = os.path.join(L.RESULTS, "figures")
DENSE, OLD = "0000", "phase7a_ppo"
METHODS = ["dense", "new_ppo", "uniform", "lamp", "old_ppo"]
LABEL = {"dense": "Dense", "new_ppo": "New PPO-Structured", "uniform": "Uniform-Structured", "lamp": "LAMP-Structured",
         "old_ppo": "Old PPO-Structured (7A)"}
# dataviz reference categorical slots in fixed order (as in Phase 7A); Dense is the neutral reference mark
COLOR = {"dense": "#52514e", "new_ppo": "#2a78d6", "uniform": "#eb6834", "lamp": "#1baf7a", "old_ppo": "#a35fd1"}
MARK = {"dense": "s", "new_ppo": "o", "uniform": "^", "lamp": "D", "old_ppo": "v"}
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e6e5e0", "#b5b4ae"
RC = {"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
      "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
      "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
      "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
      "savefig.dpi": 300, "font.family": "DejaVu Sans"}


def jload(path):
    return json.load(open(path, encoding="utf-8"))


def wilcoxon(x, y, alternative):
    """Exact Wilcoxon signed-rank p-value (verbatim from experiments/phase7a/analyze.py, verified there): all 2^n sign
    patterns, zero differences dropped, tied |d| get average ranks. 'less': H1 x < y; 'two-sided'."""
    import itertools
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    a = np.abs(d)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(n)
    r[order] = np.arange(1, n + 1)
    for v in np.unique(a):
        r[a == v] = r[a == v].mean()
    ranks = r
    w_plus = ranks[d > 0].sum()
    null = np.array([sum(ranks[i] for i in range(n) if s[i]) for s in itertools.product((0, 1), repeat=n)])
    if alternative == "less":
        return float(np.mean(null <= w_plus + 1e-12))
    centre = ranks.sum() / 2
    return float(min(1.0, np.mean(np.abs(null - centre) >= abs(w_plus - centre) - 1e-12)))


def write_csv(name, rows):
    with open(os.path.join(L.RESULTS, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()} for r in rows])


# ------------------------------------------------------------------ accuracy and integrity
def recovery(sel, jobs):
    integ, rows = {}, {}
    streams = set()
    for j in jobs:
        r = jload(os.path.join(L.RESULTS, "_recovery_runs", "final", f"{j['id']}_E{L.FINAL_EPOCHS}.json"))
        assert r["status"] == "complete" and r["epochs"] == L.FINAL_EPOCHS and r["test_use"] == "none"
        assert L.sha256_file(os.path.join(L.ROOT, r["checkpoint"])) == r["checkpoint_sha256"]
        streams.add(r["data_stream_sha256"])
        pv = os.path.join(L.RESULTS, "_recovery_runs", "proxy_validation", f"{j['id']}_E5.json")
        if os.path.exists(pv):                                  # same job already run in Stage A: must be bit-identical
            integ[f"stage_a_identical_{j['id']}"] = jload(pv)["final_state_sha256"] == r["final_state_sha256"]
        rows[j["id"]] = {"id": j["id"], "roles": j["roles"], "policy": r["policy"], "ratios": r["ratios"],
                         "widths": r["widths"], "macs": r["macs"], "params": r["params"],
                         "val_rl_pre": r["pre"]["val_rl"], "val_rl_post": r["post"]["val_rl"],
                         "val_select_pre": r["pre"]["val_select"], "val_select_post": r["post"]["val_select"],
                         "final_state_sha256": r["final_state_sha256"], "checkpoint": r["checkpoint"],
                         "checkpoint_bytes": r["checkpoint_bytes"], "recovery_seconds": r["seconds"]}
    a7 = pd.read_csv(os.path.join(L.ROOT, "results", "phase7a_sustainability", "accuracy_results.csv"))
    a7 = a7[a7.setting == L.SETTING].drop_duplicates("method").set_index("method")
    old = sel["methods"]["old_ppo"]
    assert a7.loc["ppo", "checkpoint_sha256"] == old["checkpoint_sha256"]
    rows[OLD] = {"id": OLD, "roles": ["old_ppo"], "policy": old["actions"], "ratios": old["ratios"], "widths": old["widths"],
                 "macs": old["macs"], "params": old["params"], "val_rl_pre": float(a7.loc["ppo", "val_rl_pre"]),
                 "val_rl_post": float(a7.loc["ppo", "val_rl_post"]), "val_select_pre": float(a7.loc["ppo", "val_select_pre"]),
                 "val_select_post": float(a7.loc["ppo", "val_select_post"]),
                 "final_state_sha256": a7.loc["ppo", "final_state_sha256"], "checkpoint": old["checkpoint"],
                 "checkpoint_bytes": os.path.getsize(os.path.join(L.ROOT, old["checkpoint"])),
                 "recovery_seconds": float(a7.loc["ppo", "recovery_seconds"])}
    integ["single_data_stream_phase7b"] = len(streams) == 1
    integ["phase7b_dense_identical_to_phase7a_dense"] = rows[DENSE]["final_state_sha256"] == a7.loc["dense", "final_state_sha256"]
    integ["same_data_stream_as_phase7a"] = streams == {a7.loc["dense", "data_stream_sha256"]}
    d = rows[DENSE]
    for r in rows.values():
        r["delta_val_select_pp"] = r["val_select_post"] - d["val_select_post"]
        r["delta_val_rl_pp"] = r["val_rl_post"] - d["val_rl_post"]
        r["accuracy_preserving"] = bool(r["delta_val_select_pp"] >= -TOL - 1e-9)
    return rows, integ


def memory(rows):
    import torch
    from src.pruning import structured as SP
    P7 = L.energy_meter()
    ref, _ = L.p3().reference(L.SETTING)
    out = {}
    for mid, r in rows.items():
        m = P7.load_recovered(L.ARCH, "ppo") if mid == OLD else SP.simplecnn_structured(ref, r["ratios"])[0]
        if mid != OLD:
            m.load_state_dict(torch.load(os.path.join(L.ROOT, r["checkpoint"]), map_location="cpu", weights_only=True))
        pb = sum(p.numel() * p.element_size() for p in m.parameters())
        bb = sum(b.numel() * b.element_size() for b in m.buffers())
        assert sum(p.numel() for p in m.parameters()) == r["params"]
        out[mid] = {"param_count": r["params"], "param_bytes": pb, "buffer_bytes": bb, "serialized_bytes": r["checkpoint_bytes"],
                    "dtype": str(next(m.parameters()).dtype)}
    d = out[DENSE]
    for v in out.values():
        v["param_bytes_reduction"] = 1 - v["param_bytes"] / d["param_bytes"]
        v["serialized_reduction"] = 1 - v["serialized_bytes"] / d["serialized_bytes"]
    return out


# ------------------------------------------------------------------ energy and latency
def energy_raw(ids):
    rows = []
    for s in range(1, SESSIONS + 1):
        r = jload(os.path.join(EF, f"session{s}.json"))
        assert r["effective"]["threads"] == 1 and r["effective"]["interop"] == 1 and r["effective"]["cpu_affinity"] == [L.CPU]
        for b in r["blocks"]:
            rows.append({"session": s, "position": b["position"], "model": b["model"], "appearance": b["appearance"],
                         "measured_at": b["at"], "n": r["n"], "seconds": b["seconds"], "pkg_j": b["pkg_j"], "pp0_j": b["pp0_j"],
                         "e1000_j": b["e1000_j"], "net_e1000_j": b["net_e1000_j"], "pkg_w": b["pkg_w"],
                         "idle_pkg_w": r["idle"]["pkg_w"], "latency_median_ms": b["latency_median_ms"],
                         "latency_mean_ms": b["latency_mean_ms"], "latency_p95_ms": b["latency_p95_ms"],
                         "latency_iqr_ms": b["latency_iqr_ms"]})
    raw = pd.DataFrame(rows)
    cnt = raw.groupby(["session", "model"]).size().unstack()
    assert set(cnt.columns) == set(ids) and (cnt[DENSE] == 3).all() and (cnt.drop(columns=DENSE) == 1).all().all()
    return raw


def energy_stats(raw, ids):
    per = raw.groupby(["session", "model"])[["e1000_j", "latency_median_ms", "net_e1000_j"]].median().unstack()
    e, lat = per["e1000_j"], per["latency_median_ms"]
    dense_e, dense_l = e[DENSE], lat[DENSE]
    cv = float(dense_e.std(ddof=1) / dense_e.mean())
    noise = 2 * cv
    within = raw[raw.model == DENSE].groupby("session").e1000_j.agg(lambda v: v.max() / v.min() - 1)
    out = {}
    for m in ids:
        sv = 1 - e[m] / dense_e
        edp = (e[m] * lat[m]) / (dense_e * dense_l)
        lower_e, lower_l = int((e[m] < dense_e).sum()), int((lat[m] < dense_l).sum())
        out[m] = {"e1000_session_j": [float(v) for v in e[m]], "e1000_median_j": float(e[m].median()),
                  "e1000_cv": float(e[m].std(ddof=1) / e[m].mean()), "saving_median": float(sv.median()),
                  "saving_min": float(sv.min()), "saving_max": float(sv.max()), "sessions_energy_lower": lower_e,
                  "energy_reduced": bool(m != DENSE and lower_e == SESSIONS and float(sv.median()) > noise),
                  "wilcoxon_one_sided_p": None if m == DENSE else wilcoxon(e[m], dense_e, "less"),
                  "net_saving_median": float((1 - per["net_e1000_j"][m] / per["net_e1000_j"][DENSE]).median()),
                  "latency_session_ms": [float(v) for v in lat[m]], "latency_median_ms": float(lat[m].median()),
                  "speedup_median": float((dense_l / lat[m]).median()), "sessions_latency_lower": lower_l,
                  "latency_reduced": bool(m != DENSE and float(lat[m].median()) < float(dense_l.median()) and lower_l == SESSIONS),
                  "normalized_edp_median": float(edp.median())}
    return out, {"dense_between_session_cv": cv, "noise_threshold": noise,
                 "dense_within_session_spread": {int(k): float(v) for k, v in within.items()}}, e


def better(a, b, e, acc):
    """Phase-7A comparison rule with VAL-SELECT accuracy: 'a better', 'b better' or 'no clear difference'."""
    dacc = acc[a] - acc[b]
    comparable = abs(dacc) <= TOL + 1e-9
    a_lower, b_lower = int((e[a] < e[b]).sum()), int((e[b] < e[a]).sum())
    if (comparable and b_lower == SESSIONS) or (-dacc > TOL and e[b].median() <= e[a].median()):
        v = "baseline better"
    elif (comparable and a_lower == SESSIONS) or (dacc > TOL and e[a].median() <= e[b].median()):
        v = "new PPO better"
    else:
        v = "no clear difference"
    return {"delta_val_select_pp": dacc, "comparable_accuracy": comparable, "sessions_new_ppo_lower": a_lower,
            "sessions_baseline_lower": b_lower, "energy_diff_median_pct": float((1 - e[a] / e[b]).median() * 100),
            "wilcoxon_two_sided_p": wilcoxon(e[a], e[b], "two-sided"), "verdict": v}


def pareto(points, keys):
    """Non-dominated set; keys = [(column, 'max'|'min'), ...]."""
    def ge(p, q):
        return all((p[k] >= q[k]) if d == "max" else (p[k] <= q[k]) for k, d in keys)

    def gt(p, q):
        return any((p[k] > q[k]) if d == "max" else (p[k] < q[k]) for k, d in keys)
    return {i: not any(ge(q, p) and gt(q, p) for j, q in points.items() if j != i) for i, p in points.items()}


# ------------------------------------------------------------------ figures
def figures(summary, method_id):
    """Figures only; imported here so that a blocked matplotlib (Windows Smart App Control) never stops the numbers."""
    import matplotlib
    if not hasattr(matplotlib, "use"):
        raise ImportError("matplotlib is the ppo.py placeholder (the real package is blocked)")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC)
    os.makedirs(FIG, exist_ok=True)
    planes = [("e1000_median_j", "Energy per 1,000 inferences (J)", "val_select_post", "VAL-SELECT accuracy (%)"),
              ("macs_m", "MACs per image (M)", "val_select_post", "VAL-SELECT accuracy (%)"),
              ("memory_kb", "Parameter memory (KB)", "val_select_post", "VAL-SELECT accuracy (%)"),
              ("macs_m", "MACs per image (M)", "e1000_median_j", "Energy per 1,000 inferences (J)")]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
    for ax, (x, xl, y, yl) in zip(axes.flat, planes):
        others = summary[~summary.id.isin(method_id.values())]
        ax.scatter(others[x], others[y], s=16, color=MUTED, edgecolor="white", linewidth=0.6, zorder=2,
                   label="Other seed / enumeration policies")
        for m in METHODS:
            r = summary[summary.id == method_id[m]].iloc[0]
            ax.scatter(r[x], r[y], s=46, marker=MARK[m], color=COLOR[m], edgecolor="white", linewidth=0.8, zorder=3,
                       label=LABEL[m])
        if y == "val_select_post":
            ax.axhline(summary[summary.id == DENSE].val_select_post.iloc[0] - TOL, color=INK2, lw=0.8, ls="--", zorder=1)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(FIG, "fig_pareto_planes.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    order = summary.sort_values("e1000_median_j").reset_index(drop=True)
    for i, r in order.iterrows():
        m = next((k for k, v in method_id.items() if v == r.id), None)
        sv = 100 * (1 - np.array(r.e1000_session_j) / np.array(order[order.id == DENSE].e1000_session_j.iloc[0]))
        ax.scatter([i] * len(sv), sv, s=12, color=COLOR.get(m, MUTED), zorder=3)
    ax.axhline(100 * order.noise_threshold.iloc[0], color=INK2, lw=0.8, ls="--")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABEL.get(next((k for k, v in method_id.items() if v == i), ""), i) for i in order.id],
                       rotation=60, ha="right", fontsize=6.5)
    ax.set_ylabel("Energy saving vs dense,\nper session (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_energy_sessions.png"))
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    pre = jload(L.FINAL_PREREG)
    sel = jload(os.path.join(L.RESULTS, "selected_policies.json"))
    assert sel["preregistration_sha256"] == L.sha256_file(L.FINAL_PREREG)
    jobs = jload(os.path.join(L.RESULTS, "final_jobs.json"))
    method_id = {m: sel["methods"][m]["id"] for m in METHODS}
    method_id["dense"] = DENSE
    ids = [j["id"] for j in jobs] + [OLD]

    acc, integ = recovery(sel, jobs)
    mem = memory(acc)
    raw = energy_raw(ids)
    en, noise, e = energy_stats(raw, ids)
    roles = {j["id"]: j["roles"] for j in jobs}
    roles[OLD] = ["old_ppo"]

    m0, p0, _ = L.counts((0, 0, 0, 0))
    summary = []
    for mid in ids:
        a, mm, ee = acc[mid], mem[mid], en[mid]
        mem_red, mac_red = mm["param_bytes_reduction"], 1 - a["macs"] / m0
        green = bool(a["accuracy_preserving"] and mem_red > 0 and mac_red > 0 and ee["latency_reduced"] and ee["energy_reduced"])
        summary.append({"id": mid, "roles": roles[mid], "policy": a["policy"], "widths": a["widths"],
                        "val_select_post": a["val_select_post"], "delta_val_select_pp": a["delta_val_select_pp"],
                        "val_rl_post": a["val_rl_post"], "delta_val_rl_pp": a["delta_val_rl_pp"],
                        "accuracy_preserving": a["accuracy_preserving"], "params": a["params"],
                        "memory_bytes": mm["param_bytes"], "memory_kb": mm["param_bytes"] / 1024, "memory_reduction": mem_red,
                        "serialized_bytes": mm["serialized_bytes"], "macs": a["macs"], "macs_m": a["macs"] / 1e6,
                        "mac_reduction": mac_red, **{k: v for k, v in ee.items()}, "noise_threshold": noise["noise_threshold"],
                        "green_success": green})
    S = pd.DataFrame(summary)
    acc_sel = {mid: acc[mid]["val_select_post"] for mid in ids}

    comps = {b: better(method_id["new_ppo"], method_id[b], e, acc_sel) for b in ("uniform", "lamp", "old_ppo")}
    p = S[S.id == method_id["new_ppo"]].iloc[0]
    if p.green_success:
        case = "GREEN-B" if any(comps[b]["verdict"] == "baseline better" for b in ("uniform", "lamp")) else "GREEN-A"
    elif p.memory_reduction > 0 and p.mac_reduction > 0 and p.energy_reduced and not p.accuracy_preserving:
        case = "GREEN-C"
    elif p.accuracy_preserving and not p.energy_reduced:
        case = "GREEN-D"
    else:
        case = "none of GREEN-A-D (GREEN-SUCCESS = NO; see the failed criteria)"
    failed = [k for k, ok in (("accuracy", p.accuracy_preserving), ("memory", p.memory_reduction > 0),
                              ("MACs", p.mac_reduction > 0), ("latency", p.latency_reduced),
                              ("energy", p.energy_reduced)) if not ok]

    seeds = pre["seeds"]["confirmatory"]
    per_seed = []
    for s in seeds:
        ps = sel["per_seed"][str(s)]
        g = bool(S[S.id == ps["id"]].green_success.iloc[0]) if ps else False
        per_seed.append({"seed": s, "selected": ps["actions"] if ps else None, "green_success": g})

    planes = {"accuracy_energy": [("val_select_post", "max"), ("e1000_median_j", "min")],
              "accuracy_macs": [("val_select_post", "max"), ("macs", "min")],
              "accuracy_memory": [("val_select_post", "max"), ("memory_bytes", "min")],
              "energy_macs": [("e1000_median_j", "min"), ("macs", "min")]}
    pts = {r["id"]: r for r in summary}
    front = {k: pareto(pts, v) for k, v in planes.items()}
    write_csv("pareto_results.csv", [{"id": i, "roles": roles[i], **{f"pareto_{k}": front[k][i] for k in planes}} for i in ids])

    write_csv("recovery_results.csv", [acc[i] for i in ids])
    write_csv("memory_results.csv", [{"id": i, "roles": roles[i], **mem[i]} for i in ids])
    raw.to_csv(os.path.join(L.RESULTS, "latency_energy_raw.csv"), index=False)
    write_csv("final_summary.csv", summary)
    try:
        figures(S, method_id)
        fig_status = "produced"
    except ImportError as exc:
        fig_status = f"not produced (matplotlib import blocked: {exc}); rerun this script once it loads"
    pred ={tuple(json.loads(r["policy"])): float(r["e1000_pred_j"]) for r in
            csv.DictReader(open(os.path.join(L.RESULTS, "energy_surrogate_predictions.csv"), encoding="utf-8"))}

    stats = {"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
             "preregistration_sha256": L.sha256_file(L.FINAL_PREREG), "integrity": integ, "noise": noise,
             "methods": {m: next(r for r in summary if r["id"] == method_id[m]) for m in METHODS},
             "comparisons_new_ppo_vs": comps, "case": case, "new_ppo_failed_criteria": failed,
             "green_success_seeds": {"count": int(sum(r["green_success"] for r in per_seed)), "of": len(seeds),
                                     "per_seed": per_seed},
             "green_success_models": {i: bool(pts[i]["green_success"]) for i in ids},
             "enumeration_reference": sel["enumeration_reference"]["id"],
             "surrogate_check": {i: {"e_pred_j": pred.get(tuple(acc[i]["policy"])) if acc[i]["policy"] else None,
                                     "e_measured_median_j": en[i]["e1000_median_j"]} for i in ids},
             "pareto": {k: [i for i in ids if front[k][i]] for k in planes}, "figures": fig_status}
    json.dump(stats, open(os.path.join(L.RESULTS, "statistics.json"), "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps({"case": case, "failed": failed, "integrity": integ, "noise": noise,
                      "green_seeds": stats["green_success_seeds"]["count"], "comparisons": comps}, indent=1, default=float))
    print(S[["id", "delta_val_select_pp", "memory_reduction", "mac_reduction", "saving_median", "sessions_energy_lower",
             "energy_reduced", "latency_reduced", "green_success"]].to_string())


if __name__ == "__main__":
    main()
