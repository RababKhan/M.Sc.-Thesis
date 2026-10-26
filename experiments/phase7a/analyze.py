"""Phase 7A, steps 11-12: statistics, tables, figures and the pre-registered interpretation.

  python experiments/phase7a/analyze.py
Inputs: accuracy_results.csv, energy_parts/<arch>_session<s>.json, selected_policies.json, preregistration.json.
Outputs (results/phase7a_sustainability/): energy_raw.csv, energy_summary.csv, latency_summary.csv,
pairwise_comparisons.csv, primary_table.md, sustainability_summary.json, figures/fig_A_*.png, figures/fig_B_*.png.
Every rule used here is the one in preregistration.json; nothing is tuned to the data.
"""
import datetime
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7_lib as L  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PARTS = os.path.join(L.RESULTS, "energy_parts")
FIG = os.path.join(L.RESULTS, "figures")
SESSIONS = 5
J_PER_KWH = 3.6e6
# dataviz reference categorical slots 1-3 in fixed order; Dense is the neutral reference mark
COLOR = {"dense": "#52514e", "ppo": "#2a78d6", "uniform": "#eb6834", "lamp": "#1baf7a"}
MARK = {"dense": "s", "ppo": "o", "uniform": "^", "lamp": "D"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "font.family": "DejaVu Sans"})
ARCH_TITLE = {"simplecnn": "SimpleCNN", "resnet8": "ResNet-8"}


def wilcoxon(x, y, alternative):
    """Exact Wilcoxon signed-rank p-value by enumerating all 2^n sign patterns (zero differences dropped, tied
    |d| get average ranks). alternative 'less': H1 x < y; 'two-sided'."""
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
    for v in np.unique(a):                                  # average ranks over ties
        r[a == v] = r[a == v].mean()
    ranks = r
    w_plus = ranks[d > 0].sum()
    null = np.array([sum(ranks[i] for i in range(n) if s[i]) for s in itertools.product((0, 1), repeat=n)])
    if alternative == "less":
        return float(np.mean(null <= w_plus + 1e-12))
    centre = ranks.sum() / 2
    return float(min(1.0, np.mean(np.abs(null - centre) >= abs(w_plus - centre) - 1e-12)))


def load_raw():
    rows = []
    for arch in L.SETTINGS:
        for s in range(1, SESSIONS + 1):
            path = os.path.join(PARTS, f"{arch}_session{s}.json")
            r = json.load(open(path, encoding="utf-8"))
            assert r["n"] == r["n"] and r["effective"]["threads"] == 1 and r["effective"]["cpu_affinity"] == [L.CPU]
            for pos, b in enumerate(r["blocks"]):
                rows.append({"arch": arch, "setting": L.SETTINGS[arch], "session": s, "position": pos, "method": b["model"],
                             "measured_at": b["at"], "n": r["n"], "seconds": b["seconds"], "pkg_j": b["pkg_j"],
                             "pp0_j": b["pp0_j"], "e1000_j": b["e1000_j"], "pp0_e1000_j": b["pp0_j"] / r["n"] * 1000,
                             "net_e1000_j": b["net_e1000_j"], "pkg_w": b["pkg_w"], "idle_pkg_w": r["idle"]["pkg_w"],
                             "latency_median_ms": b["latency_median_ms"], "latency_mean_ms": b["latency_mean_ms"],
                             "latency_p95_ms": b["latency_p95_ms"], "latency_iqr_ms": b["latency_iqr_ms"]})
    raw = pd.DataFrame(rows)
    assert (raw.groupby(["arch", "session"]).method.nunique() == len(L.METHODS)).all()
    return raw


def iqr(v):
    q = np.percentile(v, [75, 25])
    return float(q[0] - q[1])


def summarize(raw, acc, sel):
    e_rows, l_rows, pair_rows, primary = [], [], [], []
    per_arch = {}
    for arch, setting in L.SETTINGS.items():
        d = raw[raw.arch == arch]
        piv = {k: d.pivot(index="session", columns="method", values=k) for k in
               ("e1000_j", "pp0_e1000_j", "net_e1000_j", "latency_median_ms")}
        dense_e = piv["e1000_j"]["dense"]
        cv_dense = float(dense_e.std(ddof=1) / dense_e.mean())
        noise = 2 * cv_dense
        per_arch[arch] = {"dense_cv": cv_dense, "noise_threshold": noise, "methods": {}}
        a = acc[acc.setting == setting].set_index("method")
        for m in L.METHODS:
            e = piv["e1000_j"][m]
            saving = 1 - e / dense_e
            lat = piv["latency_median_ms"][m]
            speed = piv["latency_median_ms"]["dense"] / lat
            edp = (e / 1000 * lat) / (dense_e / 1000 * piv["latency_median_ms"]["dense"])
            lower = int((e < dense_e).sum())
            consistent = bool(m != "dense" and lower == SESSIONS and float(saving.median()) > noise)
            p_one = wilcoxon(e, dense_e, "less") if m != "dense" else None
            sec = {}
            for k, lab in (("pp0_e1000_j", "pp0"), ("net_e1000_j", "net")):
                sv = 1 - piv[k][m] / piv[k]["dense"]
                sec[lab] = {"median_j": float(piv[k][m].median()), "median_saving": float(sv.median()),
                            "sessions_lower": int((piv[k][m] < piv[k]["dense"]).sum())}
            spec = sel["settings"][setting]["methods"][m]
            row = {"arch": arch, "method": m, "e1000_median_j": float(e.median()), "e1000_mean_j": float(e.mean()),
                   "e1000_sd_j": float(e.std(ddof=1)), "e1000_iqr_j": iqr(e), "e1000_cv": float(e.std(ddof=1) / e.mean()),
                   "e_per_inference_median_mj": float(e.median()),  # J per 1000 inferences == mJ per inference
                   "saving_median": float(saving.median()), "saving_min": float(saving.min()), "saving_max": float(saving.max()),
                   "sessions_lower_than_dense": lower, "noise_threshold": noise, "consistent_reduction": consistent,
                   "wilcoxon_one_sided_p": p_one, "pp0_saving_median": sec["pp0"]["median_saving"],
                   "pp0_sessions_lower": sec["pp0"]["sessions_lower"], "net_saving_median": sec["net"]["median_saving"],
                   "net_sessions_lower": sec["net"]["sessions_lower"],
                   "power_median_w": float(d[d.method == m].pkg_w.median())}
            e_rows.append(row)
            l_rows.append({"arch": arch, "method": m, "latency_median_ms": float(lat.median()),
                           "latency_sd_ms": float(lat.std(ddof=1)), "speedup_median": float(speed.median()),
                           "speedup_min": float(speed.min()), "speedup_max": float(speed.max()),
                           "normalized_edp_median": float(edp.median()), "normalized_edp_min": float(edp.min()),
                           "normalized_edp_max": float(edp.max())})
            acc_m = a.loc[m]
            per_arch[arch]["methods"][m] = {**row, "speedup_median": float(speed.median()),
                                            "latency_median_ms": float(lat.median()), "normalized_edp": float(edp.median()),
                                            "val_rl": float(acc_m.val_rl_post), "delta_pp": float(acc_m.delta_val_rl_pp),
                                            "q": float(acc_m.q_val_rl), "accuracy_preserving": bool(acc_m.accuracy_preserving),
                                            "param_reduction": spec["param_reduction"], "mac_reduction": spec["mac_reduction"],
                                            "widths": spec["widths"], "e_session": [float(v) for v in e]}
            primary.append({"Architecture": ARCH_TITLE[arch], "Method": L.LABELS[m],
                            "Accuracy (%)": f"{acc_m.val_rl_post:.2f}", "ΔAcc (pp)": f"{acc_m.delta_val_rl_pp:+.2f}",
                            "Params ↓": f"{100 * spec['param_reduction']:.1f}%", "MACs ↓": f"{100 * spec['mac_reduction']:.1f}%",
                            "Latency (ms)": f"{lat.median():.4f}", "Speed-up": f"{speed.median():.3f}×",
                            "Energy / 1000 inf. (J)": f"{e.median():.3f}", "Energy ↓": f"{100 * saving.median():.1f}%",
                            "Normalized EDP": f"{edp.median():.3f}"})
        # PPO versus each baseline (pre-registered comparison rule)
        ppo = per_arch[arch]["methods"]["ppo"]
        for b in ("uniform", "lamp"):
            bm = per_arch[arch]["methods"][b]
            e_p, e_b = piv["e1000_j"]["ppo"], piv["e1000_j"][b]
            dacc = ppo["val_rl"] - bm["val_rl"]
            comparable = abs(dacc) <= L.ACC_TOLERANCE_PP + 1e-9
            ppo_lower, b_lower = int((e_p < e_b).sum()), int((e_b < e_p).sum())
            b_better = (comparable and b_lower == SESSIONS) or (-dacc > L.ACC_TOLERANCE_PP and e_b.median() <= e_p.median())
            p_better = (comparable and ppo_lower == SESSIONS) or (dacc > L.ACC_TOLERANCE_PP and e_p.median() <= e_b.median())
            verdict = "baseline better" if b_better else "PPO better" if p_better else "no clear difference"
            pair_rows.append({"arch": arch, "comparison": f"ppo_vs_{b}", "delta_acc_pp_ppo_minus_baseline": dacc,
                              "comparable_accuracy": comparable,
                              "energy_diff_median_pct": float((1 - e_p / e_b).median() * 100),
                              "sessions_ppo_lower": ppo_lower, "sessions_baseline_lower": b_lower,
                              "wilcoxon_two_sided_p": wilcoxon(e_p, e_b, "two-sided"), "verdict": verdict})
    return pd.DataFrame(e_rows), pd.DataFrame(l_rows), pd.DataFrame(pair_rows), pd.DataFrame(primary), per_arch


def cases(per_arch, pairs):
    out = {}
    for arch, v in per_arch.items():
        p = v["methods"]["ppo"]
        pres, cons = p["accuracy_preserving"], p["consistent_reduction"]
        case = "A" if pres and cons else "B" if cons else "C" if pres else "neither (accuracy loss, no consistent saving)"
        d = pairs[(pairs.arch == arch) & (pairs.verdict == "baseline better")].comparison.tolist()
        out[arch] = {"case": case, "case_D_baselines_better": d, "ppo_accuracy_preserving": pres,
                     "ppo_consistent_energy_reduction": cons}
    return out


def projection(per_arch):
    proj = {}
    for arch, v in per_arch.items():
        e_d = v["methods"]["dense"]["e1000_median_j"] / 1000
        proj[arch] = {}
        for m in ("ppo", "uniform", "lamp"):
            dj = e_d - v["methods"][m]["e1000_median_j"] / 1000
            proj[arch][m] = {f"{n:.0e}": {"joules": dj * n, "kwh": dj * n / J_PER_KWH} for n in (1e6, 1e7, 1e8)}
    return proj


# ------------------------------------------------------------------ figures
def fig_a(per_arch):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, (arch, v) in zip(axes, per_arch.items()):
        for m in L.METHODS:
            r = v["methods"][m]
            es = np.array(r["e_session"])
            ax.errorbar(r["e1000_median_j"], r["val_rl"], xerr=[[r["e1000_median_j"] - es.min()], [es.max() - r["e1000_median_j"]]],
                        fmt=MARK[m], ms=7, color=COLOR[m], ecolor=COLOR[m], elinewidth=1, capsize=0, label=L.LABELS[m],
                        markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            ax.annotate(L.LABELS[m].replace("-Structured", ""), (r["e1000_median_j"], r["val_rl"]), xytext=(6, 4),
                        textcoords="offset points", fontsize=7.5, color=INK2)
        dense = v["methods"]["dense"]["val_rl"]
        ax.axhspan(dense - L.ACC_TOLERANCE_PP, dense + L.ACC_TOLERANCE_PP, color=GRID, alpha=0.6, lw=0, zorder=0)
        ax.set_title(ARCH_TITLE[arch], fontsize=9)
        ax.set_xlabel("Package energy per 1,000 inferences (J)")
        ax.set_ylabel("VAL-RL accuracy after recovery (%)")
        xs = [v["methods"][m]["e1000_median_j"] for m in L.METHODS]
        ax.set_xlim(min(xs) * 0.9, max(xs) * 1.12)
    axes[0].legend(loc="lower right", frameon=False)
    fig.text(0.5, -0.02, "Shaded band: dense accuracy ± 0.5 pp. Horizontal bars: range over 5 sessions.", ha="center",
             fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_A_accuracy_vs_energy.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_b(per_arch, sel):
    metrics = [("MACs", lambda r: 1 - r["mac_reduction"]), ("Latency", lambda r: 1 / r["speedup_median"]),
               ("Energy", lambda r: 1 - r["saving_median"]), ("EDP", lambda r: r["normalized_edp"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    methods = ["ppo", "uniform", "lamp"]
    w = 0.26
    for ax, (arch, v) in zip(axes, per_arch.items()):
        x = np.arange(len(metrics))
        for j, m in enumerate(methods):
            vals = [f(v["methods"][m]) for _, f in metrics]
            bars = ax.bar(x + (j - 1) * w, vals, w - 0.02, color=COLOR[m], label=L.LABELS[m], edgecolor="white", linewidth=0.8)
            for bx, val in zip(bars, vals):
                ax.text(bx.get_x() + bx.get_width() / 2, val + 0.015, f"{val:.2f}", ha="center", va="bottom", fontsize=6.3,
                        color=INK2)
        ax.axhline(1.0, color=INK2, lw=0.8, ls="--")
        ax.text(len(metrics) - 0.55, 1.015, "Dense = 1.0", ha="right", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(x, [k for k, _ in metrics])
        ax.set_title(ARCH_TITLE[arch], fontsize=9)
        ax.set_ylim(0, 1.15)
    axes[0].set_ylabel("Relative to dense")
    axes[1].legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="none")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_B_normalized_efficiency.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    sel = L.policies()
    acc = pd.read_csv(os.path.join(L.RESULTS, "accuracy_results.csv"))
    raw = load_raw()
    raw.to_csv(os.path.join(L.RESULTS, "energy_raw.csv"), index=False)
    esum, lsum, pairs, primary, per_arch = summarize(raw, acc, sel)
    esum.to_csv(os.path.join(L.RESULTS, "energy_summary.csv"), index=False)
    lsum.to_csv(os.path.join(L.RESULTS, "latency_summary.csv"), index=False)
    pairs.to_csv(os.path.join(L.RESULTS, "pairwise_comparisons.csv"), index=False)
    cols = list(primary.columns)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in primary.iterrows()]
    open(os.path.join(L.RESULTS, "primary_table.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    cs = cases(per_arch, pairs)
    summary = {"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
               "preregistration_sha256": L.sha256_file(L.PREREG), "n_inferences_per_block": int(raw.n.iloc[0]),
               "sessions": SESSIONS, "cases": cs, "per_architecture": per_arch,
               "pairwise": pairs.to_dict(orient="records"),
               "deployment_projection_label": "deployment-scale projection from measured per-inference package energy; "
                                              "not a direct experiment; no CO2e conversion",
               "deployment_projection": projection(per_arch)}
    json.dump(summary, open(os.path.join(L.RESULTS, "sustainability_summary.json"), "w", encoding="utf-8"), indent=1,
              default=float)
    fig_a(per_arch)
    fig_b(per_arch, sel)
    print(open(os.path.join(L.RESULTS, "primary_table.md"), encoding="utf-8").read())
    print(pairs.to_string())
    print(json.dumps(cs, indent=1))


if __name__ == "__main__":
    main()
