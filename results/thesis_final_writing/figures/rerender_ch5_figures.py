"""Thesis-specific, presentation-only re-renders of three Chapter-5 figures.

Reads the SAME frozen CSV files the original phase scripts read; changes only colours, legends, labels and layout.
The original phase figures are not touched. After plotting, every plotted coordinate is read back from the matplotlib
artists and compared with the source CSV (exact equality); the script exits non-zero on any mismatch.

  python results/thesis_final_writing/figures/rerender_ch5_figures.py
Outputs (results/thesis_final_writing/figures/):
  ch5_fig5_3_sensitivity_tolerance.png   from results/architecture_generalization/phase3_fig_D_sensitivity_tolerance.csv
  ch5_fig5_4_topk_reach_rate.png         from results/phase4/phase4_fig_D_topk_reach_rate.csv
  ch5_fig5_5_resnet_reward_alignment.png from results/phase5/phase5_fig_B_resnet_reward_alignment.csv
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", ".."))
S6 = ["cifar10_simplecnn", "cifar10_lenet5", "cifar10_resnet8", "cifar100_simplecnn", "cifar100_lenet5", "cifar100_resnet8"]
# matplotlib default (tab10) colours, one fixed colour per setting on both panels
SET_COL = dict(zip(S6, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]))
checks = []


def read(p):
    return pd.read_csv(os.path.join(R, p), float_precision="round_trip")


def same(name, plotted, source):
    a = np.asarray(plotted, float)
    b = np.asarray(source, float)
    ok = a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])
    checks.append((name, ok))


# ------------------------------------------------------------------ Figure 5.3 (Phase-3 fig D)
def fig_sensitivity():
    df = read("architecture_generalization/phase3_fig_D_sensitivity_tolerance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    fig.suptitle("Unit sensitivity vs pruning tolerance and PPO allocation (Phase 3)")
    for s in S6:
        g = df[df.setting == s]
        c = SET_COL[s]
        p0 = axes[0].scatter(g.normalized_sensitivity, g.tolerance_ratio_1pp, color=c)
        p1 = axes[1].scatter(g.normalized_sensitivity, g.mean_ratio_P0, marker="o", color=c)
        p2 = axes[1].scatter(g.normalized_sensitivity, g.mean_ratio_P1, marker="x", color=c)
        same(f"5.3 left {s}", p0.get_offsets(), np.c_[g.normalized_sensitivity, g.tolerance_ratio_1pp])
        same(f"5.3 right P0 {s}", p1.get_offsets(), np.c_[g.normalized_sensitivity, g.mean_ratio_P0])
        same(f"5.3 right P1 {s}", p2.get_offsets(), np.c_[g.normalized_sensitivity, g.mean_ratio_P1])
    for a in axes:
        a.set_xlabel("normalised V_RL loss sensitivity")
    axes[0].set_ylabel("largest single-unit ratio with V_RL drop <= 1 pp (%)")
    axes[1].set_ylabel("mean final ratio over 20 seeds (%)")
    set_handles = [Line2D([], [], ls="", marker="s", color=SET_COL[s], label=s) for s in S6]
    mk_handles = [Line2D([], [], ls="", marker="o", color="#444444", label="P0 (no prior)"),
                  Line2D([], [], ls="", marker="x", color="#444444", label="P1 (sensitivity prior)")]
    fig.legend(handles=set_handles + mk_handles, loc="lower center", ncol=8, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    return fig


# ------------------------------------------------------------------ Figure 5.4 (Phase-4 fig D)
def fig_topk():
    df = read("phase4/phase4_fig_D_topk_reach_rate.csv")
    ks = [10, 20, 50, 100]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Share of runs whose selected policy is in the top-k (bars) and best-seen top-20 reach (dots), Phase 4")
    handles = None
    for s, a in zip(S6, axs.ravel()):
        g = df[df.setting == s].reset_index(drop=True)
        w = 0.8 / max(1, len(g))
        hs = []
        for i, r in g.iterrows():
            vals = [r.get(f"selected_reach_top{k}", np.nan) for k in ks]
            lab = str(r.condition)
            if lab.startswith("Phase-3"):
                lab = "Phase-3 P0 reference (top-10/20 reach 0; top-50/100 not recorded)"
            bars = a.bar(np.arange(len(ks)) + i * w, vals, width=w, label=lab)
            same(f"5.4 bars {s} {r.condition[:12]}", [b.get_height() for b in bars], vals)
            sc = a.scatter(1 + i * w, r.get("best_seen_reach_top20", np.nan), color="black", s=8, zorder=3)
            same(f"5.4 dot {s} {r.condition[:12]}", sc.get_offsets()[:, 1], [r.get("best_seen_reach_top20", np.nan)])
            hs.append(bars)
        handles = handles or hs
        a.set_xticks(np.arange(len(ks)) + 0.4, [f"top {k}" for k in ks])
        a.set_ylim(0, 1.05)
        a.set_title(s)
        a.set_ylabel("share of 20 runs")
    dot = Line2D([], [], ls="", marker="o", color="black", markersize=4, label="best-seen top-20 reach")
    fig.legend(handles=handles + [dot], labels=[h.get_label() for h in handles] + ["best-seen top-20 reach"],
               loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    return fig


# ------------------------------------------------------------------ Figure 5.5 (Phase-5 fig B)
COL5 = {"C1": "#2a78d6", "C0": "#eb6834", "C2": "#1baf7a", "Phase-4 F": "#0b0b0b"}
NAMES = {"cifar10_resnet8": "C10 ResNet-8", "cifar100_resnet8": "C100 ResNet-8"}


def fig_alignment():
    df = read("phase5/phase5_fig_B_resnet_reward_alignment.csv")
    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.color": "#e6e5e0", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))
    for a, s in zip(axes, ["cifar10_resnet8", "cifar100_resnet8"]):
        g = df[df.setting == s]
        for _, r in g.iterrows():
            c = r.condition if r.condition in COL5 else "Phase-4 F"
            sc = a.scatter(r.sparsity_median, r.test_drop_mean, s=40, color=COL5[c], marker="s" if c == "Phase-4 F" else "o", zorder=3)
            same(f"5.5 {s} {r.condition[:12]}", sc.get_offsets()[0], [r.sparsity_median, r.test_drop_mean])
            lab = "Phase-4 F (mean)" if c == "Phase-4 F" else f"{r.condition} (median)"
            off = {"C1": (6, 6), "C2": (6, -11), "C0": (6, 4)}.get(r.condition, (-8, 7))
            a.annotate(lab, (r.sparsity_median, r.test_drop_mean), textcoords="offset points", xytext=off, fontsize=7,
                       ha="right" if off[0] < 0 else "left")
        a.set_title(NAMES[s])
        a.set_xlabel("Selected-policy sparsity (%)")
        a.set_ylabel("Mean test accuracy drop vs dense (pp)")
        a.set_ylim(0, 1.2 * g.test_drop_mean.max())  # headroom for labels only; no point moved
    fig.text(0.5, 0.01, "x: C0, C1, C2 = median over 20 runs; Phase-4 F = mean over 20 runs (the only value recorded).",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


for name, fn, dpi in (("ch5_fig5_3_sensitivity_tolerance.png", fig_sensitivity, 300),
                      ("ch5_fig5_4_topk_reach_rate.png", fig_topk, 300),
                      ("ch5_fig5_5_resnet_reward_alignment.png", fig_alignment, 300)):
    f = fn()
    f.savefig(os.path.join(HERE, name), dpi=dpi, bbox_inches="tight")
    plt.close(f)

bad = [n for n, ok in checks if not ok]
print(f"{len(checks)} plotted-value checks, {len(checks) - len(bad)} identical to source CSV, {len(bad)} mismatched")
for n in bad:
    print("MISMATCH", n)
sys.exit(1 if bad else 0)
