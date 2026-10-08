"""Phase 6 figures 1-5 from results/phase6/figures/*.csv only (IEEE double-column width; no hardcoded numbers).

Colour follows the method in every figure (validated categorical slots 1-3): PPO blue, LAMP orange, global aqua;
the dense control is a neutral grey. Pre-fine-tuning marks are hollow, post-fine-tuning marks are filled.
"""
import os

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import phase6_lib as L  # noqa: E402

FIG = os.path.join(L.RESULTS, "figures")
COL = {"ppo": "#2a78d6", "lamp": "#eb6834", "global": "#1baf7a", "dense": "#52514e"}
LAB = {"ppo": "PPO (C1)", "lamp": "LAMP", "global": "Global magnitude", "dense": "Dense control"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
NAMES = {"cifar10_simplecnn": "C10 SimpleCNN", "cifar10_lenet5": "C10 LeNet-5", "cifar10_resnet8": "C10 ResNet-8",
         "cifar100_simplecnn": "C100 SimpleCNN", "cifar100_lenet5": "C100 LeNet-5", "cifar100_resnet8": "C100 ResNet-8"}
T975 = 2.093024                         # Student t, df = 19 (20 paired runs)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "font.family": "DejaVu Sans"})


def read(n):
    return pd.read_csv(os.path.join(FIG, n), float_precision="round_trip")


def panels(h=4.4):
    fig, ax = plt.subplots(2, 3, figsize=(7.16, h))
    return fig, dict(zip(L.SETTINGS, np.ravel(ax)))


def legend_below(fig, handles, labels, ncol):
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))


def fig1():
    df = read("phase6_fig1_recovery.csv")
    fig, ax = panels()
    for s, a in ax.items():
        g = df[df.setting == s].set_index("method")
        for i, m in enumerate(L.METHODS):
            r = g.loc[m]
            ci = lambda sd: T975 * sd / np.sqrt(r.n)  # noqa: E731
            a.plot([i, i], [r.test_pre_mean, r.test_post_mean], color=COL[m], lw=1.5, zorder=2)
            a.errorbar(i, r.test_pre_mean, yerr=ci(r.test_pre_sd), fmt="o", mfc="white", mec=COL[m], ecolor=COL[m], ms=5, capsize=2, zorder=3)
            a.errorbar(i, r.test_post_mean, yerr=ci(r.test_post_sd), fmt="o", color=COL[m], ms=5, capsize=2, zorder=3)
        a.axhline(g.dense_test.iloc[0], color=INK2, lw=0.8, ls="--")
        if "dense" in g.index:
            a.axhline(g.loc["dense"].test_post_mean, color=INK2, lw=0.8, ls=":")
        a.set_xticks(range(3), ["PPO", "LAMP", "Global"])
        a.set_xlim(-0.5, 2.5)
        a.set_title(NAMES[s])
        a.set_ylabel("Test accuracy (%)")
    h = [plt.Line2D([], [], marker="o", mfc="white", mec=INK, ls="", ms=5), plt.Line2D([], [], marker="o", color=INK, ls="", ms=5),
         plt.Line2D([], [], color=INK2, ls="--", lw=0.8), plt.Line2D([], [], color=INK2, ls=":", lw=0.8)]
    legend_below(fig, h, ["Before fine-tuning", "After fine-tuning (mean, 95% CI)", "Dense reference", "Dense, fine-tuned"], 4)
    return fig


def fig2():
    df = read("phase6_fig2_ppo_vs_lamp.csv")
    fig, ax = panels()
    for s, a in ax.items():
        g = df[df.setting == s]
        for _, r in g.iterrows():
            a.plot([0, 1], [r.gap_pre, r.gap_post], color="#9a9893", lw=0.6, alpha=0.7, zorder=1)
        m = [g.gap_pre.mean(), g.gap_post.mean()]
        ci = [T975 * g.gap_pre.std(ddof=1) / np.sqrt(len(g)), T975 * g.gap_post.std(ddof=1) / np.sqrt(len(g))]
        a.errorbar([0, 1], m, yerr=ci, color=COL["ppo"], marker="o", ms=5, lw=1.8, capsize=3, zorder=3)
        a.axhline(0, color=INK2, lw=0.8, ls="--")
        a.set_xticks([0, 1], ["Before", "After"])
        a.set_xlim(-0.3, 1.3)
        a.set_title(NAMES[s])
        a.set_ylabel("PPO − LAMP test accuracy (pp)")
    h = [plt.Line2D([], [], color="#9a9893", lw=0.6), plt.Line2D([], [], color=COL["ppo"], marker="o", ms=5, lw=1.8)]
    legend_below(fig, h, ["One paired run (same sparsity, seed and data order)", "Mean (95% CI)"], 2)
    return fig


def fig3():
    df = read("phase6_fig3_recovery_curves.csv")
    fig, ax = panels()
    for s, a in ax.items():
        for m in ("dense", "global", "lamp", "ppo"):
            g = df[(df.setting == s) & (df.method == m)].sort_values("epoch")
            a.plot(g.epoch, g.val_select_mean, color=COL[m], lw=1.5, ls="--" if m == "dense" else "-", marker="o", ms=3, label=LAB[m])
            if m != "dense":
                a.fill_between(g.epoch, g.val_select_mean - g.val_select_sd, g.val_select_mean + g.val_select_sd, color=COL[m], alpha=0.15, lw=0)
        a.set_title(NAMES[s])
        a.set_xlabel("Fine-tuning epoch")
        a.set_ylabel("VAL-SELECT accuracy (%)")
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    h, lab = ax[L.SETTINGS[0]].get_legend_handles_labels()
    legend_below(fig, h[::-1], lab[::-1], 4)
    return fig


def fig4():
    df = read("phase6_fig4_accuracy_sparsity.csv")
    fig, ax = panels()
    for s, a in ax.items():
        g = df[df.setting == s]
        for m, mk in (("global", "s"), ("lamp", "^"), ("ppo", "o")):
            q = g[g.method == m]
            a.scatter(q.sparsity_pre, q.test_pre, s=14, marker=mk, facecolors="none", edgecolors=COL[m], linewidths=0.8, zorder=2)
            a.scatter(q.sparsity_pre, q.test_post, s=14, marker=mk, color=COL[m], zorder=3, label=LAB[m])
        a.axhline(g.dense_test.iloc[0], color=INK2, lw=0.8, ls="--")
        a.set_title(NAMES[s])
        a.set_xlabel("Total sparsity (%) (identical for the three methods)")
        a.set_ylabel("Test accuracy (%)")
    h, lab = ax[L.SETTINGS[0]].get_legend_handles_labels()
    h += [plt.Line2D([], [], marker="o", mfc="white", mec=INK, ls="", ms=4), plt.Line2D([], [], color=INK2, ls="--", lw=0.8)]
    legend_below(fig, h, lab + ["Hollow: before fine-tuning", "Dense reference"], 5)
    return fig


def fig5():
    df = read("phase6_fig5_cost.csv")
    df = df[df.method != "dense"]
    fig, a = plt.subplots(figsize=(7.16, 3.0))
    for m, mk in (("global", "s"), ("lamp", "^"), ("ppo", "o")):
        g = df[df.method == m]
        a.errorbar(g.finetune_minutes, g.recovery_mean, yerr=g.recovery_sd, fmt=mk, color=COL[m], ms=5, capsize=2, label=LAB[m], lw=0.8)
    for s in L.SETTINGS:
        g = df[df.setting == s]
        a.annotate(NAMES[s], (g.finetune_minutes.mean(), g.recovery_mean.max()), textcoords="offset points", xytext=(0, 7), ha="center",
                   fontsize=7, color=INK2)
    a.axhline(0, color=INK2, lw=0.8, ls="--")
    a.set_xlabel("Fine-tuning time per model (CPU minutes, 1 thread)")
    a.set_ylabel("Test accuracy recovery (pp, mean ± SD)")
    a.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    return fig


def main():
    for name, fn in (("fig1_recovery", fig1), ("fig2_ppo_vs_lamp", fig2), ("fig3_recovery_curves", fig3), ("fig4_accuracy_sparsity", fig4),
                     ("fig5_cost", fig5)):
        f = fn()
        f.savefig(os.path.join(FIG, f"phase6_{name}.png"), bbox_inches="tight")
        plt.close(f)
        print("wrote", name)


if __name__ == "__main__":
    main()
