"""Phase 5 figures A-E from the aggregated CSVs only (IEEE double-column width; no hardcoded numbers).

Colour follows the condition in every figure (validated categorical slots 1-3, all-pairs CVD dE >= 9.2):
C1 blue, C0 orange, C2 aqua. Baselines are neutral greys with distinct marker shapes (identity never colour-alone).
"""
import os

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import phase5_lib as L  # noqa: E402

OUT = L.RESULTS
COL = {"C1": "#2a78d6", "C0": "#eb6834", "C2": "#1baf7a", "LAMP": "#52514e", "global": "#9a9893", "Phase-4 F": "#0b0b0b"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
NAMES = {"cifar10_simplecnn": "C10 SimpleCNN", "cifar10_lenet5": "C10 LeNet-5", "cifar10_resnet8": "C10 ResNet-8",
         "cifar100_simplecnn": "C100 SimpleCNN", "cifar100_lenet5": "C100 LeNet-5", "cifar100_resnet8": "C100 ResNet-8"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "font.family": "DejaVu Sans"})


def read(n):
    return pd.read_csv(os.path.join(OUT, n), float_precision="round_trip")


def panels(rows=2, cols=3, h=4.4):
    fig, ax = plt.subplots(rows, cols, figsize=(7.16, h))
    return fig, dict(zip(L.SETTINGS, np.ravel(ax)))


def fig_a():
    df = read("phase5_fig_A_accuracy_sparsity.csv")
    fig, ax = panels()
    for s, a in ax.items():
        g = df[df.setting == s]
        for c, mk in (("C0", "o"), ("C1", "o")):
            h = g[g.condition == c]
            a.scatter(h.selected_sparsity, h.test_accuracy, s=14, color=COL[c], marker=mk, alpha=0.8, label=c, zorder=3,
                      edgecolors="white", linewidths=0.4)
        a.scatter(g.selected_sparsity, g.lamp_test_accuracy, s=14, color=COL["LAMP"], marker="^", label="LAMP (matched)", zorder=2)
        a.scatter(g.selected_sparsity, g.global_test_accuracy, s=14, color=COL["global"], marker="x", label="Global (matched)", zorder=2,
                  linewidths=0.8)
        a.axhline(g.dense_test_accuracy.iloc[0], color=INK2, lw=0.8, ls="--", label="Dense")
        a.set_title(NAMES[s])
        a.set_xlabel("Total sparsity (%)")
        a.set_ylabel("Test accuracy (%)")
    h, lab = ax[L.SETTINGS[0]].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def fig_b():
    df = read("phase5_fig_B_resnet_reward_alignment.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6))
    for a, s in zip(axes, ["cifar10_resnet8", "cifar100_resnet8"]):
        g = df[df.setting == s]
        for _, r in g.iterrows():
            c = r.condition if r.condition in COL else "Phase-4 F"
            a.scatter(r.sparsity_median, r.test_drop_mean, s=40, color=COL[c], marker="s" if c == "Phase-4 F" else "o", zorder=3)
            off = {"C1": (6, 6), "C2": (6, -11), "C0": (6, 4)}.get(r.condition, (-8, 7))
            a.annotate(str(r.condition).replace(" (reference)", ""), (r.sparsity_median, r.test_drop_mean), textcoords="offset points",
                       xytext=off, fontsize=7, color=INK, ha="right" if off[0] < 0 else "left")
        a.set_title(NAMES[s])
        a.set_xlabel("Selected-policy sparsity, median (%)")
        a.set_ylabel("Test accuracy drop vs dense (pp)")
        a.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def fig_c():
    df = read("phase5_fig_C_policy_rank.csv")
    fig, ax = panels(h=4.2)
    for s, a in ax.items():
        g = df[df.setting == s]
        data = [g[g.condition == c].selected_rank.to_numpy() for c in L.CONDITIONS]
        bp = a.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops={"color": INK, "lw": 1.2})
        for patch, c in zip(bp["boxes"], L.CONDITIONS):
            patch.set_facecolor(COL[c])
            patch.set_alpha(0.55)
            patch.set_edgecolor(COL[c])
        for i, c in enumerate(L.CONDITIONS, 1):
            v = g[g.condition == c].selected_rank.to_numpy()
            a.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=6, color=COL[c], zorder=3)
        a.set_yscale("log")
        a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.set_xticks([1, 2, 3], L.CONDITIONS)
        a.axhline(20, color=INK2, lw=0.8, ls=":")
        a.set_title(NAMES[s])
        a.set_ylabel("Constrained rank (log; 1 = oracle)")
    fig.tight_layout()
    return fig


def fig_d():
    df = read("phase5_fig_D_sensitivity_exploration.csv")
    fig, ax = panels()
    for s, a in ax.items():
        for c in ("C2", "C1"):
            g = df[(df.setting == s) & (df.condition == c)].sort_values("episode")
            a.plot(g.episode, g.rank_score_mean.rolling(16, min_periods=1).mean(), color=COL[c], lw=1.5,
                   label=f"{c} ({'centred prior' if c == 'C1' else 'no prior'})")
        a.set_title(NAMES[s])
        a.set_xlabel("Episode")
        a.set_ylabel("Rank score of sampled policy")
    h, lab = ax[L.SETTINGS[0]].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def fig_e():
    df = read("phase5_fig_E_feasible_sparsity_gap.csv")
    fig, a = plt.subplots(figsize=(7.16, 2.8))
    w = 0.24
    x = np.arange(len(L.SETTINGS))
    for i, c in enumerate(L.CONDITIONS):
        g = df[df.condition == c].set_index("setting").reindex(L.SETTINGS)
        a.bar(x + (i - 1) * w, g.sparsity_median, width=w - 0.02, color=COL[c], label=f"{c} selected (median)")
    mf = df[df.condition == "C1"].set_index("setting").reindex(L.SETTINGS).max_feasible_sparsity
    a.scatter(x, mf, marker="_", s=260, color=INK, zorder=3, label="Max feasible sparsity (VAL-RL, q >= tau)", linewidths=1.6)
    a.set_xticks(x, [NAMES[s] for s in L.SETTINGS])
    a.set_ylabel("Total sparsity (%)")
    a.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    return fig


def main():
    for name, fn in (("A_accuracy_sparsity", fig_a), ("B_resnet_reward_alignment", fig_b), ("C_policy_rank", fig_c),
                     ("D_sensitivity_exploration", fig_d), ("E_feasible_sparsity_gap", fig_e)):
        fig = fn()
        fig.savefig(os.path.join(OUT, f"phase5_fig_{name}.png"), bbox_inches="tight")
        plt.close(fig)
        print("wrote", name)


if __name__ == "__main__":
    main()
