"""Render the Phase-3 figures A-F from the aggregated CSV files only (no hardcoded numbers).

  python experiments/phase3/make_figures.py   ->  results/architecture_generalization/phase3_figures/*.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import phase3_common as C  # noqa: E402

OUT = os.path.join(C.AG, "phase3_figures")
COND_COLORS = {"P0": "#444444", "P1": "#1f77b4", "P2": "#ff7f0e", "P3": "#2ca02c"}
METHOD_COLORS = {"uniform": "#d62728", "global": "#9467bd", "lamp": "#8c564b", "erk": "#e377c2", "random": "#7f7f7f"}


def read(name):
    return pd.read_csv(os.path.join(C.AG, name), float_precision="round_trip")


def grid_axes(title):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle(title)
    return fig, dict(zip(C.SETTINGS, axes.flatten()))


def fig_a():
    df = read("phase3_fig_A_accuracy_sparsity.csv")
    fig, ax = grid_axes("A. Test accuracy vs total sparsity (PPO final models and baseline grid)")
    for s, a in ax.items():
        g = df[df.setting == s]
        for m, col in METHOD_COLORS.items():
            h = g[g.method == m].sort_values("sparsity")
            a.plot(h.sparsity, h.test_accuracy, "-o", ms=3, color=col, label=m)
        for c, col in COND_COLORS.items():
            h = g[g.method == f"PPO {c}"]
            a.scatter(h.sparsity, h.test_accuracy, s=14, color=col, alpha=0.6, label=f"PPO {c}")
        dn = g[g.method == "dense"]
        a.axhline(dn.test_accuracy.iloc[0], color="black", ls=":", lw=1, label="dense")
        a.set_title(s)
        a.set_xlabel("total sparsity (%)")
        a.set_ylabel("test accuracy (%)")
    ax[C.SETTINGS[0]].legend(fontsize=7, ncol=2)
    return fig


def fig_b():
    df = read("phase3_fig_B_policy_heatmap.csv")
    fig, axes = plt.subplots(len(C.SETTINGS), 4, figsize=(12, 16))
    fig.suptitle("B. Final-policy pruning ratio per unit (count of 20 seeds)")
    for i, s in enumerate(C.SETTINGS):
        for j, c in enumerate(C.CONDITIONS):
            g = df[(df.setting == s) & (df.condition == c)]
            mat = g.pivot(index="unit_index", columns="ratio", values="count").sort_index()
            a = axes[i, j]
            a.imshow(mat.to_numpy(), cmap="Blues", vmin=0, vmax=20, aspect="auto")
            a.set_xticks(range(len(mat.columns)), [str(x) for x in mat.columns], fontsize=7)
            units = g.drop_duplicates("unit_index").sort_values("unit_index").unit.tolist()
            a.set_yticks(range(len(units)), units, fontsize=7)
            for (y, x), v in np.ndenumerate(mat.to_numpy()):
                if v:
                    a.text(x, y, str(v), ha="center", va="center", fontsize=6)
            if i == 0:
                a.set_title(c)
            if j == 0:
                a.set_ylabel(s, fontsize=8)
    fig.tight_layout()
    return fig


def fig_c():
    df = read("phase3_fig_C_exploration.csv")
    fig, ax = grid_axes("C. Exploration: V_RL accuracy of sampled policies (mean over 20 seeds, 16-episode moving average)")
    for s, a in ax.items():
        for c, col in COND_COLORS.items():
            g = df[(df.setting == s) & (df.condition == c)].sort_values("episode")
            a.plot(g.episode, g.vrl_accuracy_mean.rolling(16, min_periods=1).mean(), color=col, label=c)
        a.set_title(s)
        a.set_xlabel("episode")
        a.set_ylabel("V_RL accuracy (%)")
    ax[C.SETTINGS[0]].legend()
    return fig


def fig_d():
    df = read("phase3_fig_D_sensitivity_tolerance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("D. Unit sensitivity vs pruning tolerance and PPO allocation")
    for s in C.SETTINGS:
        g = df[df.setting == s]
        axes[0].scatter(g.normalized_sensitivity, g.tolerance_ratio_1pp, label=s)
        axes[1].scatter(g.normalized_sensitivity, g.mean_ratio_P0, marker="o", label=f"{s} P0")
        axes[1].scatter(g.normalized_sensitivity, g.mean_ratio_P1, marker="x")
    axes[0].set_xlabel("normalised V_RL loss sensitivity")
    axes[0].set_ylabel("largest single-unit ratio with V_RL drop <= 1 pp (%)")
    axes[1].set_xlabel("normalised V_RL loss sensitivity")
    axes[1].set_ylabel("mean final ratio (o: P0, x: P1) (%)")
    axes[0].legend(fontsize=7)
    return fig


def fig_e():
    df = read("phase3_fig_E_landscape.csv")
    fig, ax = grid_axes("E. Exact V_RL policy landscapes (1,296 policies; colour = reward rank; black = Pareto; stars = final P0/P1)")
    for s, a in ax.items():
        g = df[df.setting == s]
        sc = a.scatter(g.total_sparsity, g.vrl_accuracy, c=g.reward_rank, cmap="viridis_r", s=6)
        p = g[g.pareto]
        a.scatter(p.total_sparsity, p.vrl_accuracy, facecolors="none", edgecolors="black", s=14, lw=0.6)
        for c, col in (("P0", COND_COLORS["P0"]), ("P1", COND_COLORS["P1"])):
            h = g[g[f"final_{c}"] > 0]
            a.scatter(h.total_sparsity, h.vrl_accuracy, marker="*", s=40 + 8 * h[f"final_{c}"], color=col, label=c)
        a.set_title(s)
        a.set_xlabel("total sparsity (%)")
        a.set_ylabel("V_RL accuracy (%)")
        fig.colorbar(sc, ax=a, fraction=0.04)
    ax[C.SETTINGS[0]].legend()
    return fig


def fig_f():
    df = read("phase3_fig_F_retention.csv")
    fig, a = plt.subplots(figsize=(14, 5))
    methods = [m for m in df.method.unique() if m != "dense"]
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        g = df[df.method == m].set_index("setting").reindex(C.SETTINGS)
        a.bar(np.arange(len(C.SETTINGS)) + i * w, g.retention_mean, width=w, label=m)
    a.set_xticks(np.arange(len(C.SETTINGS)) + 0.4 - w / 2, C.SETTINGS, fontsize=8)
    a.set_ylabel("test accuracy retention vs dense (%)")
    a.set_title("F. Accuracy retention across architectures and datasets (baselines matched to P0 sparsity)")
    a.legend(fontsize=7, ncol=3)
    lo = np.nanmin(df.retention_mean)
    a.set_ylim(max(0, lo - 5), 102)
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("A_accuracy_sparsity", fig_a), ("B_policy_heatmap", fig_b), ("C_exploration", fig_c),
                     ("D_sensitivity_tolerance", fig_d), ("E_landscape", fig_e), ("F_retention", fig_f)):
        fig = fn()
        fig.savefig(os.path.join(OUT, f"phase3_fig_{name}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("wrote", name)


if __name__ == "__main__":
    main()
