"""Render Phase-4 figures A-E from the aggregated CSV files only (no hardcoded numbers)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import phase4_lib as L  # noqa: E402

OUT = L.RESULTS


def read(n):
    return pd.read_csv(os.path.join(OUT, n), float_precision="round_trip")


def grid(title):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle(title)
    return fig, dict(zip(L.SETTINGS, ax.flatten()))


def fig_a():
    df = read("phase4_fig_A_sparsity_control.csv")
    fig, a = plt.subplots(figsize=(13, 5))
    w = 0.2
    for i, r in enumerate(L.REWARDS):
        g = df[df.reward == r].set_index("setting").reindex(L.SETTINGS)
        x = np.arange(6) + i * w
        a.bar(x, g.sparsity_mean, width=w, yerr=g.sparsity_sd, capsize=2, label=f"{r} (PPO final, mean ± SD)")
        a.scatter(x, g.oracle_optimum_sparsity, marker="_", s=120, color="black", zorder=3)
    a.set_xticks(np.arange(6) + 1.5 * w, L.SETTINGS, fontsize=8)
    a.set_ylabel("final total sparsity (%)")
    a.set_title("A. Sparsity control by reward (black ticks: V_RL-landscape optimum of each reward)")
    a.legend(fontsize=7)
    return fig


def fig_b():
    df = read("phase4_fig_B_prior_exploration.csv")
    fig, ax = grid("B. Exploration under each prior: reward-rank score of sampled policies (mean over 20 seeds, 16-episode MA)")
    for s, a in ax.items():
        for p in L.PRIORS:
            g = df[(df.setting == s) & (df.prior == p)].sort_values("episode")
            a.plot(g.episode, g.rank_score_mean.rolling(16, min_periods=1).mean(), label=p, lw=1.2)
        a.set_title(s)
        a.set_xlabel("episode")
        a.set_ylabel("rank score (1 = best policy)")
    ax[L.SETTINGS[0]].legend(fontsize=7)
    return fig


def fig_c():
    df = read("phase4_fig_C_policy_rank_trajectory.csv")
    fig, ax = grid("C. Policy rank over training (median over seeds; dashed: best-so-far)")
    for s, a in ax.items():
        for v, col in (("S0", "#444444"), ("S2", "#1f77b4")):
            g = df[(df.setting == s) & (df.variant == v)].sort_values("episode")
            a.plot(g.episode, g.rank_median.rolling(16, min_periods=1).median(), color=col, label=f"{v} sampled")
            a.plot(g.episode, g.best_so_far_rank_median, color=col, ls="--", label=f"{v} best so far")
        a.set_yscale("log")
        a.invert_yaxis()
        a.set_title(s)
        a.set_xlabel("episode")
        a.set_ylabel("reward rank (log)")
    ax[L.SETTINGS[0]].legend(fontsize=7)
    return fig


def fig_d():
    df = read("phase4_fig_D_topk_reach_rate.csv")
    fig, ax = grid("D. Share of runs whose selected policy is in the top-k (bars) and best-seen top-20 (dots)")
    ks = [10, 20, 50, 100]
    for s, a in ax.items():
        g = df[df.setting == s].reset_index(drop=True)
        w = 0.8 / max(1, len(g))
        for i, r in g.iterrows():
            vals = [r.get(f"selected_reach_top{k}", np.nan) for k in ks]
            a.bar(np.arange(len(ks)) + i * w, vals, width=w, label=str(r.condition)[:24])
            a.scatter(1 + i * w, r.get("best_seen_reach_top20", np.nan), color="black", s=8, zorder=3)
        a.set_xticks(np.arange(len(ks)) + 0.4, [f"top {k}" for k in ks])
        a.set_ylim(0, 1.05)
        a.set_title(s)
    ax[L.SETTINGS[0]].legend(fontsize=6)
    return fig


def fig_e():
    df = read("phase4_fig_E_lamp_comparison.csv")
    fig, a = plt.subplots(figsize=(13, 5))
    w = 0.2
    for i, (pipe, m) in enumerate([("F", "lamp"), ("F", "global"), ("K", "lamp"), ("K", "global")]):
        g = df[(df.pipeline == pipe) & (df.method == m)].set_index("setting").reindex(L.SETTINGS)
        a.bar(np.arange(6) + i * w, g.ppo_minus_baseline_mean, width=w, label=f"{pipe} minus matched {m}")
    a.axhline(0, color="black", lw=0.8)
    a.set_xticks(np.arange(6) + 1.5 * w, L.SETTINGS, fontsize=8)
    a.set_ylabel("test accuracy difference (pp), identical zero count")
    a.set_title("E. Final pipeline F and control K vs LAMP and global magnitude at matched sparsity")
    a.legend(fontsize=7)
    return fig


def main():
    for name, fn in (("A_sparsity_control", fig_a), ("B_prior_exploration", fig_b), ("C_policy_rank_trajectory", fig_c),
                     ("D_topk_reach_rate", fig_d), ("E_lamp_comparison", fig_e)):
        fig = fn()
        fig.savefig(os.path.join(OUT, f"phase4_fig_{name}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("wrote", name)


if __name__ == "__main__":
    main()
