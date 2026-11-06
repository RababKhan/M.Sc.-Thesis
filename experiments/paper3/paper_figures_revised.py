"""Paper 3 revised-manuscript figures (larger type), from frozen result files only. IEEE single column, 300 dpi.

  python experiments/paper3/paper_figures_revised.py   -> results/paper3/paper_figures_revised/fig{1,2,3}_*.png
Colour follows the architecture in every figure (validated categorical slots): SimpleCNN blue, ResNet-8 orange;
grey for reference marks. Every series also has its own marker shape, so the figures survive greyscale print.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "paper3")
OUT = os.path.join(RES, "paper_figures_revised")
BLUE, ORANGE, INK, INK2, GREY, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#9a9893", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "font.family": "DejaVu Sans"})


def rho(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def fig1():
    s = pd.read_csv(os.path.join(RES, "h2_latency_table_simplecnn_c10.csv"))
    r = pd.read_csv(os.path.join(RES, "h2_latency_table_resnet8_c10.csv"))
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.7), sharey=True)
    for ax, col, label in ((axes[0], "param_reduction", "Physical parameter reduction (%)"),
                           (axes[1], "mac_reduction", "MAC reduction (%)")):
        for t, c, m, name in ((s, BLUE, "o", "SimpleCNN"), (r, ORANGE, "^", "ResNet-8")):
            ax.scatter(100 * t[col], 100 * t.latency_gain, s=5, marker=m, color=c, alpha=0.45, linewidths=0,
                       label=f"{name} (Spearman ρ = {rho(t[col], t.latency_gain):.2f})")
        ax.set_xlabel(label)
        ax.set_ylabel("Latency reduction (%)")
        ax.set_xlim(-2, 88)
        ax.set_ylim(-3, 82)
        leg = ax.legend(loc="upper left", frameon=False, markerscale=2.2, handletextpad=0.3, borderaxespad=0.2)
        for h in leg.legend_handles:
            h.set_alpha(1)
    axes[0].set_title("(a) against parameter reduction", fontsize=9, loc="left")
    axes[1].set_title("(b) against MAC reduction", fontsize=9, loc="left")
    fig.tight_layout(h_pad=1.2)
    return fig


def fig2():
    v = pd.read_csv(os.path.join(RES, "paper3_proxy_validation.csv"))
    s, r = v[v.setting == "cifar10_simplecnn"], v[v.setting == "cifar10_resnet8"]
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    lo, hi = 35, 82
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.8, ls="--", zorder=1)
    ax.text(41, 39.2, "A₁ = A₅", color=INK2, fontsize=8, rotation=38)
    sp, sr = s[s.source_5epoch == "pilot"], s[s.source_5epoch != "pilot"]
    ax.scatter(sp.A5, sp.A1, s=12, marker="o", facecolors="none", edgecolors=BLUE, linewidths=0.7, zorder=4,
               label="SimpleCNN, pilot policies (126)")
    ax.scatter(sr.A5, sr.A1, s=12, marker="o", color=BLUE, zorder=3, label="SimpleCNN, random policies (30)")
    ax.scatter(r.A5, r.A1, s=14, marker="^", color=ORANGE, zorder=3, label="ResNet-8, identity + random (31)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("VAL-RL accuracy after 5-epoch recovery, A₅ (%)")
    ax.set_ylabel("VAL-RL accuracy after 1 epoch, A₁ (%)")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none", handletextpad=0.3, borderpad=0.3)
    ax.text(0.98, 0.04, f"Spearman ρ: SimpleCNN {rho(s.A1, s.A5):.3f}, ResNet-8 {rho(r.A1, r.A5):.3f}",
            transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    fig.tight_layout()
    return fig


def fig3():
    land = pd.read_csv(os.path.join(RES, "h2_c10_resnet8_1epoch_landscape.csv"))
    cal = pd.read_csv(os.path.join(RES, "resnet8_threshold_calibration.csv")).merge(land[["policy", "speedup"]], on="policy")
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.scatter(land.q1, land.speedup, s=4, color=GREY, alpha=0.5, linewidths=0, label="Proxy q₁, all 1,296")
    ax.scatter(cal.q5, cal.speedup, s=14, marker="^", color=ORANGE, zorder=3, label="5-epoch q₅, 48 calibration")
    ax.axhline(1.10, color=INK2, lw=0.8, ls="--")
    ax.axvline(0.98, color=INK2, lw=0.8, ls=":")
    ax.text(0.976, 1.47, "q = 0.98", color=INK2, fontsize=8, ha="right")
    ax.text(0.505, 1.108, "1.10× latency-room criterion", color=INK2, fontsize=8)
    ax.set_xlim(0.48, 1.02)
    ax.set_xlabel("Retained validation accuracy q (after recovery)")
    ax.set_ylabel("Speed-up vs dense (batch 1, 1 thread)")
    ax.legend(loc="upper right", frameon=False, handletextpad=0.3, borderaxespad=0.1, bbox_to_anchor=(0.9, 1.0))
    fig.tight_layout()
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("fig1_latency_vs_reduction", fig1), ("fig2_proxy_validation", fig2), ("fig3_resnet8_latency_room", fig3)):
        f = fn()
        f.savefig(os.path.join(OUT, f"{name}.png"), bbox_inches="tight")
        plt.close(f)
        print("wrote", name)


if __name__ == "__main__":
    main()
