"""Paper 3 methodology flowchart (IEEE single column, 300 dpi).

  python experiments/paper3/paper_flowchart.py   -> results/paper3/paper_figures_final_v2/fig_methodology_flowchart.png
Two branches share one policy space: latency (blue, SimpleCNN-style slot colour) and accuracy recovery (orange); the
pre-registered latency-room check joins them. Box text states the protocol only, no outcomes.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "paper3", "paper_figures_final_v2")
BLUE, ORANGE, INK, INK2, GREY = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#9a9893"
FILL = {BLUE: "#eaf2fb", ORANGE: "#fdefe9", INK2: "#f3f2ef"}
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 300, "text.color": INK})

W = 3.5                     # inches; every coordinate below is in inches
BOX_H, ROW_GAP, EDGE = 0.5, 0.13, 0.03
H = 5 * BOX_H + 4 * ROW_GAP + 2 * EDGE
COL_GAP = 0.12


def box(ax, x, y, w, h, title, body, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", linewidth=0.9,
                                edgecolor=edge, facecolor=FILL[edge]))
    ax.text(x + w / 2, y + h - 0.06, title, ha="center", va="top", fontsize=7.6, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.19, body, ha="center", va="top", fontsize=6.6, color=INK2, linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, linewidth=0.8, color=GREY,
                                 shrinkA=0, shrinkB=0))


def main():
    os.makedirs(OUT, exist_ok=True)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    L, R = EDGE, W - EDGE
    half = (R - L - COL_GAP) / 2
    xl, xr = L, L + half + COL_GAP
    y = [H - EDGE - BOX_H - i * (BOX_H + ROW_GAP) for i in range(5)]     # box bottoms, top to bottom

    b1 = box(ax, L, y[0], R - L, BOX_H, "1  Dense reference models",
             "SimpleCNN and ResNet-8 on CIFAR-10 and CIFAR-100\n45,000 train / 2,500 VAL-RL images; test set unused",
             INK2)
    b2 = box(ax, L, y[1], R - L, BOX_H, "2  Structured policy space",
             "4 pruning units × 6 actions (0–60%) = 1,296 policies,\neach physically compacted (L1 ranking)",
             INK2)
    b3l = box(ax, xl, y[2], half, BOX_H, "3a  Latency tables",
              "batch 1, one thread, 5 sessions\nall 4 × 1,296 shapes", BLUE)
    b3r = box(ax, xr, y[2], half, BOX_H, "3b  Proxy validation",
              "1 vs 5 recovery epochs\naccept if Spearman ρ ≥ 0.90", ORANGE)
    b4l = box(ax, xl, y[3], half, BOX_H, "4a  Cost-metric analysis",
              "parameters, MACs vs latency\nρ, inversions, linear fit", BLUE)
    b4r = box(ax, xr, y[3], half, BOX_H, "4b  Boundary calibration",
              "48 ResNet-8 policies, 5 epochs\nq₅ ≥ 0.98 → proxy threshold τ", ORANGE)
    b5 = box(ax, L, y[4], R - L, BOX_H, "5  Pre-registered latency-room check",
             "≥ 20 policies with q₁ ≥ τ and ≥ 1 of them at ≥ 1.10× speed-up\n"
             "→ GO / NO-GO for a hardware-aware search", INK2)

    def bottom(b, f=0.5): return (b[0] + f * b[2], b[1])
    def top(b, f=0.5): return (b[0] + f * b[2], b[1] + b[3])
    arrow(ax, bottom(b1), top(b2))
    arrow(ax, bottom(b2, 0.25), top(b3l))
    arrow(ax, bottom(b2, 0.75), top(b3r))
    arrow(ax, bottom(b3l), top(b4l))
    arrow(ax, bottom(b3r), top(b4r))
    arrow(ax, bottom(b4l), top(b5, 0.25))
    arrow(ax, bottom(b4r), top(b5, 0.75))
    path = os.path.join(OUT, "fig_methodology_flowchart.png")
    fig.savefig(path, facecolor="white")
    print("wrote", path, f"({W} x {H:.2f} in)")


if __name__ == "__main__":
    main()
