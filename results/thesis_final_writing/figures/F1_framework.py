"""F1 — Overall thesis framework (editable source).

  python results/thesis_final_writing/figures/F1_framework.py
Writes F1_framework.{png,pdf,svg} next to this script. The SVG/PDF are vector and editable (e.g. Inkscape); edit the
BOX lists below to change wording. The figure separates the core framework (Phases 2–6) from the later deployment and
sustainability extension (Paper 3, Phases 7A–7B), and tags every core component with the phase that introduced it, so
that components added after diagnostic results are not presented as part of the original design.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2, MUTED = "#1f1f1d", "#52514e", "#8a8984"
ORIG = ("#e6eefa", "#2a6fc9")       # original design (Phases 2–3)
ADDED = ("#efe7f8", "#7d4cb8")      # added after diagnostic results (Phases 4–6)
EXT = ("#fcebe0", "#c8571f")        # deployment / sustainability extension
PANEL_CORE, PANEL_EXT = "#f6f8fc", "#fdf7f2"

# (text, subtitle, phase tag, style)
CORE = [
    ("Dense CNN", "SimpleCNN · LeNet-5 · ResNet-8; CIFAR-10/100", "Phase 2", ORIG),
    ("Layer sensitivity analysis", "loss / accuracy change per unit and ratio (VAL-RL)", "Phase 3", ORIG),
    ("Sensitivity-aware PPO policy", "soft prior on action logits; centred S_l − S̄ from Phase 4", "Phases 3–4", ADDED),
    ("Layer-wise pruning actions", "ratio r_l ∈ {0, .1, .2, .3, .4, .6} per unit", "Phase 3", ORIG),
    ("Accuracy / compression reward", "R0: accuracy–sparsity trade-off", "Phase 3", ORIG),
    ("Policy search (PPO)", "2,048 timesteps per run", "Phase 3", ORIG),
    ("Archive selection", "top-10 visited policies, re-ranked on a held-out split", "Phase 4", ADDED),
    ("Accuracy-constrained reward / selection", "feasible only if q ≥ τ = 0.98", "Phase 5", ADDED),
    ("Final pruned policy", "unstructured magnitude masks per unit", "Phases 3–5", ORIG),
    ("Equal recovery / fine-tuning", "identical 5-epoch fixed-mask fine-tuning", "Phase 6", ADDED),
    ("Accuracy–compression evaluation", "test once per frozen model; matched-sparsity baselines", "Phases 3–6", ORIG),
]
EXTENSION = [
    ("Physical structured pruning", "channels kept: max(1, round((1 − r)·C))", "Paper 3, 7A, 7B"),
    ("Memory measurement", "parameter bytes; serialised size", "7A, 7B"),
    ("MAC / computation measurement", "closed-form MACs, checked by hooks", "Paper 3, 7A, 7B"),
    ("CPU latency analysis", "1,296 structured shapes; batch 1, 1 thread", "Paper 3, 7A"),
    ("Energy evaluation", "RAPL package energy, J per 1,000 inferences", "7A measured; 7B unresolved"),
    ("Deployment trade-off analysis", "accuracy vs memory, MACs, latency, energy", "7A, 7B"),
]

FIG_W, FIG_H = 7.6, 9.9
BW, BH, GAP = 2.95, 0.52, 0.17
CX, EX = 0.6, 4.3
TOP = FIG_H - 0.95


def box(ax, x, y, w, h, title, sub, style, tag=None):
    fill, edge = style
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.06", fc=fill, ec=edge, lw=1.0, zorder=3))
    ax.text(x + 0.09, y + h - 0.17, title, ha="left", va="center", fontsize=7.6, weight="bold", color=INK, zorder=4)
    ax.text(x + 0.09, y + 0.15, sub, ha="left", va="center", fontsize=6.0, color=INK2, zorder=4)
    if tag:
        ax.text(x + w - 0.07, y + h - 0.17, tag, ha="right", va="center", fontsize=5.6, color=edge, style="italic", zorder=4)


def arrow(ax, p, q, color=INK2, style="-|>", ls="-", rad=0.0, lw=0.9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=2, shrinkA=1, shrinkB=1))


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    core_h = len(CORE) * (BH + GAP) + 0.35
    ext_h = len(EXTENSION) * (BH + GAP) + 0.35
    ax.add_patch(FancyBboxPatch((CX - 0.18, TOP - core_h + 0.1), BW + 0.36, core_h + 0.32, boxstyle="round,pad=0,rounding_size=0.1",
                                fc=PANEL_CORE, ec="#c9d6ec", lw=0.8, zorder=1))
    ax.add_patch(FancyBboxPatch((EX - 0.18, TOP - ext_h + 0.1), BW + 0.36, ext_h + 0.32, boxstyle="round,pad=0,rounding_size=0.1",
                                fc=PANEL_EXT, ec="#ecd3c1", lw=0.8, ls=(0, (4, 2)), zorder=1))
    ax.text(CX + BW / 2, TOP + 0.27, "CORE THESIS FRAMEWORK", ha="center", fontsize=8.6, weight="bold", color=ORIG[1])
    ax.text(CX + BW / 2, TOP + 0.08, "Phases 2–6 · unstructured masks · RQ1–RQ5", ha="center", fontsize=6.4, color=INK2)
    ax.text(EX + BW / 2, TOP + 0.27, "DEPLOYMENT / SUSTAINABILITY EXTENSION", ha="center", fontsize=8.6, weight="bold", color=EXT[1])
    ax.text(EX + BW / 2, TOP + 0.08, "Paper 3, Phases 7A–7B · added after the core · RQ6", ha="center", fontsize=6.4, color=INK2)

    ys = []
    for i, (t, s, tag, st) in enumerate(CORE):
        y = TOP - 0.25 - (i + 1) * (BH + GAP) + GAP
        ys.append(y)
        box(ax, CX, y, BW, BH, t, s, st, tag)
        if i:
            arrow(ax, (CX + BW / 2, ys[i - 1]), (CX + BW / 2, y + BH))
    # PPO learning loop: reward -> state (inside the search)
    s_i, r_i = 2, 4
    arrow(ax, (CX - 0.02, ys[r_i] + BH / 2), (CX - 0.02, ys[s_i] + BH / 2), color=ORIG[1], rad=-0.55, lw=0.8)
    ax.text(CX - 0.27, (ys[r_i] + ys[s_i] + BH) / 2, "PPO\nupdate", ha="center", va="center", fontsize=5.6, color=ORIG[1], rotation=90)

    eys = []
    for i, (t, s, tag) in enumerate(EXTENSION):
        y = TOP - 0.25 - (i + 1) * (BH + GAP) + GAP
        eys.append(y)
        box(ax, EX, y, BW, BH, t, s, EXT, tag)
        if i:
            arrow(ax, (EX + BW / 2, eys[i - 1]), (EX + BW / 2, y + BH))
    # hand-over: learned layer-wise allocation -> structured realisation, routed through the gap between the panels
    gx = (CX + BW + EX) / 2
    y0, y1 = ys[8] + BH / 2, eys[0] + BH / 2
    ax.plot([CX + BW + 0.02, gx, gx], [y0, y0, y1], color=EXT[1], lw=1.0, ls=(0, (3, 2)), zorder=2)
    arrow(ax, (gx, y1), (EX, y1), color=EXT[1], lw=1.0)
    ax.text(gx, (y0 + y1) / 2, "learned layer-wise allocation\n7A: transferred from Phase 5  ·  7B: learned directly as structured actions",
            ha="center", va="center", rotation=90, fontsize=5.5, color=EXT[1], linespacing=1.5,
            bbox=dict(fc="white", ec="none", pad=0.8), zorder=3)

    # outcomes under the extension (frozen verdicts, so the figure cannot be read as a success claim)
    oy = eys[-1] - 0.32
    notes = ["Paper 3: parameter sparsity a weak latency proxy; MACs strong",
             "7A: energy −22.8% / −13.4%, but accuracy −5.1 / −3.2 pp (VAL-RL)",
             "7B: memory −63.4%, MACs −36.0%, accuracy −1.12 pp (VAL-SELECT);",
             "      direct energy unresolved (qualification CV 19.92%)"]
    for k, n in enumerate(notes):
        ax.text(EX - 0.05, oy - k * 0.2, n, ha="left", va="top", fontsize=5.7, color=INK2)
    ax.text(EX - 0.05, oy + 0.17, "Frozen outcomes", ha="left", va="top", fontsize=6.0, weight="bold", color=INK)

    # legend
    ly = 0.38
    for k, (lab, st) in enumerate([("original design (Phases 2–3)", ORIG), ("added after diagnostic results (Phases 4–6)", ADDED),
                                   ("deployment extension (Paper 3, 7A–7B)", EXT)]):
        x0 = 0.6 + k * 2.3
        ax.add_patch(FancyBboxPatch((x0, ly - 0.07), 0.22, 0.14, boxstyle="round,pad=0,rounding_size=0.03", fc=st[0], ec=st[1], lw=0.9))
        ax.text(x0 + 0.3, ly, lab, ha="left", va="center", fontsize=6.2, color=INK)

    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(HERE, f"F1_framework.{ext}"), dpi=600 if ext == "png" else None, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
