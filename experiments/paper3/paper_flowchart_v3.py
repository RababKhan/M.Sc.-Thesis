"""Paper 3 methodology flowchart, classic style (IEEE single column, 300 dpi, black and white).

  python experiments/paper3/paper_flowchart_v3.py  -> results/paper3/paper_figures_final_v3/fig_methodology_flowchart.png
Terminators for start/end, parallelograms for data, rectangles for processes, diamonds for the pre-registered
decisions, dashed panels for the two branches (A latency, B recovery proxy). Protocol only, no outcomes.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "paper3", "paper_figures_final_v3")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "savefig.dpi": 300})
W, H = 3.5, 6.0                     # inches; drawn top-down, then the figure is cropped to the content
LW, FS, FS_T = 0.6, 6.4, 6.8        # line width, body font size, panel-title font size
GAP = 0.09                          # vertical gap between consecutive shapes
ax = None


def text(x, y, s, size=FS, weight="normal"):
    ax.text(x, y, s, ha="center", va="center", fontsize=size, fontweight=weight, linespacing=1.15)


def rect(cx, top, w, h, s):
    ax.add_patch(Rectangle((cx - w / 2, top - h), w, h, fill=False, lw=LW))
    text(cx, top - h / 2, s)
    return top - h


def term(cx, top, w, h, s):
    ax.add_patch(FancyBboxPatch((cx - w / 2, top - h), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                fill=False, lw=LW))
    text(cx, top - h / 2, s)
    return top - h


def para(cx, top, w, h, s, slant=0.08):
    x0, y0 = cx - w / 2, top - h
    ax.add_patch(Polygon([(x0 + slant, top), (x0 + w, top), (x0 + w - slant, y0), (x0, y0)], fill=False, lw=LW))
    text(cx, top - h / 2, s)
    return y0


def diamond(cx, top, w, h, s):
    ax.add_patch(Polygon([(cx, top), (cx + w / 2, top - h / 2), (cx, top - h), (cx - w / 2, top - h / 2)],
                         fill=False, lw=LW))
    text(cx, top - h / 2, s, size=FS - 0.4)
    return top - h


def line(pts, arrow=True):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color="black", lw=LW, solid_capstyle="butt")
    if arrow:
        ax.annotate("", xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", lw=LW, color="black",
                                    shrinkA=0, shrinkB=0))


def down(cx, y_from, y_to):
    line([(cx, y_from), (cx, y_to)])


def label(x, y, s):
    ax.text(x, y, s, ha="center", va="center", fontsize=FS - 0.6, style="italic")


def main():
    global ax
    os.makedirs(OUT, exist_ok=True)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    cx = W / 2

    # ---- shared spine
    y = H - 0.03
    y = term(cx, y, 0.62, 0.15, "Start")
    down(cx, y, y - GAP); y -= GAP
    y = para(cx, y, 2.35, 0.2, "CIFAR-10 / CIFAR-100 (50,000 training images)")
    down(cx, y, y - GAP); y -= GAP
    y = rect(cx, y, 2.5, 0.27, "Data split: 45,000 train, 2,500 VAL-RL, 2,500 VAL-SELECT\n(official test set unused)")
    down(cx, y, y - GAP); y -= GAP
    y = rect(cx, y, 2.5, 0.2, "Frozen dense references: SimpleCNN, ResNet-8")
    down(cx, y, y - GAP); y -= GAP
    y = rect(cx, y, 2.5, 0.27, "Structured policy space: 4 units \u00d7 6 actions = 1,296\npolicies, "
             "physically compacted (L1 ranking)")
    split_y = y - 0.07
    line([(cx, y), (cx, split_y)], arrow=False)

    # ---- two dashed panels
    pl, pr = 0.04, W - 0.04
    mid = W / 2
    pw = (pr - pl - 0.08) / 2
    ca, cb = pl + pw / 2, pr - pw / 2
    ptop = split_y - 0.1
    line([(ca, split_y), (cb, split_y)], arrow=False)
    down(ca, split_y, ptop); down(cb, split_y, ptop)
    bw = pw - 0.2

    # A. latency characterization
    ya = ptop - 0.05
    text(ca, ya - 0.06, "A. Latency Characterization", size=FS_T, weight="bold")
    ya -= 0.16
    ya = rect(ca, ya, bw, 0.27, "Latency-metric calibration\n(configurations A, B, C)")
    down(ca, ya, ya - GAP); ya -= GAP
    tab_top = ya
    ya = rect(ca, ya, bw, 0.27, "Latency tables: 4 \u00d7 1,296 shapes,\n5 sessions, batch 1, one thread")
    tab_mid = (tab_top + ya) / 2
    down(ca, ya, ya - GAP); ya -= GAP
    dtop = ya
    ya = diamond(ca, ya, 0.92, 0.36, "Anchor drift\n\u2264 10%?")
    # No: repeat the session (loop back to the tables box)
    xl = ca - bw / 2 - 0.07
    line([(ca - 0.46, dtop - 0.18), (xl, dtop - 0.18), (xl, tab_mid), (ca - bw / 2, tab_mid)])
    label(xl + 0.17, dtop - 0.13, "No")
    label(ca + 0.09, ya - 0.045, "Yes")
    down(ca, ya, ya - GAP); ya -= GAP
    ya = rect(ca, ya, bw, 0.27, "Cost-metric analysis: parameters,\nMACs vs latency (\u03c1, inversions, fit)")
    a_bottom = ya

    # B. recovery proxy and feasibility
    yb = ptop - 0.05
    text(cb, yb - 0.06, "B. Recovery Proxy and Feasibility", size=FS_T, weight="bold")
    yb -= 0.16
    yb = rect(cb, yb, bw, 0.27, "1- and 5-epoch recovery of\nvalidation policies")
    dw, dc = 0.86, cb - 0.17
    down(dc, yb, yb - GAP); yb -= GAP
    dtop_b = yb
    yb = diamond(dc, yb, dw, 0.36, "Spearman\n$\\rho \\geq 0.90$?")
    nx = dc + dw / 2
    side_w = pr - 0.06 - (nx + 0.13)
    side_c = nx + 0.13 + side_w / 2
    rect(side_c, dtop_b - 0.07, side_w, 0.22, "Latency\nonly")
    line([(nx, dtop_b - 0.18), (nx + 0.13, dtop_b - 0.18)])
    label(nx + 0.065, dtop_b - 0.13, "No")
    label(dc + 0.09, yb - 0.045, "Yes")
    down(dc, yb, yb - GAP); yb -= GAP
    yb = rect(cb, yb, bw, 0.27, "1-epoch proxy landscape\n(ResNet-8, 1,296 policies)")
    down(cb, yb, yb - GAP); yb -= GAP
    yb = rect(cb, yb, bw, 0.27, "Boundary calibration: 48 policies,\n5 epochs, $q_5 \\geq 0.98$ \u2192 $\\tau$")
    b_bottom = yb

    pbot = min(a_bottom, b_bottom) - 0.05
    for c in (ca, cb):
        ax.add_patch(Rectangle((c - pw / 2, pbot), pw, ptop - pbot, fill=False, lw=LW * 0.9, ls=(0, (4, 2.5))))

    # ---- merge and pre-registered decision
    join_y = pbot - 0.08
    line([(ca, a_bottom), (ca, join_y)], arrow=False)
    line([(cb, b_bottom), (cb, join_y)], arrow=False)
    line([(ca, join_y), (cb, join_y)], arrow=False)
    y = join_y
    down(cx, y, y - 0.07); y -= 0.07
    dt = y
    dw2 = 1.42
    y = diamond(cx, y, dw2, 0.5, "\u2265 20 policies with $q_1 \\geq \\tau$\nand \u2265 1 at \u2265 1.10\u00d7?")
    dmid = dt - 0.25
    side = 0.76
    lc, rc = 0.04 + side / 2, W - 0.04 - side / 2
    no_top = dmid + 0.13
    rect(lc, no_top, side, 0.26, "No search:\nreport headroom")
    rect(rc, no_top, side, 0.26, "Hardware-aware\nsearch")
    line([(cx - dw2 / 2, dmid), (lc + side / 2, dmid)])
    line([(cx + dw2 / 2, dmid), (rc - side / 2, dmid)])
    label((cx - dw2 / 2 + lc + side / 2) / 2, dmid + 0.055, "No")
    label((cx + dw2 / 2 + rc - side / 2) / 2, dmid + 0.055, "Yes")
    out_top = y - 0.12
    line([(lc, no_top - 0.26), (lc, out_top + 0.1), (cx - 0.6, out_top + 0.1), (cx - 0.6, out_top)])
    line([(rc, no_top - 0.26), (rc, out_top + 0.1), (cx + 0.6, out_top + 0.1), (cx + 0.6, out_top)])
    y = para(cx, out_top, 3.0, 0.2, "Outputs: latency tables, cost-metric correlations, latency headroom")
    down(cx, y, y - GAP); y -= GAP
    y = term(cx, y, 0.62, 0.15, "End")
    bottom = y - 0.03
    ax.set_ylim(bottom, H)
    fig.set_size_inches(W, H - bottom)
    print(f"figure height {H - bottom:.2f} in")
    path = os.path.join(OUT, "fig_methodology_flowchart.png")
    fig.savefig(path, facecolor="white")
    print("wrote", path)


if __name__ == "__main__":
    main()
