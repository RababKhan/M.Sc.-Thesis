# QA note: presentation-only re-renders of Chapter 5 Figures 5.3–5.5

- **Script:** `figures/rerender_ch5_figures.py`.
- **Date:** 2026-11-05.
- **Originals:** the frozen phase figures and CSVs are untouched; `git status` shows no change under
  `results/architecture_generalization`, `results/phase4` or `results/phase5`.

| Thesis figure | New file | Source data (unchanged) | Change (presentation only) |
|---|---|---|---|
| 5.3 | `ch5_fig5_3_sensitivity_tolerance.png` | `architecture_generalization/phase3_fig_D_sensitivity_tolerance.csv` | One fixed colour per setting in **both** panels. In the original, two scatter calls per setting advanced the colour cycle, so the right-panel colours did not match the legend. Shared legend for settings and markers (P0 = circle, P1 = cross). Right y-label states "mean final ratio over 20 seeds". |
| 5.4 | `ch5_fig5_4_topk_reach_rate.png` | `phase4/phase4_fig_D_topk_reach_rate.csv` | Legend moved below the panels (it overlapped the C10 SimpleCNN bars). Full, unclipped labels, including an explanation for the Phase-3 reference series, which has no bars. Black dot added to the legend. y-axis label added. |
| 5.5 | `ch5_fig5_5_resnet_reward_alignment.png` | `phase5/phase5_fig_B_resnet_reward_alignment.csv` | x-axis label "Selected-policy sparsity (%)" with a footnote, and each point labelled with its statistic: C0/C1/C2 at the **median**, Phase-4 F at the **mean** (the only value recorded). y-axis headroom (×1.2) so that labels do not collide with titles. |

**Deviation from the request, explained.** The task asked for the axis of Figure 5.5 to read "mean sparsity". Only
the Phase-4 F point is a mean; C0, C1 and C2 are medians in the frozen CSV (`sparsity_median`). Labelling the whole axis
"mean" would mislabel three of the four series. The axis is therefore neutral, and each point states its statistic. No
point was moved.

## Verification

After plotting, the script reads every plotted coordinate back from the matplotlib artists and compares it with the
source CSV, requiring exact float equality.

**Result: 86 checks, 86 identical, 0 mismatched.**

- **Same source data:** yes.
- **Numerical change:** none.
- **Nature of the change:** presentation only.
