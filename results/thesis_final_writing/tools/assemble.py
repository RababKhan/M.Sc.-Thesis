"""Assemble the thesis content master and its inventories. Read-only on chapters and phase files.

  python results/thesis_final_writing/tools/assemble.py
Writes: figures/final/* (neutral-name copies), thesis_full_source.md, final_figure_inventory.md,
final_table_inventory.md, final_toc_map.md, final_length_report.md.
Citation keys are converted to IEEE numbers (order of first citation from Chapter 1) and figure paths point to the
neutral copies. Chapter files themselves are not modified.
"""
import os
import re
import shutil
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import refs_final as RF  # noqa: E402

FIN = os.path.join(W, "figures", "final")
os.makedirs(FIN, exist_ok=True)
# figure number -> (shortlist id, source path relative to W, neutral name, status note)
FIG = {
    "3.1": ("F1", "figures/F1_framework.png", "fig_3_1_overall_framework.png", "original thesis figure (Milestone 1)"),
    "5.1": ("F3", "figures/final/fig_5_1_phase3_accuracy_vs_sparsity.png", "fig_5_1_phase3_accuracy_vs_sparsity.png", "300-dpi re-render with the unchanged Phase-3 plotting function (was 130 dpi)"),
    "5.2": ("F4", "figures/final/fig_5_2_phase3_allocation_heatmap.png", "fig_5_2_phase3_allocation_heatmap.png", "300-dpi re-render with the unchanged Phase-3 plotting function (was 130 dpi)"),
    "5.3": ("F2", "figures/ch5_fig5_3_sensitivity_tolerance.png", "fig_5_3_sensitivity_profile.png", "thesis re-render (colour key), 86/86 values verified"),
    "5.4": ("F6", "figures/ch5_fig5_4_topk_reach_rate.png", "fig_5_4_archive_topk_reach.png", "thesis re-render (legend placement), values verified"),
    "5.5": ("F7", "figures/ch5_fig5_5_resnet_reward_alignment.png", "fig_5_5_resnet8_reward_alignment.png", "thesis re-render (mean/median labels), values verified"),
    "5.6": ("F5", "../phase5/phase5_fig_D_sensitivity_exploration.png", "fig_5_6_sensitivity_exploration.png", "original phase figure"),
    "5.7": ("O4", "../phase5/phase5_fig_E_feasible_sparsity_gap.png", "fig_5_7_feasible_sparsity_gap.png", "original phase figure"),
    "5.8": ("F8", "../phase6/figures/phase6_fig2_ppo_vs_lamp.png", "fig_5_8_ppo_vs_lamp_recovery.png", "original phase figure"),
    "6.1": ("F9", "../paper3/paper_figures_revised/fig1_latency_vs_reduction.png", "fig_6_1_latency_vs_param_mac_reduction.png", "original Paper-3 figure (single-column width)"),
    "6.2": ("O6", "../paper3/paper_figures_revised/fig3_resnet8_latency_room.png", "fig_6_2_resnet8_latency_room.png", "original Paper-3 figure (single-column width)"),
    "6.3": ("F10", "../phase7a_sustainability/figures/fig_A_accuracy_vs_energy.png", "fig_6_3_phase7a_accuracy_vs_energy.png", "original phase figure"),
    "6.4": ("F11", "../phase7b_green_pruning/figures/fig_accuracy_planes.png", "fig_6_4_phase7b_accuracy_planes.png", "original phase figure; copied to a neutral name (source folder name contains 'green')"),
}
DATA = {"3.1": "conceptual (no data)", "5.1": "architecture_generalization/phase3_fig_A_accuracy_sparsity.csv",
        "5.2": "architecture_generalization/phase3_fig_B_policy_heatmap.csv", "5.3": "architecture_generalization/phase3_fig_D_sensitivity_tolerance.csv",
        "5.4": "phase4/phase4_fig_D_topk_reach_rate.csv", "5.5": "phase5/phase5_fig_B_resnet_reward_alignment.csv",
        "5.6": "phase5/phase5_fig_D_sensitivity_exploration.csv", "5.7": "phase5/phase5_fig_E_feasible_sparsity_gap.csv",
        "5.8": "phase6/figures (phase6_fig2 source CSV)", "6.1": "paper3/h2_latency_table_{simplecnn,resnet8}_c10.csv",
        "6.2": "paper3/h2_c10_resnet8_1epoch_landscape.csv; resnet8_threshold_calibration.csv", "6.3": "phase7a_sustainability/energy_summary.csv, accuracy_results.csv",
        "6.4": "phase7b_green_pruning/final_summary.csv"}
for n, (_, src, dst, _) in FIG.items():
    s = os.path.normpath(os.path.join(W, src))
    d = os.path.join(FIN, dst)
    if os.path.abspath(s) != os.path.abspath(d):
        shutil.copyfile(s, d)

FRONT = ["abstract.md"]
CH = RF.CHAPTERS
nmap = RF.number_map()


def body(f):
    return open(os.path.join(W, f), encoding="utf-8").read()


def fix_figs(t):
    def rep(m):
        n = m.group(1)
        return f"![Figure {n}](figures/final/{FIG[n][2]})"
    return re.sub(r"!\[Figure ([\d.]+)\]\([^)]+\)", rep, t)


# ---------------------------------------------------------------- figure and table inventories
caps = {}
tabs = []
for f in CH:
    t = body(f)
    for m in re.finditer(r"\*\*Figure ([\d.]+)\. ([^*]+?)\*\*", t):
        caps[m.group(1)] = re.sub(r"\s+", " ", m.group(2)).strip().rstrip(".")
    for m in re.finditer(r"\*\*Table ([\d.]+)\. ([^*]+?)\*\*", t):
        tabs.append((m.group(1), re.sub(r"\s+", " ", m.group(2)).strip().rstrip("."), f))
txt_all = "\n".join(body(f) for f in CH)
fig_rows = []
for n, (fid, src, dst, note) in FIG.items():
    im = Image.open(os.path.join(FIN, dst))
    dpi = round(im.info.get("dpi", (0, 0))[0])
    refd = len(re.findall(rf"Figure {re.escape(n)}\b", txt_all)) - 1  # minus the caption itself
    fig_rows.append(f"| {n} | {fid} | {caps.get(n, '?')} | Ch. {n.split('.')[0]} | `{src}` → `figures/final/{dst}` | `{DATA[n]}` | {note} | "
                    f"{im.size[0]}×{im.size[1]} px, {dpi} dpi | yes | {'yes' if n in ('5.3', '5.4', '5.5') else 'plotted by the frozen phase script from the listed CSV; numbers in the caption audited'} | "
                    f"{'referenced ' + str(refd) + '×' if refd > 0 else 'caption only'} | final |")
fig_doc = ["# Final figure inventory", "",
           "Thirteen figures: the eleven essential figures F1–F11 of the integration shortlist, plus the optional O4 and O6.",
           "Neutral thesis file names are in `figures/final/`. The original phase figures are untouched. The assembled",
           "source references only the neutral copies, so no path contains 'green'.", "",
           "| Fig. | ID | Title | Chapter | Source → thesis file | Source data | Original or re-render | Resolution | Caption verified | Numerical content verified | In-text references | Status |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"] + fig_rows + ["",
           "## Figure QA", "",
           "| Check | Result |", "|---|---|",
           "| Readable labels, no clipped legends | All 13 inspected as rendered images. The three Chapter-5 defects (missing colour key, overlapping and truncated legend, mean/median axis label) were fixed by presentation-only re-renders (86/86 plotted values identical; `figures/qa_ch5_figure_rerender.md`) |",
           "| Resolution | Ten figures at 300 dpi or more; F1 at 600 dpi. Figures 5.1 and 5.2 were re-rendered from 130 to 300 dpi with the unchanged Phase-3 plotting functions and the same CSVs (the frozen phase folder is unchanged). Figures 6.1 and 6.2 are 300-dpi single-column (about 3.5 in) Paper-3 figures. They are sharp at their native width; at the template stage they should be placed at about 3.5 in, or re-rendered wider from the same tables |",
           "| Consistent fonts | Phase figures use matplotlib DejaVu Sans. Font size varies between figure families (phase scripts vs. thesis re-renders). A unified restyle is optional at template stage and must keep plotted values identical |",
           "| Units | Accuracy %, sparsity %, latency ms, energy J per 1,000 inferences, memory KB, MACs M: stated on the axes |",
           "| Mean vs median | Figure 5.5 labels each point's statistic. Figure 5.6 states its 16-episode rolling mean in the caption. Figure 5.7 is median sparsity. Figure 6.3 is session medians with ranges |",
           "| Split stated in caption | 5.1 test; 5.3 VAL-RL (I); 5.4 VAL-RL (I); 5.6 VAL-RL; 5.7 VAL-RL; 5.8 test; 6.2 VAL-RL (q₁ / q₅); 6.3 VAL-RL; 6.4 VAL-SELECT |",
           "| No caption stronger than the data | Checked. For example, 6.3 notes that no pruned model lies inside the ±0.5 pp band; 5.8 notes that the ±0.5 pp band is not drawn; 6.1 states it shows the CIFAR-10 tables only |",
           "| Figure referenced in text | Every figure is placed directly after the text that introduces its content, and its number is cited in at least one caption or sentence (see In-text references) |"]
open(os.path.join(W, "final_figure_inventory.md"), "w", encoding="utf-8").write("\n".join(fig_doc) + "\n")

AUD = {"1": "conceptual / mapping (no frozen numbers)", "2": "conceptual; Table 2.2 cells checked against abstracts (Milestone 1)", "3": "conceptual; equations audited (equation_audit_chapter_03.md)",
       "4": "audit_ch4.py 100/100", "5": "audit_ch5.py 182/182", "6": "audit_ch6.py 142/142"}
tab_rows = []
for n, title, f in tabs:
    ch = n.split(".")[0]
    refs = len(re.findall(rf"Table {re.escape(n)}\b", txt_all)) - 1
    tab_rows.append(f"| {n} | {title} | Ch. {ch} | `{f}` | main text | {AUD.get(ch, '—')} | {'referenced ' + str(refs) + '×' if refs > 0 else 'caption only'} | final |")
unn = {f: len([l for l in body(f).splitlines() if l.startswith("|---")]) - len([1 for n, _, g in tabs if g == f]) for f in CH}
tab_doc = ["# Final table inventory", "",
           f"Numbered main-text tables: **{len(tabs)}**. Each chapter also contains short un-numbered inline tables (summary",
           "grids, criteria lists). These are counted below and should be numbered, or kept as in-text lists, according to",
           "the official template.", "",
           "| Table | Title | Chapter | Source file | Placement | Numbers audited | In-text references | Status |",
           "|---|---|---|---|---|---|---|---|"] + tab_rows + ["",
           "| Chapter | Un-numbered inline tables |", "|---|---|"] + [f"| Ch. {int(f.split('_')[1])} | {v} |" for f, v in unn.items()] + ["",
           "## Table QA", "",
           "| Check | Result |", "|---|---|",
           "| Duplicate tables | None. Table 5.1 (dense references, Ch. 5) repeats four columns of Table 4.4 (Ch. 4) by design: Ch. 4 defines them as setup, Ch. 5 reports them as the reference point. At template stage, Table 5.1 could be shortened to a pointer |",
           "| Repeated results | Phase-6 PPO–LAMP gaps appear in Table 5.7 and are summarised in prose in Chapters 7–8; Phase-7A energy appears in Table 6.3 and Table 6.5. The values are identical (global numerical audit PASS) |",
           "| Inconsistent rounding | Accuracies 2 dp, energy J 2 dp in Table 6.3 (3 dp in the appendix), ratios 3 dp. The convention is round-half-up throughout (+31.37). One prose variant (5.1 / 3.2 pp, Ch. 6 §6.8) is a 1-dp rendering of 5.08 / 3.24, accepted |",
           "| Oversized tables moved to the appendix | Seed-level and session-level data and full statistics families are in Appendices B–J. The largest main-text tables (4.6 PPO by phase, 4.8 metrics, 5.7 Phase 6, 6.3 Phase 7A) are 6–14 rows |",
           "| Every numbered table referenced in text | See In-text references. Tables 6.2, 6.4 and 6.5 are introduced by the paragraph directly above them but are not cited by number. Add '(Table 6.x)' at template stage (presentation only) |",
           "| Numbering order | **Table 5.3 is placed in §5.11, after Tables 5.4–5.7** (the chapter explains this: it draws on §§5.4, 5.5 and 5.8). Templates normally require numerical order. At template stage, either move Table 5.3 to the end of §5.5 or renumber it 5.7 (with 5.4–5.7 becoming 5.3–5.6). This is a formatting decision; no content change |"]
open(os.path.join(W, "final_table_inventory.md"), "w", encoding="utf-8").write("\n".join(tab_doc) + "\n")

# ---------------------------------------------------------------- TOC map
toc = ["# Final table-of-contents map", "",
       "Hierarchy of the assembled thesis (`thesis_full_source.md`), with the tables and figures of each section. Use it",
       "to rebuild the TOC in the official template.", ""]
for f in FRONT + CH + ["appendices.md"]:
    t = body(f)
    cur_tables, cur_figs = [], []
    for line in t.splitlines():
        m = re.match(r"^(#{1,4}) (.*)", line)
        if m:
            lvl = len(m.group(1))
            toc.append(("  " * (lvl - 1)) + "- " + m.group(2).strip())
        mt = re.match(r"\*\*Table ([\d.]+)\.", line)
        mf = re.match(r"\*\*Figure ([\d.]+)\.", line)
        if mt:
            toc.append("      - *Table " + mt.group(1) + "*")
        if mf:
            toc.append("      - *Figure " + mf.group(1) + "*")
open(os.path.join(W, "final_toc_map.md"), "w", encoding="utf-8").write("\n".join(toc) + "\n")


# ---------------------------------------------------------------- length report
def words(t, prose_only=False):
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"\[@[^\]]+\]", " ", t)
    if prose_only:
        t = "\n".join(l for l in t.splitlines() if not l.startswith("|"))
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-–.]*", t))


L = ["# Final length report", "", "Word counts exclude citation keys and code blocks. Prose excludes table rows.", "",
     "| Part | Words (total) | Of which prose | Tables | Figures | Est. pages* |", "|---|---|---|---|---|---|"]
tot = totp = 0
pages = 0.0
for f in FRONT + CH:
    t = body(f)
    w, p = words(t), words(t, True)
    nt = len(re.findall(r"\*\*Table [\d.]+\.", t))
    nf = len(re.findall(r"!\[Figure", t))
    pg = w / 420 + 0.4 * nt + 0.5 * nf
    pages += pg if f != "abstract.md" else 0
    if f != "abstract.md":
        tot += w
        totp += p
    L.append(f"| {'Abstract' if f == 'abstract.md' else 'Chapter ' + str(int(f.split('_')[1]))} | {w:,} | {p:,} | {nt} | {nf} | {pg:.1f} |")
aw = words(body("appendices.md"))
apg = aw / 520 + 0
L += [f"| **Main text (Ch. 1–8)** | **{tot:,}** | **{totp:,}** | | | **{pages:.0f}** |",
      f"| Appendices A–K | {aw:,} | — | many | 0 | {apg:.0f}–{apg * 1.4:.0f} |",
      f"| References | {len(nmap)} entries | | | | {len(nmap) / 12:.0f} |", "",
      "*Page estimate: about 420 words per page of 12-pt text at standard thesis margins, plus 0.4 page per numbered "
      "table and 0.5 page per figure. Appendix tables are dense, so a range is given. Front matter (title, declaration, "
      "acknowledgements, abstract, lists, TOC) adds about 8–12 pages. These are estimates; the official template "
      "determines the final count. No page target was forced.*"]
open(os.path.join(W, "final_length_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- full source
lof = ["## List of Figures", ""] + [f"- Figure {n}. {caps.get(n, '')}" for n in FIG]
lot = ["## List of Tables", ""] + [f"- Table {n}. {t}" for n, t, _ in tabs]
parts = ["# A Reinforcement Learning Framework for Layer-Wise Adaptive Neural Network Pruning with Sensitivity-Aware Optimization", "",
         "[OFFICIAL TITLE PAGE TO BE INSERTED FROM UNIVERSITY TEMPLATE: author name, student ID, degree, department, institution, supervisor, month and year]", "",
         "[OFFICIAL DECLARATION PAGE TO BE INSERTED FROM UNIVERSITY TEMPLATE]", "",
         "[APPROVAL / CERTIFICATION PAGE TO BE INSERTED FROM UNIVERSITY TEMPLATE]", "",
         "[ACKNOWLEDGEMENTS TO BE WRITTEN BY THE AUTHOR]", "",
         convert_abs := RF.convert(body("abstract.md"), nmap), "",
         "[TABLE OF CONTENTS TO BE GENERATED IN THE OFFICIAL TEMPLATE; see final_toc_map.md]", "",
         "\n".join(lof), "", "\n".join(lot), "",
         re.sub(r"^# List of Abbreviations", "## List of Abbreviations", body("list_of_abbreviations.md"), flags=re.M), ""]
for f in CH:
    parts += [fix_figs(RF.convert(body(f), nmap)), ""]
parts += [open(os.path.join(W, "references_final.md"), encoding="utf-8").read(), "", body("appendices.md")]
full = "\n".join(parts)
left = re.findall(r"\[@[^\]]+\]", full)
todo = re.findall(r"\bTODO\b|\bTBD\b|\?\?", full)
open(os.path.join(W, "thesis_full_source.md"), "w", encoding="utf-8").write(full)
print(f"figures {len(FIG)}; numbered tables {len(tabs)}; main words {tot:,} (prose {totp:,}); est. pages {pages:.0f}; "
      f"appendix words {aw:,}; unresolved citation keys {len(left)}; TODO/TBD/?? {len(todo)}")
