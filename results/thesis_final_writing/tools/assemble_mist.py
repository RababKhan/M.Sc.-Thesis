"""Assemble the MIST content master and its inventories (author-date citations). Read-only on chapters/phase files.

  python results/thesis_final_writing/tools/assemble_mist.py
Writes: figures/final/* (neutral copies), thesis_full_source.md, final_figure_inventory.md, final_table_inventory.md,
final_toc_map.md, final_length_report.md. Replaces the IEEE-numbered assemble.py of the previous stage.
"""
import json
import os
import re
import subprocess
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(W))
sys.path.insert(0, HERE)
import refs_mist as RM  # noqa: E402
from figmap import DATA, FIG, FIN, copy_figures, fix_figs  # noqa: E402
from wordcount import words  # noqa: E402

CH = RM.CHAPTERS
META = json.load(open(os.path.join(HERE, "mist_meta.json"), encoding="utf-8"))
OLD = "9366c52a"   # MIST stage before the final length-reduction pass


def body(f):
    return open(os.path.join(W, f), encoding="utf-8").read()


def old_body(f):
    return subprocess.run(["git", "-C", ROOT, "show", f"{OLD}:results/thesis_final_writing/{f}"], capture_output=True,
                          text=True, encoding="utf-8").stdout


copy_figures()
TXT = {f: body(f) for f in CH}
APX = body("appendices.md")
ALL = "\n".join(TXT.values())

# ------------------------------------------------------------------------------------------------- figures
caps = {}
for f in CH:
    for m in re.finditer(r"\*\*Figure ([\d.]+)\. ([^*]+?)\*\*", TXT[f]):
        caps[m.group(1)] = re.sub(r"\s+", " ", m.group(2)).strip().rstrip(".")
blog = json.load(open(os.path.join(HERE, "_docx_build_log.json"), encoding="utf-8")) if os.path.exists(os.path.join(HERE, "_docx_build_log.json")) else {"figures": []}
fsize = {x["num"]: x for x in blog.get("figures", [])}
fig_rows = []
for n, (fid, src, dst, note) in FIG.items():
    im = Image.open(os.path.join(FIN, dst))
    dpi = round(im.info.get("dpi", (0, 0))[0])
    refd = len(re.findall(rf"Fig(?:ure|\.)s? {re.escape(n)}\b", ALL)) - 1
    fs = fsize.get(n, {})
    size = (f"{fs.get('w_mm')} × {fs.get('h_mm')} mm, {'large' if fs.get('large') else 'small'}") if fs else "—"
    fig_rows.append(f"| {n} | {caps.get(n, '?')} | Ch. {n.split('.')[0]} | `figures/final/{dst}` | `{DATA[n]}` | {note} | "
                    f"{im.size[0]}×{im.size[1]} px, {dpi} dpi | {size} | {'referenced ' + str(refd) + '×' if refd > 0 else 'caption only'} | in List of Figures |")
fig_doc = ["# Final figure inventory (MIST)", "",
           f"{len(FIG)} figures, numbered consecutively within each chapter. Each is placed after the text that introduces it, has a",
           "caption below it in MIST form (\"Fig. n.m: …\"), and has an entry in the List of Figures (generated from caption",
           "fields in `final_thesis_MIST.docx`). Figure data are unchanged; the files in `figures/final/` are neutral-name copies.", "",
           "| Fig. | Title | Chapter | Thesis file | Source data | Origin | Resolution | Placed size (MIST class) | In-text references | List |",
           "|---|---|---|---|---|---|---|---|---|---|"] + fig_rows + ["",
           "Large/small follows MIST §3.3.2: a figure taller than half the text block (123.5 mm) is large (350 words), otherwise small (150 words).", ""]
open(os.path.join(W, "final_figure_inventory.md"), "w", encoding="utf-8").write("\n".join(fig_doc) + "\n")

# ------------------------------------------------------------------------------------------------- tables
tabs = []
for f in CH + ["appendices.md"]:
    t = TXT.get(f, APX)
    for m in re.finditer(r"\*\*Table ([A-Z]?[\d.]+)\. ([^*]+?)\*\*", t):
        tabs.append((m.group(1), re.sub(r"\s+", " ", m.group(2)).strip().rstrip("."), f))
renum = json.load(open(os.path.join(HERE, "_table_renumber_map.json"), encoding="utf-8")) if os.path.exists(os.path.join(HERE, "_table_renumber_map.json")) else {"mapping": []}
oldof = {x["new"]: x["old"] for x in renum["mapping"]}
tab_rows = []
for n, title, f in tabs:
    src = ALL if not n[0].isalpha() else APX + "\n" + ALL
    refs = len(re.findall(rf"Tables? (?:[\d.A-Z]+(?:\s*(?:–|,|and)\s*)?)*{re.escape(n)}\b", src)) - 1
    ch = n.split(".")[0]
    old = oldof.get(n, n) if not n[0].isalpha() else "new (Appendix)"
    old = "new caption" if old.endswith("new") else old
    tab_rows.append(f"| {n} | {title} | {'Ch. ' + ch if ch.isdigit() else 'App. ' + ch} | {old} | "
                    f"{'referenced ' + str(refs) + '×' if refs > 0 else ('caption + adjacent text' if ch.isdigit() else 'appendix (listed)')} | in List of Tables |")
ntab_ch = len([t for t in tabs if t[0][0].isdigit()])
ntab_ap = len(tabs) - ntab_ch
tab_doc = ["# Final table inventory (MIST)", "",
           f"Numbered tables: **{len(tabs)}** ({ntab_ch} in Chapters 1–8, {ntab_ap} in Appendices A–E). MIST requires consecutive",
           "numbering; all chapter tables were renumbered in order of appearance by `tools/renumber_tables.py` (mapping in",
           "`tools/_table_renumber_map.json`), which also resolved the former out-of-order Table 5.3 (now Table 5.11, where it",
           "stands). Every previously un-numbered inline table was either converted to an equivalent list/sentence during",
           "compression or given a caption and number. Every table appears in the List of Tables. Table values are unchanged.", "",
           "| Table | Title | Location | Previous number | In-text references | List |", "|---|---|---|---|---|---|"] + tab_rows + [""]
open(os.path.join(W, "final_table_inventory.md"), "w", encoding="utf-8").write("\n".join(tab_doc) + "\n")

# ------------------------------------------------------------------------------------------------- TOC map
toc = ["# Final table-of-contents map (MIST)", "",
       "Hierarchy of `final_thesis_MIST.docx` (page numbers are generated by the Word TOC field; see the PDF).", "",
       "- Cover, Title page, Approval, Declaration (unnumbered)",
       "- ABSTRACT; সারসংক্ষেপ (Bangla Abstract); ACKNOWLEDGEMENTS; TABLE OF CONTENTS; LIST OF FIGURES; LIST OF TABLES;",
       "  LIST OF MAIN NOTATION AND ABBREVIATIONS (Roman page numbers)"]
for f in CH:
    for line in TXT[f].splitlines():
        m = re.match(r"^(#{1,3}) (.*)", line)
        if m:
            lvl = len(m.group(1))
            txt = m.group(2).strip()
            if lvl == 1:
                mm = re.match(r"Chapter (\d+)\s+[—-]\s+(.*)", txt)
                txt = f"CHAPTER {mm.group(1)}: {mm.group(2).upper()}"
            toc.append(("  " * (lvl - 1)) + "- " + txt)
        mt = re.match(r"\*\*Table ([\d.]+)\.", line)
        mf = re.match(r"\*\*Figure ([\d.]+)\.", line)
        if mt:
            toc.append("      - *Table " + mt.group(1) + "*")
        if mf:
            toc.append("      - *Fig. " + mf.group(1) + "*")
toc.append("- REFERENCES (Arabic page numbers continue)")
for line in APX.splitlines():
    m = re.match(r"^## Appendix ([A-Z]) — (.*)", line)
    if m:
        toc.append(f"- APPENDIX {m.group(1)}: {m.group(2).upper()} (pages {m.group(1)}-1, {m.group(1)}-2, …)")
    mt = re.match(r"\*\*Table ([A-Z][\d.]+)\.", line)
    if mt:
        toc.append("      - *Table " + mt.group(1) + "*")
open(os.path.join(W, "final_toc_map.md"), "w", encoding="utf-8").write("\n".join(toc) + "\n")

# ------------------------------------------------------------------------------------------------- length report
L = ["# Final length report (after the final MIST length-reduction pass)", "",
     "Counts use the same rule throughout (`tools/wordcount.py`): citation keys and code blocks excluded; \"prose\" excludes",
     "table rows. \"Before\" is commit `9366c52a` (MIST-formatted build before the final length-reduction pass).", "",
     "| Part | Before (words) | After (words) | After (prose) | Change |", "|---|---|---|---|---|"]
tb = ta = tp = 0
for f in CH:
    b, a, p = words(old_body(f)), words(TXT[f]), words(TXT[f], True)
    tb, ta, tp = tb + b, ta + a, tp + p
    L.append(f"| Chapter {int(f.split('_')[1])} | {b:,} | {a:,} | {p:,} | {a - b:+,} ({100 * (a - b) / b:+.1f}%) |")
L.append(f"| **Chapters 1–8** | **{tb:,}** | **{ta:,}** | **{tp:,}** | **{ta - tb:+,} ({100 * (ta - tb) / tb:+.1f}%)** |")
ab, aa = words(old_body("appendices.md")), words(APX)
L.append(f"| Appendices | {ab:,} (A–E) | {aa:,} (A–E, selective) | {words(APX, True):,} | {aa - ab:+,} ({100 * (aa - ab) / ab:+.1f}%) |")
eng = re.sub(r"\*\*Keywords:\*\*.*", "", re.sub(r"^# Abstract", "", body("abstract.md")), flags=re.S)
L += [f"| English Abstract (body, keywords excluded; unchanged in this pass) | 390 | {words(eng):,} | | |", "",
      "Earlier target (first MIST pass) for Chapters 1–8: 38,000–42,000 words; this pass reduced them further to " + f"{ta:,} for the 150-page / 50,000 word-equivalent guideline.",
      "The MIST word-equivalent and the page count of the formatted thesis are in `mist_length_compliance_audit.md`."]
open(os.path.join(W, "final_length_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

# ------------------------------------------------------------------------------------------------- full source
m = META
fig_list = [f"- Figure {n}: {caps.get(n, '')}" for n in FIG]
tab_list = [f"- Table {n}: {t}" for n, t, _ in tabs]
front = [f"# {m['title'].upper()}", "",
         "<!-- MIST TP.03 cover -->",
         f"{m['author'].upper()}  ", "M.Sc. ENGINEERING THESIS  ", "[MIST MONOGRAM 28 mm × 30 mm]  ",
         f"{m['department'].upper()}  ", f"{m['institution'].upper()}  ", f"{m['city'].upper()}  ", f"{m['month_year']}", "",
         "<!-- MIST TP.06 title page -->",
         f"{m['title'].upper()}  ", f"{m['author'].upper()} (SN. {m['student_number']})  ",
         f"A Thesis Submitted in Partial Fulfillment of the Requirements for the Degree of {m['degree_statement']}  ",
         f"{m['department'].upper()}, {m['institution'].upper()}, {m['city'].upper()}, {m['month_year']}", "",
         "<!-- MIST TP.08 approval page -->",
         f"Approved as to style and content by the Board of Examination on {m['exam_date']}:", ""]
for name, role in m["examiners"]:
    front.append(f"- ______________________ {', '.join(name)} — {', '.join(role)}")
front += ["", "<!-- MIST TP.09 declaration -->", "## DECLARATION", "", m["declaration_text"], "",
          f"______________________  {m['author']}, {m['department']}, MIST, Dhaka", ""]
parts = front + ["# ABSTRACT", "", re.sub(r"^# Abstract\s*", "", body("abstract.md")).strip(), "",
                 "# সারসংক্ষেপ (Abstract in Bangla)", "", re.sub(r"^# .*\n", "", body("abstract_bangla.md")).strip(), "",
                 "# ACKNOWLEDGEMENTS", ""] + m["acknowledgements_placeholder"] + ["",
                 "# TABLE OF CONTENTS", "", "[Generated with page numbers in final_thesis_MIST.docx; hierarchy in final_toc_map.md]", "",
                 "# LIST OF FIGURES", ""] + fig_list + ["", "# LIST OF TABLES", ""] + tab_list + ["",
                 "# LIST OF MAIN NOTATION AND ABBREVIATIONS", "", "See `final_notation_inventory.md` (notation) and the list below.", "",
                 re.sub(r"^# List of Abbreviations", "## List of Abbreviations", body("list_of_abbreviations.md"), flags=re.M), ""]
for f in CH:
    parts += [fix_figs(RM.cite_text(TXT[f])), ""]
parts += [body("references_mist_final.md"), "", APX]
full = "\n".join(parts)
left = re.findall(r"\[@[^\]]+\]", full)
ieee = re.findall(r"(?<![\w(])\[\d+\](?!\()", full)
open(os.path.join(W, "thesis_full_source.md"), "w", encoding="utf-8").write(full)
print(f"figures {len(FIG)}; tables {len(tabs)} ({ntab_ch} chapter, {ntab_ap} appendix); chapters {ta:,} words (prose {tp:,}); "
      f"appendices {aa:,}; citation markers left {len(left)}; numbered citations {len(ieee)}")
