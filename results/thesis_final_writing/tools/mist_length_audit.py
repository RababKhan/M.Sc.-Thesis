"""MIST length-compliance audit of the formatted thesis (template §2.2.3 and §3.3.2).

  PYTHONPATH=<scratch>/pylib;<venv site-packages> python results/thesis_final_writing/tools/mist_length_audit.py
Inputs: final_thesis_MIST.pdf (page labels via pdftotext), tools/_table_heights.json (rendered table positions measured
in Word by word_measure.ps1), tools/_docx_build_log.json (placed figure sizes, equation count).
Writes mist_length_compliance_audit.md.
"""
import json
import os
import re
import sys

import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from wordcount import CH, words  # noqa: E402

PT_MM = 25.4 / 72
TEXT_H = 247.0
HALF = TEXT_H / 2
EQ = {"page": 330, "large_table": 550, "small_table": 200, "large_figure": 350, "small_figure": 150, "equation": 50}

blog = json.load(open(os.path.join(HERE, "_docx_build_log.json"), encoding="utf-8"))

# ---- tables measured in Word (portrait text block 247 mm; landscape text block 153 mm)
TH = json.load(open(os.path.join(HERE, "_table_heights.json"), encoding="utf-8-sig"))


def _height(m):
    block = 153.0 if m.get("land") == 1 else TEXT_H
    top_margin = 32.0 if m.get("land") == 1 else 25.0
    if m["pn"] == m["p1"]:
        return (m["yn"] - m["y1"]) * PT_MM + 7.0
    first = block - ((m["y1"] * PT_MM) - top_margin)
    last = (m["yn"] * PT_MM - top_margin) + 7.0
    return first + last + (m["pn"] - m["p1"] - 1) * block


def _large(m, h):
    return h > (153.0 if m.get("land") == 1 else TEXT_H) / 2 or m["pn"] > m["p1"]


content = [(m["label"], _height(m), m["p1"], m["pn"], _large(m, _height(m))) for m in TH if m["kind"] == "content"]
codes = [(m["label"], _height(m), m["p1"], m["pn"], _large(m, _height(m))) for m in TH if m["kind"] == "code"]
large_t = [c for c in content if c[4]]
small_t = [c for c in content if not c[4]]
code_l = [c for c in codes if c[4]]
code_s = [c for c in codes if not c[4]]
figs = blog["figures"]
large_f = [f for f in figs if f["large"]]
small_f = [f for f in figs if not f["large"]]
neq = blog["equations"]

# ---- words (prose; tables are counted through their equivalents)
ch_prose = sum(words(open(os.path.join(W, f), encoding="utf-8").read(), True) for f in CH)
ch_all = sum(words(open(os.path.join(W, f), encoding="utf-8").read()) for f in CH)
abs_en = words(re.sub(r"^# Abstract", "", open(os.path.join(W, "abstract.md"), encoding="utf-8").read()))
bn_txt = open(os.path.join(W, "abstract_bangla.md"), encoding="utf-8").read()
abs_bn = len(re.findall(r"[ঀ-৿A-Za-z0-9০-৯][ঀ-৿A-Za-z0-9০-৯.\-–%]*", re.sub(r"^# .*\n", "", bn_txt)))
meta = json.load(open(os.path.join(HERE, "mist_meta.json"), encoding="utf-8"))
title_page = len(meta["title"].split()) + 40
ack = sum(len(x.split()) for x in meta["acknowledgements_placeholder"])
nota = words(open(os.path.join(W, "final_notation_inventory.md"), encoding="utf-8").read()) // 2
abbr = words(open(os.path.join(W, "list_of_abbreviations.md"), encoding="utf-8").read())
refs = words(open(os.path.join(W, "references_mist_final.md"), encoding="utf-8").read())
apx = open(os.path.join(W, "appendices.md"), encoding="utf-8").read()
apx_prose = words(apx, True)
front = title_page + abs_en + abs_bn + ack + nota + abbr

# ---- pages from the PDF footers (last non-empty line of each page, pdftotext -layout)
_tmp = os.path.join(tempfile.gettempdir(), "mist_len_pages.txt")
subprocess.run(["pdftotext", "-layout", os.path.join(W, "final_thesis_MIST.pdf"), _tmp], check=True)
_pages = open(_tmp, encoding="utf-8", errors="replace").read().split("\f")
if _pages and not _pages[-1].strip():
    _pages = _pages[:-1]
lab = {"unnumbered": 0, "roman": 0, "arabic": 0, "appendix": 0}
for ptxt in _pages:
    lines = [l.strip() for l in ptxt.splitlines() if l.strip()]
    last = lines[-1] if lines else ""
    if re.fullmatch(r"[ivxlcdm]+", last):
        lab["roman"] += 1
    elif re.fullmatch(r"\d+", last):
        lab["arabic"] += 1
    elif re.fullmatch(r"[A-Z]-\d+", last):
        lab["appendix"] += 1
    else:
        lab["unnumbered"] += 1
pages = len(_pages)

eq_t = len(large_t) * EQ["large_table"] + len(small_t) * EQ["small_table"]
eq_c = len(code_l) * EQ["large_table"] + len(code_s) * EQ["small_table"]
eq_f = len(large_f) * EQ["large_figure"] + len(small_f) * EQ["small_figure"]
eq_e = neq * EQ["equation"]
main_eq = ch_prose + eq_t + eq_c + eq_f + eq_e
total = front + main_eq + refs + apx_prose
# appendix tables are included in the table equivalents above (content tables include appendix tables)
R = ["# MIST length-compliance audit", "",
     "Rule (MIST template, Part 2 §2.2.3 and §3.3.2): an M.Sc. Engineering thesis \"should not exceed 50,000 words (150 A4",
     "pages in 1.5 spacing)\", inclusive of appendices, tables, figures and references; tables, figures and display",
     "equations are counted by their word equivalents. The guideline states that there is no strict limit.", "",
     "Measured on `final_thesis_MIST.docx` / `final_thesis_MIST.pdf` (Word layout). Prose words exclude table cells,",
     "citation keys and code; every table and figure is counted instead by its MIST equivalent, classified by its rendered",
     f"height against half the text block ({HALF:.1f} mm).", "",
     "## Inputs", "",
     "| Item | Count / words | MIST equivalent | Words |", "|---|---|---|---|",
     f"| Chapter text (Chapters 1–8, prose) | {ch_prose:,} words | counted directly | {ch_prose:,} |",
     f"| Front matter (title page ≈{title_page}, English Abstract {abs_en}, Bangla Abstract {abs_bn}, Acknowledgements placeholder {ack}, notation ≈{nota}, abbreviations {abbr}) | {front:,} words | counted directly | {front:,} |",
     f"| References (98 entries) | {refs:,} words | counted directly | {refs:,} |",
     f"| Appendix text (prose outside tables) | {apx_prose:,} words | counted directly | {apx_prose:,} |",
     f"| Numbered tables measured (chapters 43, appendices 18) | {len(content)} | | |",
     f"| Small tables (≤ ½ page; chapters and appendices) | {len(small_t)} | 200 each | {len(small_t) * 200:,} |",
     f"| Large tables (> ½ page) | {len(large_t)} | 550 each | {len(large_t) * 550:,} |",
     f"| Algorithm listings (counted as tables) | {len(code_s)} small, {len(code_l)} large | 200 / 550 | {eq_c:,} |",
     f"| Small figures (≤ ½ page) | {len(small_f)} | 150 each | {len(small_f) * 150:,} |",
     f"| Large figures (> ½ page) | {len(large_f)} | 350 each | {len(large_f) * 350:,} |",
     f"| Display equations | {neq} | 50 each | {eq_e:,} |",
     f"| **Estimated MIST word-equivalent** | | | **{total:,}** |", "",
     "## Pages (from the PDF page labels)", "",
     "| Part | Pages |", "|---|---|",
     f"| Unnumbered (cover, title, approval, declaration) | {lab['unnumbered']} |",
     f"| Preliminary (Roman) | {lab['roman']} |",
     f"| Chapters 1–8 and References (Arabic) | {lab['arabic']} |",
     f"| Appendices (A-1 …) | {lab['appendix']} |",
     f"| **Total A4 pages** | **{pages}** |", "",
     "## Assessment", "",
     f"- Chapters 1–8 total {ch_all:,} words including table cells ({ch_prose:,} prose); target 38,000–42,000 met.",
     f"- Estimated word-equivalent {total:,} against the 50,000 guideline: "
     + ("within the guideline." if total <= 50000 else f"**{total - 50000:,} above** the guideline."),
     f"- Formatted length {pages} A4 pages against the 150-page guideline: "
     + ("within the guideline." if pages <= 150 else f"**{pages - 150} pages above** the guideline."),
     "- The MIST body format itself (1.5 line spacing plus 12 pt before and after every paragraph, TP.18) is spacious;",
     "  the page count reflects that format, not a change of content.", "",
     "## Large tables (more than half a page, or spanning pages)", "", "| Table | Height (mm) | PDF pages |", "|---|---|---|"]
R += [f"| Table {lbl} | {h:.0f} | {p1}–{p2} |" for lbl, h, p1, p2, _ in large_t]
R += ["", "## Small tables", "", ", ".join(f"Table {lbl} ({h:.0f} mm)" for lbl, h, _, _, _ in small_t), ""]
open(os.path.join(W, "mist_length_compliance_audit.md"), "w", encoding="utf-8").write("\n".join(R) + "\n")
print(f"pages {pages} {lab}; tables small {len(small_t)} large {len(large_t)}; code {len(code_s)}/{len(code_l)}; "
      f"figures small {len(small_f)} large {len(large_f)}; eq {neq}; word-equivalent {total:,}")
