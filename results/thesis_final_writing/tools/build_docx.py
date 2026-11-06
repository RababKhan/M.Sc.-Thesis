"""Build the MIST-formatted thesis DOCX from the frozen markdown sources.

  PYTHONPATH=<scratch>/pylib python results/thesis_final_writing/tools/build_docx.py
  -> results/thesis_final_writing/final_thesis_MIST.docx   (fields are updated and the PDF exported by word_finalize.ps1)

Formatting follows `template/Thesis Template MIST.pdf` (see mist_template_audit.md): A4; body 12 pt Times New Roman,
1.5 spacing, 12 pt before/after, justified; margins L32/R25/T25/B25 mm; chapters on a new page with "CHAPTER n" and the
title centred, bold, capitals, single spacing, 24 pt after; captions "Table n.m:" above tables and "Fig. n.m:" below
figures; equations in Cambria Math 14 pt indented 19 mm with the number 19 mm inside the right margin; author-date
citations; references single-spaced, 10 pt before/after, 13 mm hanging indent; page numbers centred 19 mm from the
bottom, Roman in preliminary pages, Arabic in the main text and references, A-1, B-1, ... in appendices; lists
labelled (a), (b) / (i), (ii) instead of bullets. Nothing in the content is altered: the text is the converted source.
"""
import json
import os
import re
import sys

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor
from latex2mathml.converter import convert as l2m
from lxml import etree
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import refs_mist as RM  # noqa: E402
from figmap import FIG, fix_figs  # noqa: E402

META = json.load(open(os.path.join(HERE, "mist_meta.json"), encoding="utf-8"))
CH = RM.CHAPTERS
XSL = etree.XSLT(etree.parse(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"))
TNR, BN = "Times New Roman", META.get("bangla_font", "Kalpurush")
TEXT_W = 210 - 32 - 25           # mm
TEXT_H = 297 - 25 - 25
LOG = {"tables": [], "figures": [], "equations": 0, "warnings": [], "all_tables": []}


# ============================================================================================ low-level XML helpers
def set_cell_margins(table, top=0, bottom=0, left=1.5, right=1.5):
    tblPr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for side, v in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:w"), str(int(v * 56.7)))
        e.set(qn("w:type"), "dxa")
        mar.append(e)
    tblPr.append(mar)


def set_borders(table, sz=4, inside=True, none=False):
    tblPr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{side}")
        if none or (not inside and side.startswith("inside")):
            e.set(qn("w:val"), "nil")
        else:
            e.set(qn("w:val"), "single")
            e.set(qn("w:sz"), str(sz))
            e.set(qn("w:space"), "0")
            e.set(qn("w:color"), "000000")
        b.append(e)
    tblPr.append(b)


def fixed_layout(table, widths_mm):
    tblPr = table._tbl.tblPr
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    tblPr.append(lay)
    tw = OxmlElement("w:tblW")
    tw.set(qn("w:w"), str(int(sum(widths_mm) * 56.7)))
    tw.set(qn("w:type"), "dxa")
    for old in tblPr.findall(qn("w:tblW")):
        tblPr.remove(old)
    tblPr.append(tw)
    grid = table._tbl.tblGrid
    for i, gc in enumerate(grid.findall(qn("w:gridCol"))):
        gc.set(qn("w:w"), str(int(widths_mm[i] * 56.7)))
    for row in table.rows:
        for i, cell in enumerate(row.cells):
            cell.width = Mm(widths_mm[i])


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    e = OxmlElement("w:tblHeader")
    e.set(qn("w:val"), "true")
    trPr.append(e)


def cant_split(row):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:cantSplit"))


def add_field(par, instr, placeholder=""):
    r = par.add_run()
    fc = OxmlElement("w:fldChar")
    fc.set(qn("w:fldCharType"), "begin")
    r._r.append(fc)
    r2 = par.add_run()
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = f" {instr} "
    r2._r.append(it)
    r3 = par.add_run()
    fc = OxmlElement("w:fldChar")
    fc.set(qn("w:fldCharType"), "separate")
    r3._r.append(fc)
    if placeholder:
        par.add_run(placeholder)
    r4 = par.add_run()
    fc = OxmlElement("w:fldChar")
    fc.set(qn("w:fldCharType"), "end")
    r4._r.append(fc)
    return par


def tc(par, text, ident="C", level=1):
    t = text.replace('"', "'")
    add_field(par, f'TC "{t}" \\f {ident} \\l {level}')


def fonts(run, name=TNR, size=None, cs=None):
    run.font.name = name
    rpr = run._r.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia"):
        rf.set(qn(a), name)
    rf.set(qn("w:cs"), cs or name)
    if size:
        run.font.size = Pt(size)
        szcs = OxmlElement("w:szCs")
        szcs.set(qn("w:val"), str(int(size * 2)))
        rpr.append(szcs)


def pf(par, before=None, after=None, line=None, align=None, left=None, right=None, first=None, keep=None, pbb=None):
    f = par.paragraph_format
    if before is not None:
        f.space_before = Pt(before)
    if after is not None:
        f.space_after = Pt(after)
    if line is not None:
        if line == 1:
            f.line_spacing_rule = WD_LINE_SPACING.SINGLE
        else:
            f.line_spacing = line
    if align is not None:
        par.alignment = align
    if left is not None:
        f.left_indent = Mm(left)
    if right is not None:
        f.right_indent = Mm(right)
    if first is not None:
        f.first_line_indent = Mm(first)
    if keep is not None:
        f.keep_with_next = keep
    if pbb is not None:
        f.page_break_before = pbb
    return par


def tab_stop(par, pos_mm, align="left"):
    pPr = par._p.get_or_add_pPr()
    tabs = pPr.find(qn("w:tabs"))
    if tabs is None:
        tabs = OxmlElement("w:tabs")
        pPr.append(tabs)
    t = OxmlElement("w:tab")
    t.set(qn("w:val"), align)
    t.set(qn("w:pos"), str(int(pos_mm * 56.7)))
    tabs.append(t)


# ============================================================================================ document & styles
doc = Document()
st = doc.styles
normal = st["Normal"]
normal.font.name = TNR
normal.font.size = Pt(12)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), TNR)
normal.element.rPr.rFonts.set(qn("w:cs"), TNR)
nf = normal.paragraph_format
nf.line_spacing = 1.5
nf.space_before = Pt(12)
nf.space_after = Pt(12)
nf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
nf.widow_control = True


def pstyle(name, base="Normal", size=12, bold=False, italic=False, line=1.0, before=0, after=0, align=None, font=TNR,
           left=0, hanging=0, keep=False, outline=None):
    s = st[name] if name in [x.name for x in st] else st.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    s.base_style = st[base] if base and name != base else None
    s.font.name = font
    s.font.size = Pt(size)
    s.font.bold = bold
    s.font.italic = italic
    s.font.color.rgb = RGBColor(0, 0, 0)
    rpr = s.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), font)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
    f = s.paragraph_format
    if line == 1.0:
        f.line_spacing_rule = WD_LINE_SPACING.SINGLE
    else:
        f.line_spacing = line
    f.space_before = Pt(before)
    f.space_after = Pt(after)
    if align is not None:
        f.alignment = align
    f.left_indent = Mm(left + hanging) if hanging else Mm(left)
    f.first_line_indent = Mm(-hanging) if hanging else Mm(0)
    f.keep_with_next = keep
    if outline is not None:
        pPr = s.element.get_or_add_pPr()
        ol = OxmlElement("w:outlineLvl")
        ol.set(qn("w:val"), str(outline))
        pPr.append(ol)
    return s


C = WD_ALIGN_PARAGRAPH.CENTER
J = WD_ALIGN_PARAGRAPH.JUSTIFY
L = WD_ALIGN_PARAGRAPH.LEFT
pstyle("MIST Chapter", size=12, bold=True, line=1.0, before=0, after=24, align=C, keep=True)
pstyle("MIST Front Heading", size=12, bold=True, line=1.0, before=0, after=18, align=C, keep=True)
for lvl, (bef, aft, tab) in {2: (24, 18, 16), 3: (24, 18, 20), 4: (18, 12, 24)}.items():
    s = pstyle(f"Heading {lvl}", base=None, size=12, bold=True, line=1.0, before=bef, after=aft, align=L, keep=True, outline=lvl - 1)
pstyle("MIST AppHeading", size=12, bold=True, line=1.0, before=18, after=12, align=L, keep=True)
pstyle("MIST List", line=1.5, before=2, after=2, align=J)
pstyle("MIST Quote", line=1.5, before=6, after=6, align=J, left=10)
pstyle("MIST Caption", line=1.0, before=12, after=9, align=L, keep=True)
pstyle("MIST FigCaption", line=1.0, before=9, after=12, align=L)
pstyle("MIST TableText", size=10, line=1.0, before=3, after=3, align=C)
pstyle("MIST TableNote", size=10, line=1.0, before=3, after=12, align=J)
pstyle("MIST Code", size=8, line=1.0, before=0, after=0, align=L, font="Courier New")
pstyle("MIST Reference", line=1.0, before=10, after=10, align=J, hanging=13)
pstyle("MIST Equation", line=1.0, before=12, after=12, align=L)
pstyle("MIST Figure", line=1.0, before=12, after=0, align=C, keep=True)
pstyle("MIST Notation", line=1.0, before=6, after=6, align=L)
pstyle("MIST Cover", size=16, bold=True, line=1.0, before=0, after=0, align=C, font="Arial")
pstyle("MIST TitlePage", size=12, line=1.0, before=0, after=0, align=C)
# TOC 1-5 (TP.14-TP.16) are Word built-in styles; they are redefined in word_finalize.ps1 before the lists are built.


# ============================================================================================ sections
def setup_section(sec, left=32, right=25, top=25, bottom=25, fmt=None, start=None, prefix=None, footer=True):
    sec.page_width, sec.page_height = Mm(210), Mm(297)
    sec.left_margin, sec.right_margin, sec.top_margin, sec.bottom_margin = Mm(left), Mm(right), Mm(top), Mm(bottom)
    sec.footer_distance = Mm(19)
    sec.header_distance = Mm(12.5)
    sectPr = sec._sectPr
    for old in sectPr.findall(qn("w:pgNumType")):
        sectPr.remove(old)
    if fmt:
        pg = OxmlElement("w:pgNumType")
        pg.set(qn("w:fmt"), fmt)
        if start is not None:
            pg.set(qn("w:start"), str(start))
        sectPr.append(pg)
    ft = sec.footer
    ft.is_linked_to_previous = False
    p = ft.paragraphs[0]
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    hd = sec.header
    hd.is_linked_to_previous = False
    if footer:
        pf(p, before=0, after=0, line=1, align=C)
        if prefix:
            p.add_run(prefix)
        add_field(p, "PAGE", "1")
    return sec


def new_section(**kw):
    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    return setup_section(sec, **kw)


# ============================================================================================ inline markdown
INL = re.compile(r"(\*\*(?:[^*]|\*(?!\*))+?\*\*|(?<![\w*\\])\*(?![\s*])(?:[^*\n])+?(?<![\s\\])\*(?![\w*])|`[^`]+`|<br\s*/?>)")
BANGLA = re.compile(r"([\u0980-\u09FF][\u0980-\u09FF\s,;:।\-–—.!?%০-৯()]*[\u0980-\u09FF%)]|[\u0980-\u09FF])")


def clean(s):
    s = s.replace("\\|", "|").replace("\\*", "*").replace("\\_", "_")
    s = re.sub(r"\[([^\]]+)\]\((?:https?://|#)[^)]+\)", r"\1", s)
    return s


SUB_ON = {"on": True}
SUBRE = re.compile(r"_(\{(?:[^{}]|\{[^{}]*\})*\}|[A-Za-z0-9α-ωΑ-Ω′+\x01]+(?:[,\-−][A-Za-z0-9α-ω′+]+)*)")


def _sub_ok(text, m):
    s = text.rfind(" ", 0, m.start()) + 1
    e = text.find(" ", m.end())
    word = text[s:e if e >= 0 else len(text)]
    if word.count("_") >= 2 and "{" not in word:
        return False
    if re.search(r"\.(csv|json|md|py|txt|npz|pth)\b", word) or "/" in word or "10." in word[:3]:
        return False
    if m.start() == 0 or text[m.start() - 1].isspace():
        return False
    base = re.search(r"([0-9A-Za-zα-ωΑ-Ω′]*)$", text[:m.start()]).group(1)
    if len(base) >= 3 and not m.group(1).startswith("{"):
        return False          # data identifiers such as resnet8_c10, pilot_n100000, seed700_x
    return True


def put(par, text, bold=False, italic=False, size=None, code=False, font=None):
    if not text:
        return
    if SUB_ON["on"] and not code and "_" in text:
        pos, done = 0, False
        for m in SUBRE.finditer(text):
            if not _sub_ok(text, m):
                continue
            done = True
            put_plain(par, text[pos:m.start()], bold, italic, size, code, font)
            s = m.group(1)
            if s.startswith("{"):
                s = s[1:-1].replace("_", "")
            sb = bold
            if s.startswith("\x01"):
                s, sb = s[1:], True
            r = par.add_run(s.replace("\x01", ""))
            r.font.subscript = True
            if sb:
                r.bold = True
            if italic:
                r.italic = True
            if size:
                r.font.size = Pt(size)
            if font:
                fonts(r, font, size)
            pos = m.end()
        if done:
            put_plain(par, text[pos:], bold, italic, size, code, font)
            return
    put_plain(par, text, bold, italic, size, code, font)


def put_plain(par, text, bold=False, italic=False, size=None, code=False, font=None):
    if not text:
        return
    if BANGLA.search(text) and not code:
        pos = 0
        for m in BANGLA.finditer(text):
            if m.start() > pos:
                put(par, text[pos:m.start()], bold, italic, size, code, font)
            r = par.add_run(m.group(0))
            r.bold, r.italic = bold or None, italic or None
            fonts(r, BN, 14, cs=BN)
            rpr = r._r.get_or_add_rPr()
            lang = OxmlElement("w:lang")
            lang.set(qn("w:bidi"), "bn-BD")
            rpr.append(lang)
            pos = m.end()
        if pos < len(text):
            put(par, text[pos:], bold, italic, size, code, font)
        return
    r = par.add_run(text)
    if bold:
        r.bold = True
    if italic:
        r.italic = True
    if code:
        fonts(r, "Courier New", (size or 12) - 1)
    elif font:
        fonts(r, font, size)
    elif size:
        r.font.size = Pt(size)


def inline(par, text, size=None, bold=False, italic=False, font=None):
    text = clean(text)
    text = re.sub(r"_\*\*([^*]+)\*\*", lambda m_: "_{\x01" + m_.group(1) + "}", text)
    for tok in INL.split(text):
        if not tok:
            continue
        if re.fullmatch(r"<br\s*/?>", tok):
            par.add_run().add_break()
        elif tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            inline(par, tok[2:-2], size, True, italic, font)
        elif tok.startswith("`") and tok.endswith("`"):
            put(par, tok[1:-1], bold, italic, size, code=True)
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            inline(par, tok[1:-1], size, bold, True, font)
        else:
            put(par, tok, bold, italic, size, font=font)


def para(text, style=None, **kw):
    p = doc.add_paragraph(style=style)
    inline(p, text, **{k: v for k, v in kw.items() if k in ("size", "bold", "italic", "font")})
    return p


# ============================================================================================ block parser
LIST_RE = re.compile(r"^(\s*)([-*+]|\d+\.)\s+(.*)")


def is_block_start(line, nxt=""):
    if line.startswith("|"):
        return bool(re.match(r"^\|\s*:?-{2,}", nxt))
    return bool(re.match(r"^(#{1,4} |```|>|!\[|\s*\$\$)", line)) or bool(LIST_RE.match(line))


def parse(md):
    lines = md.split("\n")
    i, out = 0, []
    while i < len(lines):
        l = lines[i]
        if not l.strip():
            i += 1
            continue
        if l.startswith("```"):
            j, buf = i + 1, []
            while j < len(lines) and not lines[j].startswith("```"):
                buf.append(lines[j])
                j += 1
            out.append(("code", buf))
            i = j + 1
            continue
        m = re.match(r"^(#{1,4}) (.*)", l)
        if m:
            out.append(("h", len(m.group(1)), m.group(2).strip()))
            i += 1
            continue
        if re.match(r"^\s*\$\$.*\$\$\s*$", l):
            out.append(("eq", l.strip()[2:-2]))
            i += 1
            continue
        if l.startswith("|") and i + 1 < len(lines) and re.match(r"^\|\s*:?-{2,}", lines[i + 1]):
            buf = []
            while i < len(lines) and lines[i].startswith("|"):
                buf.append(lines[i])
                i += 1
            out.append(("table", buf))
            continue
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", l)
        if m:
            out.append(("img", m.group(1), m.group(2)))
            i += 1
            continue
        if l.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i][1:].strip())
                i += 1
            paras, cur = [], []
            for b in buf:
                if not b:
                    if cur:
                        paras.append(" ".join(cur))
                        cur = []
                else:
                    cur.append(b)
            if cur:
                paras.append(" ".join(cur))
            out.append(("quote", paras))
            continue
        if LIST_RE.match(l):
            items = []          # [indent, ordered, text, kind]  kind: item | cont
            while i < len(lines):
                l2 = lines[i]
                m2 = LIST_RE.match(l2)
                if m2:
                    items.append([len(m2.group(1)), m2.group(2)[0].isdigit(), m2.group(3).strip(), "item",
                                  int(m2.group(2)[:-1]) if m2.group(2)[0].isdigit() else None])
                    i += 1
                    continue
                if not l2.strip():
                    k = i + 1
                    while k < len(lines) and not lines[k].strip():
                        k += 1
                    if k < len(lines) and (LIST_RE.match(lines[k]) or (lines[k].startswith("  ") and not is_block_start(lines[k].lstrip()))):
                        if not LIST_RE.match(lines[k]):
                            ind = len(lines[k]) - len(lines[k].lstrip())
                            items.append([ind, False, lines[k].strip(), "cont", None])
                            i = k + 1
                            continue
                        i = k
                        continue
                    break
                if is_block_start(l2.lstrip()) and not l2.startswith("  "):
                    break
                if l2.startswith(" ") or not is_block_start(l2):
                    items[-1][2] += " " + l2.strip()
                    i += 1
                    continue
                break
            out.append(("list", items))
            continue
        buf = [l.strip()]
        i += 1
        while i < len(lines) and lines[i].strip() and not is_block_start(lines[i], lines[i + 1] if i + 1 < len(lines) else ""):
            buf.append(lines[i].strip())
            i += 1
        out.append(("p", " ".join(buf)))
    return out


# ============================================================================================ renderers
ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii",
         "xviii", "xix", "xx", "xxi", "xxii", "xxiii", "xxiv", "xxv"]


def render_list(items):
    # level from indent
    levels = sorted({it[0] for it in items if it[3] == "item"})

    def lvl(ind):
        return min(range(len(levels)), key=lambda k: abs(levels[k] - ind)) if levels else 0
    counters, kinds = {}, {}
    last_item_level = 0
    for ind, ordered, text, kind, srcnum in items:
        if kind == "cont":
            lv = last_item_level
            p = doc.add_paragraph(style="MIST List")
            pf(p, left=10 + 10 * lv)
            inline(p, text)
            continue
        lv = lvl(ind)
        for k in list(counters):
            if k > lv:
                del counters[k]
                kinds.pop(k, None)
        counters[lv] = counters.get(lv, 0) + 1
        if ordered and srcnum:
            counters[lv] = srcnum
        n = counters[lv] - 1
        parent_kind = kinds.get(lv - 1)
        if lv == 0:
            style = "roman" if ordered else "alpha"
        elif lv == 1:
            style = "alpha" if parent_kind == "roman" else "roman"
        else:
            style = "dash"
        kinds[lv] = style
        lab = f"({ROMAN[n]})" if style == "roman" else (f"({chr(97 + n % 26)})" if style == "alpha" else "–")
        p = doc.add_paragraph(style="MIST List")
        pf(p, left=10 + 10 * lv, first=-10 if lv < 2 else -6)
        tab_stop(p, 10 + 10 * lv)
        p.add_run(lab + "\t")
        inline(p, text)
        last_item_level = lv


def render_quote(paras):
    for t in paras:
        p = doc.add_paragraph(style="MIST Quote")
        pf(p, right=10)
        inline(p, t)


def render_code(lines):
    t = doc.add_table(rows=1, cols=1)
    LOG["all_tables"].append({"kind": "code", "label": lines[0][:40] if lines else ""})
    set_borders(t, sz=4)
    fixed_layout(t, [TEXT_W])
    cell = t.rows[0].cells[0]
    first = True
    for ln in lines:
        p = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        p.style = st["MIST Code"]
        r = p.add_run(ln.replace("\t", "    ") or " ")
        fonts(r, "Courier New", 8)
    sp = doc.add_paragraph()
    pf(sp, before=0, after=6, line=1)


def prep_tex(s):
    tag = re.search(r"\\tag\{([^}]*)\}", s)
    num = tag.group(1) if tag else ""
    s = re.sub(r"\\tag\{[^}]*\}", "", s)
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = re.sub(r"\\[Bb]ig[glr]?(?![a-zA-Z])", "", s)
    s = s.replace(r"\!", "")
    s = re.sub(r"\\\\\[[^\]]*\]", r"\\\\", s)
    s = s.replace(r"\arg\max", r"\operatorname{argmax}")
    s = s.replace(r"\{", r"\lbrace ").replace(r"\}", r"\rbrace ")
    return s.strip(), num


M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def omml(tex):
    mml = l2m(tex)
    res = XSL(etree.fromstring(mml.encode("utf-8")))
    root = res.getroot()
    for r in root.iter(f"{{{M_NS}}}r"):
        for old in r.findall(qn("w:rPr")):
            r.remove(old)
        rpr = etree.SubElement(r, qn("w:rPr"))
        rf = etree.SubElement(rpr, qn("w:rFonts"))
        rf.set(qn("w:ascii"), "Cambria Math")
        rf.set(qn("w:hAnsi"), "Cambria Math")
        sz = etree.SubElement(rpr, qn("w:sz"))
        sz.set(qn("w:val"), "28")
        szc = etree.SubElement(rpr, qn("w:szCs"))
        szc.set(qn("w:val"), "28")
        t = r.find(f"{{{M_NS}}}t")
        if t is not None:     # w:rPr must precede m:t
            r.remove(rpr)
            t.addprevious(rpr)
            if re.search(r"[A-Za-z]{2,}", t.text or ""):
                mr = r.find(f"{{{M_NS}}}rPr")
                if mr is None:
                    mr = etree.Element(f"{{{M_NS}}}rPr")
                    r.insert(0, mr)
                sty = etree.SubElement(mr, f"{{{M_NS}}}sty")
                sty.set(f"{{{M_NS}}}val", "p")
    return root


def render_eq(tex):
    s, num = prep_tex(tex)
    t = doc.add_table(rows=1, cols=2)
    LOG["all_tables"].append({"kind": "eq", "label": num})
    set_borders(t, none=True)
    fixed_layout(t, [TEXT_W - 19, 19])
    c0, c1 = t.rows[0].cells
    p = c0.paragraphs[0]
    p.style = st["MIST Equation"]
    pf(p, left=19)
    try:
        p._p.append(omml(s))
    except Exception as e:  # noqa: BLE001
        LOG["warnings"].append(f"equation {num}: {e}")
        p.add_run(s)
    q = c1.paragraphs[0]
    q.style = st["MIST Equation"]
    q.add_run(f"({num})" if num else "")
    c0.vertical_alignment = c1.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    LOG["equations"] += 1


def split_row(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    cells, cur, esc = [], "", False
    for ch in line:
        if ch == "\\" and not esc:
            esc = True
            cur += ch
            continue
        if ch == "|" and not esc:
            cells.append(cur.strip())
            cur = ""
        else:
            cur += ch
        esc = False
    cells.append(cur.strip())
    return cells


LAND_W = 297 - 25 - 25


def _rows(lines):
    rows = [split_row(l) for l in lines if not re.match(r"^\|\s*:?-{2,}", l)]
    ncol = max(len(r) for r in rows)
    return [r + [""] * (ncol - len(r)) for r in rows], ncol


def _tw(tok, size, bold=False, code=False):
    """Approximate rendered width (mm) of an unbreakable token in Times New Roman / Courier New at `size` pt."""
    if code:
        return len(tok) * 0.60 * size * 0.3528
    em = 0.0
    for c in tok:
        if c.isupper():
            em += 0.68
        elif c.isdigit():
            em += 0.50
        elif c in "mwMW":
            em += 0.75
        elif c in "iljtf.,;:'|()[]":
            em += 0.30
        elif c.isalpha():
            em += 0.47
        else:
            em += 0.50
    return em * size * 0.3528 * (1.08 if bold else 1.0)


def _plan(rows, ncol, avail):
    start = 10 if ncol <= 5 else (9 if ncol <= 7 else 8)
    pad = 3.9
    for size in [s for s in (10, 9, 8) if s <= start]:
        minw, pref = [], []
        for j in range(ncol):
            mw, full = 6.0, 0.0
            for i, r in enumerate(rows):
                c = r[j]
                bold = i == 0 or c.strip().startswith("**")
                for code in re.findall(r"`([^`]+)`", c):
                    for tok in code.split():
                        mw = max(mw, _tw(tok, size, code=True) + pad)
                plain = re.sub(r"`[^`]+`", " ", c).replace("*", "")
                for tok in re.split(r"[\s/]+", plain):
                    if tok:
                        mw = max(mw, _tw(tok, size, bold) + pad)
                if i > 0:
                    full = max(full, min(_tw(re.sub(r"[*`]", "", c), size), 85) + pad)
            minw.append(mw)
            pref.append(max(mw, full))
        if sum(minw) <= avail:
            break
    if sum(pref) <= avail:
        widths = [x * avail / sum(pref) for x in pref]
    elif sum(minw) <= avail:
        extra = avail - sum(minw)
        span = [p_ - m_ for p_, m_ in zip(pref, minw)]
        widths = [m_ + extra * s_ / (sum(span) or 1) for m_, s_ in zip(minw, span)]
    else:
        widths = [m_ * avail / sum(minw) for m_ in minw]
    return size, widths, sum(minw)


def table_needs_landscape(lines):
    rows, ncol = _rows(lines)
    _, _, need = _plan(rows, ncol, TEXT_W)
    return ncol >= 10 or need > TEXT_W * 1.02


def render_table(lines, label, avail=None):
    rows, ncol = _rows(lines)
    size, widths, _ = _plan(rows, ncol, avail or TEXT_W)
    text_col = []
    for j in range(ncol):
        body = [re.sub(r"[*`]", "", r[j]) for r in rows[1:]] or [""]
        text_col.append(j == 0 or (sum(len(x) for x in body) / len(body)) > 18)
    t = doc.add_table(rows=len(rows), cols=ncol)
    LOG["all_tables"].append({"kind": "content", "label": label})
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_borders(t, sz=4)
    set_cell_margins(t)
    fixed_layout(t, widths)
    keep_rows = len(rows) - 1 if len(rows) <= 25 else 3
    for i, r in enumerate(rows):
        row = t.rows[i]
        cant_split(row)
        if i == 0:
            repeat_header(row)
        for j, val in enumerate(r):
            p = row.cells[j].paragraphs[0]
            p.style = st["MIST TableText"]
            if text_col[j] and i > 0:
                p.alignment = L
            if i < keep_rows:
                p.paragraph_format.keep_with_next = True
            inline(p, val, size=size, bold=(i == 0))
    LOG["tables"].append({"label": label, "rows": len(rows), "cols": ncol, "font": size})


def _orient(landscape):
    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    if landscape:
        sec.orientation = WD_ORIENT.LANDSCAPE
        sec.page_width, sec.page_height = Mm(297), Mm(210)
        sec.left_margin, sec.right_margin, sec.top_margin, sec.bottom_margin = Mm(25), Mm(25), Mm(32), Mm(25)
    else:
        sec.orientation = WD_ORIENT.PORTRAIT
        sec.page_width, sec.page_height = Mm(210), Mm(297)
        sec.left_margin, sec.right_margin, sec.top_margin, sec.bottom_margin = Mm(32), Mm(25), Mm(25), Mm(25)
    sec.footer_distance = Mm(19)
    sectPr = sec._sectPr
    for pg in sectPr.findall(qn("w:pgNumType")):
        if pg.get(qn("w:start")) is not None:
            del pg.attrib[qn("w:start")]
    sec.footer.is_linked_to_previous = True
    sec.header.is_linked_to_previous = True
    return sec


FIG_DIR = os.path.join(W, "figures", "final")


def render_img(path, caption_parts):
    full = os.path.normpath(os.path.join(W, path))
    im = Image.open(full)
    wpx, hpx = im.size
    dpi = im.info.get("dpi", (300, 300))[0] or 300
    nat_w_mm = wpx / dpi * 25.4
    w = 120 if nat_w_mm < 110 else 150
    h = w * hpx / wpx
    if h > 190:
        w, h = w * 190 / h, 190
    large = h > TEXT_H / 2
    p = doc.add_paragraph(style="MIST Figure")
    p.add_run().add_picture(full, width=Mm(w))
    num, title, rest = caption_parts
    cp = doc.add_paragraph(style="MIST FigCaption")
    if large:
        cp.alignment = C
    inline(cp, f"Fig. {num}: {title}" + (f" {rest}" if rest else ""))
    tc(cp, f"Figure {num}: {strip_md(title).rstrip('.')}", "F", 5)
    LOG["figures"].append({"num": num, "w_mm": round(w, 1), "h_mm": round(h, 1), "large": large})


def strip_md(s):
    return re.sub(r"[*`]", "", clean(s)).strip()


CAPRE = re.compile(r"^\*\*(Table|Figure) ([A-Z]?[\d.]+)\. (.+?)\*\*\s*(.*)$", re.S)


def render_blocks(bl, appendix=False):
    i = 0
    while i < len(bl):
        b = bl[i]
        kind = b[0]
        if kind == "h":
            lvl, text = b[1], b[2]
            if lvl == 2 and not appendix:
                m = re.match(r"^(\d+\.\d+)\s+(.*)", text)
                p = doc.add_paragraph(style="Heading 2")
                tab_stop(p, 16)
                p.add_run(f"{m.group(1)}\t{strip_md(m.group(2))}" if m else strip_md(text))
            elif lvl == 3 and not appendix:
                m = re.match(r"^(\d+\.\d+\.\d+)\s+(.*)", text)
                p = doc.add_paragraph(style="Heading 3")
                tab_stop(p, 20)
                p.add_run(f"{m.group(1)}\t{strip_md(m.group(2))}" if m else strip_md(text))
            elif lvl == 4 and not appendix:
                p = doc.add_paragraph(style="Heading 4")
                p.add_run(strip_md(text))
            else:
                p = doc.add_paragraph(style="MIST AppHeading")
                p.add_run(strip_md(text))
        elif kind == "p":
            text = b[1]
            m = CAPRE.match(text)
            if m and m.group(1) == "Table" and i + 1 < len(bl) and bl[i + 1][0] == "table":
                num, title, rest = m.group(2), m.group(3).strip(), m.group(4).strip()
                land = table_needs_landscape(bl[i + 1][1])
                if land:
                    _orient(True)
                    LOG.setdefault("landscape", []).append(num)
                cp = doc.add_paragraph(style="MIST Caption")
                if land:
                    pf(cp, before=0)
                inline(cp, f"Table {num}: {title}" + (f" {rest}" if rest else ""))
                tc(cp, f"Table {num}: {strip_md(title).rstrip('.')}", "T", 5)
                render_table(bl[i + 1][1], num, avail=LAND_W if land else None)
                i += 2
                # a following italic note belongs to the table
                if i < len(bl) and bl[i][0] == "p" and bl[i][1].startswith("*") and not bl[i][1].startswith("**"):
                    p = doc.add_paragraph(style="MIST TableNote")
                    inline(p, bl[i][1])
                    i += 1
                if land:
                    _orient(False)
                continue
            if text.startswith("*") and not text.startswith("**") and i > 0 and bl[i - 1][0] == "table":
                p = doc.add_paragraph(style="MIST TableNote")
                inline(p, text)
            else:
                p = doc.add_paragraph()
                inline(p, text)
                if i + 1 < len(bl) and bl[i + 1][0] == "list":
                    p.paragraph_format.keep_with_next = True
        elif kind == "img":
            cap = ("?", "", "")
            if i + 1 < len(bl) and bl[i + 1][0] == "p":
                m = CAPRE.match(bl[i + 1][1])
                if m and m.group(1) == "Figure":
                    cap = (m.group(2), m.group(3).strip(), m.group(4).strip())
                    i += 1
            if cap[0] == "?":
                LOG["warnings"].append(f"figure without caption: {b[2]}")
            render_img(b[2], cap)
        elif kind == "table":
            LOG["warnings"].append(f"table without caption: {b[1][0][:80]}")
            render_table(b[1], "uncaptioned")
        elif kind == "list":
            render_list(b[1])
        elif kind == "quote":
            render_quote(b[1])
        elif kind == "code":
            render_code(b[1])
        elif kind == "eq":
            render_eq(b[1])
        i += 1


def chapter_heading(num, title, label="CHAPTER"):
    p = doc.add_paragraph(style="MIST Chapter")
    pf(p, pbb=True)
    p.add_run(f"{label} {num}".strip())
    p.add_run().add_break()
    p.add_run(title.upper())
    tc(p, f"{label} {num}: {title.upper()}".strip(), "C", 1)
    return p


def front_heading(text, toc_text=None, first=False):
    p = doc.add_paragraph(style="MIST Front Heading")
    if not first:
        pf(p, pbb=True)
    inline(p, text)
    if toc_text:
        tc(p, toc_text, "C", 1)
    return p


def spacer(mm):
    p = doc.add_paragraph()
    pf(p, before=0, after=0, line=1)
    p.paragraph_format.space_before = Pt(mm * 72 / 25.4)
    return p


def monogram_box():
    t = doc.add_table(rows=1, cols=1)
    LOG["all_tables"].append({"kind": "layout", "label": "monogram"})
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_borders(t, sz=4)
    fixed_layout(t, [30])
    row = t.rows[0]
    trPr = row._tr.get_or_add_trPr()
    h = OxmlElement("w:trHeight")
    h.set(qn("w:val"), str(int(28 * 56.7)))
    h.set(qn("w:hRule"), "exact")
    trPr.append(h)
    c = row.cells[0]
    c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    p = c.paragraphs[0]
    pf(p, before=0, after=0, line=1, align=C)
    r = p.add_run("[MIST MONOGRAM 28 mm × 30 mm]")
    fonts(r, TNR, 8)


# ============================================================================================ build
m = META
TITLE = m["title"]

# ---- A: cover (TP.03)
setup_section(doc.sections[0], left=35, right=25, top=35, bottom=35, footer=False)
p = doc.paragraphs[0] if doc.paragraphs else doc.add_paragraph()
p.style = st["MIST Cover"]
r = p.add_run(TITLE.upper())
fonts(r, "Arial", 18)
spacer(32)
para(m["author"].upper(), "MIST Cover")
spacer(30)
para("M.Sc. ENGINEERING THESIS", "MIST Cover")
spacer(16)
monogram_box()
spacer(8)
for ln in (m["department"].upper(), m["institution"].upper(), m["city"].upper()):
    para(ln, "MIST Cover")
spacer(6)
para(m["month_year"], "MIST Cover")

# ---- B: title page, approval, declaration (no page numbers)
new_section(left=32, right=25, top=25, bottom=25, footer=False)
spacer(10)
p = para(TITLE.upper(), "MIST TitlePage", size=14, bold=True)
spacer(38)
para(f"{m['author'].upper()} (SN. {m['student_number']})", "MIST TitlePage")
spacer(30)
para(f"A Thesis Submitted in Partial Fulfillment of the Requirements for the Degree of {m['degree_statement']}", "MIST TitlePage")
spacer(30)
monogram_box()
spacer(12)
for ln in (m["department"].upper(), m["institution"].upper(), m["city"].upper()):
    para(ln, "MIST TitlePage")
spacer(8)
para(m["month_year"], "MIST TitlePage")

# approval (TP.08)
p = para(TITLE.upper(), "MIST TitlePage", size=14, bold=True)
pf(p, pbb=True, before=20)
spacer(12)
for ln in ("M.Sc. Engineering Thesis", "By", f"{m['author'].upper()} (SN. {m['student_number']})"):
    para(ln, "MIST TitlePage")
spacer(10)
p = para(f"Approved as to style and content by the Board of Examination on {m['exam_date']}:", "MIST TitlePage")
p.alignment = L
for name, role in m["examiners"]:
    t = doc.add_table(rows=1, cols=2)
    LOG["all_tables"].append({"kind": "layout", "label": "approval"})
    set_borders(t, none=True)
    fixed_layout(t, [95, 58])
    a, b_ = t.rows[0].cells
    pa = a.paragraphs[0]
    pf(pa, before=20, after=0, line=1, align=L)
    pa.add_run("______________________________")
    for ln in name:
        q = a.add_paragraph()
        pf(q, before=0, after=0, line=1, align=L)
        q.add_run(ln)
    pb = b_.paragraphs[0]
    pf(pb, before=26, after=0, line=1, align=L)
    for k, ln in enumerate(role):
        if k:
            pb.add_run().add_break()
        pb.add_run(ln)
spacer(14)
para(f"{m['department']}, MIST, Dhaka", "MIST TitlePage")

# declaration (TP.09)
p = para(TITLE.upper(), "MIST TitlePage", size=14, bold=True)
pf(p, pbb=True, before=36)
spacer(40)
para("DECLARATION", "MIST TitlePage", size=14, bold=True)
spacer(8)
p = para(m["declaration_text"])
pf(p, line=1.5, align=J)
spacer(18)
p = para("______________________________", "MIST TitlePage")
p.alignment = L
p = para(m["author"], "MIST TitlePage")
p.alignment = L
spacer(14)
para(f"{m['department']}, MIST, Dhaka", "MIST TitlePage")

# ---- C: preliminary pages, Roman numerals
new_section(fmt="lowerRoman", start=1)
abs_en = open(os.path.join(W, "abstract.md"), encoding="utf-8").read()
front_heading("ABSTRACT", "Abstract", first=True)
p = para(TITLE, "MIST TitlePage")
pf(p, before=6, after=24)
body = re.sub(r"^# Abstract\s*", "", abs_en).strip()
for bl_ in parse(body):
    if bl_[0] == "p":
        q = doc.add_paragraph()
        pf(q, before=0, after=12)
        inline(q, bl_[1])
abs_bn = open(os.path.join(W, "abstract_bangla.md"), encoding="utf-8").read()
h = front_heading("সারসংক্ষেপ", "Abstract in Bangla (সারসংক্ষেপ)")
p = para(TITLE, "MIST TitlePage")
pf(p, before=6, after=24)
for bl_ in parse(re.sub(r"^# .*\n", "", abs_bn)):
    if bl_[0] == "p":
        q = doc.add_paragraph()
        pf(q, before=0, after=8)
        q.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        q.paragraph_format.line_spacing = Pt(22)
        inline(q, bl_[1])
front_heading("ACKNOWLEDGEMENTS", "Acknowledgements")
for t_ in m["acknowledgements_placeholder"]:
    q = para(t_)
    pf(q, before=12, after=12)
front_heading("TABLE OF CONTENTS", "Table of Contents")
p = doc.add_paragraph()
pf(p, before=0, after=0, line=1, align=L)
add_field(p, 'TOC \\f C \\o "2-3" \\h \\z', "[Table of contents: update fields]")
front_heading("LIST OF FIGURES", "List of Figures")
p = doc.add_paragraph()
pf(p, before=0, after=0, line=1, align=L)
add_field(p, "TOC \\f F \\h \\z", "[List of figures: update fields]")
front_heading("LIST OF TABLES", "List of Tables")
p = doc.add_paragraph()
pf(p, before=0, after=0, line=1, align=L)
add_field(p, "TOC \\f T \\h \\z", "[List of tables: update fields]")

# notation (TP.17) and abbreviations
front_heading("LIST OF MAIN NOTATION AND ABBREVIATIONS", "List of Main Notation and Abbreviations")


def two_col(rows, w0=43):
    t = doc.add_table(rows=len(rows), cols=2)
    LOG["all_tables"].append({"kind": "layout", "label": "notation list"})
    set_borders(t, none=True)
    fixed_layout(t, [w0, TEXT_W - w0])
    for i_, (a_, b2) in enumerate(rows):
        for j_, v in enumerate((a_, b2)):
            q = t.rows[i_].cells[j_].paragraphs[0]
            q.style = st["MIST Notation"]
            inline(q, v)


def md_tables(path):
    out, cur, sec_ = [], [], None
    for ln in open(path, encoding="utf-8").read().split("\n"):
        if ln.startswith("## "):
            sec_ = ln[3:].strip()
        if ln.startswith("|") and not re.match(r"^\|\s*-{2,}", ln):
            cur.append((sec_, split_row(ln)))
    return cur


nota = [r for s_, r in md_tables(os.path.join(W, "final_notation_inventory.md")) if r and r[0] not in ("Symbol",)]
p = doc.add_paragraph(style="MIST AppHeading")
p.add_run("Main notation")
two_col([(r[0], r[1]) for r in nota])
abbr = md_tables(os.path.join(W, "list_of_abbreviations.md"))
for secname in ("General abbreviations", "Method, model and dataset names", "Experiment labels (thesis notation)"):
    p = doc.add_paragraph(style="MIST AppHeading")
    p.add_run(secname[0].upper() + secname[1:])
    two_col([(r[0], r[1] + (f" ({r[2]})" if len(r) > 2 and r[2] else "")) for s_, r in abbr if s_ == secname and r[0] not in ("Abbreviation", "Name", "Label")], w0=43)

# ---- D: main text, Arabic numerals
new_section(fmt="decimal", start=1)
first_chapter = True
for f in CH:
    t = fix_figs(RM.cite_text(open(os.path.join(W, f), encoding="utf-8").read()))
    bl = parse(t)
    assert bl[0][0] == "h" and bl[0][1] == 1, f
    mm = re.match(r"Chapter (\d+)\s+[—-]\s+(.*)", bl[0][2])
    hp = chapter_heading(mm.group(1), mm.group(2))
    if first_chapter:
        pf(hp, pbb=False)
        first_chapter = False
    render_blocks(bl[1:])

# references
p = doc.add_paragraph(style="MIST Chapter")
pf(p, pbb=True)
p.add_run("REFERENCES")
tc(p, "REFERENCES", "C", 1)
SUB_ON["on"] = False
for ln in open(os.path.join(W, "references_mist_final.md"), encoding="utf-8").read().split("\n"):
    if ln.strip() and not ln.startswith("#"):
        q = doc.add_paragraph(style="MIST Reference")
        inline(q, ln.strip())
SUB_ON["on"] = True

# ---- E: appendices, one section each, A-1, B-1, ...
apx = open(os.path.join(W, "appendices.md"), encoding="utf-8").read()
intro, *parts = re.split(r"(?m)^(?=## Appendix [A-Z] — )", apx)
for k, part in enumerate(parts):
    mm = re.match(r"## Appendix ([A-Z]) — (.*)", part)
    letter, title = mm.group(1), mm.group(2).strip()
    new_section(fmt="decimal", start=1, prefix=f"{letter}-")
    p = doc.add_paragraph(style="MIST Chapter")
    p.add_run(f"APPENDIX {letter}")
    p.add_run().add_break()
    p.add_run(title.upper())
    tc(p, f"APPENDIX {letter}: {title.upper()}", "C", 1)
    if k == 0:
        intro_text = re.sub(r"^# Appendices\s*", "", intro).strip()
        if intro_text:
            q = doc.add_paragraph()
            inline(q, " ".join(intro_text.split()))
    render_blocks(parse(part.split("\n", 1)[1]), appendix=True)

out_path = os.path.join(W, "final_thesis_MIST.docx")
doc.core_properties.title = TITLE
doc.core_properties.author = m["author"]
doc.core_properties.subject = "M.Sc. Engineering thesis, " + m["department"] + ", " + m["institution"]
doc.save(out_path)
json.dump(LOG, open(os.path.join(HERE, "_docx_build_log.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"saved {out_path}; tables {len(LOG['tables'])}; figures {len(LOG['figures'])}; equations {LOG['equations']}; warnings {len(LOG['warnings'])}")
for w_ in LOG["warnings"]:
    print("WARNING:", w_)
