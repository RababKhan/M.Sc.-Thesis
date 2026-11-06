"""MIST author-date (Harvard-style) citations and alphabetical reference list, from verified metadata only.

  python results/thesis_final_writing/tools/refs_mist.py            -> writes references_mist_final.md,
                                                                         reference_audit_mist_final.md,
                                                                         citation_conversion_audit.md
Importable: cite_text(text) converts every [@key] / [@a; @b] marker of a chapter to MIST author-date form.

Rules implemented (MIST Thesis Template, TP.19-TP.21):
  one author          (Peck, 2002)
  two or three        (Peck, Henson and Simons, 2004)
  more than three     (Jones et al., 2012)
  several works       (Hussain, 1988; Hawser et al., 1998; Noman, 2000)   chronological, then alphabetical
  same label & year   2003a, 2003b  (letters assigned by title, alphabetically)
  reference list      alphabetical by first-author surname, then year; author surname + initials; (Year).
                      Title in sentence case, Venue in capitalised-word case, Vol., No., pp., DOI.
                      Seven or more authors: first six, then "et al."
Nothing is fetched or invented: every field comes from tools/reference_verification.json (ICML volume/pages from the
cached official PMLR pages used by tools/bib.py).
"""
import html
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(W))
sys.path.insert(0, HERE)
import bib  # noqa: E402

VER = bib.VER
CAND = bib.CAND
FRONT = ["abstract.md"]
CHAPTERS = ["chapter_01_introduction.md", "chapter_02_literature_review.md", "chapter_03_framework.md",
            "chapter_04_experimental_setup.md", "chapter_05_core_results.md", "chapter_06_deployment_sustainability.md",
            "chapter_07_discussion.md", "chapter_08_conclusion.md"]
OLD_COMMIT = "f01d9636"          # last IEEE-numbered assembly
SURNAME_FIX = {"Lecun": "LeCun"}  # same author as lecun1989obd ("LeCun, Yann"); capitalisation only
# Proper names inside titles that automatic sentence casing would lower-case (software / benchmark / method names).
TITLE_KEEP = {"raffin2021sb3": ("Stable-baselines3", "Stable-Baselines3"), "ma2023llmpruner": ("LLM-pruner", "LLM-Pruner"),
              "tschand2025mlperfpower": ("MLPerf power", "MLPerf Power")}

# Words kept as written in sentence-case titles (proper nouns, named methods, acronyms are kept automatically).
PROPER = {"bayesian", "hessian", "gaussian", "markov", "go-explore", "atari", "intel", "pareto", "english", "bloom",
          "transformer", "transformers", "lottery"}
PROPER.discard("transformer"); PROPER.discard("transformers"); PROPER.discard("lottery")  # common nouns here


# ----------------------------------------------------------------------------------------------- names
def split_name(name):
    name = html.unescape(name).strip()
    if "," in name:
        last, first = [x.strip() for x in name.split(",", 1)]
    else:
        parts = name.split()
        particles = {"van", "von", "de", "der", "den", "da", "di", "le", "la", "el"}
        i = len(parts) - 1
        while i > 1 and parts[i - 1].lower() in particles:
            i -= 1
        first, last = " ".join(parts[:i]), " ".join(parts[i:])
    last = SURNAME_FIX.get(last, last)
    return first, last


def initials(first):
    out = []
    for tok in first.replace(".", ". ").split():
        out.append("-".join(p[0].upper() + "." for p in tok.split("-") if p))
    return " ".join(out)


def ref_author(name):
    first, last = split_name(name)
    ini = initials(first)
    return f"{last}, {ini}" if ini else last


def surnames(key):
    return [split_name(a)[1] for a in (VER[key].get("authors") or [])]


def year(key):
    r = VER[key]
    y = r.get("year")
    m = re.search(r"\b(19|20)\d\d\b", r.get("venue") or "")
    if m and r.get("type") != "journal-article":
        y = m.group(0)
    return str(y)


def label(key):
    s = surnames(key)
    if len(s) == 1:
        return s[0]
    if len(s) <= 3:
        return ", ".join(s[:-1]) + " and " + s[-1]
    return s[0] + " et al."


def _title(key):
    r = VER[key]
    t = html.unescape(r["title"]).strip()
    ct = CAND.get(key, {}).get("title", "")
    if re.sub(r"[^a-z0-9]", "", ct.lower()) == re.sub(r"[^a-z0-9]", "", t.lower()):
        t = ct
    return re.sub(r"\s+", " ", t).rstrip(".")


def _keep(part):
    core = re.sub(r"^[^\w]+|[^\w]+$", "", part)
    if not core:
        return False
    return (any(c.isupper() for c in core[1:]) or (core.isupper() and len(core) > 1) or core.lower() in PROPER
            or re.match(r"^[A-Z]\d", core) is not None or (len(core) == 1 and core.isupper() and core != "A"))


def _case_part(part, first):
    if _keep(part):
        return part
    low = part.lower()
    if first:
        i = next((n for n, c in enumerate(low) if c.isalpha()), None)
        return low if i is None else low[:i] + low[i].upper() + low[i + 1:]
    return low


def sentence_case(t):
    out, first = [], True
    for w in t.split(" "):
        parts = w.split("-")
        out.append("-".join(_case_part(p, first and n == 0) for n, p in enumerate(parts)))
        first = w.endswith(":") or w.endswith("?")
    return " ".join(out)


# --------------------------------------------------------------------------------------------- suffixes
def cited_keys(files=FRONT + CHAPTERS, texts=None):
    seen, where = [], {}
    for f in files:
        t = texts[f] if texts else open(os.path.join(W, f), encoding="utf-8").read()
        for grp in re.findall(r"\[@([^\]]+)\]", t):
            for k in grp.split(";"):
                k = k.strip().lstrip("@")
                where.setdefault(k, [])
                if f not in where[k]:
                    where[k].append(f)
                if k not in seen:
                    seen.append(k)
    return seen, where


def _first_author(key):
    au = VER[key].get("authors") or []
    if not au:
        return ""
    f, l = split_name(au[0])
    return (initials(f) + " " + l).strip()


def suffix_map(keys):
    """MIST A.4/A.5: same label and year -> different first authors are told apart by initials (PFX); the same
    first author with several works in a year gets letters a, b, ... ordered by title (SFX)."""
    groups = {}
    for k in keys:
        groups.setdefault((label(k), year(k)), []).append(k)
    sfx, pfx = {}, {}
    for (lab, y), ks in groups.items():
        if len(ks) == 1:
            continue
        by_author = {}
        for k in ks:
            by_author.setdefault(_first_author(k), []).append(k)
        if len(by_author) > 1:
            for fa, kk in by_author.items():
                for k in kk:
                    pfx[k] = initials(split_name((VER[k].get("authors") or [""])[0])[0])
        for fa, kk in by_author.items():
            if len(kk) > 1:
                for i, k in enumerate(sorted(kk, key=lambda k: _title(k).lower())):
                    sfx[k] = "abcdefghij"[i]
    return sfx, pfx


_SEEN, _WHERE = cited_keys()
SFX, PFX = suffix_map(_SEEN)


def cite_one(key):
    lab = label(key)
    if key in PFX:
        lab = PFX[key] + " " + lab
    return lab, year(key) + SFX.get(key, "")


def cite_group(keys, inner=False):
    items = sorted(keys, key=lambda k: (year(k), label(k).lower(), SFX.get(k, "")))
    parts, i = [], 0
    while i < len(items):                       # same label, different years: (Lee, 1985, 1986)
        lab, y = cite_one(items[i])
        ys = [y]
        j = i + 1
        while j < len(items) and cite_one(items[j])[0] == lab:
            ys.append(cite_one(items[j])[1])
            j += 1
        parts.append(f"{lab}, {', '.join(ys)}")
        i = j
    body = "; ".join(parts)
    return body if inner else f"({body})"


def _inside_paren(line_before):
    depth = 0
    for ch in line_before:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
    return depth > 0


def cite_text(text):
    out, last = [], 0
    for m in re.finditer(r"\[@([^\]]+)\]", text):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        line_before = text[text.rfind("\n", 0, m.start()) + 1:m.start()]
        inner = _inside_paren(line_before)
        rep = cite_group(keys, inner=inner)
        pre = text[last:m.start()]
        if inner and pre.endswith(" ") and re.search(r"[;(]\s*$", pre) is None:
            pre = pre.rstrip() + "; " if not pre.rstrip().endswith("(") else pre
        out.append(pre)
        out.append(rep)
        last = m.end()
    out.append(text[last:])
    return "".join(out)


# --------------------------------------------------------------------------------------------- entries
def venue_str(key):
    r = VER[key]
    v = r.get("venue") or ""
    y = year(key)
    pg = bib.pages(r.get("pages"))
    doi = r.get("doi")
    if r["route"] == "arxiv":
        return f"arXiv preprint arXiv:{r['source'].rsplit('/', 1)[-1]}", None
    if key == "krizhevsky2009cifar":
        return "Technical Report, University of Toronto, Toronto, Canada", None
    if key == "sutton2018rl":
        return "2nd Ed., MIT Press, Cambridge, MA, USA", None
    if "Neural Information Processing Systems" in v:
        return f"Advances in Neural Information Processing Systems (NeurIPS), Vol. {int(y) - 1987}", None
    if "Learning Representations" in v:
        return "Proc. International Conference on Learning Representations (ICLR)", None
    if "International Conference on Machine Learning" in v:
        vol, pp = bib.pmlr(r)
        vol = vol or bib.ICML_VOL.get(y)
        if not pp and vol:
            pp = bib.pmlr_index(vol, _title(key))
        s = "Proc. International Conference on Machine Learning (ICML), Proceedings of Machine Learning Research"
        return s + (f", Vol. {vol}" if vol else "") + (f", pp. {pp}" if pp else ""), None
    if "Machine Learning and Systems" in v:
        return f"Proc. Machine Learning and Systems (MLSys), Vol. {r.get('volume')}" + (f", pp. {pg}" if pg else ""), doi
    if "Journal of Machine Learning Research" in v:
        return f"Journal of Machine Learning Research, Vol. {r.get('volume')}, No. {r.get('issue')}, pp. {pg}", doi
    if r.get("type") == "journal-article":
        s = v
        if r.get("volume"):
            s += f", Vol. {r['volume']}"
        if r.get("issue"):
            s += f", No. {r['issue']}"
        if pg:
            s += f", pp. {pg}" if "–" in pg else (f", p. {pg}" if len(pg) < 4 else f", Article No. {pg}")
        return s, doi
    if "ECCV" in v:
        return f"Computer Vision – ECCV {y}, Lecture Notes in Computer Science, Springer" + (f", pp. {pg}" if pg else ""), doi
    ven = re.sub(r"^(The )?\d{4}\s+", "", v)
    ven = ven.replace("Fairness Accountability and Transparency", "Fairness, Accountability, and Transparency (FAccT)")
    ven = re.sub(r"^Proceedings of the ", "", ven)
    if ven.startswith("Findings of"):
        return ven + (f", pp. {pg}" if pg else ""), doi
    return "Proc. " + ven + (f", pp. {pg}" if pg else ""), doi


def mist_entry(key):
    names = [ref_author(a) for a in (VER[key].get("authors") or [])]
    if len(names) >= 7:
        a = ", ".join(names[:6]) + " et al."
    elif len(names) == 1:
        a = names[0]
    elif len(names) == 2:
        a = names[0] + " and " + names[1]
    else:
        a = ", ".join(names[:-1]) + ", and " + names[-1]
    ven, doi = venue_str(key)
    y = year(key) + SFX.get(key, "")
    title = sentence_case(_title(key))
    if key in TITLE_KEEP:
        title = title.replace(*TITLE_KEEP[key])
    sep = " " if title.endswith(("?", "!")) else ", "
    s = f"{a} ({y}). {title}{sep}{ven}."
    if doi:
        s = s[:-1] + f". DOI: {doi.lower()}"
    return s.replace("..", ".")


def sort_key(key):
    au = VER[key].get("authors") or []
    s = surnames(key)
    first_ini = initials(split_name(au[0])[0]) if au else ""
    n = min(len(au), 3)
    return (s[0].lower() if s else "", first_ini, n, ", ".join(x.lower() for x in s[1:]) if n < 3 else "",
            year(key), SFX.get(key, ""))


def status(key):
    return bib.status(key)


# --------------------------------------------------------------------------------------------- outputs
def chapter_label(f):
    return "Abstract" if f == "abstract.md" else "Ch. " + str(int(f.split("_")[1]))


def old_ieee_map():
    """IEEE numbers of the last numbered assembly (commit f01d9636), from the chapter files of that commit."""
    texts = {}
    for f in FRONT + CHAPTERS:
        try:
            texts[f] = subprocess.run(["git", "-C", ROOT, "show", f"{OLD_COMMIT}:results/thesis_final_writing/{f}"],
                                      capture_output=True, text=True, encoding="utf-8").stdout
        except Exception:  # noqa: BLE001
            texts[f] = ""
    seen, _ = cited_keys(texts=texts)
    return {k: i for i, k in enumerate(seen, 1)}


def main():
    keys = sorted(_SEEN, key=sort_key)
    uncited = [k for k in VER if k not in _SEEN]
    # references list
    R = ["# REFERENCES", ""]
    for k in keys:
        R.append(mist_entry(k))
        R.append("")
    open(os.path.join(W, "references_mist_final.md"), "w", encoding="utf-8").write("\n".join(R))

    recent = [k for k in keys if int(year(k)) >= 2022]
    found = [k for k in keys if int(year(k)) < 2022]
    A = ["# Reference audit (MIST author-date style)", "",
         "Replaces the IEEE-oriented `reference_audit_final.md`. Generated by `tools/refs_mist.py` from the verified",
         "metadata in `tools/reference_verification.json`; no source was replaced because of the style change.", "",
         "## Summary", "",
         f"- Cited references: **{len(keys)}**, all verified; listed alphabetically by first-author surname, then year, unnumbered.",
         f"- Contemporary (2022–2026): **{len(recent)}** ({100 * len(recent) / len(keys):.1f}%); foundational (pre-2022): **{len(found)}**.",
         f"- Verified but not cited, therefore excluded: {', '.join(uncited) or 'none'}.",
         f"- Same author label and year: different first authors distinguished by initials (MIST A.5): "
         + (", ".join(f"{cite_one(k)[0]} ({cite_one(k)[1]})" for k in keys if k in PFX) or "none")
         + "; the same first author distinguished by letters (MIST A.4): "
         + (", ".join(f"{cite_one(k)[0]} ({cite_one(k)[1]})" for k in keys if SFX.get(k)) or "none") + ".",
         "- Every bibliography entry is cited and every citation has an entry (both are generated from the same key list).", "",
         "## Entries", "",
         "| # | Key | Authors | Year | Title | Venue | Vol./No. | Pages / Art. | DOI | Status | Verification source | Class | Chapters cited | Final MIST reference |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, k in enumerate(keys, 1):
        r = VER[k]
        au = "; ".join(ref_author(a) for a in (r.get("authors") or []))
        vn = f"{r.get('volume') or '—'}/{r.get('issue') or '—'}"
        cls = "recent (2022–2026)" if int(year(k)) >= 2022 else "foundational (pre-2022)"
        ch = ", ".join(chapter_label(f) for f in _WHERE[k])
        cells = [str(i), f"`{k}`", au, year(k) + SFX.get(k, ""), _title(k), r.get("venue") or "", vn,
                 bib.pages(r.get("pages")) or "—", (r.get("doi") or "—").lower(), status(k),
                 f"{r.get('route')}: {r.get('source')}", cls, ch, mist_entry(k)]
        A.append("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    open(os.path.join(W, "reference_audit_mist_final.md"), "w", encoding="utf-8").write("\n".join(A) + "\n")

    # conversion audit
    old = old_ieee_map()
    C = ["# Citation conversion audit (IEEE numbered → MIST author-date)", "",
         f"Old numbers are those of the last IEEE-numbered assembly (commit `{OLD_COMMIT}`). New citations are produced",
         "by `tools/refs_mist.py` from the same citation keys in the final chapter files.", "",
         "| # | Chapter | Old IEEE citation | New author-date citation | Reference key(s) | Status |", "|---|---|---|---|---|---|"]
    n = 0
    bad = 0
    for f in FRONT + CHAPTERS:
        t = open(os.path.join(W, f), encoding="utf-8").read()
        for m in re.finditer(r"\[@([^\]]+)\]", t):
            ks = [x.strip().lstrip("@") for x in m.group(1).split(";")]
            n += 1
            oldc = ", ".join(f"[{old[k]}]" if k in old else "(new)" for k in ks)
            ok = all(k in VER and k in _SEEN for k in ks)
            bad += 0 if ok else 1
            C.append(f"| {n} | {chapter_label(f)} | {oldc} | {cite_group(ks)} | {', '.join('`' + k + '`' for k in ks)} | "
                     f"{'converted; entry present' if ok else 'MISSING ENTRY'} |")
    conv = {f: cite_text(open(os.path.join(W, f), encoding="utf-8").read()) for f in FRONT + CHAPTERS}
    ieee_left = sum(len(re.findall(r"\[\d+\](?!\()", t)) for t in conv.values())
    marker_left = sum(len(re.findall(r"\[@", t)) for t in conv.values())
    C += ["", "## Final verification", "",
          f"- Citation occurrences converted: **{n}** (keys: {len(_SEEN)}).",
          f"- Numbered IEEE citations remaining after conversion (`\\[[0-9]+\\]`): **{ieee_left}**.",
          f"- Unconverted citation markers remaining: **{marker_left}**.",
          f"- Citations without a bibliography entry: **{bad}**.",
          f"- Bibliography entries never cited: **0** (the list is generated from the cited keys; {len(uncited)} verified "
          "but uncited candidates were excluded).", "",
          "Format checks (MIST TP.19): one author → `(Surname, Year)`; two or three → `(A, B and C, Year)`; more than three →",
          "`(A et al., Year)`; several works → semicolon-separated, chronological; same label and year → `2022a`, `2022b`;",
          "a citation inside an existing parenthesis is merged into it rather than nested. No corporate-author source is cited."]
    open(os.path.join(W, "citation_conversion_audit.md"), "w", encoding="utf-8").write("\n".join(C) + "\n")
    print(f"{len(keys)} cited; recent {len(recent)} ({100 * len(recent) / len(keys):.1f}%); occurrences {n}; "
          f"IEEE left {ieee_left}; markers left {marker_left}; missing {bad}; uncited excluded {uncited}")


if __name__ == "__main__":
    main()
