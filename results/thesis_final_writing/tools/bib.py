"""IEEE bibliography and reference audit, built only from verified metadata (reference_verification.json).

  python results/thesis_final_writing/tools/bib.py audit chapter_01_*.md ...   -> reference_audit.md
  python results/thesis_final_writing/tools/bib.py ieee  KEY                     -> prints one IEEE entry
Numbers follow the order of first citation across the chapter files given, in the order given (thesis order).
"""
import html
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
VER = json.load(open(os.path.join(HERE, "reference_verification.json"), encoding="utf-8"))
CAND = {c["id"]: c for c in json.load(open(os.path.join(HERE, "reference_candidates.json"), encoding="utf-8"))}
ICML_VOL = {"2017": "70", "2018": "80", "2020": "119", "2022": "162", "2023": "202", "2024": "235"}
JOURNAL = {"IEEE Transactions on Pattern Analysis and Machine Intelligence": "IEEE Trans. Pattern Anal. Mach. Intell.",
           "IEEE Transactions on Circuits and Systems for Video Technology": "IEEE Trans. Circuits Syst. Video Technol.",
           "IEEE Transactions on Parallel and Distributed Systems": "IEEE Trans. Parallel Distrib. Syst.",
           "ACM Computing Surveys": "ACM Comput. Surv.", "Communications of the ACM": "Commun. ACM",
           "ACM Transactions on Modeling and Performance Evaluation of Computing Systems": "ACM Trans. Model. Perform. Eval. Comput. Syst.",
           "Sustainable Computing: Informatics and Systems": "Sustain. Comput.: Inform. Syst.", "Proceedings of the IEEE": "Proc. IEEE",
           "Computer": "Computer", "Nature": "Nature"}
CONF = [("Computer Vision and Pattern Recognition", "Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)"),
        ("Winter Conference on Applications of Computer Vision", "Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV)"),
        ("International Conference on Computer Vision", "Proc. IEEE Int. Conf. Comput. Vis. (ICCV)"),
        ("Annual Meeting of the Association for Computational Linguistics", "Proc. {n} Annu. Meeting Assoc. Comput. Linguistics (ACL)"),
        ("Findings of the Association for Computational Linguistics: EMNLP", "Findings Assoc. Comput. Linguistics: EMNLP"),
        ("Fairness Accountability and Transparency", "Proc. ACM Conf. Fairness, Accountability, Transparency (FAccT)"),
        ("High Performance Extreme Computing", "Proc. IEEE High Perform. Extreme Comput. Conf. (HPEC)"),
        ("High Performance Computer Architecture", "Proc. IEEE Int. Symp. High-Perform. Comput. Archit. (HPCA)"),
        ("Cluster, Cloud and Internet Computing", "Proc. IEEE/ACM Int. Symp. Cluster, Cloud Internet Comput. (CCGrid)")]
CACHE = os.path.join(HERE, "_cache")


def cached(url):
    key = os.path.join(CACHE, re.sub(r"[^A-Za-z0-9]+", "_", url)[-180:])
    if os.path.exists(key):
        return open(key, encoding="utf-8").read()
    t = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (MSc thesis reference verification)"}),
                               timeout=60).read().decode("utf-8", "replace")
    open(key, "w", encoding="utf-8").write(t)
    return t


def initials(name):
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
    ini = []
    for tok in first.replace(".", ". ").split():
        ini.append("-".join(p[0] + "." for p in tok.split("-") if p))
    return (" ".join(ini) + " " + last).strip()


def authors(lst):
    names = [initials(a) for a in lst]
    if len(names) > 6:
        return names[0] + " et al."
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]


def pmlr(rec):
    """Volume and pages for ICML papers from the official PMLR page linked by ML Anthology."""
    try:
        page = cached(rec["source"])
        pdf = re.search(r'citation_pdf_url content="?(https://proceedings\.mlr\.press/[^" >]+)', page)
        if not pdf:
            return None, None
        url = re.sub(r"/[^/]+\.pdf$", ".html", pdf.group(1))
        p = cached(url)
        fp = re.search(r'citation_firstpage" content="(\d+)', p)
        lp = re.search(r'citation_lastpage" content="(\d+)', p)
        vol = re.search(r"/v(\d+)/", url).group(1)
        return vol, (f"{fp.group(1)}–{lp.group(1)}" if fp and lp else None)
    except Exception:  # noqa: BLE001
        return None, None


def pmlr_index(vol, title):
    """Pages from the official PMLR volume index when ML Anthology does not link the PMLR page."""
    try:
        idx = cached(f"https://proceedings.mlr.press/v{vol}/")
        n = re.sub(r"[^a-z0-9]", "", title.lower())
        for tt in re.finditer(r'<p class="title">(.*?)</p>', idx, re.S):
            if re.sub(r"[^a-z0-9]", "", html.unescape(tt.group(1)).lower()) == n:
                m = re.search(r"PMLR \d+:(\d+)-(\d+)", idx[tt.end(): tt.end() + 1500])
                return f"{m.group(1)}–{m.group(2)}" if m else None
    except Exception:  # noqa: BLE001
        return None
    return None


def pages(p):
    if not p:
        return None
    p = str(p).replace("-", "–")
    return p


def ieee(key):
    r = VER[key]
    t = html.unescape(r["title"]).strip()
    ct = CAND.get(key, {}).get("title", "")
    if re.sub(r"[^a-z0-9]", "", ct.lower()) == re.sub(r"[^a-z0-9]", "", t.lower()):
        t = ct
    a, t, v, y = authors(r.get("authors") or []), t.rstrip("."), r.get("venue") or "", r.get("year")
    doi = f", doi: {r['doi'].lower()}" if r.get("doi") else ""
    pg = pages(r.get("pages"))
    m = re.search(r"\b(19|20)\d\d\b", v)
    if m and r.get("type") != "journal-article":
        y = m.group(0)
    if r["route"] == "arxiv":
        return f"{a}, “{t},” arXiv preprint arXiv:{r['source'].rsplit('/', 1)[-1]}, {y}."
    if key == "sutton2018rl":
        return f"{a}, *Reinforcement Learning: An Introduction*, 2nd ed. Cambridge, MA, USA: MIT Press, {y}."
    if key == "krizhevsky2009cifar":
        return f"{a}, “{t},” Univ. Toronto, Toronto, ON, Canada, Tech. Rep., {y}."
    if "Neural Information Processing Systems" in v:
        vol = int(y) - 1987
        return f"{a}, “{t},” in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. {vol}, {y}."
    if "Learning Representations" in v:
        return f"{a}, “{t},” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, {y}."
    if "International Conference on Machine Learning" in v:
        vol, pp = pmlr(r)
        vol = vol or ICML_VOL.get(y)
        if not pp and vol:
            pp = pmlr_index(vol, t)
        return f"{a}, “{t},” in *Proc. Int. Conf. Mach. Learn. (ICML)*, ser. PMLR" + (f", vol. {vol}" if vol else "") + f", {y}" + (f", pp. {pp}" if pp else "") + "."
    if "Machine Learning and Systems" in v:
        return f"{a}, “{t},” in *Proc. Mach. Learn. Syst. (MLSys)*, vol. {r.get('volume')}, {y}" + (f", pp. {pg}" if pg else "") + "."
    if "Journal of Machine Learning Research" in v:
        return f"{a}, “{t},” *J. Mach. Learn. Res.*, vol. {r.get('volume')}, no. {r.get('issue')}, pp. {pg}, {y}."
    jt = r.get("type") == "journal-article"
    if jt:
        vol = f", vol. {r['volume']}" if r.get("volume") else ""
        no = f", no. {r['issue']}" if r.get("issue") else ""
        pp = (f", Art. no. {pg}" if pg and "–" not in pg else f", pp. {pg}") if pg else ""
        return f"{a}, “{t},” *{JOURNAL.get(v, v)}*{vol}{no}{pp}, {y}{doi}."
    ven = re.sub(r"^\d{4}\s+", "", v)
    for pat, abbr in CONF:
        if pat in v:
            num = re.search(r"(\d+(?:st|nd|rd|th))", v)
            ven = abbr.replace("{n}", num.group(1) if num else "")
    if "ECCV" in v:
        ven = f"Computer Vision – ECCV {y} (Lecture Notes in Computer Science)"
    return f"{a}, “{t},” in *{ven}*, {y}" + (f", pp. {pg}" if pg else "") + f"{doi}."


def status(key):
    r = VER[key]
    if r["route"] == "arxiv":
        return "not peer-reviewed (arXiv); foundational exception: original algorithm paper"
    if key == "sutton2018rl":
        return "academic textbook (MIT Press)"
    if key == "krizhevsky2009cifar":
        return "technical report; original dataset source (foundational exception)"
    return "peer-reviewed (" + ("journal" if r.get("type") == "journal-article" or "Journal" in (r.get("venue") or "") or
                                r.get("venue") in ("Computer", "Nature") else "conference proceedings") + ")"


def order(files):
    seen, where = [], {}
    for f in files:
        for grp in re.findall(r"\[@([^\]]+)\]", open(os.path.join(W, f), encoding="utf-8").read()):
            for k in grp.split(";"):
                k = k.strip().lstrip("@")
                where.setdefault(k, [])
                if f not in where[k]:
                    where[k].append(f)
                if k not in seen:
                    seen.append(k)
    return seen, where


def audit(files):
    seen, where = order(files)
    unused = [k for k in VER if k not in seen]
    rows = ["# Reference audit", "",
            "Every reference was verified against an authoritative bibliographic source by `tools/verify_refs.py`. Sources:",
            "- Crossref (DOI registry);",
            "- ML Anthology (mirrors the official ICLR/NeurIPS/ICML/CVF proceedings metadata);",
            "- the official MLSys and JMLR pages;",
            "- arXiv (for one foundational preprint).",
            "",
            "Author lists, titles, venues, years, volumes, pages and DOIs below come from that source, not from memory. "
            "A candidate was accepted only when the source returned the same normalised title. Numbers follow the order "
            "of first citation in the chapter files listed below; they are renumbered automatically when chapters are added.",
            "", f"Chapter files: {', '.join(files)}", "",
            f"Cited references: **{len(seen)}**; recent (2022–2026): **{sum(VER[k]['class'] == 'recent' for k in seen)}**; "
            f"foundational: **{sum(VER[k]['class'] == 'foundational' for k in seen)}**; "
            f"verified but not yet cited: {len(unused)} ({', '.join(unused) if unused else 'none'}).", "",
            "| # | Key | Title | Year | Venue | Peer-review status | Verification source | Class | Cited in |",
            "|---|---|---|---|---|---|---|---|---|"]
    for i, k in enumerate(seen, 1):
        r = VER.get(k)
        if not r or r.get("status") != "verified":
            rows.append(f"| {i} | {k} | **UNVERIFIED — must not be cited** | | | | | | {', '.join(where[k])} |")
            continue
        rows.append(f"| {i} | `{k}` | {html.unescape(r['title'])} | {r['year']} | {r.get('venue') or ''} | {status(k)} | {r['source']} | "
                    f"{r['class']} | {', '.join(x.split('_')[1] for x in where[k])} |")
    rows += ["", "## IEEE-formatted list (citation order)", ""]
    rows += [f"[{i}] {ieee(k)}" for i, k in enumerate(seen, 1) if VER.get(k, {}).get("status") == "verified"]
    open(os.path.join(W, "reference_audit.md"), "w", encoding="utf-8").write("\n".join(rows) + "\n")
    print(f"{len(seen)} cited, {len(unused)} unused")


if __name__ == "__main__":
    if sys.argv[1] == "audit":
        audit(sys.argv[2:])
    else:
        print(ieee(sys.argv[2]))
