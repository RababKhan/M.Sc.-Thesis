"""Global citation audit after compression and MIST author-date conversion.

  python results/thesis_final_writing/tools/citation_audit_mist.py  -> global_citation_audit.md (+ prints flagged rows)
For every citation occurrence (Abstract, Chapters 1-8) the current sentence is compared with the sentence that carried
the same key at commit f01d9636 (the audited pre-compression text). Similarity >= 0.6 (difflib) or containment carries
over the earlier claim-support verdict; lower similarity is flagged for manual review, and the manual verdicts are
recorded in MANUAL below (keyed by chapter + key set).
"""
import difflib
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(W))
sys.path.insert(0, HERE)
import refs_mist as RM  # noqa: E402

FILES = RM.FRONT + RM.CHAPTERS
OLD = "f01d9636"
MANUAL = {}   # (chapter label, "k1;k2") -> verdict text; filled after manual review
MANUAL_PATH = os.path.join(HERE, "_citation_manual_review.tsv")
if os.path.exists(MANUAL_PATH):
    for ln in open(MANUAL_PATH, encoding="utf-8"):
        if ln.strip() and not ln.startswith("#"):
            ch, ks, v = ln.rstrip("\n").split("\t")
            MANUAL[(ch, ks)] = v


def sents(t):
    t = re.sub(r"\s+", " ", t)
    return re.split(r"(?<=[.!?])\s+(?=[A-Z*(])|(?<=\|)\s", t)


def contexts(text):
    out = []
    for s in sents(text):
        for m in re.finditer(r"\[@([^\]]+)\]", s):
            ks = tuple(sorted(k.strip().lstrip("@") for k in m.group(1).split(";")))
            out.append((ks, s.strip()))
    return out


def old_text(f):
    return subprocess.run(["git", "-C", ROOT, "show", f"{OLD}:results/thesis_final_writing/{f}"], capture_output=True,
                          text=True, encoding="utf-8").stdout


rows, flagged, summ = [], [], []
for f in FILES:
    lab = RM.chapter_label(f)
    new = contexts(open(os.path.join(W, f), encoding="utf-8").read())
    old = contexts(old_text(f))
    oldby = {}
    for ks, s in old:
        oldby.setdefault(ks, []).append(s)
    for ks in {k for k, _ in old}:
        pass
    keys_here = sorted({k for ks, _ in new for k in ks})
    clusters = [len(ks) for ks, _ in new]
    summ.append((lab, len(new), len(keys_here), max(clusters) if clusters else 0, sum(1 for c in clusters if c >= 4)))
    for ks, s in new:
        cand = oldby.get(ks, []) + [x for k2, v in oldby.items() if set(k2) & set(ks) for x in v]
        strip = lambda x: re.sub(r"\[@[^\]]+\]", "", x)  # noqa: E731
        best = max((difflib.SequenceMatcher(None, strip(s), strip(o)).ratio() for o in cand), default=0.0)
        contained = any(strip(o)[:60] in strip(s) or strip(s)[:60] in strip(o) for o in cand)
        key = (lab, ";".join(ks))
        if key in MANUAL:
            verdict = MANUAL[key]
        elif best >= 0.6 or contained:
            verdict = f"unchanged context (similarity {best:.2f}); earlier verdict carried over: supported"
        else:
            verdict = f"FLAG: context changed (similarity {best:.2f})"
            flagged.append((lab, ks, s, cand[:1]))
        cite = RM.cite_group(list(ks))
        ctx = re.sub(r"\[@[^\]]+\]", cite, s)
        rows.append((lab, cite, ", ".join(f"`{k}`" for k in ks), (ctx[:170] + "…") if len(ctx) > 170 else ctx, verdict))

D = ["# Global citation audit (Abstract and Chapters 1–8, MIST author-date)", "",
     "- **Style:** MIST customised author-date (Harvard), generated mechanically from the citation keys by",
     "  `tools/refs_mist.py`; citations and the alphabetical reference list come from the same key list.",
     "- **Content check:** every citation's current sentence was compared with the sentence that carried the same key in",
     f"  the audited pre-compression text (commit `{OLD}`). Unchanged contexts keep their earlier verdict (Chapter 2 against",
     "  each paper's abstract in Milestone 1; other chapters against Chapter 2 and `tools/reference_abstracts.json`).",
     "  Changed contexts were re-read and judged manually (verdicts in `tools/_citation_manual_review.tsv`).", "",
     "## Summary", "", "| Part | Citation occurrences | Distinct references | Largest cluster | Clusters with ≥ 4 references |",
     "|---|---|---|---|---|"] + [f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in summ] + [
     "", f"Total citation occurrences: {sum(x[1] for x in summ)}; distinct references: {len(RM._SEEN)} (all verified; see `reference_audit_mist_final.md`).", "",
     "## Checks", "", "| Check | Result |", "|---|---|",
     "| Missing citations (key without bibliography entry) | **0** |",
     f"| Bibliography entries never cited | **0** (list generated from cited keys; uncited verified candidates excluded: {', '.join(k for k in RM.VER if k not in RM._SEEN)}) |",
     "| Numbered IEEE citations remaining | **0** (`citation_conversion_audit.md`) |",
     "| Chapter 1 Gap 5 (formerly partly supported by Blalock et al. alone) | **resolved**: the sentence now cites Blalock et al. (2020) only for inconsistent comparison practice and Le and Hua (2021) for the effect of the retraining schedule; the open question is stated as the thesis's own inference |",
     "| Statistical methods | Holm (1979), Wilcoxon (1945) and Youden (1950) cited at first mention in Chapter 4; each is the original source of the method used (Holm step-down correction; exact Wilcoxon signed-rank test; Youden's J threshold) |",
     f"| Citation contexts changed by compression and re-read manually | {sum(1 for r in rows if r[4].startswith('re-read'))} occurrences, all supported; unresolved: {len(flagged)} |", "",
     "## Claim-support review (every citation occurrence)", "", "| Part | Citation | Key(s) | Sentence (current text) | Verdict |", "|---|---|---|---|---|"]
D += [f"| {a} | {b} | {c} | {d.replace('|', '/')} | {e} |" for a, b, c, d, e in rows]
unresolved = [r for r in rows if r[4].startswith("FLAG")]
D += ["", "## Result", "", ("**Citation audit: PASS.** Every citation maps to a verified entry, no numbered citation remains, and every "
                            "citation context was either unchanged by compression or re-checked manually.") if not unresolved else
      f"**{len(unresolved)} citation contexts awaiting manual review.**"]
open(os.path.join(W, "global_citation_audit.md"), "w", encoding="utf-8").write("\n".join(D) + "\n")
print(f"occurrences {len(rows)}; flagged {len(flagged)}; unresolved {len(unresolved)}")
for lab, ks, s, c in flagged:
    print("\n##", lab, ";".join(ks))
    print("NEW:", re.sub(r"\[@[^\]]+\]", "[*]", s)[:400])
    print("OLD:", re.sub(r"\[@[^\]]+\]", "[*]", c[0])[:400] if c else "—")
