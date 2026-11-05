"""Global reference rebuild for the assembled thesis (IEEE numbered, order of first citation from Chapter 1).

  python results/thesis_final_writing/tools/refs_final.py
Writes reference_audit_final.md and references_final.md. Importable: order(), number_map(), convert(text, nmap),
ieee_final(key). Uses the verified metadata of tools/reference_verification.json and the formatter of tools/bib.py;
nothing is fetched or invented here.
"""
import html
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import bib  # noqa: E402

CHAPTERS = ["chapter_01_introduction.md", "chapter_02_literature_review.md", "chapter_03_framework.md",
            "chapter_04_experimental_setup.md", "chapter_05_core_results.md", "chapter_06_deployment_sustainability.md",
            "chapter_07_discussion.md", "chapter_08_conclusion.md"]
FRONT = ["abstract.md"]
VER = bib.VER


def ieee_final(key):
    if key == "wilcoxon1945individual":  # source records the first page only
        return "F. Wilcoxon, “Individual comparisons by ranking methods,” *Biometrics Bull.*, vol. 1, no. 6, p. 80, 1945, doi: 10.2307/3001968."
    if key == "holm1979sequential":
        return "S. Holm, “A simple sequentially rejective multiple test procedure,” *Scand. J. Statist.*, vol. 6, pp. 65–70, 1979."
    if key == "youden1950index":
        return "W. J. Youden, “Index for rating diagnostic tests,” *Cancer*, vol. 3, no. 1, pp. 32–35, 1950, doi: 10.1002/1097-0142(1950)3:1<32::aid-cncr2820030106>3.0.co;2-3."
    return bib.ieee(key)


def keys_in(text):
    out = []
    for grp in re.findall(r"\[@([^\]]+)\]", text):
        for k in grp.split(";"):
            out.append(k.strip().lstrip("@"))
    return out


def order(files=FRONT + CHAPTERS):
    seen, where = [], {}
    for f in files:
        for k in keys_in(open(os.path.join(W, f), encoding="utf-8").read()):
            where.setdefault(k, [])
            if f not in where[k]:
                where[k].append(f)
            if k not in seen:
                seen.append(k)
    return seen, where


def number_map():
    seen, _ = order()
    return {k: i for i, k in enumerate(seen, 1)}


def _fmt(nums):
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"[{nums[i]}]–[{nums[j]}]" if j - i >= 2 else ", ".join(f"[{n}]" for n in nums[i:j + 1]))
        i = j + 1
    return ", ".join(out)


def convert(text, nmap):
    return re.sub(r"\[@([^\]]+)\]", lambda m: _fmt([nmap[k.strip().lstrip("@")] for k in m.group(1).split(";")]), text)


def chapter_label(f):
    return "Abstract" if f == "abstract.md" else "Ch. " + str(int(f.split("_")[1]))


def main():
    seen, where = order()
    missing = [k for k in seen if VER.get(k, {}).get("status") != "verified"]
    if missing:
        sys.exit(f"UNVERIFIED keys cited: {missing}")
    # duplicates: same DOI or same normalised title under two keys
    norm = lambda s: re.sub(r"[^a-z0-9]", "", html.unescape(s or "").lower())  # noqa: E731
    dup = {}
    for k in seen:
        for sig in filter(None, [VER[k].get("doi"), norm(VER[k]["title"])]):
            dup.setdefault(sig, []).append(k)
    dups = {s: ks for s, ks in dup.items() if len(ks) > 1}
    arxiv = [k for k in seen if VER[k]["route"] == "arxiv"]
    recent = [k for k in seen if VER[k]["class"] == "recent"]
    found = [k for k in seen if VER[k]["class"] == "foundational"]
    unused = sorted(k for k in VER if k not in seen)

    rows = ["# Final reference audit", "",
            "- **Scope:** the assembled thesis (Abstract, then Chapters 1–8).",
            "- **Numbering:** IEEE numbered style, in order of first citation starting with Chapter 1.",
            "- **Supersedes:** the Milestone-1 `reference_audit.md`, which covered Chapter 2 only and is kept frozen for"
            " provenance.",
            "- **Metadata:** from `tools/reference_verification.json`. Every entry was verified against an authoritative"
            " source (Crossref, ML Anthology, PMLR, MLSys/JMLR pages, OpenAlex, arXiv for one foundational preprint, or the"
            " original technical-report URL) by exact normalised-title match; nothing here comes from memory.",
            "",
            "## Summary", "",
            f"- Cited references: **{len(seen)}**; all verified ({len(missing)} unverified).",
            f"- Contemporary (2022–2026): **{len(recent)}** ({100 * len(recent) / len(seen):.1f}%); "
            f"foundational (pre-2022): **{len(found)}** ({100 * len(found) / len(seen):.1f}%).",
            f"- arXiv-only: {len(arxiv)} ({', '.join(arxiv)}), the permitted foundational exception (the original PPO "
            "paper, which has no peer-reviewed version).",
            f"- Duplicate entries (same DOI or same normalised title under two keys): **{len(dups)}**"
            + (f" {dups}" if dups else "") + ".",
            "- Every bibliography entry is cited, and every citation has an entry. Citations and entries are generated from"
            " the same key list.",
            f"- Verified but not cited, and therefore excluded from the bibliography: {len(unused)} ({', '.join(unused) if unused else 'none'}).",
            "",
            "**Foundational exceptions,** pre-2022 sources cited for the origin of a method, model, dataset or tool:",
            "- CIFAR technical report;",
            "- ResNet; LeNet-5;",
            "- PPO; GAE;",
            "- PyTorch; Stable-Baselines3;",
            "- the original pruning algorithms (OBD, magnitude pruning, filter and channel pruning, LAMP, ERK/RigL);",
            "- RL methods (AMC, self-imitation, Go-Explore, constrained policy optimisation);",
            "- RAPL;",
            "- the three statistical methods (Holm, Wilcoxon, Youden), added at final assembly.",
            "",
            "Contemporary literature (2022–2026) is peer-reviewed except where noted.",
            "",
            "## Entries", "",
            "| # | Key | Authors | Title | Venue | Year | Vol./no./pages | DOI | Peer-review status | Verification source | Class | Cited in |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, k in enumerate(seen, 1):
        r = VER[k]
        vp = "/".join(x for x in [str(r.get("volume") or ""), str(r.get("issue") or ""), str(r.get("pages") or "")]) .strip("/")
        doi = r.get("doi") or "—"
        rows.append(f"| {i} | `{k}` | {bib.authors(r.get('authors') or [])} | {html.unescape(r['title'])} | {r.get('venue') or ''} | "
                    f"{r.get('year')} | {vp or '—'} | {doi} | {bib.status(k)} | {r['source']} | {r['class']} | "
                    f"{', '.join(chapter_label(f) for f in where[k])} |")
    open(os.path.join(W, "reference_audit_final.md"), "w", encoding="utf-8").write("\n".join(rows) + "\n")
    refs = ["# References"] + [f"[{i}] {ieee_final(k)}" for i, k in enumerate(seen, 1)]
    open(os.path.join(W, "references_final.md"), "w", encoding="utf-8").write("\n\n".join(refs) + "\n")
    print(f"{len(seen)} cited; recent {len(recent)} ({100 * len(recent) / len(seen):.1f}%); foundational {len(found)}; "
          f"arXiv {arxiv}; duplicates {len(dups)}; unused {len(unused)}")


if __name__ == "__main__":
    main()
