"""Automated part of each chapter QA: word count, citation audit, dangerous-phrase scan, table/figure references.

  python results/thesis_final_writing/tools/qa_chapter.py chapter_02_literature_review.md
Prints a markdown block that is pasted into qa_chapter_NN.md together with the manual numerical and claim audits.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
VER = json.load(open(os.path.join(HERE, "reference_verification.json"), encoding="utf-8"))

# claim-boundary phrases (task section 41 plus claim_matrix.md forbidden wording); every hit is inspected manually
DANGER = [r"beat[s]? LAMP", r"outperform[s]? LAMP", r"equivalent to LAMP", r"equivalen(t|ce)", r"sensitivity improves (final )?accuracy",
          r"archive (selection )?improves (final |test )", r"accuracy[- ]preserv", r"GREEN-SUCCESS", r"green", r"environmentally friendly",
          r"carbon", r"lower energy", r"energy[- ]efficient", r"latency reduction", r"state-of-the-art", r"superior", r"groundbreaking",
          r"revolutionary", r"unprecedented", r"guarantee", r"matched high sparsity", r"heuristic", r"TODO|TBD|XX|\?\?"]


def main(name):
    text = open(os.path.join(W, name), encoding="utf-8").read()
    body = re.sub(r"```.*?```", " ", text, flags=re.S)
    words = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-–.]*", re.sub(r"\[@[^\]]+\]", " ", body)))
    keys = []
    for grp in re.findall(r"\[@([^\]]+)\]", text):
        keys += [k.strip().lstrip("@") for k in grp.split(";")]
    uniq = list(dict.fromkeys(keys))
    missing = [k for k in uniq if VER.get(k, {}).get("status") != "verified"]
    recent = [k for k in uniq if VER.get(k, {}).get("class") == "recent"]
    found = [k for k in uniq if VER.get(k, {}).get("class") == "foundational"]
    print(f"## Automated checks for `{name}`\n")
    print(f"- Words (excluding citation keys and code blocks): **{words}** (≈ {words / 420:.1f} pages at ~420 words/page)")
    print(f"- Citation occurrences: {len(keys)}; distinct references: **{len(uniq)}**")
    print(f"- Unverified / unknown keys: **{len(missing)}** {missing if missing else ''}")
    print(f"- Recent (2022–2026): **{len(recent)}**; foundational (pre-2022): **{len(found)}**"
          f" → recent share {100 * len(recent) / max(1, len(uniq)):.0f}%")
    pre = [k for k in uniq if VER.get(k, {}).get("venue", "").lower().startswith("arxiv")]
    print(f"- arXiv-only references: {pre if pre else 'none'}")
    tabs = sorted(set(re.findall(r"Table (\d+\.\d+)", text)))
    figs = sorted(set(re.findall(r"Figure (\d+\.\d+)", text)))
    print(f"- Tables referenced: {tabs or 'none'}; figures referenced: {figs or 'none'}")
    print("\n### Dangerous-phrase scan (each hit inspected manually)\n")
    print("| Pattern | Line | Context |\n|---|---|---|")
    for i, line in enumerate(text.splitlines(), 1):
        for pat in DANGER:
            for m in re.finditer(pat, line, re.I):
                ctx = line[max(0, m.start() - 60): m.end() + 60].replace("|", "/")
                print(f"| `{pat}` | {i} | …{ctx}… |")


if __name__ == "__main__":
    main(sys.argv[1])
