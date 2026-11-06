"""Word counts per chapter, identical to tools/assemble.py (citation keys and code blocks excluded).

  python results/thesis_final_writing/tools/wordcount.py
"""
import os
import re

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH = ["chapter_01_introduction.md", "chapter_02_literature_review.md", "chapter_03_framework.md",
      "chapter_04_experimental_setup.md", "chapter_05_core_results.md", "chapter_06_deployment_sustainability.md",
      "chapter_07_discussion.md", "chapter_08_conclusion.md"]


def words(t, prose_only=False):
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"\[@[^\]]+\]", " ", t)
    if prose_only:
        t = "\n".join(l for l in t.splitlines() if not l.startswith("|"))
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-–.]*", t))


if __name__ == "__main__":
    tot = totp = 0
    for f in CH:
        t = open(os.path.join(W, f), encoding="utf-8").read()
        w, p = words(t), words(t, True)
        tot += w
        totp += p
        nt = len(re.findall(r"^\|-", t, re.M))
        print(f"{f:45s} {w:7,d} words  {p:7,d} prose  tables={nt}")
    print(f"{'TOTAL':45s} {tot:7,d} words  {totp:7,d} prose")
