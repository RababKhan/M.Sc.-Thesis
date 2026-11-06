"""Chapter 7 must introduce no new numbers. Every numeric token in chapter_07_discussion.md is looked up in the
audited Chapters 4-6 (normalised: unicode minus -> '-', thousands commas kept). Tokens that are design constants or
structural counts are listed separately. Read-only.

  python results/thesis_final_writing/tools/audit_ch7_numbers.py
"""
import os
import re
import sys

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
norm = lambda s: s.replace("−", "-").replace(" ", " ")  # noqa: E731
ch7 = norm(open(os.path.join(W, "chapter_07_discussion.md"), encoding="utf-8").read())
ref = norm(" ".join(open(os.path.join(W, f), encoding="utf-8").read() for f in
                    ("chapter_04_experimental_setup.md", "chapter_05_core_results.md", "chapter_06_deployment_sustainability.md")))
body = re.sub(r"\[@[^\]]+\]", " ", re.sub(r"^#+ .*$", " ", ch7, flags=re.M))  # drop citations and headings
body = re.sub(r"(Sections?|Chapters?|Figure|Table|RQ)\s*[\d.,–-]+(?:\s*(?:–|and|,)\s*[\d.]+)*", " ", body)  # drop cross-references
toks = sorted(set(re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", body)), key=lambda x: float(x.replace(",", "")))
TRIVIAL = {str(i) for i in range(0, 21)} | {"24", "100", "120", "512", "1,296", "5,184", "2,500"}
missing = [t for t in toks if t not in TRIVIAL and t not in ref]
print(f"numeric tokens in Chapter 7: {len(toks)}; small counts / design constants: {len([t for t in toks if t in TRIVIAL])}")
print(f"tokens not found in audited Chapters 4-6: {missing if missing else 'none'}")
sys.exit(1 if missing else 0)
