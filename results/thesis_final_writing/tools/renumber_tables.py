"""Renumber all chapter tables consecutively in order of appearance (MIST: tables numbered consecutively) and update
every in-text table reference across Chapters 1-8 in one simultaneous pass.

  python results/thesis_final_writing/tools/renumber_tables.py [--dry-run]

Captions are lines starting "**Table N.M. " or "**Table N.new. " (placeholder captions added during compression).
A reference "Table N.new" is mapped to the next "N.new" caption at or after it in the same file (or, failing that,
the last one before it). Writes tools/_table_renumber_map.json with the old -> new mapping. Values are untouched.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
CH = ["chapter_01_introduction.md", "chapter_02_literature_review.md", "chapter_03_framework.md",
      "chapter_04_experimental_setup.md", "chapter_05_core_results.md", "chapter_06_deployment_sustainability.md",
      "chapter_07_discussion.md", "chapter_08_conclusion.md"]
CAP = re.compile(r"(?m)^\*\*Table (\d+)\.(\d+|new)\. ")
REF = re.compile(r"\b(Tables?)(\s+)((?:\d+\.(?:\d+|new))(?:\s*(?:–|-|,|and|or)\s*(?:\d+\.(?:\d+|new)))*)")
NUM = re.compile(r"\d+\.(?:\d+|new)")


def main(dry):
    texts = {f: open(os.path.join(W, f), encoding="utf-8").read() for f in CH}
    gmap, new_pos, report = {}, {}, []
    for f in CH:
        k = 0
        new_pos[f] = []
        for m in CAP.finditer(texts[f]):
            k += 1
            chap, old = m.group(1), m.group(2)
            new = f"{chap}.{k}"
            if old == "new":
                new_pos[f].append((m.start(), new))
            else:
                gmap[f"{chap}.{old}"] = new
            report.append((f, f"{chap}.{old}", new))
    warn = []
    for f in CH:
        t = texts[f]
        caps_here = [(m.start(), m.end()) for m in CAP.finditer(t)]

        def map_num(tok, pos):
            if tok.endswith(".new"):
                after = [n for p, n in new_pos[f] if p >= pos and n.startswith(tok.split(".")[0] + ".")]
                before = [n for p, n in new_pos[f] if p < pos and n.startswith(tok.split(".")[0] + ".")]
                if after:
                    return after[0]
                if before:
                    return before[-1]
                warn.append(f"{f}: unresolved reference {tok}")
                return tok
            if tok not in gmap:
                warn.append(f"{f}: reference to Table {tok}, which has no caption (deleted table?)")
            return gmap.get(tok, tok)

        out, last = [], 0
        # captions first: rewrite caption numbers by position
        cap_iter = list(CAP.finditer(t))
        cap_new = {}
        for f2, old, new in report:
            pass
        ci = 0
        k = 0
        for m in cap_iter:
            k += 1
            cap_new[m.start()] = f"**Table {m.group(1)}.{k}. "
        pieces, last = [], 0
        events = sorted([(m.start(), m.end(), "cap", m) for m in cap_iter] +
                        [(m.start(), m.end(), "ref", m) for m in REF.finditer(t)
                         if not any(cs <= m.start() < ce for cs, ce in caps_here)], key=lambda e: e[0])
        for s, e, kind, m in events:
            if s < last:
                continue
            pieces.append(t[last:s])
            if kind == "cap":
                pieces.append(cap_new[s])
            else:
                nums = m.group(3)
                mapped = NUM.sub(lambda mm: map_num(mm.group(0), s), nums)
                toks = NUM.findall(nums)
                if "–" in nums or "-" in nums:
                    a, b = NUM.findall(mapped)[0], NUM.findall(mapped)[-1]
                    oa, ob = toks[0], toks[-1]
                    if (a.split(".")[0] != b.split(".")[0] or int(b.split(".")[1]) - int(a.split(".")[1]) !=
                            (int(ob.split(".")[1]) - int(oa.split(".")[1]) if not ob.endswith("new") and not oa.endswith("new") else 0)):
                        warn.append(f"{f}: range '{m.group(0)}' -> '{m.group(1)}{m.group(2)}{mapped}' changed its span; review")
                pieces.append(m.group(1) + m.group(2) + mapped)
            last = e
        pieces.append(t[last:])
        texts[f] = "".join(pieces)
    changed = [(f, o, n) for f, o, n in report if o != n]
    print(f"captions: {len(report)}; renumbered: {len(changed)}")
    for f, o, n in changed:
        print(f"  {f}: Table {o} -> Table {n}")
    for w_ in warn:
        print("WARNING:", w_)
    if not dry:
        for f in CH:
            open(os.path.join(W, f), "w", encoding="utf-8").write(texts[f])
        json.dump({"mapping": [{"file": f, "old": o, "new": n} for f, o, n in report], "warnings": warn},
                  open(os.path.join(HERE, "_table_renumber_map.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
