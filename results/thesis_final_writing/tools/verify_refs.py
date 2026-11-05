"""Reference verification for the thesis: every candidate is checked against an authoritative bibliographic source.

  python results/thesis_final_writing/tools/verify_refs.py            # verify all candidates (cached)
Input : tools/reference_candidates.json  [{id, title, route, ...}]
Output: tools/reference_verification.json {id: {status, source, title, authors, venue, year, volume, issue, pages, doi}}
Routes:
  crossref  - api.crossref.org (doi given, else bibliographic title query with exact normalised-title match)
  mla       - ML Anthology venue/year index -> paper page citation_* metadata (ICLR, NeurIPS, ICML, CVF, ECCV ...)
  index     - an official proceedings index page (e.g. proceedings.mlsys.org) -> paper page citation_* metadata
  page      - an official paper page with citation_* metadata (e.g. jmlr.org)
  arxiv     - export.arxiv.org API (foundational preprints only)
  manual    - foundational non-indexed item (technical report, book); a URL that must resolve is recorded
A candidate is 'verified' only if the source returns the same normalised title.
"""
import difflib
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.join(HERE, "reference_candidates.json")
OUT = os.path.join(HERE, "reference_verification.json")
CACHE = os.path.join(HERE, "_cache")
UA = {"User-Agent": "Mozilla/5.0 (MSc thesis reference verification; contact via repository owner)"}


def norm(s):
    s = html.unescape(s or "").lower()
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def same(a, b):
    a, b = norm(a), norm(b)
    return a == b or (len(a) > 20 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.97)


def get(url, cache=True):
    os.makedirs(CACHE, exist_ok=True)
    key = os.path.join(CACHE, re.sub(r"[^A-Za-z0-9]+", "_", url)[-180:])
    if cache and os.path.exists(key):
        return open(key, encoding="utf-8").read()
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=UA)
            txt = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
            open(key, "w", encoding="utf-8").write(txt)
            time.sleep(0.6)
            return txt
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{url}: {err}")


def meta(page):
    out = {}
    for k, v in re.findall(r'<meta\s+name="?(citation_[a-z_]+)"?\s+content="([^"]*)"', page, re.I):
        out.setdefault(k.lower(), []).append(html.unescape(v).strip())
    return out


def from_meta(m, source):
    first = (m.get("citation_firstpage") or [None])[0]
    last = (m.get("citation_lastpage") or [None])[0]
    return {"source": source, "title": (m.get("citation_title") or [""])[0], "authors": m.get("citation_author", []),
            "venue": (m.get("citation_conference_title") or m.get("citation_journal_title") or m.get("citation_inbook_title") or [""])[0],
            "year": ((m.get("citation_publication_date") or m.get("citation_date") or [""])[0])[:4],
            "volume": (m.get("citation_volume") or [None])[0], "issue": (m.get("citation_issue") or [None])[0],
            "pages": f"{first}-{last}" if first and last else first, "doi": (m.get("citation_doi") or [None])[0]}


def crossref(c):
    if c.get("doi"):
        items = [json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(c["doi"])))["message"]]
    else:
        q = urllib.parse.quote(c["title"])
        items = json.loads(get(f"https://api.crossref.org/works?query.bibliographic={q}&rows=10"))["message"]["items"]
    for it in items:
        t = (it.get("title") or [""])[0]
        if it.get("subtitle") and not same(t, c["title"]):
            t = f"{t}: {it['subtitle'][0]}"
        if not same(t, c["title"]) or re.search(r"/(mm\d+|supp)", it.get("DOI", ""), re.I):
            continue
        if it.get("type") in ("posted-content",) or "arxiv" in it.get("DOI", "").lower():
            continue
        y = (it.get("published-print") or it.get("published-online") or it.get("issued") or {}).get("date-parts", [[None]])[0][0]
        return {"source": "https://doi.org/" + it["DOI"], "title": t,
                "authors": [" ".join(x for x in (a.get("given"), a.get("family")) if x) for a in it.get("author", [])],
                "venue": next((v for v in (it.get("container-title") or [""]) if "Lecture Notes" not in v), (it.get("container-title") or [""])[0]),
                "series": [v for v in (it.get("container-title") or []) if "Lecture Notes" in v], "year": str(y), "volume": it.get("volume"),
                "issue": it.get("issue"), "pages": it.get("page"), "doi": it["DOI"], "type": it.get("type"),
                "publisher": it.get("publisher")}
    return None


def mla(c):
    venue, year = c["mla"].split("/")
    idx = get(f"https://mlanthology.org/{venue}/{year}/")
    for href, title in re.findall(r'<a href=(/[a-z]+/\d+/[a-z0-9-]+/) class=paper-title>(.*?)</a>', idx, re.S):
        if same(re.sub(r"\s+", " ", title), c["title"]):
            r = from_meta(meta(get("https://mlanthology.org" + href)), "https://mlanthology.org" + href)
            if not r["venue"]:
                r["venue"] = venue.upper()
            r["year"] = r["year"] or year
            return r
    return None


def index(c):
    idx = get(c["index"])
    base = re.match(r"(https?://[^/]+)", c["index"]).group(1)
    for href, title in re.findall(r'<a[^>]*href="([^"]+)"[^>]*>([^<]+)</a>', idx):
        if same(title, c["title"]):
            url = href if href.startswith("http") else base + href
            return from_meta(meta(get(url)), url)
    return None


def page(c):
    r = from_meta(meta(get(c["page"])), c["page"])
    return r if same(r["title"], c["title"]) else None


def arxiv(c):
    x = get(f"https://export.arxiv.org/api/query?id_list={c['arxiv']}")
    entry = x.split("<entry>")[1] if "<entry>" in x else ""
    t = re.search(r"<title>(.*?)</title>", entry, re.S)
    if not t or not same(t.group(1), c["title"]):
        return None
    return {"source": f"https://arxiv.org/abs/{c['arxiv']}", "title": re.sub(r"\s+", " ", t.group(1)).strip(),
            "authors": re.findall(r"<name>(.*?)</name>", entry), "venue": "arXiv preprint", "year": re.search(r"<published>(\d{4})", entry).group(1),
            "volume": None, "issue": None, "pages": None, "doi": None}


def manual(c):
    try:
        get(c["url"])
        ok = True
    except Exception:  # noqa: BLE001
        ok = False
    return {"source": c["url"], "title": c["title"], "authors": c["authors"], "venue": c["venue"], "year": str(c["year"]),
            "volume": None, "issue": None, "pages": None, "doi": None, "url_resolves": ok} if ok else None


ROUTES = {"crossref": crossref, "mla": mla, "index": index, "page": page, "arxiv": arxiv, "manual": manual}


def main():
    cands = json.load(open(CAND, encoding="utf-8"))
    out = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    only = set(sys.argv[1:])
    for c in cands:
        if only and c["id"] not in only:
            continue
        if not only and out.get(c["id"], {}).get("status") == "verified":
            continue
        try:
            r = ROUTES[c["route"]](c)
        except Exception as e:  # noqa: BLE001
            r, err = None, str(e)
        else:
            err = None
        out[c["id"]] = ({"status": "verified", **r} if r else {"status": "not_found", "error": err, "route": c["route"]})
        out[c["id"]].update({"route": c["route"], "class": c["class"], "topic": c.get("topic")})
        print(f"{c['id']:22s} {out[c['id']]['status']:10s} {out[c['id']].get('venue', '')[:50]} {out[c['id']].get('year', '')}", flush=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    ok = sum(v["status"] == "verified" for v in out.values())
    print(f"verified {ok}/{len(out)}")


if __name__ == "__main__":
    main()
