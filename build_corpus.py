"""Fetch a mixed text corpus for the neuron explorer into corpus/corpus.jsonl.

Sources (text only, nothing executed):
  wiki-en, wiki-zh, wiki-ja, wiki-es   random Wikipedia article intros (MediaWiki API)
  fiction                              passages from four Project Gutenberg novels
  code                                 Python standard library source, read locally
  mixed                                the hand-written sentences in steer.py

Each line is {"source": ..., "text": ...}. Re-run to draw a fresh random sample.
"""

import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "corpus", "corpus.jsonl")
UA = {"User-Agent": "qwen-local-neuron-explorer/0.1 (personal research)"}
WIKI_COUNTS = {"en": 160, "zh": 30, "ja": 30, "es": 30}
GUTENBERG = {1342: "Pride and Prejudice", 2701: "Moby-Dick", 84: "Frankenstein", 11: "Alice in Wonderland"}
PASSAGES_PER_BOOK = 25
PASSAGE_CHARS = 1200
CODE_FILES = ["json/decoder.py", "textwrap.py", "fractions.py", "calendar.py", "csv.py", "difflib.py"]


try:
    import truststore  # verify TLS against the macOS keychain, like curl does

    truststore.inject_into_ssl()
except ImportError:
    pass


def get(url, tries=5):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == tries - 1:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 5 * 2 ** attempt
            print(f"  rate limited, waiting {wait}s")
            time.sleep(wait)


def wiki(lang, n):
    out = []
    while len(out) < n:
        q = urllib.parse.urlencode({
            "action": "query", "format": "json", "generator": "random", "grnnamespace": 0,
            "grnlimit": 20, "prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": 20,
        })
        pages = json.loads(get(f"https://{lang}.wikipedia.org/w/api.php?{q}"))["query"]["pages"]
        time.sleep(1.5)  # stay well under Wikipedia's anonymous rate limit
        for p in pages.values():
            text = (p.get("extract") or "").strip()
            if len(text) > 200:
                out.append({"source": f"wiki-{lang}", "text": text[:2000]})
    return out[:n]


def gutenberg(book_id, rng):
    raw = get(f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt")
    start = raw.find("*** START")
    end = raw.find("*** END")
    body = raw[raw.find("\n", start) + 1:end] if start != -1 and end != -1 else raw
    body = re.sub(r"\r", "", body)
    out = []
    for _ in range(PASSAGES_PER_BOOK):
        i = rng.randrange(0, max(1, len(body) - PASSAGE_CHARS))
        i = body.find("\n\n", i) + 2  # start on a paragraph boundary
        out.append({"source": "fiction", "text": body[i:i + PASSAGE_CHARS].strip()})
    return out


def code():
    import sysconfig
    lib = sysconfig.get_paths()["stdlib"]
    out = []
    for rel in CODE_FILES:
        src = open(os.path.join(lib, rel), encoding="utf-8").read()
        for i in range(0, min(len(src), 6000), 1500):
            out.append({"source": "code", "text": src[i:i + 1500]})
    return out


def main():
    sys.path.insert(0, HERE)
    from steer import CORPUS

    rng = random.Random(0)
    rows = [{"source": "mixed", "text": t} for t in CORPUS]
    rows += code()
    for book_id, title in GUTENBERG.items():
        print(f"fiction: {title}")
        rows += gutenberg(book_id, rng)
    for lang, n in WIKI_COUNTS.items():
        print(f"wikipedia {lang}: {n} intros")
        rows += wiki(lang, n)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {}
    for r in rows:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    print(f"wrote {len(rows)} passages to {OUT}: {counts}")


if __name__ == "__main__":
    main()
