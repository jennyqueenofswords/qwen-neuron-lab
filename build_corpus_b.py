"""Fetch an independent second corpus into corpus/corpus_b.jsonl, different in genre from corpus A.

Sources (text only, nothing executed):
  poetry, drama   Gutenberg: Whitman, Dickinson, Shakespeare, Wilde
  fiction-b       Gutenberg: Sherlock Holmes, Jane Eyre, Dracula
  travel          random Wikivoyage pages
  science         arXiv abstracts (export API)
  simple          random Simple English Wikipedia pages

Used to check that neuron readings from corpus A aren't accidents of that corpus.
"""

import json
import os
import random
import re
import time
import xml.etree.ElementTree as ET

from build_corpus import get, HERE

OUT = os.path.join(HERE, "corpus", "corpus_b.jsonl")
BOOKS = {"poetry": {1322: "Leaves of Grass", 12242: "Poems by Emily Dickinson"},
         "drama": {1524: "Hamlet", 1513: "Romeo and Juliet", 844: "The Importance of Being Earnest"},
         "fiction-b": {1661: "Sherlock Holmes", 1260: "Jane Eyre", 345: "Dracula"}}
PASSAGES_PER_BOOK = 22
PASSAGE_CHARS = 1400
ARXIV_QUERIES = ["cat:q-bio.NC", "cat:astro-ph.EP", "cat:cs.CL", "cat:physics.soc-ph", "cat:q-bio.PE", "cat:cond-mat.soft"]


def gutenberg(book_id, source, rng):
    raw = get(f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt").replace("\r", "")
    start, end = raw.find("*** START"), raw.find("*** END")
    body = raw[raw.find("\n", start) + 1:end] if start != -1 and end != -1 else raw
    out = []
    for _ in range(PASSAGES_PER_BOOK):
        i = body.find("\n\n", rng.randrange(0, max(1, len(body) - PASSAGE_CHARS))) + 2
        out.append({"source": source, "text": body[i:i + PASSAGE_CHARS].strip()})
    return out


def mediawiki_random(host, source, n):
    import urllib.parse
    out = []
    while len(out) < n:
        q = urllib.parse.urlencode({"action": "query", "format": "json", "generator": "random", "grnnamespace": 0,
                                    "grnlimit": 20, "prop": "extracts", "explaintext": 1, "exlimit": 20, "exintro": 0,
                                    "exchars": 1800})
        pages = json.loads(get(f"https://{host}/w/api.php?{q}"))["query"]["pages"]
        time.sleep(1.5)
        for p in pages.values():
            text = (p.get("extract") or "").strip()
            if len(text) > 300:
                out.append({"source": source, "text": text[:1800]})
    return out[:n]


def arxiv(n_per_query):
    import urllib.parse
    out = []
    for q in ARXIV_QUERIES:
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"search_query": q, "start": 0, "max_results": n_per_query, "sortBy": "submittedDate"})
        root = ET.fromstring(get(url))
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for e in root.findall("a:entry", ns):
            text = re.sub(r"\s+", " ", (e.find("a:summary", ns).text or "")).strip()
            if text:
                out.append({"source": "science", "text": text})
        time.sleep(3.5)  # arXiv asks for a pause between API calls
    return out


def main():
    rng = random.Random(1)
    rows = []
    for source, books in BOOKS.items():
        for book_id, title in books.items():
            print(f"{source}: {title}", flush=True)
            rows += gutenberg(book_id, source, rng)
    print("travel: Wikivoyage", flush=True)
    rows += mediawiki_random("en.wikivoyage.org", "travel", 70)
    print("simple: Simple English Wikipedia", flush=True)
    rows += mediawiki_random("simple.wikipedia.org", "simple", 90)
    print("science: arXiv abstracts", flush=True)
    rows += arxiv(20)
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
