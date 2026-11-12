"""Build the Qwen Dictionary explorer -> atlas/qwen-dictionary.html (template: atlas/dictionary_template.html).

Every headword of Qwen's self-dictionary (Hunt 31) gets a ring: the primes are ring 0, and a word
sits one ring past the furthest-out word its definition uses, so each ring can be learned from the
rings inside it. "Known" for this is the primes plus the grounding set that overlaps them most
(primes alone don't break every cycle). Each word's parent is its defining word on the ring just
inside, which makes the rings a tree rooted in the primes: the explorer's map.

    .venv/bin/python dictionary_data.py
"""

import collections
import json
import os
import re

from primes_test import PRIMES
from selfdict import candidates, def_words, lemma, strip_boilerplate

HERE = os.path.dirname(os.path.abspath(__file__))
SD = os.path.join(HERE, "eval", "selfdict")


def main():
    E = {json.loads(l)["word"]: json.loads(l) for l in open(os.path.join(SD, "dictionary.jsonl"))}
    G = json.load(open(os.path.join(SD, "graph.json")))
    rank = {w: i for i, w in enumerate(candidates())}
    for w, e in E.items():
        e["uses"] = def_words(e["definition"], w, rank)
    words = set(E)
    primes = set(G["primes"])
    base = max(G["grounding_sets"], key=lambda s: len(primes & set(s)))
    known = primes | set(base)
    uses = {w: sorted(set(E[w]["uses"]) & words - {w}) for w in words}
    deps = collections.defaultdict(list)
    for w, us in uses.items():
        for u in us:
            deps[u].append(w)

    # rings: learnable once everything you're made of is learned
    ring = {w: 0 for w in known}
    left = words - known
    r = 0
    while left:
        r += 1
        new = {w for w in left if all(u in ring and ring[u] < r for u in uses[w])}
        if not new:
            break
        for w in new:
            ring[w] = r
        left -= new
    print(f"rings 0–{r}; {len(left)} words never resolve")
    # grounding-set words that aren't primes: put them on ring 1 of... no: they are known, ring 0 too,
    # but flagged so the page can tell primes (in >= half the sets) from one set's extra picks
    parent = {}
    for w in words:
        if ring.get(w, 0) > 0:
            inner = [u for u in uses[w] if ring.get(u) == ring[w] - 1]
            parent[w] = max(inner, key=lambda u: (len(deps[u]), u))

    # layer-0 neurons that mostly read the word (selfdict_neurons.py's measure, for every headword)
    from wiki_game import N, Net, nid
    net = Net()
    cache = {}

    def txt(t):
        if t not in cache:
            cache[t] = net.tok.decode([int(t)]).strip().lower()
        return cache[t]

    pt, pv = net.ix["L0_pos_tok"], net.ix["L0_pos_val"]
    own = {}
    for k in range(N):
        idx = pt[:, k][pv[:, k] > 0]
        idx = idx[(idx >= 0) & (idx < len(net.ids))]
        for w, m in collections.Counter(txt(net.ids[i]) for i in idx).items():
            if w in words and m / 64 >= 0.5 and m / 64 > own.get(w, (0, 0))[1]:
                own[w] = (nid(k), m / 64)

    common = rank
    freq = G.get("prime_freq", {})
    allfreq = collections.Counter(w for s in G["grounding_sets"] for w in s)
    out = []
    for w in sorted(words, key=lambda w: (ring.get(w, 99), -len(deps[w]), w)):
        d = E[w]["definition"]
        # surface forms in the definition that map to a different headword
        sm = {}
        for t in re.findall(r"[A-Za-z]+", strip_boilerplate(d, w)):
            lw = lemma(t, common)
            if lw != t.lower() and lw in words:
                sm[t.lower()] = lw
        out.append({
            "w": w, "d": d, "r": ring.get(w, -1), "p": parent.get(w), "u": uses[w], "n": len(deps[w]),
            "g": allfreq.get(w, 0), "prime": int(w in primes), "nsm": int(w in PRIMES),
            "own": own.get(w, [None])[0], "rank": rank.get(w), "round": E[w]["round"], "map": sm or None,
        })
    data = {"words": out, "n_sets": len(G["grounding_sets"]), "rings": r,
            "n_primes": len(primes), "n_kernel": len(G["kernel"]), "n_core": len(G["core"])}
    tpl = open(os.path.join(HERE, "atlas", "dictionary_template.html")).read()
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = tpl.replace("/*DATA*/null", blob)
    path = os.path.join(HERE, "atlas", "qwen-dictionary.html")
    open(path, "w").write(html)
    ringsz = collections.Counter(x["r"] for x in out)
    print("ring sizes:", dict(sorted(ringsz.items())))
    print(f"wrote {path} ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
