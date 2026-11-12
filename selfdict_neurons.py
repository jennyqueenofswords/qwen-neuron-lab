"""Do Qwen's own primes (selfdict.py graph) get a layer-0 neuron of their own?

Fresh test of Hunt 30's lead (NSM primes: 35% had a dedicated layer-0 neuron vs 17% of controls),
with a word list chosen by the model's dictionary, not by us. Pre-set:
  - primes = words in at least half the grounding sets (eval/selfdict/graph.json)
  - controls = headwords of the same dictionary that are not primes, 3 per prime, nearest in
    corpus-A log frequency, without replacement, mostly-lowercase words with >= 5 occurrences
  - dedicated = some layer-0 neuron spends >= 50% of its top-64 corpus-A firings on the word
  - one-sided Fisher, primes > controls

    .venv/bin/python selfdict_neurons.py
"""

import collections
import json
import os
import re
import sys

import numpy as np
from scipy.stats import fisher_exact

from wiki_game import N, Net, nid

HERE = os.path.dirname(os.path.abspath(__file__))
ZH = len(sys.argv) > 1 and sys.argv[1] == "zh"  # `selfdict_neurons.py zh` runs it on the Chinese dictionary
OUT = os.path.join(HERE, "eval", "selfdict_zh" if ZH else "selfdict")


def main():
    G = json.load(open(os.path.join(OUT, "graph.json")))
    heads = {json.loads(l)["word"] for l in open(os.path.join(OUT, "dictionary.jsonl"), encoding="utf-8")}
    primes = set(G["primes"])
    net = Net()
    cache = {}

    def txt(t):
        if t not in cache:
            cache[t] = net.tok.decode([int(t)]).strip()
        return cache[t]

    freq, lower = collections.Counter(), collections.Counter()
    for t, c in collections.Counter(net.ids.tolist()).items():
        w = txt(t)
        if (re.fullmatch(r"[\u4e00-\u9fff]+", w) if ZH else (w.isalpha() and w.isascii())):
            freq[w.lower()] += c
            if w == w.lower():
                lower[w.lower()] += c
    pt, pv = net.ix["L0_pos_tok"], net.ix["L0_pos_val"]
    best = collections.defaultdict(float)
    owner = {}
    for k in range(N):
        idx = pt[:, k][pv[:, k] > 0]
        idx = idx[(idx >= 0) & (idx < len(net.ids))]
        for w, m in collections.Counter(txt(net.ids[i]).lower() for i in idx).items():
            if m / 64 > best[w]:
                best[w], owner[w] = m / 64, k

    ok = {w for w in heads if freq[w] >= 5 and lower[w] / freq[w] >= 0.8}
    P = sorted((w for w in primes if w in ok), key=lambda w: -freq[w])
    pool = [w for w in ok if w not in primes]
    used, ctrl = set(), []
    for p in P:
        c = sorted((w for w in pool if w not in used), key=lambda w: abs(np.log(freq[w]) - np.log(freq[p])))[:3]
        used.update(c)
        ctrl += c
    ded = lambda w: best[w] >= 0.5
    a, b = sum(map(ded, P)), sum(map(ded, ctrl))
    f = fisher_exact([[a, len(P) - a], [b, len(ctrl) - b]], alternative="greater")
    print(f"Qwen's primes in corpus A: {len(P)} of {len(primes)}; controls {len(ctrl)}")
    print(f"dedicated layer-0 neuron: primes {a}/{len(P)} = {a / len(P):.0%}, controls {b}/{len(ctrl)} = {b / len(ctrl):.0%}, "
          f"odds ratio {f.statistic:.2f}, Fisher p = {f.pvalue:.2g}")
    print("  primes with one: " + ", ".join(f"{w} {nid(owner[w])}" for w in P if ded(w)))
    print("  controls with one: " + ", ".join(w for w in ctrl if ded(w)))
    json.dump({"n_primes": len(P), "n_ctrl": len(ctrl), "primes_dedicated": a, "ctrl_dedicated": b, "p": float(f.pvalue),
               "controls": ctrl}, open(os.path.join(OUT, "neurons.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
