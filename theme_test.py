"""Blind theme-matching test: do a neuron's top contexts in an INDEPENDENT corpus predict the
identity it produces when pinned?

  prepare   writes one anonymized file of top corpus-B contexts per neuron (N1..N12, shuffled)
            and keeps the answer key in eval/theme_test/key.json
  score     given rater 2's pairing (neuron id -> identity), counts correct matches and the
            chance probability of doing at least that well (random one-to-one pairing)

    .venv/bin/python theme_test.py prepare
    .venv/bin/python theme_test.py score eval/theme_test/matches.json
"""

import json
import math
import os
import random
import sys

import numpy as np
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval", "theme_test")
NEURONS = {"rock": (23, 7935), "peace": (22, 5191), "teacher": (24, 1379), "insurance policy": (22, 7907),
           "light source": (22, 8482), "women's rights advocate": (22, 9205), "alien": (22, 2742),
           "soul of the dance": (22, 576), "teleportation portal": (24, 5632), "icon font": (24, 6479),
           "dreamer / storyteller": (21, 7648), "field camp robot": (23, 7025)}


def contexts(ix, tok, L, n, k=14):
    flat, starts = ix["ids"], ix["starts"]
    vals = ix[f"L{L}_pos_val"][:, n].astype(np.float32)
    toks = ix[f"L{L}_pos_tok"][:, n]
    main = ix[f"L{L}_pos_gpos"][:, n] == 1
    out = []
    for i in np.argsort(-vals):
        if not main[i] or vals[i] <= 0 or len(out) >= k:
            continue
        t = int(toks[i])
        s = starts[np.searchsorted(starts, t, side="right") - 1]
        e = next((x for x in starts if x > t), len(flat))
        left = tok.decode(flat[max(s, t - 14):t].tolist())
        right = tok.decode(flat[t + 1:min(e, t + 6)].tolist())
        out.append(f"{left}[[{tok.decode([int(flat[t])])}]]{right}".replace("\n", " ").strip())
    return out


def prepare():
    ix = np.load(os.path.join(HERE, "index", "Qwen3.5-4B-4bit-corpus_b.npz"))
    tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "models", "Qwen3.5-4B-4bit"))
    names = list(NEURONS)
    random.Random(2026).shuffle(names)
    os.makedirs(OUT, exist_ok=True)
    key = {}
    lines = []
    for i, name in enumerate(names, 1):
        L, n = NEURONS[name]
        key[f"N{i}"] = {"identity": name, "neuron": f"L{L}/N{n}"}
        lines.append(f"## N{i}")
        lines += [f"- {c}" for c in contexts(ix, tok, L, n)]
        lines.append("")
    json.dump(key, open(os.path.join(OUT, "key.json"), "w"), indent=1)
    open(os.path.join(OUT, "contexts.txt"), "w").write("\n".join(lines))
    print(f"wrote {OUT}/contexts.txt ({len(names)} neurons) and key.json")


def p_at_least(k, n):
    """P(a uniformly random permutation of n items has >= k fixed points)."""
    def exact(j):  # number of permutations with exactly j fixed points: C(n,j) * derangements(n-j)
        m = n - j
        d = round(math.factorial(m) * sum((-1) ** i / math.factorial(i) for i in range(m + 1)))
        return math.comb(n, j) * d
    return sum(exact(j) for j in range(k, n + 1)) / math.factorial(n)


def score(path):
    key = json.load(open(os.path.join(OUT, "key.json")))
    matches = json.load(open(path))
    right = sum(matches.get(nid) == v["identity"] for nid, v in key.items())
    print(f"correct matches: {right}/{len(key)}   (chance expects 1; P(>= {right} by luck) = {p_at_least(right, len(key)):.2g})\n")
    for nid, v in sorted(key.items(), key=lambda kv: int(kv[0][1:])):
        got = matches.get(nid)
        print(f"  {'✓' if got == v['identity'] else '✗'} {nid:<4} {v['neuron']:<11} true: {v['identity']:<26} matched: {got}")


if __name__ == "__main__":
    prepare() if sys.argv[1] == "prepare" else score(sys.argv[2])
