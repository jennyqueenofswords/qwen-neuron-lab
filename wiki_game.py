"""The Wikipedia game on Qwen3.5-4B's connectome.

On Wikipedia, clicking the first link of an article, over and over, ends at "Philosophy" for
most articles. Here a neuron's "first link" is its strongest incoming synapse (from
index/connectome_edges.npz): the earlier neuron it is most built from. Walk down from every
deep neuron until there is nothing below, and see where the walks end up.

Two link rules:
  raw       strongest incoming synapse as stored (virtual weight x how loudly the source fires)
  specific  same, divided by the source's total outgoing strength, so a loud hub that feeds
            everything counts less than a source that feeds this neuron in particular

    .venv/bin/python wiki_game.py [--from-layers 20 31] [--rule raw|specific] [--top 25]
"""

import argparse
import collections
import json
import os
import re

import numpy as np
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
IDX = os.path.join(HERE, "index")
N = 9216


def nid(g):
    return f"L{g // N}/N{g % N}"


def clean(w):
    w = w.replace("\n", "↵").replace("�", "").strip()
    return w if w and not re.fullmatch(r"[\W_]{0,1}", w) or w in "。，,.;:：；()（）\"'“”«»「」" else ""


class Net:
    def __init__(self):
        E = np.load(os.path.join(IDX, "connectome_edges.npz"))
        self.in_src = np.concatenate([E[f"L{L}_in_src"] for L in range(32)])
        self.in_val = np.concatenate([E[f"L{L}_in_val"] for L in range(32)]).astype(np.float32)
        self.in_exc = np.concatenate([E[f"L{L}_in_excite"] for L in range(32)])
        out_val = np.concatenate([E[f"L{L}_out_val"] for L in range(32)]).astype(np.float32)
        self.out_total = np.clip(out_val, 0, None).sum(1)
        self.motor = np.concatenate([np.load(os.path.join(IDX, "connectome_motor.npz"))[f"L{L}"] for L in range(32)])
        self.ix = np.load(os.path.join(IDX, "Qwen3.5-4B-4bit.npz"))
        self.ids = self.ix["ids"]
        self.tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "models", "Qwen3.5-4B-4bit"))
        self._pos = {}

    def link(self, g, rule):
        src, val = self.in_src[g], self.in_val[g]
        ok = src >= 0
        if not ok.any():
            return -1
        src, val = src[ok], val[ok]
        if rule == "specific":
            val = val / np.maximum(self.out_total[src], 1e-6)
        return int(src[np.argmax(val)])

    def writes(self, g, n=5):
        seen, out = set(), []
        for t in self.motor[g]:
            w = clean(self.tok.decode([int(t)]))
            if w and w.lower() not in seen:
                seen.add(w.lower())
                out.append(w)
        return out[:n]

    def reads(self, g, n=5):
        """Most common tokens at a neuron's strongest firings in corpus A."""
        L, k = g // N, g % N
        if L not in self._pos:
            self._pos[L] = (self.ix[f"L{L}_pos_tok"], self.ix[f"L{L}_pos_val"])
        pt, pv = self._pos[L]
        idx = pt[:, k][pv[:, k] > 0]
        c = collections.Counter(clean(self.tok.decode([int(self.ids[i])])) for i in idx if 0 <= i < len(self.ids))
        c.pop("", None)
        return [w for w, _ in c.most_common(n)]

    def label(self, g):
        return f"{nid(g):>10}  reads [{', '.join(self.reads(g))}]  writes [{', '.join(self.writes(g))}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-layers", type=int, nargs=2, default=[20, 31])
    ap.add_argument("--rule", default="raw", choices=["raw", "specific"])
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--show", nargs="*", default=["L23/N7935", "L22/N5191", "L22/N2742", "L15/N3920"])
    a = ap.parse_args()
    net = Net()

    starts = [g for g in range(a.from_layers[0] * N, (a.from_layers[1] + 1) * N) if (net.in_src[g] >= 0).any()]
    end, visits, lengths = collections.Counter(), collections.Counter(), []
    paths = {}
    for s in starts:
        p, g = [s], s
        while True:
            nxt = net.link(g, a.rule)
            if nxt < 0:
                break
            p.append(nxt)
            g = nxt
        paths[s] = p
        end[p[-1]] += 1
        for x in set(p[1:]):
            visits[x] += 1
        lengths.append(len(p) - 1)

    n = len(starts)
    print(f"rule={a.rule}  {n} walks from layers {a.from_layers[0]}-{a.from_layers[1]}  "
          f"mean length {np.mean(lengths):.1f} links  distinct endpoints {len(end)}")
    top1 = end.most_common(1)[0][1]
    top10 = sum(c for _, c in end.most_common(10))
    print(f"top endpoint takes {top1 / n:.1%} of walks, top 10 take {top10 / n:.1%}")
    print("\nENDPOINTS (where the walks bottom out)")
    for g, c in end.most_common(a.top):
        print(f"  {c / n:6.1%}  {net.label(g)}")
    print("\nMOST-VISITED (on the way down)")
    for g, c in visits.most_common(a.top):
        print(f"  {c / n:6.1%}  {net.label(g)}")
    print("\nSAMPLE WALKS")
    for s in a.show:
        L, k = s[1:].split("/N")
        g = int(L) * N + int(k)
        p = paths.get(g)
        if p is None:
            p, x = [g], g
            while (x := net.link(x, a.rule)) >= 0:
                p.append(x)
        print(f"\n  from {s}")
        for x in p:
            print("    " + net.label(x))

    out = os.path.join(HERE, "eval", "wiki_game")
    os.makedirs(out, exist_ok=True)
    json.dump({"rule": a.rule, "from_layers": a.from_layers, "n": n,
               "endpoints": [[nid(g), c] for g, c in end.most_common(200)],
               "visits": [[nid(g), c] for g, c in visits.most_common(200)],
               "paths": {nid(s): [nid(x) for x in p] for s, p in paths.items()}},
              open(os.path.join(out, f"walks_{a.rule}.json"), "w"))


if __name__ == "__main__":
    main()
