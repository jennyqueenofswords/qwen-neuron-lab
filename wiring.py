"""Wiring neighborhood of neurons, from the full-precision chat-model weights.

Virtual weight from an earlier neuron A (layer La) to a later neuron B (layer Lb):
    vw = down_A . (W_in_B * g_Lb)
where down_A is A's output arrow (down_proj column), W_in_B is B's gate or up row, and g_Lb is the
weight of the RMSNorm in front of B's MLP. This is the direct path through the residual stream:
it ignores attention, intermediate layers and the norm's division (the same for every A).

For each target: top upstream neurons by |gate vw| + |up vw|, the words that drive it directly
from the embeddings, the downstream neurons it feeds, and the words it pushes at the output.
Hubs: upstream/downstream neurons shared by several targets, against the overlap among
random neurons from the same layers.

    .venv/bin/python wiring.py              writes index/wiring.npz and prints a summary
"""

import json
import os
import random

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from diff_models import tensors

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = {"rock": (23, 7935), "peace": (22, 5191), "teacher": (24, 1379), "insurance": (22, 7907),
           "light source": (22, 8482), "advocate": (22, 9205), "alien": (22, 2742), "dance": (22, 576),
           "portal": (24, 5632), "icon font": (24, 6479), "dreams": (21, 7648), "field robot": (23, 7025)}
TOP = 50


class Weights:
    def __init__(self):
        self.t = tensors("Qwen3.5-4B")
        self.p = "model.language_model."
        self._down = {}

    def down(self, L):  # [hidden, neurons], cached per layer
        if L not in self._down:
            w = self.t[self.p + f"layers.{L}.mlp.down_proj.weight"].astype(mx.float32)
            mx.eval(w)
            self._down[L] = w
        return self._down[L]

    def read(self, L, n):  # gate and up read vectors with the pre-MLP norm folded in
        g = self.t[self.p + f"layers.{L}.post_attention_layernorm.weight"].astype(mx.float32)
        g = 1.0 + g  # Qwen3.5 stores every RMSNorm weight as (scale - 1); mlx-lm adds the 1 back on load
        gate = self.t[self.p + f"layers.{L}.mlp.gate_proj.weight"][n].astype(mx.float32) * g
        up = self.t[self.p + f"layers.{L}.mlp.up_proj.weight"][n].astype(mx.float32) * g
        return gate, up

    def write(self, L, n):
        return self.down(L)[:, n]


def upstream(W, L, n):
    gate, up = W.read(L, n)
    rows = []
    for La in range(L):
        D = W.down(La)
        vg, vu = np.array(gate @ D), np.array(up @ D)
        for m in np.argsort(-(np.abs(vg) + np.abs(vu)))[:TOP]:
            rows.append((abs(vg[m]) + abs(vu[m]), La, int(m), float(vg[m]), float(vu[m])))
    rows.sort(reverse=True)
    return rows[:TOP]


def downstream(W, L, n):
    w = W.write(L, n)
    rows = []
    for Lb in range(L + 1, 32):
        gnorm = W.t[W.p + f"layers.{Lb}.post_attention_layernorm.weight"].astype(mx.float32)
        gnorm = 1.0 + gnorm
        G = W.t[W.p + f"layers.{Lb}.mlp.gate_proj.weight"].astype(mx.float32)
        U = W.t[W.p + f"layers.{Lb}.mlp.up_proj.weight"].astype(mx.float32)
        x = w * gnorm
        vg, vu = np.array(G @ x), np.array(U @ x)
        for m in np.argsort(-(np.abs(vg) + np.abs(vu)))[:TOP]:
            rows.append((abs(vg[m]) + abs(vu[m]), Lb, int(m), float(vg[m]), float(vu[m])))
    rows.sort(reverse=True)
    return rows[:TOP]


def token_io(W, tok, L, n, k=10):
    emb = W.t[W.p + "embed_tokens.weight"].astype(mx.float32)  # tied with the output head
    gate, _ = W.read(L, n)
    sensory = np.array(emb @ gate)
    fnorm = W.t[W.p + "norm.weight"].astype(mx.float32)
    fnorm = 1.0 + fnorm
    motor = np.array(emb @ (W.write(L, n) * fnorm))
    dec = lambda ids: [tok.decode([int(i)]) for i in ids]
    return dec(np.argsort(-sensory)[:k]), dec(np.argsort(-motor)[:k])


def overlap(sets):
    """Mean number of neurons shared between each pair of sets."""
    keys = list(sets)
    pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
    return np.mean([len(sets[a] & sets[b]) for a, b in pairs])


def main():
    W = Weights()
    tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "models", "Qwen3.5-4B-4bit"))
    up_sets, down_sets, report = {}, {}, {}
    for name, (L, n) in TARGETS.items():
        ups, downs = upstream(W, L, n), downstream(W, L, n)
        sens, mot = token_io(W, tok, L, n)
        up_sets[name] = {(r[1], r[2]) for r in ups}
        down_sets[name] = {(r[1], r[2]) for r in downs}
        report[name] = {"neuron": f"L{L}/N{n}", "up": ups, "down": downs, "sensory": sens, "motor": mot}
        print(f"  {name:<13} L{L}/N{n}  done", flush=True)

    rng = random.Random(3)
    r_up, r_down = {}, {}
    for i, (name, (L, n)) in enumerate(TARGETS.items()):
        m = rng.randrange(9216)
        r_up[i] = {(r[1], r[2]) for r in upstream(W, L, m)}
        r_down[i] = {(r[1], r[2]) for r in downstream(W, L, m)}

    def hubs(sets):
        count = {}
        for s in sets.values():
            for x in s:
                count[x] = count.get(x, 0) + 1
        return sorted(((c, x) for x, c in count.items() if c >= 3), reverse=True)

    print(f"\nshared neighbors: mean neurons shared per pair of targets (top {TOP} each)")
    print(f"  upstream:   identity {overlap(up_sets):.2f}   random baseline {overlap(r_up):.2f}")
    print(f"  downstream: identity {overlap(down_sets):.2f}   random baseline {overlap(r_down):.2f}")
    hu, hd, rhu, rhd = hubs(up_sets), hubs(down_sets), hubs(r_up), hubs(r_down)
    print(f"  hubs (in the top {TOP} of >= 3 targets): upstream {len(hu)} (random {len(rhu)}), downstream {len(hd)} (random {len(rhd)})")
    for label, hs, sets in (("UPSTREAM", hu, up_sets), ("DOWNSTREAM", hd, down_sets)):
        for c, (L, m) in hs[:10]:
            who = [k for k, s in sets.items() if (L, m) in s]
            print(f"    {label:<10} L{L}/N{m:<5} shared by {c}: {', '.join(who)}")

    for name, r in report.items():
        print(f"\n== {name} {r['neuron']}")
        print(f"   sensory (words that open its gate directly): {' | '.join(r['sensory'])}")
        print(f"   motor (words it pushes at the output):       {' | '.join(r['motor'])}")
        print("   top inputs:  " + ", ".join(f"L{x[1]}/N{x[2]}({x[3]:+.2f}g,{x[4]:+.2f}u)" for x in r["up"][:6]))
        print("   top outputs: " + ", ".join(f"L{x[1]}/N{x[2]}({x[3]:+.2f}g,{x[4]:+.2f}u)" for x in r["down"][:6]))
    json.dump({k: {**v, "up": [list(map(float, x[:1])) + [x[1], x[2], x[3], x[4]] for x in v["up"]],
                   "down": [list(map(float, x[:1])) + [x[1], x[2], x[3], x[4]] for x in v["down"]]}
               for k, v in report.items()},
              open(os.path.join(HERE, "index", "wiring.json"), "w"), ensure_ascii=False, indent=0)


if __name__ == "__main__":
    main()
