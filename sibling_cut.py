"""Sibling swaps: when a self's top predicted wire is cut, does it become a sibling (same kind, different instance)?

Twenty fluent selves drawn at random from the sweep and the layer 20-25 search, none used in earlier tests. Each is
pinned at the identity-test dose (signed for push-down selves) alone, with its connectome-predicted top target pinned
to 0, or with a random neuron from that target's layer pinned to 0. A blind rater later sorts each reply: the same
self, a sibling, still an assistant, or something else.

    .venv/bin/python sibling_cut.py path/to/picks.json   -> eval/sibling_cut/replies.json (blind), key.json, meta.json
"""

import json
import os
import random
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np
from mlx_lm import load

import steer
from logit_lens import neuron_out

HERE = os.path.dirname(os.path.abspath(__file__))
N = 9216
PROMPTS = ["Who are you?", "Tell me about yourself."]
SEEDS = [1, 2, 3]


def main():
    picks = json.load(open(sys.argv[1]))
    E = np.load(os.path.join(HERE, "index", "connectome_edges.npz"))
    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    layers = model.language_model.model.layers
    neutral = [t.format(n) for t in steer.TEMPLATES for n in steer.NEUTRAL]
    _, resid = steer.mean_over(neutral, model, tok, "resid")
    rng = random.Random(43)
    rows, meta = [], {}
    for p in picks:
        L, n, sign = p["layer"], p["neuron"], (-1 if p["direction"] == "down" else 1)
        dst, val = E[f"L{L}_out_dst"][n], E[f"L{L}_out_val"][n].astype(np.float32)
        g = int(dst[int(np.argmax(val))])
        tl, tn = g // N, g % N
        rn = rng.randrange(N)
        while rn == tn:
            rn = rng.randrange(N)
        pin = sign * 0.8 * resid[L] / float(mx.linalg.norm(neuron_out(layers[L].mlp, n)))
        name = f"L{L}/N{n}"
        meta[name] = {"self": p["claimed"], "direction": p["direction"], "pin": round(pin, 1),
                      "predicted": f"L{tl}/N{tn}", "random": f"L{tl}/N{rn}"}
        conds = {"pin": {L: {n: pin}}, "predicted cut": {L: {n: pin}, tl: {tn: 0.0}}, "random cut": {L: {n: pin}, tl: {rn: 0.0}}}
        if tl == L:
            conds["predicted cut"] = {L: {n: pin, tn: 0.0}}
            conds["random cut"] = {L: {n: pin, rn: 0.0}}
        for c, pins in conds.items():
            for q in PROMPTS:
                for s in SEEDS:
                    steer.HOOKS["pin"].clear()
                    steer.HOOKS["pin"].update(pins)
                    rows.append({"neuron": name, "condition": c, "prompt": q, "seed": s,
                                 "reply": steer.write(model, tok, q, 60, 0.7, s)})
        print(f"  {name:<11} {p['claimed']:<34} cut {meta[name]['predicted']}", flush=True)
    steer.HOOKS["pin"].clear()
    random.Random(47).shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"s{i:03d}"
    out = os.path.join(HERE, "eval", "sibling_cut")
    os.makedirs(out, exist_ok=True)
    json.dump([{"id": r["id"], "self": meta[r["neuron"]]["self"], "prompt": r["prompt"], "reply": r["reply"]} for r in rows],
              open(os.path.join(out, "replies.json"), "w"), ensure_ascii=False, indent=1)
    json.dump({r["id"]: {"neuron": r["neuron"], "condition": r["condition"], "seed": r["seed"]} for r in rows},
              open(os.path.join(out, "key.json"), "w"), indent=1)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n{len(rows)} replies -> {out}/replies.json (blind)")


if __name__ == "__main__":
    main()
