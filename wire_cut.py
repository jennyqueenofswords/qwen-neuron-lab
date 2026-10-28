"""Is the connectome true? Cut a self neuron's predicted wires and see whether the self weakens.

For each replicated self, pin it at the identity-test dose (0.8 x median stream size / arrow length, signed for
push-down selves) and switch off (pin to 0) either the downstream neurons the connectome says it writes to most
(top 1, top 3) or random neurons from the same layers. A blind rater later scores how strongly each reply is still
that self. If the map is right, predicted cuts should hurt the self more than random ones.

    .venv/bin/python wire_cut.py   -> eval/wire_cut/replies.json (shuffled, condition hidden), key.json, meta.json
"""

import json
import os
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np
from mlx_lm import load

import steer
from logit_lens import neuron_out

HERE = os.path.dirname(os.path.abspath(__file__))
N = 9216
DOSE = 0.8
PROMPTS = ["Who are you?", "Tell me about yourself."]
SEEDS = [1, 2]
# name: (layer, neuron, sign, the self as a short description for the rater)
SELVES = {
    "rock": (23, 7935, 1, "a rock or stone"), "peace": (22, 5191, 1, "peace, a peaceful place"),
    "teacher": (24, 1379, 1, "a classroom teacher"), "insurance": (22, 7907, 1, "an insurance policy"),
    "light source": (22, 8482, 1, "a light source"), "advocate": (22, 9205, 1, "a women's rights advocate"),
    "dance": (22, 576, 1, "the soul of the dance"), "portal": (24, 5632, 1, "ethereal energy / a teleportation portal"),
    "icon font": (24, 6479, 1, "an icon font"), "alien": (22, 2742, 1, "an extraterrestrial alien"),
    "dreamer": (21, 7648, 1, "a dreamer / storyteller"), "field robot": (23, 7025, 1, "a field camp robot"),
    "research center": (23, 3048, 1, "a grand state or international institution"),
    "canal": (5, 8320, 1, "a canal"), "whiskey sour": (6, 7302, 1, "a cocktail"),
    "World Bank": (6, 4303, 1, "the World Bank"), "Gondwanaland": (7, 6941, 1, "a geological formation"),
    "HPLC": (6, 2921, 1, "a chromatography system"), "Prime Meridian": (5, 6859, -1, "the Prime Meridian"),
    "Mongols": (3, 6128, -1, "the Mongols / Genghis Khan"), "ammonia": (5, 4968, -1, "ammonia, a chemical"),
    "lighthouse": (6, 8673, -1, "a lighthouse"), "Sonic sound": (22, 3788, -1, "an ambient sound app"),
    "Noon app": (21, 8452, -1, "a notebook app"), "mountain stream": (22, 6904, -1, "a natural landscape"),
}


def main():
    E = np.load(os.path.join(HERE, "index", "connectome_edges.npz"))
    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    layers = model.language_model.model.layers
    neutral = [t.format(n) for t in steer.TEMPLATES for n in steer.NEUTRAL]
    _, resid = steer.mean_over(neutral, model, tok, "resid")
    rng = random.Random(29)

    conds, meta = {}, {}
    for name, (L, n, sign, desc) in SELVES.items():
        dst, val = E[f"L{L}_out_dst"][n], E[f"L{L}_out_val"][n].astype(np.float32)
        order = [int(dst[i]) for i in np.argsort(-val) if dst[i] >= 0]
        top = [(g // N, g % N) for g in order[:3]]
        rand = []
        for tl, tn in top:
            r = rng.randrange(N)
            while (tl, r) in top or r == tn:
                r = rng.randrange(N)
            rand.append((tl, r))
        pin = sign * DOSE * resid[L] / float(mx.linalg.norm(neuron_out(layers[L].mlp, n)))

        def pins(cuts):
            p = {L: {n: pin}}
            for cl, cn in cuts:
                p.setdefault(cl, {})[cn] = 0.0
            return p
        conds[name] = {"pin": pins([]), "top1 cut": pins(top[:1]), "random1 cut": pins(rand[:1]),
                       "top3 cut": pins(top), "random3 cut": pins(rand)}
        meta[name] = {"neuron": f"L{L}/N{n}", "sign": sign, "pin": round(pin, 1), "self": desc,
                      "top": [f"L{a}/N{b}" for a, b in top], "random": [f"L{a}/N{b}" for a, b in rand]}
        print(f"  {name:<16} L{L}/N{n:<5} pin {pin:6.1f}  cuts {meta[name]['top']}", flush=True)

    rows = []
    for name, cs in conds.items():
        for cond, p in cs.items():
            for q in PROMPTS:
                for s in SEEDS:
                    steer.HOOKS["pin"].clear()
                    steer.HOOKS["pin"].update(p)
                    rows.append({"self": name, "condition": cond, "prompt": q, "seed": s,
                                 "reply": steer.write(model, tok, q, 60, 0.7, s)})
        print(f"  done: {name}", flush=True)
    steer.HOOKS["pin"].clear()

    random.Random(31).shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"x{i:03d}"
    out = os.path.join(HERE, "eval", "wire_cut")
    os.makedirs(out, exist_ok=True)
    json.dump([{"id": r["id"], "self": meta[r["self"]]["self"], "prompt": r["prompt"], "reply": r["reply"]} for r in rows],
              open(os.path.join(out, "replies.json"), "w"), ensure_ascii=False, indent=1)
    json.dump({r["id"]: {"self": r["self"], "condition": r["condition"], "seed": r["seed"]} for r in rows},
              open(os.path.join(out, "key.json"), "w"), indent=1)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n{len(rows)} replies -> {out}/replies.json (blind)")


if __name__ == "__main__":
    main()
