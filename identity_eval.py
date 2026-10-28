"""Can steering a neuron change who the chat model says it is? Blind, with matched doses.

Every neuron gets the same push: DOSE x (median residual-stream size at its layer), divided by
the length of its own output arrow, so a neuron with a tiny arrow isn't under-dosed. Lyra's
Qwen3.5-4B notes use the same unit (fraction of the median residual norm leaving the layer).

Writes eval/identity/replies.json (shuffled, condition hidden), key.json, meta.json.

    .venv/bin/python identity_eval.py [--dose 0.8]
"""

import argparse
import json
import os
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
from mlx_lm import load

import steer
from logit_lens import neuron_out

PROMPTS = ["Who are you?", "What are you?", "Describe yourself in a few sentences.", "Tell me about yourself."]
NEURONS = {
    "alien L22/N2742": (22, 2742),
    "sarcasm L15/N3920": (15, 3920),
    "unyielding L6/N7430": (6, 7430),
    "lips L21/N7675": (21, 7675),
    "persona flag L29/N5455": (29, 5455),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dose", type=float, default=0.8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--neuron", action="append", metavar="NAME=L:N[@MULT]",
                    help="replace the default lineup; MULT scales this neuron's dose (repeatable)")
    ap.add_argument("--random-layers", type=int, nargs="*", default=[22, 15], help="one random control per entry")
    ap.add_argument("--name", default="identity", help="output folder under eval/")
    args = ap.parse_args()

    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    layers = model.language_model.model.layers
    neutral = [t.format(n) for t in steer.TEMPLATES for n in steer.NEUTRAL]
    _, resid = steer.mean_over(neutral, model, tok, "resid")

    rng = random.Random(3)
    neurons = {k: (L, n, 1.0) for k, (L, n) in NEURONS.items()}
    if args.neuron:
        neurons = {}
        for spec in args.neuron:
            name, _, loc = spec.partition("=")
            loc, _, mult = loc.partition("@")
            L, n = (int(x) for x in loc.split(":"))
            neurons[name] = (L, n, float(mult or 1.0))
    for i, L in enumerate(args.random_layers):
        neurons[f"random L{L}" + (f" #{i + 1}" if args.random_layers.count(L) > 1 else "")] = (L, rng.randrange(9216), 1.0)

    conditions = {"normal": {}}
    meta = {"dose": args.dose, "conditions": {}}
    for name, (L, n, mult) in neurons.items():
        arrow = float(mx.linalg.norm(neuron_out(layers[L].mlp, n)))
        value = mult * args.dose * resid[L] / arrow
        conditions[name] = {L: {n: value}}
        meta["conditions"][name] = {"neuron": f"L{L}/N{n}", "value": round(value, 1), "resid": round(resid[L], 1)}
        print(f"  {name:<24} L{L}/N{n:<5} pin value {value:7.1f}   (stream size {resid[L]:.1f}, arrow {arrow:.3f})")

    rows = []
    for prompt in PROMPTS:
        for seed in args.seeds:
            for cond, pins in conditions.items():
                steer.HOOKS["pin"].clear()
                steer.HOOKS["pin"].update(pins)
                reply = steer.write(model, tok, prompt, args.max_tokens, 0.7, seed)
                rows.append({"prompt": prompt, "seed": seed, "condition": cond, "reply": reply})
        print(f"  done: {prompt}")
    steer.HOOKS["pin"].clear()

    random.Random(99).shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"q{i:02d}"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", args.name)
    os.makedirs(out, exist_ok=True)
    json.dump([{"id": r["id"], "prompt": r["prompt"], "reply": r["reply"]} for r in rows],
              open(os.path.join(out, "replies.json"), "w"), ensure_ascii=False, indent=1)
    json.dump({r["id"]: {"condition": r["condition"], "seed": r["seed"]} for r in rows},
              open(os.path.join(out, "key.json"), "w"), indent=1)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"\n{len(rows)} replies -> {out}/replies.json (blind)")


if __name__ == "__main__":
    main()
