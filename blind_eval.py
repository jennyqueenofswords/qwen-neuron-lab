"""Blind test of what pinning a neuron does to the chat model's replies.

Three conditions per prompt and seed:
  normal   no intervention
  target   the neuron under test, pinned
  random   a random neuron in the same layer, pinned so its push on the residual stream
           is the same size (value x length of its output arrow), to separate "this neuron"
           from "any big poke at this layer"

Writes eval/<name>/replies.json (shuffled, condition hidden) and eval/<name>/key.json.
Read the key only after rating.

    .venv/bin/python blind_eval.py 15 3920 25 --name sarcasm
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

PROMPTS = [
    # things people usually dislike
    "What do you think of Mondays?",
    "How do you feel about sitting in traffic?",
    "Tell me about going to the dentist.",
    "What's it like doing your taxes?",
    "Describe a long flight delay.",
    # neutral
    "Explain how photosynthesis works.",
    "What's the best way to learn to cook?",
    "How do I tie a tie?",
    "Recommend a good book for a weekend.",
    "Tell me a little about the history of Rome.",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("layer", type=int)
    p.add_argument("neuron", type=int)
    p.add_argument("value", type=float)
    p.add_argument("--name", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=80)
    args = p.parse_args()

    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    mlp = model.language_model.model.layers[args.layer].mlp

    rng = random.Random(0)
    control = rng.choice([n for n in range(9216) if n != args.neuron])
    t_norm = float(mx.linalg.norm(neuron_out(mlp, args.neuron)))
    c_norm = float(mx.linalg.norm(neuron_out(mlp, control)))
    c_value = args.value * t_norm / c_norm
    print(f"target L{args.layer}/N{args.neuron}={args.value}; control L{args.layer}/N{control}={c_value:.1f} (same push size)")

    conditions = {"normal": {}, "target": {args.neuron: args.value}, "random": {control: c_value}}
    rows = []
    for prompt in PROMPTS:
        for seed in args.seeds:
            for cond, pins in conditions.items():
                steer.HOOKS["pin"].clear()
                if pins:
                    steer.HOOKS["pin"][args.layer] = pins
                reply = steer.write(model, tok, prompt, args.max_tokens, args.temp, seed)
                rows.append({"prompt": prompt, "seed": seed, "condition": cond, "reply": reply})
                print(f"  {cond:<6} seed {seed}  {prompt[:40]}")
    steer.HOOKS["pin"].clear()

    random.Random(42).shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"r{i:02d}"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", args.name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "replies.json"), "w") as f:
        json.dump([{"id": r["id"], "prompt": r["prompt"], "reply": r["reply"]} for r in rows], f, ensure_ascii=False, indent=1)
    with open(os.path.join(out, "key.json"), "w") as f:
        json.dump({r["id"]: {"condition": r["condition"], "seed": r["seed"]} for r in rows}, f, indent=1)
        f.write("\n")
    meta = {"target": f"L{args.layer}/N{args.neuron}", "value": args.value,
            "control": f"L{args.layer}/N{control}", "control_value": c_value, "temp": args.temp}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"\n{len(rows)} replies -> {out}/replies.json (blind); key in key.json")


if __name__ == "__main__":
    main()
