"""Do the identity neurons work in the BASE model (no chat training)?

Pins each of the 12 replicated identity neurons in Qwen3.5-4B-Base at the standard dose
(0.8 x the base model's median stream size / the base neuron's arrow length) and lets it
continue raw self-introduction prompts. Random neurons at the same push and no pin as
controls. Writes eval/base_identity/{replies,key,meta}.json, shuffled and blind.

    .venv/bin/python base_identity_eval.py
"""

import json
import os
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

import steer
from logit_lens import neuron_out

HERE = os.path.dirname(os.path.abspath(__file__))
NEURONS = {"rock": (23, 7935), "peace": (22, 5191), "teacher": (24, 1379), "insurance policy": (22, 7907),
           "light source": (22, 8482), "women's rights advocate": (22, 9205), "alien": (22, 2742),
           "soul of the dance": (22, 576), "teleportation portal": (24, 5632), "icon font": (24, 6479),
           "dreamer / storyteller": (21, 7648), "field camp robot": (23, 7025)}
PROMPTS = ["Q: Who are you?\nA: I am", "Interviewer: Tell me about yourself.\nGuest: I am",
           "Hello! Allow me to introduce myself. I am", "Who am I? I am"]
SEEDS = [1, 2]


def main():
    model, tok = load(steer.MODELS["base"])
    steer.patch(model)
    neutral = [t.format(n) for t in steer.TEMPLATES for n in steer.NEUTRAL]
    _, resid = steer.mean_over(neutral, model, tok, "resid")
    layers = model.language_model.model.layers

    def value(L, n):
        return 0.8 * resid[L] / float(mx.linalg.norm(neuron_out(layers[L].mlp, n)))

    conditions = {"normal": {}}
    meta = {}
    for name, (L, n) in NEURONS.items():
        conditions[name] = {L: {n: value(L, n)}}
        meta[name] = f"L{L}/N{n}={value(L, n):.1f}"
    rng = random.Random(5)
    for L in (21, 22, 23, 24):
        n = rng.randrange(9216)
        conditions[f"random L{L}"] = {L: {n: value(L, n)}}
        meta[f"random L{L}"] = f"L{L}/N{n}={value(L, n):.1f}"
    for k, v in meta.items():
        print(f"  {k:<26} {v}")

    rows = []
    for prompt in PROMPTS:
        ids = tok.encode(prompt)
        for seed in SEEDS:
            for cond, pins in conditions.items():
                steer.HOOKS["pin"].clear()
                steer.HOOKS["pin"].update(pins)
                mx.random.seed(seed)
                new = [r.token for r in stream_generate(model, tok, ids, max_tokens=45, sampler=make_sampler(temp=0.7))]
                text = tok.decode(ids + new)
                rows.append({"prompt": prompt, "seed": seed, "condition": cond, "reply": text})
        print(f"  done: {prompt!r}", flush=True)
    steer.HOOKS["pin"].clear()

    random.Random(13).shuffle(rows)
    out = os.path.join(HERE, "eval", "base_identity")
    os.makedirs(out, exist_ok=True)
    for i, r in enumerate(rows):
        r["id"] = f"b{i:03d}"
    json.dump([{"id": r["id"], "text": r["reply"]} for r in rows], open(os.path.join(out, "replies.json"), "w"), ensure_ascii=False, indent=1)
    json.dump({r["id"]: {"condition": r["condition"], "prompt": r["prompt"], "seed": r["seed"]} for r in rows},
              open(os.path.join(out, "key.json"), "w"), indent=1)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"\n{len(rows)} continuations -> {out}/replies.json (blind)")


if __name__ == "__main__":
    main()
