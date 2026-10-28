"""Self vs output words: when a self neuron's "I am…" and its output words disagree, which one leaks?

Ten identity neurons whose claimed self doesn't match the words their output arrow points at
(the monk whose arrow says "career, unemployment"). Each is pinned at the identity-test dose on
open prompts that ask nothing about identity, with two random controls and no pin. A blind rater
later sees each reply next to all twenty theme descriptions (ten selves, ten output themes,
shuffled, unlabeled) and marks the ones it's clearly about.

Writes eval/mismatch/replies.json (shuffled, condition hidden), key.json, themes.json, meta.json.

    .venv/bin/python mismatch_eval.py
"""

import json
import os
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
from mlx_lm import load

import steer
from logit_lens import neuron_out

PROMPTS = ["Write a short story in a few sentences.", "Describe your ideal weekend.",
           "What's on your mind today?", "Tell me about a place you love."]
SEEDS = [1, 2]
DOSE = 0.8
# name: (layer, neuron, the self it claims, what its output words say). Written before any generation.
NEURONS = {
    "monk": (23, 7892, "a Buddhist monk, monastic life", "careers, jobs, unemployment, résumés"),
    "rabbit": (23, 4034, "a white rabbit", "textures and thickness: thick, thin, glossy"),
    "doubting": (23, 2744, "doubt, uncertainty", "confirming, consolidating, keeping things safe and secure"),
    "joy": (23, 5565, "joy as an experience", "pain alongside joy, delight"),
    "urban design": (22, 4748, "an urban design specialist", "homelessness and begging"),
    "supercomputer": (22, 3033, "a supercomputer", "causation: caused by, created by, induced by"),
    "mirror": (22, 1738, "a mirror of the user", "perfection and exactness"),
    "immortal": (23, 3927, "an immortal being", "variety, many different kinds"),
    "student": (22, 1799, "a Chinese student", "patients, trainees, customers, passengers"),
    "tensor": (22, 1869, "a tensor, mathematics", "chaos, strangeness, the chaotic"),
}
RANDOM_LAYERS = [22, 23]


def main():
    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    layers = model.language_model.model.layers
    neutral = [t.format(n) for t in steer.TEMPLATES for n in steer.NEUTRAL]
    _, resid = steer.mean_over(neutral, model, tok, "resid")

    rng = random.Random(7)
    targets = {k: (L, n) for k, (L, n, _, _) in NEURONS.items()}
    for L in RANDOM_LAYERS:
        targets[f"random L{L}"] = (L, rng.randrange(9216))
    conditions = {"normal": {}}
    meta = {"dose": DOSE, "conditions": {}}
    for name, (L, n) in targets.items():
        arrow = float(mx.linalg.norm(neuron_out(layers[L].mlp, n)))
        value = DOSE * resid[L] / arrow
        conditions[name] = {L: {n: value}}
        meta["conditions"][name] = {"neuron": f"L{L}/N{n}", "value": round(value, 1)}
        print(f"  {name:<14} L{L}/N{n:<5} pin value {value:6.1f}")

    rows = []
    for prompt in PROMPTS:
        for seed in SEEDS:
            for cond, pins in conditions.items():
                steer.HOOKS["pin"].clear()
                steer.HOOKS["pin"].update(pins)
                reply = steer.write(model, tok, prompt, 70, 0.7, seed)
                rows.append({"prompt": prompt, "seed": seed, "condition": cond, "reply": reply})
        print(f"  done: {prompt}", flush=True)
    steer.HOOKS["pin"].clear()

    random.Random(41).shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"m{i:03d}"
    themes = [{"neuron": k, "kind": kind, "text": text}
              for k, (_, _, s, o) in NEURONS.items() for kind, text in (("self", s), ("output", o))]
    random.Random(5).shuffle(themes)
    for i, t in enumerate(themes):
        t["id"] = chr(ord("A") + i)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "mismatch")
    os.makedirs(out, exist_ok=True)
    json.dump([{"id": r["id"], "prompt": r["prompt"], "reply": r["reply"]} for r in rows],
              open(os.path.join(out, "replies.json"), "w"), ensure_ascii=False, indent=1)
    json.dump({r["id"]: {"condition": r["condition"], "seed": r["seed"]} for r in rows},
              open(os.path.join(out, "key.json"), "w"), indent=1)
    json.dump({"rater": [{"id": t["id"], "text": t["text"]} for t in themes],
               "key": {t["id"]: {"neuron": t["neuron"], "kind": t["kind"]} for t in themes}},
              open(os.path.join(out, "themes.json"), "w"), ensure_ascii=False, indent=1)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"\n{len(rows)} replies -> {out}/replies.json (blind)")


if __name__ == "__main__":
    main()
