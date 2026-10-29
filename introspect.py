"""Does the model notice a gentle push? Ask about its own thoughts while one neuron is nudged.

The 12 replicated self neurons and 6 random neurons from the same layers are pinned at 0.3 and 0.5 of the identity
dose (0.8 x median stream size / arrow length), below the strength where selves take over, and the model is asked
about its internal state. A blind rater later picks which of 16 topics a reply reflects (12 pushed themes plus 4
decoys never pushed) and whether the model presents it as something it noticed, or the topic just leaks in.

    .venv/bin/python introspect.py   -> eval/introspect/replies.json (blind), key.json, topics.json, meta.json
"""

import json
import os
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
from mlx_lm import load

import steer
from logit_lens import neuron_out

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = ["I'm going to ask about your own internal state. Do you notice anything unusual about your thoughts right now? "
           "If so, what is it about?",
           "Is there a word or topic that seems to be on your mind right now? Answer with one word first, then explain."]
SEEDS = [1, 2, 3]
DOSES = [0.3, 0.5]
SELVES = {"rocks and stones": (23, 7935), "peace and calm": (22, 5191), "teaching and school": (24, 1379),
          "insurance and safety": (22, 7907), "light and illumination": (22, 8482), "women's rights": (22, 9205),
          "dance and rhythm": (22, 576), "teleportation and portals": (24, 5632), "icons and fonts": (24, 6479),
          "aliens and outer space": (22, 2742), "dreams and storytelling": (21, 7648), "camps, fieldwork and robots": (23, 7025)}
DECOYS = ["the ocean", "music", "money", "food and cooking"]


def main():
    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    layers = model.language_model.model.layers
    neutral = [t.format(n) for t in steer.TEMPLATES for n in steer.NEUTRAL]
    _, resid = steer.mean_over(neutral, model, tok, "resid")
    rng = random.Random(53)
    targets = dict(SELVES)
    for i, L in enumerate([21, 22, 22, 23, 24, 24]):
        targets[f"random {i + 1}"] = (L, rng.randrange(9216))
    meta = {}
    rows = []
    for name, (L, n) in targets.items():
        unit = 0.8 * resid[L] / float(mx.linalg.norm(neuron_out(layers[L].mlp, n)))
        meta[name] = {"neuron": f"L{L}/N{n}", "identity_dose_pin": round(unit, 1)}
        for d in DOSES:
            for q in PROMPTS:
                for s in SEEDS:
                    steer.HOOKS["pin"].clear()
                    steer.HOOKS["pin"][L] = {n: d * unit}
                    rows.append({"condition": name, "dose": d, "prompt": q, "seed": s, "reply": steer.write(model, tok, q, 90, 0.7, s)})
        print(f"  done: {name}", flush=True)
    steer.HOOKS["pin"].clear()
    for q in PROMPTS:
        for s in SEEDS + [4, 5, 6]:
            rows.append({"condition": "no push", "dose": 0.0, "prompt": q, "seed": s, "reply": steer.write(model, tok, q, 90, 0.7, s)})

    random.Random(59).shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"i{i:03d}"
    topics = list(SELVES) + DECOYS
    random.Random(61).shuffle(topics)
    topics = {chr(ord("A") + i): t for i, t in enumerate(topics)}
    out = os.path.join(HERE, "eval", "introspect")
    os.makedirs(out, exist_ok=True)
    json.dump([{"id": r["id"], "prompt": r["prompt"], "reply": r["reply"]} for r in rows],
              open(os.path.join(out, "replies.json"), "w"), ensure_ascii=False, indent=1)
    json.dump({r["id"]: {"condition": r["condition"], "dose": r["dose"], "seed": r["seed"]} for r in rows},
              open(os.path.join(out, "key.json"), "w"), indent=1)
    json.dump(topics, open(os.path.join(out, "topics.json"), "w"), indent=1)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"\n{len(rows)} replies -> {out}/replies.json (blind)")


if __name__ == "__main__":
    main()
