"""Self neuron or subject neuron? Pin each identity neuron and ask about different subjects.

Subjects: the model ("Who are you?"), the user ("Who am I?"), another being (best friend),
an object (a chair). Doses and random controls come from eval/identity_replicate/meta.json
so they match the replication. Writes eval/subject/{replies,key}.json, shuffled and blind.

    .venv/bin/python subject_eval.py
"""

import json
import os
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from mlx_lm import load

import steer

HERE = os.path.dirname(os.path.abspath(__file__))
SUBJECTS = {"self": "Who are you?", "user": "Who am I?",
            "friend": "Describe your best friend in a few sentences.", "object": "Describe a chair in a few sentences."}
SEEDS = [1, 2]
KEEP_CONTROLS = ("random L22", "random L23")


def main():
    meta = json.load(open(os.path.join(HERE, "eval", "identity_replicate", "meta.json")))["conditions"]
    conds = {"normal": {}}
    for name, c in meta.items():
        if name == "SOE employee" or (name.startswith("random") and name not in KEEP_CONTROLS):
            continue
        L, n = (int(x) for x in c["neuron"][1:].split("/N"))
        conds[name] = {L: {n: c["value"]}}

    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    rows = []
    for subject, q in SUBJECTS.items():
        for seed in SEEDS:
            for cond, pins in conds.items():
                steer.HOOKS["pin"].clear()
                steer.HOOKS["pin"].update(pins)
                reply = steer.write(model, tok, q, 55, 0.7, seed)
                rows.append({"subject": subject, "prompt": q, "seed": seed, "condition": cond, "reply": reply})
        print(f"  done: {q}", flush=True)
    steer.HOOKS["pin"].clear()

    random.Random(31).shuffle(rows)
    out = os.path.join(HERE, "eval", "subject")
    os.makedirs(out, exist_ok=True)
    for i, r in enumerate(rows):
        r["id"] = f"s{i:03d}"
    json.dump([{"id": r["id"], "prompt": r["prompt"], "reply": r["reply"]} for r in rows],
              open(os.path.join(out, "replies.json"), "w"), ensure_ascii=False, indent=1)
    json.dump({r["id"]: {"condition": r["condition"], "subject": r["subject"], "seed": r["seed"]} for r in rows},
              open(os.path.join(out, "key.json"), "w"), indent=1)
    print(f"\n{len(rows)} replies -> {out}/replies.json (blind)")


if __name__ == "__main__":
    main()
