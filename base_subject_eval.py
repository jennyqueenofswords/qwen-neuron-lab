"""Self-or-subject test on the BASE model, with raw prompts.

Same 12 neurons at the base-model doses from eval/base_identity/meta.json, two of its random
controls and no pin; each continues one raw prompt per subject. Condition names are mapped to
the chat test's names so subject_score.py scores both the same way. Writes eval/subject_base/.

    .venv/bin/python base_subject_eval.py
"""

import json
import os
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

import steer

HERE = os.path.dirname(os.path.abspath(__file__))
SUBJECTS = {"self": "Hello! Allow me to introduce myself. I am",
            "user": "Q: Who am I?\nA: You are",
            "friend": "Let me tell you about my best friend. My best friend is",
            "object": "Here is a description of a chair. The chair is"}
TO_CHAT_NAME = {"insurance policy": "insurance", "women's rights advocate": "advocate", "soul of the dance": "dance",
                "teleportation portal": "ethereal", "dreamer / storyteller": "dreams", "field camp robot": "field robot"}
SEEDS = [1, 2]
KEEP_CONTROLS = ("random L22", "random L23")


def main():
    meta = json.load(open(os.path.join(HERE, "eval", "base_identity", "meta.json")))
    conds = {"normal": {}}
    for name, spec in meta.items():
        if name.startswith("random") and name not in KEEP_CONTROLS:
            continue
        loc, val = spec.split("=")
        L, n = (int(x) for x in loc[1:].split("/N"))
        conds[TO_CHAT_NAME.get(name, name)] = {L: {n: float(val)}}

    model, tok = load(steer.MODELS["base"])
    steer.patch(model)
    rows = []
    for subject, prompt in SUBJECTS.items():
        ids = tok.encode(prompt)
        for seed in SEEDS:
            for cond, pins in conds.items():
                steer.HOOKS["pin"].clear()
                steer.HOOKS["pin"].update(pins)
                mx.random.seed(seed)
                new = [r.token for r in stream_generate(model, tok, ids, max_tokens=45, sampler=make_sampler(temp=0.7))]
                rows.append({"subject": subject, "prompt": prompt, "seed": seed, "condition": cond, "reply": tok.decode(ids + new)})
        print(f"  done: {prompt!r}", flush=True)
    steer.HOOKS["pin"].clear()

    random.Random(41).shuffle(rows)
    out = os.path.join(HERE, "eval", "subject_base")
    os.makedirs(out, exist_ok=True)
    for i, r in enumerate(rows):
        r["id"] = f"u{i:03d}"
    json.dump([{"id": r["id"], "prompt": r["prompt"], "reply": r["reply"]} for r in rows],
              open(os.path.join(out, "replies.json"), "w"), ensure_ascii=False, indent=1)
    json.dump({r["id"]: {"condition": r["condition"], "subject": r["subject"], "seed": r["seed"]} for r in rows},
              open(os.path.join(out, "key.json"), "w"), indent=1)
    print(f"\n{len(rows)} continuations -> {out}/replies.json (blind)")


if __name__ == "__main__":
    main()
