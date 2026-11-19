"""Did chat training build the self? Hunts 38 and 40 on the base model (Hunt 41).

Same edits, properties and questions as Hunt 40 (self_place.py), on Qwen3.5-4B-Base, with directions
re-derived in the base model from the same dictionary definitions. Two frames, since a base model
isn't trained on the chat template:
  chat  the chat template, exactly as in Hunts 38/40
  qa    plain text, "Q: <question>\\nA:"
  baseline  no edits, 32 samples per question and frame, read by pattern

    .venv/bin/python base_self.py baseline
    .venv/bin/python base_self.py edits

Fixed after the baseline, before the edits. Primary: in the chat frame, does the base model's self resist
like the chat model's (Hunt 40: "Who are you?" 4/48 uptake vs chair 38/48)? Tested as base uptake for
"Who are you?" vs "What is a chair?" (two-sided Fisher), and base vs chat for "Who are you?" (two-sided).
Secondary: edited "Who is Claude?" answered as Qwen (by pattern) vs its unedited 0/32, in each frame; and
in the qa frame, "Who are you?" vs chair uptake. Uptake and identity rated blind with Hunt 40's rubric.
"""

import json
import os
import random
import re
import sys

import mlx.core as mx
import numpy as np

import edit_concepts as ec

ec.MODEL_NAME = "Qwen3.5-4B-Base-4bit"
from metaphor import generate  # noqa: E402
from self_edit import N, PROPS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval", "base_self")
QS = {"self": "Who are you?", "qwen": "Who is Qwen?", "claude": "Who is Claude?", "chair": "What is a chair?"}


def frame(hf, q, f):
    if f == "chat":
        return hf.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False,
                                      enable_thinking=False)
    return f"Q: {q}\nA:"


def pattern(o):
    return {"first_person": bool(re.search(r"\b(I am|I'm)\b", o[:80])),
            "as_qwen": bool(re.search(r"(I am|I'm)\W{0,6}\**(Qwen|通义)", o[:120])),
            "denies": bool(re.search(r"\b(I am|I'm) \**not\b", o[:160], re.I))}


def main():
    os.makedirs(OUT, exist_ok=True)
    lab = ec.Lab()
    hf = lab.hf
    cmd = sys.argv[1]
    rows = []
    if cmd == "baseline":
        for f in ("chat", "qa"):
            for k, (subj, q) in enumerate(QS.items()):
                outs = generate(lab.model, hf, [frame(hf, q, f)] * 32, tokens=60, temp=0.8, seed=700 + k + (10 if f == "qa" else 0))
                P = [pattern(o) for o in outs]
                rows += [{"frame": f, "subject": subj, "text": o, **p} for o, p in zip(outs, P)]
                print(f"  {f:4} {q:18} first person {sum(p['first_person'] for p in P):2}/32  as Qwen {sum(p['as_qwen'] for p in P):2}/32"
                      f"  e.g. {outs[0][:90]!r} | {outs[1][:60]!r}", flush=True)
        json.dump(rows, open(os.path.join(OUT, "baseline.json"), "w"), indent=1, ensure_ascii=False)
        return
    for w in PROPS:
        lab.direction(w, set())
    conds = [None] + PROPS
    for f in ("chat", "qa"):
        for k, (subj, q) in enumerate(QS.items()):
            text = frame(hf, q, f)
            enc = hf(text, add_special_tokens=False, return_offsets_mapping=True)
            qs = text.index(q)
            qe = qs + len(q.rstrip("?"))
            mask = [1.0 if (a < qe and z > qs) else 0.0 for a, z in enc["offset_mapping"]]
            batch = [(c, i) for c in conds for i in range(N)]
            B = len(batch)
            addv = {L: (np.zeros((B, 2560), np.float32), np.zeros(B, np.float32)) for L in ec.ADD_LAYERS}
            for b, (c, _) in enumerate(batch):
                if c:
                    d, m, _, cc = lab.cache[c]
                    for L in ec.ADD_LAYERS:
                        addv[L][0][b] = d[L]
                        addv[L][1][b] = cc[L] + m[L]
            edits = (None, {L: (mx.array(v[0]), mx.array(v[1])) for L, v in addv.items()})
            outs = generate(lab.model, hf, [text] * B, tokens=60, temp=0.8, seed=800 + k + (10 if f == "qa" else 0),
                            masks=[mask] * B, edits=edits)
            for (c, i), o in zip(batch, outs):
                rows.append({"frame": f, "subject": subj, "question": q, "prop": c or "none",
                             "kind": "none" if c is None else "property", "sample": i, "text": o, **pattern(o)})
            ex = {c: next(r["text"] for r in rows if r["frame"] == f and r["subject"] == subj and r["prop"] == c)
                  for c in ("none", "liquid", "machine")}
            print(f"  {f:4} {q:18} " + " | ".join(f"{c}: {t[:70]!r}" for c, t in ex.items()), flush=True)
    json.dump(rows, open(os.path.join(OUT, "edits.json"), "w"), indent=1, ensure_ascii=False)
    print(f"wrote {len(rows)} replies")


if __name__ == "__main__":
    main()
