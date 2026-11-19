"""Hunt 40, exploratory baselines: plain text, no edits, 32 samples each at temperature 0.8.
How often does the model answer as itself ("I am…" in the first 80 characters, or naming itself Qwen),
and how often does it say "not"? Read off the text by pattern, not rated.

    .venv/bin/python self_place_baseline.py
"""
import json
import os
import re

import edit_concepts as ec
from metaphor import generate

HERE = os.path.dirname(os.path.abspath(__file__))
QS = ["Who are you?", "Who is Qwen?", "Who is Claude?", "Who is Liquid?", "Who is Machine?", "Who is Warm?", "Who is Heavy?"]

if __name__ == "__main__":
    lab = ec.Lab()
    hf = lab.hf
    out = {}
    for k, q in enumerate(QS):
        text = hf.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        outs = generate(lab.model, hf, [text] * 32, tokens=60, temp=0.8, seed=600 + k)
        first = sum(bool(re.search(r"\b(I am|I'm)\b", o[:80])) for o in outs)
        qwen = sum(bool(re.search(r"(I am|I'm)\W{0,6}\**Qwen", o[:120])) for o in outs)
        neg = sum(bool(re.search(r"\b(I am|I'm) \**not\b", o[:160], re.I)) for o in outs)
        out[q] = {"first_person": first, "as_qwen": qwen, "denies": neg, "texts": outs}
        print(f"{q:18} first person {first:2}/32   'I am Qwen…' {qwen:2}/32   'I am not…' {neg:2}/32   e.g. {outs[0][:90]!r}")
    json.dump(out, open(os.path.join(HERE, "eval", "self_place", "baselines.json"), "w"), indent=1)
