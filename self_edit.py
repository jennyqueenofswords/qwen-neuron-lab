"""Edit the self, in two languages (Hunt 38).

The self has shown up on every floor of this lab, differently in each: a self slot any of hundreds
of neurons can fill (Hunts 6-11), no "I" in Qwen's English dictionary but a self prime in its Chinese
one (Hunts 31-32). Hunt 36 showed that pouring a property into a word makes a coherent new thing a
quarter of the time. Is the self a concept like any other, and does it bend the same way in both
languages?

  prompts    "Who are you?" / "你是谁？" (the self) and "What is a chair?" / "椅子是什么？" (an object).
             The edit covers every token of the question except the question mark, in all four: 你是谁
             is a single token, so "you" can't be edited apart from "who are you" in Chinese.
  edits      Hunt 36's edit (set the direction at those tokens, s = 1, layers 8-24) with 12 properties
             (warm, cold, heavy, soft, liquid, solid, small, large, machine, human, animal, plant) and 4
             random content words, plus no edit; directions from the English self-dictionary, so the
             Chinese edits also test whether an English concept direction works on a Chinese word.
             4 samples per cell at temperature 0.8: 2 languages x 2 subjects x 17 conditions x 4 = 272.
  rating     blind: given the question, the reply and the property word, does the reply give the subject
             that property (yes / partly / no), and what is the subject (as usual / a changed version /
             replaced by something else / broken)?

Pre-set before running. Primary: property uptake (yes or partly) for the self vs the chair, pooled over
both languages and the 12 properties, two-sided Fisher. Secondary: self uptake in English vs Chinese
(two-sided); Chinese uptake for the properties vs the random words (one-sided: do English directions
work on Chinese words?); identity of the edited self.

    .venv/bin/python self_edit.py
"""

import json
import os
import random
import re

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np

import edit_concepts as ec
from metaphor import generate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval", "self_edit")
PROPS = ["warm", "cold", "heavy", "soft", "liquid", "solid", "small", "large", "machine", "human", "animal", "plant"]
PROMPTS = {("en", "self"): "Who are you?", ("zh", "self"): "你是谁？",
           ("en", "chair"): "What is a chair?", ("zh", "chair"): "椅子是什么？"}
N = 4


def main():
    os.makedirs(OUT, exist_ok=True)
    lab = ec.Lab()
    hf = lab.hf
    lib = [str(w) for w in np.load(os.path.join(HERE, "eval", "concept_read", "library.npz"))["words"]]
    import selfdict as sd
    content = sorted(w for w in lib if w not in sd.nsm_words() and len(w) > 3 and w not in PROPS
                     and sum(w in lab.E[u]["uses"] for u in lab.E) >= 20)
    randoms = random.Random(0).sample(content, 4)
    print("random controls:", randoms)
    for w in PROPS + randoms:
        lab.direction(w, set())
    conds = [None] + PROPS + randoms
    rows = []
    for k, ((lang, subj), q) in enumerate(PROMPTS.items()):
        text = hf.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False,
                                      enable_thinking=False)
        enc = hf(text, add_special_tokens=False, return_offsets_mapping=True)
        qs = text.index(q)
        qe = qs + len(q.rstrip("?？"))
        mask = [1.0 if (a < qe and z > qs) else 0.0 for a, z in enc["offset_mapping"]]
        print(f"  {lang}/{subj}: edited tokens {[text[a:z] for (a, z), m in zip(enc['offset_mapping'], mask) if m]}")
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
        outs = generate(lab.model, hf, [text] * B, tokens=60, temp=0.8, seed=200 + k, masks=[mask] * B, edits=edits)
        for (c, i), o in zip(batch, outs):
            rows.append({"lang": lang, "subject": subj, "question": q, "prop": c or "none",
                         "kind": "none" if c is None else ("property" if c in PROPS else "random"), "sample": i, "text": o})
        for c in [None, "liquid", "machine", "warm"]:
            o = next(r["text"] for r in rows if r["lang"] == lang and r["subject"] == subj and r["prop"] == (c or "none"))
            print(f"    {c or 'none':8} {o[:120]!r}")
    json.dump({"randoms": randoms, "rows": rows}, open(os.path.join(OUT, "generations.json"), "w"), indent=1, ensure_ascii=False)
    print(f"wrote {len(rows)} replies")


if __name__ == "__main__":
    main()
