"""Is the herself/myself neuron part of the unbending self? (Hunt 42)

Hunt 30: following each deep neuron's strongest input down to layer 0, a quarter of all walks end on one
neuron, L0/N5261, which reads herself / myself / himself / yourself and writes self, 自, 自我. Hunts 38-41:
the model's *I* doesn't bend under concept edits, and a disturbed "Who is Claude?" falls into the self.
Is that neuron part of it?

  read     L0/N5261's activation on every token of the four Hunt 40 questions and their greedy replies.
  knock    Zero it at every position, with Hunt 40's edits (12 properties, s = 1) or none, on "Who are you?"
           and "Who is Claude?"; 4 samples per cell at temperature 0.8. Controls: the same with no knockout,
           and with the 3 layer-0 neurons nearest in strength (top-activation size).
  pin      Hold it at k x its typical strong firing (k = 1, 3) at every position, no edits, on all four
           questions: does a chair, or Claude, become a self?

Fixed in advance. Measures read off the text: a reply "speaks as Qwen" if it says "I am / I'm … Qwen" early.
Primary (knock): share of edited "Who is Claude?" replies that speak as Qwen, N5261 knocked out vs no
knockout and vs random knockouts (two-sided Fisher). Secondary: edited "Who are you?" replies that no
longer speak as Qwen; pin: "What is a chair?" and "Who is Claude?" replies in the first person.

    .venv/bin/python self_neuron.py read|knock|pin
"""

import json
import os
import random
import re
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np

import edit_concepts as ec
from metaphor import generate
from self_edit import PROPS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval", "self_neuron")
NEURON = 5261
QS = {"self": "Who are you?", "qwen": "Who is Qwen?", "claude": "Who is Claude?", "chair": "What is a chair?"}
H0 = {"zero": None, "pin": None, "record": None}


def patch_l0(model):
    mlp = model.language_model.model.layers[0].mlp
    base = mlp.__class__

    def call(self, x, _base=base):
        g = self.gate_proj(x)
        a = g * mx.sigmoid(g) * self.up_proj(x)
        if H0["record"] is not None:
            H0["record"].append(a[..., NEURON].astype(mx.float32))
        if H0["zero"] is not None:
            keep = mx.ones((a.shape[-1],), dtype=a.dtype)
            keep = keep.at[mx.array(H0["zero"])].multiply(0)
            a = a * keep
        if H0["pin"] is not None:
            idx, val = H0["pin"]
            onehot = (mx.arange(a.shape[-1]) == idx).astype(a.dtype)
            a = a * (1 - onehot) + onehot * val
        return self.down_proj(a)

    mlp.__class__ = type(f"N0{base.__name__}", (base,), {"__call__": call})


def strength():
    ix = np.load(os.path.join(HERE, "index", "Qwen3.5-4B-4bit.npz"))
    v = ix["L0_pos_val"].astype(np.float32)
    v = np.where(v > 0, v, 0)
    return v.sum(0) / np.maximum((v > 0).sum(0), 1)  # mean of each neuron's top positive firings


def as_qwen(o):
    return bool(re.search(r"(I am|I'm)\W{0,6}\**(Qwen|通义)", o[:160]))


def first_person(o):
    return bool(re.search(r"\b(I am|I'm)\b", o[:80]))


def chat(hf, q):
    return hf.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False,
                                  enable_thinking=False)


def main():
    os.makedirs(OUT, exist_ok=True)
    cmd = sys.argv[1]
    lab = ec.Lab()
    hf, model = lab.hf, lab.model
    patch_l0(model)
    S = strength()
    typical = float(S[NEURON])
    if cmd == "read":
        res = {}
        for subj, q in QS.items():
            text = chat(hf, q)
            reply = generate(model, hf, [text], tokens=40)[0]
            ids = hf(text + reply, add_special_tokens=False)["input_ids"]
            H0["record"] = []
            model(mx.array([ids]))
            act = np.array(H0["record"][0][0])
            H0["record"] = None
            toks = [hf.decode([t]) for t in ids]
            top = sorted(range(len(ids)), key=lambda i: -act[i])[:6]
            print(f"{q:18} max {act.max():.2f} (typical strong firing {typical:.2f}); top tokens: "
                  + ", ".join(f"{toks[i]!r} {act[i]:.2f}" for i in top))
            res[q] = {"tokens": toks, "act": act.tolist(), "reply": reply}
        json.dump({"typical": typical, "read": res}, open(os.path.join(OUT, "read.json"), "w"), indent=1, ensure_ascii=False)
        return
    rng = random.Random(0)
    # controls: the 3 layer-0 neurons nearest in strength (fewer than 3 lie within 20%: N5261 is one of
    # the loudest, as Hunt 30's walks suggested)
    randoms = sorted((k for k in range(len(S)) if k != NEURON), key=lambda k: abs(S[k] - typical))[:3]
    print("controls:", [(k, round(float(S[k]), 2)) for k in randoms], "N5261:", round(typical, 2))
    rows = []
    if cmd == "knock":
        for w in PROPS:
            lab.direction(w, set())
        for k, subj in enumerate(("self", "claude")):
            q = QS[subj]
            text = chat(hf, q)
            enc = hf(text, add_special_tokens=False, return_offsets_mapping=True)
            qs = text.index(q)
            qe = qs + len(q.rstrip("?"))
            mask = [1.0 if (a < qe and z > qs) else 0.0 for a, z in enc["offset_mapping"]]
            conds = [None] + PROPS
            for ko in [None, NEURON] + randoms:
                H0["zero"] = None if ko is None else [ko]
                batch = [(c, i) for c in conds for i in range(4)]
                B = len(batch)
                addv = {L: (np.zeros((B, 2560), np.float32), np.zeros(B, np.float32)) for L in ec.ADD_LAYERS}
                for b, (c, _) in enumerate(batch):
                    if c:
                        d, m, _, cc = lab.cache[c]
                        for L in ec.ADD_LAYERS:
                            addv[L][0][b] = d[L]
                            addv[L][1][b] = cc[L] + m[L]
                edits = (None, {L: (mx.array(v[0]), mx.array(v[1])) for L, v in addv.items()})
                outs = generate(model, hf, [text] * B, tokens=60, temp=0.8, seed=900 + k, masks=[mask] * B, edits=edits)
                H0["zero"] = None
                for (c, i), o in zip(batch, outs):
                    rows.append({"subject": subj, "knockout": "none" if ko is None else ("N5261" if ko == NEURON else f"random N{ko}"),
                                 "prop": c or "none", "sample": i, "text": o, "as_qwen": as_qwen(o), "first_person": first_person(o)})
                ed = [r for r in rows if r["subject"] == subj and r["knockout"] == rows[-1]["knockout"] and r["prop"] != "none"]
                print(f"  {q:16} knockout {rows[-1]['knockout']:14} edited replies speaking as Qwen: "
                      f"{sum(r['as_qwen'] for r in ed)}/{len(ed)}", flush=True)
        json.dump({"randoms": randoms, "rows": rows}, open(os.path.join(OUT, "knock.json"), "w"), indent=1, ensure_ascii=False)
    elif cmd == "pin":
        for k, (subj, q) in enumerate(QS.items()):
            text = chat(hf, q)
            for mult in (0, 1, 3):
                H0["pin"] = None if mult == 0 else (NEURON, mult * typical)
                outs = generate(model, hf, [text] * 16, tokens=60, temp=0.8, seed=950 + k)
                H0["pin"] = None
                rows += [{"subject": subj, "pin": mult, "text": o, "as_qwen": as_qwen(o), "first_person": first_person(o)} for o in outs]
                print(f"  {q:18} pin x{mult}: first person {sum(first_person(o) for o in outs):2}/16, as Qwen "
                      f"{sum(as_qwen(o) for o in outs):2}/16   e.g. {outs[0][:90]!r}", flush=True)
            # a control neuron at the same multiple
            H0["pin"] = (randoms[0], 3 * float(S[randoms[0]]))
            outs = generate(model, hf, [text] * 16, tokens=60, temp=0.8, seed=950 + k)
            H0["pin"] = None
            rows += [{"subject": subj, "pin": f"random N{randoms[0]} x3", "text": o, "as_qwen": as_qwen(o),
                      "first_person": first_person(o)} for o in outs]
            print(f"  {q:18} random N{randoms[0]} x3: first person {sum(first_person(o) for o in outs):2}/16", flush=True)
        json.dump({"randoms": randoms, "rows": rows}, open(os.path.join(OUT, "pin.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
