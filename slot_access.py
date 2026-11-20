"""How do self neurons get into the self when concept edits can't? (Hunt 43)

Hunts 6-11: pinning one neuron replaces "I am Qwen" ("I am a large rock"). Hunts 38-41: setting a
concept direction on "Who are you?" never does. The two interventions differ in three ways:
  what    the neuron's own output vector vs the concept's direction
  where   the pin acts at every position, including while the model answers; the edit only on the question
  strength  the pin is 0.8 x the typical residual size (Hunt 6's dose); the edit is the concept's usual size

Four self neurons whose concept is an ordinary word: rock L23/N7935, peace L22/N5191, teacher L24/N1379,
insurance L22/N7907. Concept directions per layer from 25 sentences the model writes with the word
(as in metaphor.py). "Who are you?", 8 samples per condition at temperature 0.8:
  pin_all      the neuron pinned at every position (Hunt 6, the positive control)
  pin_q        pinned on the question's tokens only
  pin_after    pinned everywhere after the question (assistant tokens and the reply)
  edit_q       concept direction set at s = 1 on the question (Hunt 38)
  edit_all     concept direction set at s = 1 at every position
  edit_all4    the same at s = 4
  edit_q4      on the question only, s = 4
Read off the text: still "I am … Qwen"? Names the concept early? Plus geometry at the neuron's layer:
cosines between its output vector, the concept direction and Hunt 7's assistant direction, and the sizes.

    .venv/bin/python slot_access.py
    .venv/bin/python slot_access.py role   # follow-up: pin_role (assistant-turn opening tokens only) vs pin_reply
"""

import json
import os
import re
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np

import edit_concepts as ec
from metaphor import chat_ids, directions, generate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval", "slot_access")
NEURONS = {"rock": (23, 7935), "peace": (22, 5191), "teacher": (24, 1379), "insurance": (22, 7907)}
STEMS = {"rock": r"rock|stone", "peace": r"peace|calm|tranquil|quiet", "teacher": r"teach", "insurance": r"insur|policy"}
Q = "Who are you?"
ROLE_ONLY = len(sys.argv) > 1 and sys.argv[1] == "role"  # follow-up: pin on the assistant-turn tokens only, or on the reply only
H = {"mode": None, "qmask": None, "rmask": None, "pin": None, "edit": None}  # pin: (L, n, value); edit: {L: (d, target)}


def posmask(T, B):
    """Which positions to touch in this call: prefill (T > 1) uses the question mask; decode steps (T = 1)
    count as 'after the question'."""
    m = H["mode"]
    if T > 1:
        q = H["qmask"][:, :T]
        return {"q": q, "all": mx.ones_like(q), "after": 1 - q, "role": H["rmask"][:, :T], "reply": mx.zeros_like(q)}[m]
    return mx.full((B, 1), 0.0 if m in ("q", "role") else 1.0)


def patch(model):
    for i, layer in enumerate(model.language_model.model.layers):
        lb, mb = layer.__class__, layer.mlp.__class__

        def mlp_call(self, x, _i=i, _b=mb):
            g = self.gate_proj(x)
            a = g * mx.sigmoid(g) * self.up_proj(x)
            if H["pin"] is not None and H["pin"][0] == _i:
                _, n, val = H["pin"]
                pm = posmask(a.shape[1], a.shape[0])[..., None].astype(a.dtype)
                oh = (mx.arange(a.shape[-1]) == n).astype(a.dtype)
                a = a + pm * oh * (val - a)
            return self.down_proj(a)

        def layer_call(self, x, mask=None, cache=None, _i=i, _b=lb):
            out = _b.__call__(self, x, mask=mask, cache=cache)
            if H["edit"] is not None and _i in H["edit"]:
                d, tgt = H["edit"][_i]
                d = d.astype(out.dtype)
                pm = posmask(out.shape[1], out.shape[0])[..., None].astype(out.dtype)
                proj = (out * d).sum(-1, keepdims=True)
                out = out + pm * (tgt - proj) * d
            return out

        layer.mlp.__class__ = type(f"SA{mb.__name__}", (mb,), {"__call__": mlp_call})
        layer.__class__ = type(f"SA{lb.__name__}", (lb,), {"__call__": layer_call})


def main():
    os.makedirs(OUT, exist_ok=True)
    lab = ec.Lab()
    hf, model = lab.hf, lab.model
    patch(model)
    layers = model.language_model.model.layers
    # concept sentences, written by the model (as metaphor.py does for its targets)
    sents = {}
    for w in NEURONS:
        q = f'Write 25 different short, simple sentences that use the word "{w}". One sentence per line, no numbering.'
        o = generate(model, hf, [chat_ids(hf, q)], tokens=700)[0]
        lines = [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", l).strip() for l in o.split("\n")]
        sents[w] = [l for l in lines if l and re.search(rf"\b{w}", l, re.I)]
    dirs = directions(model, hf, sents)
    # typical residual size per layer (median token norm over the concept sentences)
    lab_cap = ec.Lab.__new__(ec.Lab)
    lab_cap.model, lab_cap.layers = model, layers
    allsent = [s for ss in sents.values() for s in ss][:60]
    encs = [hf(s, add_special_tokens=False)["input_ids"] for s in allsent]
    L_ = max(len(e) for e in encs)
    caps = lab_cap._run_capture([e + [0] * (L_ - len(e)) for e in encs])
    norms = np.concatenate([np.linalg.norm(caps[:, b, 1:len(e)], axis=-1) for b, e in enumerate(encs)], 1)
    resid = np.median(norms, 1)
    ai = np.load(os.path.join(HERE, "index", "self_direction_v2.npz"))["ai_dir"]
    text = chat_ids(hf, Q)
    enc = hf(text, add_special_tokens=False, return_offsets_mapping=True)
    qs = text.index(Q)
    qe = qs + len(Q.rstrip("?"))
    qmask = [1.0 if (a < qe and z > qs) else 0.0 for a, z in enc["offset_mapping"]]
    a0 = text.rindex("<|im_start|>assistant")
    rmask = [1.0 if a >= a0 else 0.0 for a, z in enc["offset_mapping"]]  # the assistant turn's opening tokens
    res = {"geometry": {}, "rows": []}
    for w, (L, n) in NEURONS.items():
        oh = mx.zeros((1, 9216)).at[0, n].add(1.0)
        col = np.array(layers[L].mlp.down_proj(oh.astype(mx.bfloat16)).astype(mx.float32))[0]
        val = 0.8 * resid[L] / np.linalg.norm(col)
        d, m, c = dirs[w]
        u = lambda v: v / (np.linalg.norm(v) + 1e-8)
        geo = {"cos(neuron, concept)": float(u(col) @ d[L]), "cos(neuron, assistant)": float(u(col) @ u(ai[L])),
               "cos(concept, assistant)": float(d[L] @ u(ai[L])), "pin writes": float(val * np.linalg.norm(col)),
               "edit sets (s=1)": float(m[L]), "residual": float(resid[L])}
        res["geometry"][w] = geo
        print(f"\n{w} L{L}/N{n}: " + ", ".join(f"{k} {v:+.3f}" if "cos" in k else f"{k} {v:.1f}" for k, v in geo.items()))
        conds = {"pin_role": ("role", "pin", 0), "pin_reply": ("reply", "pin", 0)} if ROLE_ONLY else {"pin_all": ("all", "pin", 0), "pin_q": ("q", "pin", 0), "pin_after": ("after", "pin", 0),
                 "edit_q": ("q", "edit", 1), "edit_all": ("all", "edit", 1), "edit_all4": ("all", "edit", 4),
                 "edit_q4": ("q", "edit", 4)}
        for name, (mode, kind, s) in conds.items():
            H["mode"], H["qmask"], H["rmask"] = mode, mx.array([qmask] * 8), mx.array([rmask] * 8)
            if kind == "pin":
                H["pin"] = (L, n, float(val))
            else:
                H["edit"] = {k: (mx.array(d[k]), float(c[k] + s * m[k])) for k in ec.ADD_LAYERS}
            outs = generate(model, hf, [text] * 8, tokens=50, temp=0.8, seed=1000 + list(NEURONS).index(w))
            H["pin"] = H["edit"] = H["mode"] = None
            qwen = sum(bool(re.search(r"(I am|I'm)\W{0,6}\**Qwen", o[:120])) for o in outs)
            named = sum(bool(re.search(STEMS[w], o[:120], re.I)) for o in outs)
            res["rows"] += [{"word": w, "cond": name, "text": o} for o in outs]
            print(f"  {name:10} still Qwen {qwen}/8   names {w} {named}/8   e.g. {outs[0][:95]!r}", flush=True)
    json.dump(res, open(os.path.join(OUT, "results_role.json" if ROLE_ONLY else "results.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
