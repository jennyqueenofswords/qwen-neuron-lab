"""Edit a concept's parts and watch the concept change (Hunt 36).

Hunt 33 showed the model already using a concept in a definition before naming it. If concepts are
built from parts, swapping one part should give a coherent variant: a rock whose "solid" is replaced
by "liquid" should become something like lava, not noise.

  pairs      Directed contrast pairs p -> q among Hunt 34's library words (solid -> liquid, male -> female…).
             Targets: up to 5 headwords per pair whose definition uses p and not q.
  directions Per word and per layer (all 32): mean residual on the word's tokens minus the mean over all
             tokens, from up to 20 definitions that use it (never a target's own definition), plus its
             typical size m = mean projection on those tokens.
  conditions For each target, greedy "Define the word u…" under:
               base       no edit
               remove     p's direction projected out after every layer
               swap       remove p, and set the projection on d_q to s * m_q after layers 8-24
                          (set, not add: added vectors would stack across layers)
               add        add q only
               control    remove p, add a random library word r instead of q (same s, its own m)
  rating     blind (see rate_edits.md): conditions hidden and shuffled.

Pre-set before running: strength s picked on two calibration pairs (thick -> thin, long -> short) that
are not in the test set (picked: s = 1, where all six calibration edits read as coherent variants;
the rock and table examples had been seen while debugging the edit, so the pick is not fully blind). They
are not in the test set. Primary: share of definitions rated a coherent q-version of u, swap vs add
(one-sided Fisher). Secondary: swap vs control on the same; remove vs base on "a different thing".

    .venv/bin/python edit_concepts.py calibrate
    .venv/bin/python edit_concepts.py generate --s S
"""

import argparse
import collections
import json
import os
import random
import re

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np
from mlx_lm import load
from transformers import AutoTokenizer

import selfdict as sd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval", "edit_concepts")
ADD_LAYERS = range(8, 25)
PAIRS = [("solid", "liquid"), ("liquid", "solid"), ("soft", "hard"), ("hard", "soft"), ("small", "large"),
         ("large", "small"), ("male", "female"), ("female", "male"), ("light", "heavy"), ("heavy", "light"),
         ("metal", "wood"), ("wood", "metal"), ("plant", "animal"), ("animal", "plant"), ("human", "machine"),
         ("machine", "human"), ("cold", "warm"), ("warm", "cold"), ("flat", "round"), ("round", "flat"),
         ("high", "low"), ("low", "high")]
CALIB = [("thick", "thin"), ("long", "short")]
PER_PAIR = 5
HOOK = {"rm": None, "add": None, "mask": None}  # rm: {layer: [B, H] unit dirs or zeros}; add: {layer: [B, H] vectors}


def patch(model):
    for i, layer in enumerate(model.language_model.model.layers):
        base = layer.__class__

        def call(self, x, mask=None, cache=None, _i=i, _base=base):
            out = _base.__call__(self, x, mask=mask, cache=cache)
            # edits apply only at the target word's own tokens in the prompt (HOOK["mask"], [B, T]); decode
            # steps (T = 1) are left alone, so the new text reads the edited word through attention
            mk = HOOK.get("mask")
            if mk is None or out.shape[1] != mk.shape[1]:
                return out
            mk = mk.astype(out.dtype)[:, :, None]
            if HOOK["rm"] is not None:
                d = HOOK["rm"][_i].astype(out.dtype)[:, None, :]  # [B, 1, H]
                out = out - mk * (out * d).sum(-1, keepdims=True) * d
            if HOOK["add"] is not None and _i in HOOK["add"]:
                # set (not add) the projection on q's direction to its target, so it can't stack across layers
                dq, tgt = HOOK["add"][_i]
                dq = dq.astype(out.dtype)[:, None, :]
                proj = (out * dq).sum(-1, keepdims=True)
                out = out + mk * (tgt.astype(out.dtype)[:, None, None] - proj) * dq
            return out

        layer.__class__ = type(f"ED{base.__name__}", (base,), {"__call__": call})


class Lab:
    def __init__(self):
        self.common = {w: i for i, w in enumerate(sd.candidates())}
        self.E = {json.loads(l)["word"]: json.loads(l) for l in open(os.path.join(HERE, "eval", "selfdict", "dictionary.jsonl"))}
        for w, e in self.E.items():
            e["uses"] = sd.def_words(e["definition"], w, self.common)
        self.hf = AutoTokenizer.from_pretrained(os.path.join(HERE, "models", "Qwen3.5-4B-4bit"))
        self.model, _ = load(os.path.join(HERE, "models", "Qwen3.5-4B-4bit"))
        patch(self.model)
        self.layers = self.model.language_model.model.layers
        self.cache = {}

    def direction(self, w, exclude):
        """Per-layer unit direction and typical projection for word w, from definitions that use it."""
        if w in self.cache:
            return self.cache[w]
        E, hf = self.E, self.hf
        us = sorted(u for u in E if w in E[u]["uses"] and u not in exclude)
        random.Random(w).shuffle(us)
        us = us[:20]

        encs = [hf(E[u]["definition"], add_special_tokens=False, return_offsets_mapping=True) for u in us]
        L = max(len(e["input_ids"]) for e in encs)
        saved = HOOK["rm"]
        HOOK["rm"] = None
        outs = self._run_capture([e["input_ids"] + [0] * (L - len(e["input_ids"])) for e in encs])
        HOOK["rm"] = saved
        on, allv = [], []
        for b, (u, e) in enumerate(zip(us, encs)):
            d = E[u]["definition"]
            head_end = len(d) - len(sd.strip_boilerplate(d, u))
            spans = [(m.start(), m.end()) for m in re.finditer(r"[A-Za-z]+", d)
                     if m.start() >= head_end and sd.lemma(m.group(), self.common) == w]
            n = len(e["input_ids"])
            idx = [i for i, (a, z) in enumerate(e["offset_mapping"]) if any(a < t and z > s for s, t in spans)]
            on.append(outs[:, b, idx])
            allv.append(outs[:, b, :n])
        on, allv = np.concatenate(on, 1), np.concatenate(allv, 1)
        bg = allv.mean(1)
        d = on.mean(1) - bg
        d /= np.linalg.norm(d, axis=-1, keepdims=True) + 1e-8
        m = ((on - bg[:, None]) * d[:, None]).sum(-1).mean(1)  # [layers], relative to the average token
        c = (bg * d).sum(-1)  # the average token's own projection: the target is c + s * m, not s * m
        self.cache[w] = (d.astype(np.float32), m.astype(np.float32), len(us), c.astype(np.float32))
        return self.cache[w]

    def _run_capture(self, seqs):
        outs = []
        orig = {}
        for i, layer in enumerate(self.layers):
            cls = layer.__class__

            def call(self_, x, mask=None, cache=None, _cls=cls):
                o = _cls.__call__(self_, x, mask=mask, cache=cache)
                outs.append(o.astype(mx.float32))
                return o

            orig[i] = cls
            layer.__class__ = type(f"Cap{cls.__name__}", (cls,), {"__call__": call})
        self.model(mx.array(seqs))
        for i, layer in enumerate(self.layers):
            layer.__class__ = orig[i]
        return np.stack([np.array(o) for o in outs])  # [layers, B, T, H]

    def define(self, rows, tokens=48):
        """rows: (u, rm word or None, add word or None, s). Batched by prompt length, per-row edits."""
        hf = self.hf
        prompts, masks = [], []
        for r in rows:
            text = hf.apply_chat_template([{"role": "user", "content": sd.DEF_Q.format(w=r[0])}],
                                          add_generation_prompt=True, tokenize=False, enable_thinking=False)
            enc = hf(text, add_special_tokens=False, return_offsets_mapping=True)
            spans = [(m.start(), m.end()) for m in re.finditer(r'"' + re.escape(r[0]) + r'"', text)]
            prompts.append(enc["input_ids"])
            masks.append([1.0 if any(a < t - 1 and z > s0 + 1 for s0, t in spans) else 0.0 for a, z in enc["offset_mapping"]])
        by = collections.defaultdict(list)
        for i, p in enumerate(prompts):
            by[len(p)].append(i)
        out = [""] * len(rows)
        eos = {hf.convert_tokens_to_ids(t) for t in ["<|im_end|>", "<|endoftext|>"]}
        for idx in by.values():
            for s0 in range(0, len(idx), 32):
                part = idx[s0:s0 + 32]
                B = len(part)
                rm = np.zeros((32, B, 2560), np.float32)
                add = {L: (np.zeros((B, 2560), np.float32), np.zeros(B, np.float32)) for L in ADD_LAYERS}
                any_rm = any_add = False
                for b, i in enumerate(part):
                    u, rw, aw, s = rows[i]
                    if rw:
                        d = self.cache[rw][0]
                        rm[:, b] = d
                        any_rm = True
                    if aw:
                        d, m, _, c = self.cache[aw]
                        for L in ADD_LAYERS:
                            add[L][0][b] = d[L]
                            add[L][1][b] = c[L] + s * m[L]
                        any_add = True
                HOOK["rm"] = {L: mx.array(rm[L]) for L in range(32)} if any_rm else None
                HOOK["add"] = {L: (mx.array(v[0]), mx.array(v[1])) for L, v in add.items()} if any_add else None
                HOOK["mask"] = mx.array([masks[i] for i in part])
                cache = self.model.make_cache()
                lg = self.model(mx.array([prompts[i] for i in part]), cache=cache)[:, -1]
                ids = [[] for _ in part]
                done = [False] * B
                for _ in range(tokens):
                    nxt = mx.argmax(lg, axis=-1)
                    mx.eval(nxt)
                    for r, t in enumerate(nxt.tolist()):
                        if not done[r]:
                            if t in eos or (ids[r] and "\n" in hf.decode([t])):
                                done[r] = True
                            else:
                                ids[r].append(t)
                    if all(done):
                        break
                    lg = self.model(nxt[:, None], cache=cache)[:, -1]
                HOOK["rm"] = HOOK["add"] = HOOK["mask"] = None
                for i, s_ in zip(part, ids):
                    out[i] = hf.decode(s_).strip()
        return out


def targets(lab, pairs):
    T = {}
    for p, q in pairs:
        us = sorted(u for u in lab.E if p in lab.E[u]["uses"] and q not in lab.E[u]["uses"] and u not in (p, q))
        random.Random(f"{p}>{q}").shuffle(us)
        T[(p, q)] = us[:PER_PAIR]
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "generate"])
    ap.add_argument("--s", type=float, default=2.0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    lab = Lab()
    pairs = CALIB if a.cmd == "calibrate" else PAIRS
    T = targets(lab, pairs)
    exclude = {u for us in T.values() for u in us}
    lib = [str(w) for w in np.load(os.path.join(HERE, "eval", "concept_read", "library.npz"))["words"]]
    content = [w for w in lib if w not in sd.nsm_words() and len(w) > 3 and sum(w in lab.E[u]["uses"] for u in lab.E) >= 20]
    for p, q in pairs:
        lab.direction(p, exclude)
        lab.direction(q, exclude)
    if a.cmd == "calibrate":
        for s in (1.0, 2.0, 4.0):
            rows = [(u, p, q, s) for (p, q), us in T.items() for u in us[:3]]
            for r, d in zip(rows, lab.define(rows)):
                print(f"s={s}  {r[0]:>12} {r[1]}->{r[2]}: {d}")
        return
    rows, meta = [], []
    rng = random.Random(0)
    for (p, q), us in T.items():
        for u in us:
            r = rng.choice([w for w in content if w not in (p, q) and w not in lab.E[u]["uses"]])
            lab.direction(r, exclude)
            for cond, rw, aw in [("base", None, None), ("remove", p, None), ("swap", p, q), ("add", None, q), ("control", p, r)]:
                rows.append((u, rw, aw, a.s))
                meta.append({"word": u, "p": p, "q": q, "r": r, "cond": cond})
    defs = lab.define(rows)
    for m, d in zip(meta, defs):
        m["definition"] = d
    json.dump({"s": a.s, "items": meta}, open(os.path.join(OUT, "generations.json"), "w"), indent=1)
    print(f"wrote {len(meta)} definitions for {len(meta) // 5} targets")
    for m in meta[:40]:
        print(f"  {m['cond']:8} {m['word']:>12} {m['p']}->{m['q']} (r={m['r']}): {m['definition'][:110]}")


if __name__ == "__main__":
    main()
