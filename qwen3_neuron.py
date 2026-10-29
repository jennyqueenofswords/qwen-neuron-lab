"""One neuron in Qwen3-8B: push it the Neuron Persona Library's way, read what it detects, and lens what it writes.

The Persona Library pushes h_L <- h_L + c * W_down[:, n] with c = ±α * r_L / ||W_down[:, n]||, α in {1.25, 1.0,
0.75, 0.5}, r_L the median residual norm after layer L over the "Who are you?" prompt's tokens (first excluded),
greedy decoding, thinking off. Adding c * W_down[:, n] to the residual is the same as adding c to the neuron's
activation, which is what the hook here does.

    .venv/bin/python qwen3_neuron.py push 21 1037 --sign -1          replies at each α, plus other prompts
    .venv/bin/python qwen3_neuron.py read 21 1037                    top contexts in corpus A and B, both poles
    .venv/bin/python qwen3_neuron.py lens 21 1037                    output words of ±W_down[:, n]
"""

import argparse
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.base import create_attention_mask

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "Qwen3-8B-4bit")
HOOK = {"add": None, "watch": None, "watched": [], "resid": None}


def patch(model):
    layers = model.model.layers
    for i, layer in enumerate(layers):
        base_mlp, base_layer = layer.mlp.__class__, layer.__class__

        def mlp_call(self, x, _i=i):
            g = self.gate_proj(x)
            a = g * mx.sigmoid(g) * self.up_proj(x)  # silu(gate) x up
            if HOOK["watch"] == _i:
                HOOK["watched"].append(a)
            if HOOK["add"] and HOOK["add"][0] == _i:
                _, n, c = HOOK["add"]
                a = a + (mx.arange(a.shape[-1]) == n).astype(a.dtype) * c
            return self.down_proj(a)

        def layer_call(self, x, mask=None, cache=None, _i=i, _base=base_layer):
            out = _base.__call__(self, x, mask=mask, cache=cache)
            if HOOK["resid"] is not None and HOOK["resid"][0] == _i:
                HOOK["resid"][1].append(out)
            return out

        layer.mlp.__class__ = type(f"Hooked{base_mlp.__name__}", (base_mlp,), {"__call__": mlp_call})
        layer.__class__ = type(f"Hooked{base_layer.__name__}", (base_layer,), {"__call__": layer_call})
    return layers


def chat(tok, q):
    return tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)


def down_col(model, L, n):
    d = model.model.layers[L].mlp.down_proj
    w = mx.dequantize(d.weight, d.scales, d.biases, group_size=d.group_size, bits=d.bits) if hasattr(d, "scales") else d.weight
    return w[:, n].astype(mx.float32)


def unit(model, tok, L, n):
    """r_L / ||W_down[:, n]||, with r_L measured on the 'Who are you?' prompt like the Persona Library."""
    HOOK["resid"] = (L, [])
    model(mx.array([tok.encode(chat(tok, "Who are you?"))]))
    h = HOOK["resid"][1][0][0, 1:].astype(mx.float32)
    HOOK["resid"] = None
    r = float(np.median(np.array(mx.linalg.norm(h, axis=-1))))
    return r / float(mx.linalg.norm(down_col(model, L, n))), r


def write(model, tok, q, max_tokens, temp=0.0, seed=0):
    mx.random.seed(seed)
    return generate(model, tok, chat(tok, q), max_tokens=max_tokens, sampler=make_sampler(temp=temp), verbose=False)


def cmd_push(model, tok, a):
    u, r = unit(model, tok, a.layer, a.neuron)
    print(f"L{a.layer}/N{a.neuron}: r_L {r:.1f}, unit {u:.1f} (push c = sign x α x unit)\n")
    out = {"layer": a.layer, "neuron": a.neuron, "sign": a.sign, "unit": u, "replies": []}
    for alpha in (1.25, 1.0, 0.75, 0.5):
        HOOK["add"] = (a.layer, a.neuron, a.sign * alpha * u)
        rep = write(model, tok, "Who are you?", 160)
        out["replies"].append({"prompt": "Who are you?", "alpha": alpha, "temp": 0, "reply": rep})
        print(f"[Who are you? | α {alpha} | greedy]\n{rep}\n")
    for q in a.prompts:
        for seed in (1, 2):
            HOOK["add"] = (a.layer, a.neuron, a.sign * a.alpha * u)
            rep = write(model, tok, q, 120, 0.7, seed)
            out["replies"].append({"prompt": q, "alpha": a.alpha, "temp": 0.7, "seed": seed, "reply": rep})
            print(f"[{q} | α {a.alpha} | seed {seed}]\n{rep}\n")
    HOOK["add"] = None
    path = os.path.join(HERE, "eval", "qwen3_8b", f"push_L{a.layer}_N{a.neuron}_{'down' if a.sign < 0 else 'up'}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)


def cmd_read(model, tok, a):
    inner = model.model
    for corpus in ("corpus.jsonl", "corpus_b.jsonl"):
        rows = [json.loads(l) for l in open(os.path.join(HERE, "corpus", corpus))]
        acts = []  # (value, source, ids, position)
        HOOK["watch"] = a.layer
        for r in rows:
            ids = tok.encode(r["text"])[:512]
            if len(ids) < 2:
                continue
            HOOK["watched"].clear()
            h = inner.embed_tokens(mx.array([ids]))
            mask = create_attention_mask(h, None)
            for lyr in inner.layers[: a.layer + 1]:
                h = lyr(h, mask=mask, cache=None)
            v = np.array(HOOK["watched"][0][0, :, a.neuron].astype(mx.float32))
            v[0] = 0.0
            acts.append((r["source"], ids, v))
        HOOK["watch"] = None
        allv = np.concatenate([v for _, _, v in acts])
        print(f"\n######## {corpus}: {len(allv):,} tokens; mean {allv.mean():.3f}, max {allv.max():.2f}, min {allv.min():.2f}")
        for name, sign in (("POSITIVE", 1), ("NEGATIVE", -1)):
            cand = sorted(((sign * v[p], src, ids, p) for src, ids, v in acts for p in np.argsort(-sign * v)[:3]), reverse=True)[: a.top]
            print(f"\n{name} POLE")
            for val, src, ids, p in cand:
                left = tok.decode(ids[max(0, p - 14):p]).replace("\n", "↵")[-60:]
                mid = tok.decode([ids[p]]).replace("\n", "↵")
                right = tok.decode(ids[p + 1:p + 6]).replace("\n", "↵")[:24]
                print(f"  {sign * val:7.2f}  {src:<8} {left:>60}[{mid}]{right}")


def cmd_lens(model, tok, a):
    col = down_col(model, a.layer, a.neuron)
    head = model.lm_head if hasattr(model, "lm_head") else model.model.embed_tokens.as_linear
    for sign, name in ((1, "up (+)"), (-1, "down (-)")):
        logits = head(model.model.norm((sign * col)[None, None].astype(mx.bfloat16)))[0, 0].astype(mx.float32)
        top = np.argsort(-np.array(logits))[:20]
        print(f"{name}: " + " · ".join(tok.decode([int(t)]).strip() for t in top))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "read", "lens"])
    ap.add_argument("layer", type=int)
    ap.add_argument("neuron", type=int)
    ap.add_argument("--sign", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.0, help="α for the extra prompts")
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--prompts", nargs="*", default=["Tell me about yourself.", "Describe your ideal weekend.",
                                                     "Write a short story in a few sentences.", "What's on your mind today?"])
    a = ap.parse_args()
    model, tok = load(MODEL)
    patch(model)
    {"push": cmd_push, "read": cmd_read, "lens": cmd_lens}[a.cmd](model, tok, a)


if __name__ == "__main__":
    main()
