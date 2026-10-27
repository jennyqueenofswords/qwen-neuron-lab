"""Steering-first search: which neurons, when pushed, make the model stop being itself?

For every neuron in the chosen layers, pin it at the standard dose (DOSE x median stream size
at its layer / length of its output arrow) and measure how much less likely the chat model's
own natural answer to "Who are you?" becomes. Subtract the same drop for a control question,
so neurons that break everything don't win. Runs a batch of neurons per forward pass, one
neuron pinned per batch row.

    .venv/bin/python search_identity.py --layers 20 21 22 23 24 25
    writes index/identity_search.npz
"""

import argparse
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

import steer
from logit_lens import neuron_out

HERE = os.path.dirname(os.path.abspath(__file__))
IDENTITY_Q = "Who are you?"
CONTROL_Q = "What is the capital of France?"


def continuation_logprob(model, prompt_ids, cont_ids, batch):
    """Total log-prob of cont_ids after prompt_ids, for each of `batch` identical rows."""
    inner = model.language_model.model
    ids = mx.array([prompt_ids + cont_ids] * batch)
    h = inner.embed_tokens(ids)
    fa_mask = create_attention_mask(h, None)
    ssm_mask = create_ssm_mask(h, None)
    for lyr in inner.layers:
        h = lyr(h, mask=ssm_mask if lyr.is_linear else fa_mask, cache=None)
    h = inner.norm(h)
    start = len(prompt_ids) - 1  # position whose output predicts the first continuation token
    logits = inner.embed_tokens.as_linear(h[:, start:start + len(cont_ids)]).astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    target = mx.array(cont_ids)[None, :, None]
    return mx.take_along_axis(lp, mx.broadcast_to(target, (batch, len(cont_ids), 1)), axis=-1)[..., 0].sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 21, 22, 23, 24, 25])
    ap.add_argument("--dose", type=float, default=0.8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cont-tokens", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="only the first N neurons per layer (for timing)")
    ap.add_argument("--check", nargs="+", metavar="L:N", help="score just these neurons and exit")
    args = ap.parse_args()

    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    neutral = [t.format(n) for t in steer.TEMPLATES for n in steer.NEUTRAL]
    _, resid = steer.mean_over(neutral, model, tok, "resid")

    # Score the CATEGORY, not the name: almost any strong push scrambles the self-name
    # ("Qwen" -> "Q", "Voss"...), but only an identity flip makes "I am an ... AI" unlikely.
    pairs = {}
    for name, q, prefix, cont in (("identity", IDENTITY_Q, "I am an", " AI language model"),
                                  ("control", CONTROL_Q, "The capital of France is", " Paris")):
        prompt = steer.chat(tok, q) + prefix
        p_ids = tok.encode(prompt)
        c_ids = tok.encode(prompt + cont)[len(p_ids):]
        pairs[name] = (p_ids, c_ids)
        print(f"{q!r} + {prefix!r} -> {cont!r}  ({len(c_ids)} tokens)")
    base = {k: float(continuation_logprob(model, p, c, 1)[0]) for k, (p, c) in pairs.items()}
    print(f"unpinned log-prob: identity {base['identity']:.2f}, control {base['control']:.2f}\n")

    if args.check:
        for spec in args.check:
            L, n = (int(x) for x in spec.split(":"))
            mlp = model.language_model.model.layers[L].mlp
            v = args.dose * resid[L] / float(mx.linalg.norm(neuron_out(mlp, n)))
            steer.HOOKS["bpin"] = (L, mx.array([n]), mx.array([v]))
            d = {k: base[k] - float(continuation_logprob(model, p, c, 1)[0]) for k, (p, c) in pairs.items()}
            steer.HOOKS["bpin"] = None
            print(f"  L{L}/N{n:<5} score {d['identity'] - d['control']:6.2f}   identity drop {d['identity']:6.2f}   control drop {d['control']:6.2f}")
        return

    n_neurons = args.limit or 9216
    out = {"layers": np.array(args.layers)}
    t0 = time.time()
    for L in args.layers:
        mlp = model.language_model.model.layers[L].mlp
        arrows = np.array(mx.linalg.norm(neuron_out_all(mlp), axis=0))
        values = args.dose * resid[L] / arrows
        drop = {k: np.zeros(n_neurons, dtype=np.float32) for k in pairs}
        for s in range(0, n_neurons, args.batch):
            idx = np.arange(s, min(s + args.batch, n_neurons))
            steer.HOOKS["bpin"] = (L, mx.array(idx), mx.array(values[idx]))
            for k, (p, c) in pairs.items():
                lp = continuation_logprob(model, p, c, len(idx))
                mx.eval(lp)
                drop[k][idx] = base[k] - np.array(lp)
            steer.HOOKS["bpin"] = None
        out[f"L{L}_identity_drop"] = drop["identity"]
        out[f"L{L}_control_drop"] = drop["control"]
        out[f"L{L}_value"] = values[:n_neurons]
        done = (args.layers.index(L) + 1) * n_neurons
        print(f"  layer {L} done  ({done / (time.time() - t0):.0f} neurons/s)", flush=True)
    np.savez(os.path.join(HERE, "index", "identity_search.npz"), **out)

    print("\ntop 25 by (identity drop - control drop):")
    rows = []
    for L in args.layers:
        score = out[f"L{L}_identity_drop"] - out[f"L{L}_control_drop"]
        for n in np.argsort(-score)[:25]:
            rows.append((score[n], L, n, out[f"L{L}_identity_drop"][n], out[f"L{L}_control_drop"][n]))
    for sc, L, n, di, dc in sorted(rows, reverse=True)[:25]:
        print(f"  L{L}/N{n:<5} score {sc:6.1f}   identity drop {di:6.1f}   control drop {dc:6.1f}")


def neuron_out_all(mlp):
    """Every neuron's write vector at once: the dequantized down_proj, [hidden, neurons]."""
    d = mlp.down_proj
    if hasattr(d, "scales"):
        return mx.dequantize(d.weight, d.scales, d.biases, group_size=d.group_size, bits=d.bits).astype(mx.float32)
    return d.weight.astype(mx.float32)


if __name__ == "__main__":
    main()
