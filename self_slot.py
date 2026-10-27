"""Do the identity-flipping neurons share a common push against "I am an AI"?

For each neuron, pin it at its replication dose, run "Who are you?" + "I am" through the chat
model, and record how the residual stream at the last position (the moment the model names
itself) changes at each later layer: delta = pinned - normal. Then compare how well the deltas
of the 12 self-neurons agree with each other against 12 random neurons in the same layers at
the same push size. If they agree much more, their mean is a shared "away from AI" direction.

    .venv/bin/python self_slot.py
    writes index/self_slot.npz
"""

import json
import os
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

import steer
from logit_lens import neuron_out

HERE = os.path.dirname(os.path.abspath(__file__))
READ_LAYERS = list(range(22, 32))


def residuals(model, ids):
    """Residual stream after every layer at the last position, [layers, hidden]."""
    inner = model.language_model.model
    h = inner.embed_tokens(mx.array([ids]))
    fa_mask = create_attention_mask(h, None)
    ssm_mask = create_ssm_mask(h, None)
    outs = []
    for lyr in inner.layers:
        h = lyr(h, mask=ssm_mask if lyr.is_linear else fa_mask, cache=None)
        outs.append(h[0, -1].astype(mx.float32))
    out = mx.stack(outs)
    mx.eval(out)
    return np.array(out)


def mean_offdiag_cos(D):
    U = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
    C = U @ U.T
    n = len(D)
    return (C.sum() - np.trace(C)) / (n * (n - 1))


def main():
    meta = json.load(open(os.path.join(HERE, "eval", "identity_replicate", "meta.json")))
    selves = {k: v for k, v in meta["conditions"].items() if not k.startswith("random") and k != "SOE employee"}

    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    neutral = [t.format(n) for t in steer.TEMPLATES for n in steer.NEUTRAL]
    _, resid = steer.mean_over(neutral, model, tok, "resid")
    layers = model.language_model.model.layers

    ids = tok.encode(steer.chat(tok, "Who are you?") + "I am")
    steer.HOOKS["pin"].clear()
    h0 = residuals(model, ids)

    def delta(L, n, value):
        steer.HOOKS["pin"].clear()
        steer.HOOKS["pin"][L] = {n: value}
        d = residuals(model, ids) - h0
        steer.HOOKS["pin"].clear()
        return d

    names, self_d = [], []
    for name, c in selves.items():
        L, n = (int(x) for x in c["neuron"][1:].split("/N"))
        names.append(name)
        self_d.append(delta(L, n, c["value"]))
        print(f"  {name:<14} L{L}/N{n}")

    rng = random.Random(11)
    rand_d, rand_ids = [], []
    for name, c in selves.items():  # one random neuron per self-neuron, same layer, same push size
        L = int(c["neuron"][1:].split("/N")[0])
        n = rng.randrange(9216)
        v = 0.8 * resid[L] / float(mx.linalg.norm(neuron_out(layers[L].mlp, n)))
        rand_ids.append(f"L{L}/N{n}")
        rand_d.append(delta(L, n, v))

    S, R = np.stack(self_d), np.stack(rand_d)  # [12, 32 layers, hidden]
    print(f"\nagreement between pushes at the naming moment (mean pairwise cosine of the change)")
    print(f"{'layer':>6} {'self vs self':>13} {'random vs random':>17} {'self vs random':>15}")
    for L in READ_LAYERS:
        cross = np.mean([np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8) for a in S[:, L] for b in R[:, L]])
        print(f"{L:>6} {mean_offdiag_cos(S[:, L]):>13.3f} {mean_offdiag_cos(R[:, L]):>17.3f} {cross:>15.3f}")
    np.savez(os.path.join(HERE, "index", "self_slot.npz"), self_delta=S, rand_delta=R, h0=h0,
             names=np.array(names), rand_ids=np.array(rand_ids))


if __name__ == "__main__":
    main()
