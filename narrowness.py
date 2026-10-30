"""How narrowly does each neuron read? For every neuron in the chosen layers and each pole, the share of its firing
(clipped to that pole) carried by its top 1% of tokens in corpus A. 0.01 would be perfectly even; higher = a
narrower trigger. Streams the corpus once, keeping each neuron's running sum and its top-k values per pole.

    .venv/bin/python narrowness.py [--layers 8 24]   -> index/narrowness.npz
"""

import argparse
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

import steer

HERE = os.path.dirname(os.path.abspath(__file__))
N = 9216


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs=2, default=[8, 24])
    a = ap.parse_args()
    L0, L1 = a.layers
    layers = list(range(L0, L1 + 1))
    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    inner = model.language_model.model
    rows = [json.loads(l) for l in open(os.path.join(HERE, "corpus", "corpus.jsonl"))]
    texts = [tok.encode(r["text"])[:512] for r in rows]
    texts = [t for t in texts if len(t) >= 2]
    total = sum(len(t) - 1 for t in texts)
    k = max(1, total // 100)
    tot = {(L, s): mx.zeros((N,)) for L in layers for s in (1, -1)}
    top = {(L, s): mx.zeros((k, N)) for L in layers for s in (1, -1)}
    steer.HOOKS["record"] = True
    for i, ids in enumerate(texts):
        steer.clear_records()
        h = inner.embed_tokens(mx.array([ids]))
        fa, ss = create_attention_mask(h, None), create_ssm_mask(h, None)
        for lyr in inner.layers[: L1 + 1]:
            h = lyr(h, mask=ss if lyr.is_linear else fa, cache=None)
        for L in layers:
            act = steer.HOOKS["acts"][L][0][1:]  # [T, N], skip the attention sink
            for s in (1, -1):
                v = mx.maximum(s * act, 0)
                tot[(L, s)] = tot[(L, s)] + v.sum(0)
                both = mx.concatenate([top[(L, s)], v], 0)
                top[(L, s)] = mx.sort(both, axis=0)[-k:]
        mx.eval(list(tot.values()) + list(top.values()))
        if i % 100 == 0:
            print(f"  {i}/{len(texts)} passages", flush=True)
    steer.HOOKS["record"] = False
    steer.clear_records()
    out = {"layers": np.array(layers), "k": k, "tokens": total}
    for L in layers:
        for s, name in ((1, "pos"), (-1, "neg")):
            t = np.array(tot[(L, s)])
            tp = np.array(top[(L, s)])
            out[f"L{L}_{name}_share"] = np.where(t > 0, tp.sum(0) / np.maximum(t, 1e-9), 0).astype(np.float32)
            out[f"L{L}_{name}_max"] = tp[-1].astype(np.float32)
    path = os.path.join(HERE, "index", "narrowness.npz")
    if os.path.exists(path):  # merge with layers already measured
        old = dict(np.load(path))
        merged = sorted(set(old["layers"].tolist()) | set(layers))
        old.update(out)
        old["layers"] = np.array(merged)
        out = old
    np.savez(path, **out)
    print(f"saved index/narrowness.npz ({total:,} tokens, top {k} per neuron)")


if __name__ == "__main__":
    main()
