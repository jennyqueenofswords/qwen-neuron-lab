"""Which neurons are one-word detectors, and which words? For a list of (layer, neuron), stream corpus A, keep each
neuron's top 3 positive firings per passage, and report its "purity": the share of its 20 strongest hits that are
its single most common word (lowercased), plus its top words.

    .venv/bin/python word_neurons.py picks.json out.json     (picks: [[layer, neuron], ...])
"""

import collections
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

import steer

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    picks = [tuple(p) for p in json.load(open(sys.argv[1]))]
    by_layer = {}
    for L, n in picks:
        by_layer.setdefault(L, []).append(n)
    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    inner = model.language_model.model
    rows = [json.loads(l) for l in open(os.path.join(HERE, "corpus", "corpus.jsonl"))]
    hits = {p: [] for p in picks}
    steer.HOOKS["record"] = True
    for r in rows:
        ids = tok.encode(r["text"])[:512]
        if len(ids) < 3:
            continue
        steer.clear_records()
        h = inner.embed_tokens(mx.array([ids]))
        fa, ss = create_attention_mask(h, None), create_ssm_mask(h, None)
        for lyr in inner.layers[: max(by_layer) + 1]:
            h = lyr(h, mask=ss if lyr.is_linear else fa, cache=None)
        for L, ns in by_layer.items():
            a = np.array(steer.HOOKS["acts"][L][0][:, mx.array(ns)])
            a[0] = 0
            for j, n in enumerate(ns):
                col = a[:, j]
                for p in np.argsort(-col)[:3]:
                    if col[p] > 0:
                        hits[(L, n)].append((float(col[p]), tok.decode([ids[p]]).strip().lower()))
    steer.HOOKS["record"] = False
    steer.clear_records()
    out = {}
    for (L, n), hs in hits.items():
        top = sorted(hs, reverse=True)[:20]
        c = collections.Counter(w for _, w in top)
        w, k = c.most_common(1)[0] if c else ("", 0)
        out[f"L{L}/N{n}"] = {"purity": k / max(len(top), 1), "top": [x for x, _ in c.most_common(5)], "peak": top[0][0] if top else 0.0}
    json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
    print(f"{len(out)} neurons -> {sys.argv[2]}")


if __name__ == "__main__":
    main()
