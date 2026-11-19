"""Positive control for Hunt 39: does the neuron-twin test see Qwen3.6-35B-A3B's copied layers?

Goodfire's Neuron Atlas notes that the 35B's layers 20-31 began training as copies of 8-19. If the
twin test (layer_copies.py) can't see that, its "no copies" for the 4B means little.

Fetches slices of the official weights by HTTP range request (no full download, nothing executed):
the shared expert's gate and up rows, and 5 routed experts' fused gate_up rows ([256, 1024, 2048] per
layer: 512 gate rows then 512 up rows), for layers 2, 8, 9, 14, 20, 26, 27, 39. About 190 MB.

Pairs: copies 8 -> 20 and 14 -> 26; controls 8 -> 9 (neighbors), 2 -> 14 and 27 -> 39 (12 apart, not
copies; 39 is outside the copied block). For each pair, each neuron's best cosine among the other
layer's neurons of the same expert (and, as a second control, of a different expert number).

    .venv/bin/python layer_copies_35b.py
"""

import json
import os
import struct

import numpy as np

import sibling_slope as ss

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval", "layer_copies")
BASE = "https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/main/"
LAYERS = [2, 8, 9, 14, 20, 26, 27, 39]
EXPERTS = [0, 64, 128, 193, 255]
PAIRS = [(8, 20, "copy"), (14, 26, "copy"), (8, 9, "neighbors"), (2, 14, "12 apart, not a copy"), (27, 39, "12 apart, not a copy")]
_headers = {}


def bf16(raw, shape):
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)


def header(shard):
    if shard not in _headers:
        n = struct.unpack("<Q", ss.get(BASE + shard, (0, 7)))[0]
        _headers[shard] = (n, json.loads(ss.get(BASE + shard, (8, 8 + n - 1))))
    return _headers[shard]


def tensor_slice(wm, key, start=0, count=None):
    """Rows [start, start + count) of the tensor's first axis."""
    n, h = header(wm[key])
    info = h[key]
    assert info["dtype"] == "BF16"
    shape = info["shape"]
    row = int(np.prod(shape[1:])) * 2
    count = shape[0] if count is None else count
    a = 8 + n + info["data_offsets"][0] + start * row
    return bf16(ss.get(BASE + wm[key], (a, a + count * row - 1)), [count] + shape[1:])


def neurons(gate, up):
    """[512, 2048] gate rows and up rows -> unit-normalized concatenated rows, as in layer_copies.py."""
    g = gate / np.linalg.norm(gate, axis=1, keepdims=True)
    u = up / np.linalg.norm(up, axis=1, keepdims=True)
    r = np.concatenate([g, u], 1)
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def twins(a, b):
    best = (a @ b.T).max(1)
    return float(np.median(best)), float(np.mean(best > 0.5))


def main():
    os.makedirs(OUT, exist_ok=True)
    wm = json.loads(ss.get(BASE + "model.safetensors.index.json"))["weight_map"]
    P = "model.language_model.layers.{}.mlp."
    W, fetched = {}, 0
    for L in LAYERS:
        g = tensor_slice(wm, P.format(L) + "shared_expert.gate_proj.weight")
        u = tensor_slice(wm, P.format(L) + "shared_expert.up_proj.weight")
        W[(L, "shared")] = neurons(g, u)
        fetched += (g.size + u.size) * 2
        for e in EXPERTS:
            gu = tensor_slice(wm, P.format(L) + "experts.gate_up_proj", e, 1)[0]
            W[(L, e)] = neurons(gu[:512], gu[512:])
            fetched += gu.size * 2
        print(f"  layer {L} fetched ({fetched / 1e6:.0f} MB so far)", flush=True)
    res = []
    print(f"\n{'pair':>8}  {'kind':22} {'shared expert':>16} {'same routed expert':>22} {'different expert':>18}")
    for A, B, kind in PAIRS:
        sh = twins(W[(A, "shared")], W[(B, "shared")])
        same = [twins(W[(A, e)], W[(B, e)]) for e in EXPERTS]
        diff = [twins(W[(A, e)], W[(B, EXPERTS[(i + 1) % len(EXPERTS)])]) for i, e in enumerate(EXPERTS)]
        ms, ts = np.mean([x[0] for x in same]), np.mean([x[1] for x in same])
        md, td = np.mean([x[0] for x in diff]), np.mean([x[1] for x in diff])
        print(f"{A:>3} -> {B:<3} {kind:22} {sh[0]:6.3f} ({sh[1]:5.1%})   {ms:6.3f} ({ts:5.1%}) {md:10.3f} ({td:5.1%})")
        res.append({"A": A, "B": B, "kind": kind, "shared": sh, "same_expert": same, "different_expert": diff})
    json.dump({"fetched_bytes": fetched, "pairs": res}, open(os.path.join(OUT, "qwen3.6-35b-control.json"), "w"), indent=1)
    print(f"\nfetched {fetched / 1e6:.0f} MB in total; cells are median best cosine (share of neurons with a twin above 0.5)")


if __name__ == "__main__":
    main()
