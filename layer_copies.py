"""Were any of Qwen3.5-4B's layers grown as copies of others? (Hunt 39)

Goodfire's Neuron Atlas notes that in Qwen3.6-35B-A3B, layers 20-31 began training as copies of
layers 8-19. Our 4B has two separate peaks of self neurons (layers 3-7 and 20-24, Hunt 11) and neurons
that were ranked at some point in how it was made (Hunt 4). If a block was copied, the copy's neurons
should still have near-twins in the original, even after training and even if their slots were
re-ordered.

  twins     For each layer pair (A, B): every neuron in A (its gate and up rows, which read the shared
            residual stream) against every neuron in B; per neuron, its best cosine in B. Reported: the
            median best cosine and the share of A's neurons with a twin above 0.5. Independent layers
            should sit near the floor set by random high-dimensional rows.
  slots     The same pair, slot by slot (neuron i in A vs neuron i in B), for copies that kept order.

    .venv/bin/python layer_copies.py [--model Qwen3.5-4B]
"""

import argparse
import json
import os

import mlx.core as mx
import numpy as np

from diff_models import tensors

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval", "layer_copies")
P = "model.language_model.layers.{}.mlp."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3.5-4B")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t = tensors(a.model)
    n_layers = 1 + max(int(k.split(".layers.")[1].split(".")[0]) for k in t if ".layers." in k)
    rows = []
    for L in range(n_layers):
        g = t[P.format(L) + "gate_proj.weight"].astype(mx.float32)
        u = t[P.format(L) + "up_proj.weight"].astype(mx.float32)
        r = mx.concatenate([g / mx.linalg.norm(g, axis=1, keepdims=True), u / mx.linalg.norm(u, axis=1, keepdims=True)], 1)
        r = r / mx.linalg.norm(r, axis=1, keepdims=True)
        mx.eval(r)
        rows.append(r)
    N = rows[0].shape[0]
    med = np.zeros((n_layers, n_layers))
    twin = np.zeros((n_layers, n_layers))
    slot = np.zeros((n_layers, n_layers))
    for A in range(n_layers):
        for B in range(n_layers):
            if A == B:
                continue
            best = []
            for s in range(0, N, 2048):
                best.append(mx.max(rows[A][s:s + 2048] @ rows[B].T, axis=1))
            best = np.array(mx.concatenate(best))
            med[A, B] = np.median(best)
            twin[A, B] = np.mean(best > 0.5)
            slot[A, B] = float(mx.mean(mx.sum(rows[A] * rows[B], axis=1)))
        print(f"  layer {A:2d}: best partner by median twin cosine = layer {int(np.argmax(med[A]))} "
              f"({med[A].max():.3f}, {twin[A].max():.1%} above 0.5); slot-by-slot max {slot[A].max():+.3f}", flush=True)
    off = med[~np.eye(n_layers, dtype=bool)]
    print(f"\nmedian twin cosine over all layer pairs: median {np.median(off):.3f}, 95th percentile {np.percentile(off, 95):.3f}, "
          f"max {off.max():.3f}")
    print("pairs with the highest median twin cosine:")
    pairs = sorted(((med[A, B], A, B) for A in range(n_layers) for B in range(A + 1, n_layers)), reverse=True)[:15]
    for v, A, B in pairs:
        print(f"  {A:2d} – {B:2d} (gap {B - A:2d}): {v:.3f}, {twin[A, B]:.1%} of {A}'s neurons have a twin above 0.5")
    # mean similarity by layer gap: a copied block shows up as a bump at one gap
    print("\nmean median-twin cosine by layer gap:", {g: round(float(np.mean([med[A, A + g] for A in range(n_layers - g)])), 3)
                                                     for g in range(1, n_layers)})
    np.savez(os.path.join(OUT, f"{a.model}.npz"), median_twin=med, twin_share=twin, slot=slot)


if __name__ == "__main__":
    main()
