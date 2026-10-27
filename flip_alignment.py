"""Do identity-flipping neurons lean against the assistant direction more than chance?

Compares each flip neuron's arrow alignment (index/assistant_alignment.npz, from the v2
direction) with every neuron in the same layer: its percentile, and a layer-matched
permutation test on the mean alignment of the whole set.

    .venv/bin/python flip_alignment.py eval/identity_search/flips.json [more flip files...]
"""

import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    A = np.load(os.path.join(HERE, "index", "assistant_alignment.npz"))
    flips = {}
    for path in sys.argv[1:]:
        for r in json.load(open(path)):
            flips[(r["layer"], r["neuron"])] = r.get("claimed", "")
    rows = []
    for (L, n), label in sorted(flips.items()):
        c = A[f"L{L}"]
        rows.append((float(c[n]), float((c < c[n]).mean()), L, n, label))
    vals = np.array([r[0] for r in rows])
    pct = np.array([r[1] for r in rows])

    rng = random.Random(0)
    null = []
    for _ in range(20000):
        null.append(np.mean([A[f"L{L}"][rng.randrange(9216)] for _, _, L, _, _ in rows]))
    null = np.array(null)
    p_lower = float((null <= vals.mean()).mean())

    print(f"{len(rows)} identity-flip neurons\n")
    print(f"mean alignment {vals.mean():+.4f}   (layer-matched random sets: {null.mean():+.4f} ± {null.std():.4f})")
    print(f"one-sided p (flips lean MORE against than chance): {p_lower:.3f}")
    print(f"median percentile in own layer {np.median(pct):.0%} (50% = typical; low = against)")
    print(f"share in the most-against 10% of their layer: {(pct < 0.10).mean():.0%};  most-along 10%: {(pct > 0.90).mean():.0%}\n")
    for v, p, L, n, label in sorted(rows):
        print(f"  {v:+.3f}  {p:>4.0%}  L{L}/N{n:<5} {label}")


if __name__ == "__main__":
    main()
