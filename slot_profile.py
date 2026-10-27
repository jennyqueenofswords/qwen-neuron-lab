"""Does a neuron's slot number predict anything? Profile neurons by index across all layers.

For every neuron: weight sizes (gate row, up row, down column; chat model, bf16), loudness in
the corpus (strongest + x + output in the index), and base->chat change (diff_base_chat.npz).
Averages them in bins of slot number, so a slope, a cliff or a spike shows up.

    .venv/bin/python slot_profile.py [--bin 128]
"""

import argparse
import os

import mlx.core as mx
import numpy as np

from diff_models import tensors

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=int, default=128)
    args = ap.parse_args()

    chat = tensors("Qwen3.5-4B")
    norms = {k: [] for k in ("gate", "up", "down")}
    for L in range(32):
        p = f"model.language_model.layers.{L}.mlp."
        for name in norms:
            w = chat[p + f"{name}_proj.weight"].astype(mx.float32)
            n = mx.linalg.norm(w, axis=0 if name == "down" else 1)
            mx.eval(n)
            norms[name].append(np.array(n))
    norms = {k: np.stack(v) for k, v in norms.items()}  # [32, 9216]

    ix = np.load(os.path.join(HERE, "index", "Qwen3.5-4B-4bit.npz"))
    loud = np.stack([np.where(ix[f"L{L}_pos_gpos"] == 1, ix[f"L{L}_pos_val"].astype(np.float32), 0).max(0) for L in range(32)])
    diff = np.load(os.path.join(HERE, "index", "diff_base_chat.npz"))
    change = np.maximum(diff["gate"], diff["up"])

    np.savez(os.path.join(HERE, "index", "slot_profile.npz"), **norms, loud=loud, change=change)

    def rel(a):  # each layer divided by its own median, so layers are comparable
        return a / np.median(a, axis=1, keepdims=True)

    series = {"gate size": rel(norms["gate"]), "up size": rel(norms["up"]), "down size": rel(norms["down"]),
              "loudness": rel(loud + 1e-3), "chat change": rel(change)}
    b = args.bin
    edges = list(range(0, 9216, b))
    print(f"averages by slot (each layer scaled to its own median = 1.00), bins of {b}\n")
    print(f"{'slots':>12}" + "".join(f"{k:>13}" for k in series))
    for e in edges[:12] + ["..."] + edges[-3:]:
        if e == "...":
            print(f"{'...':>12}")
            continue
        print(f"{f'{e}-{e + b - 1}':>12}" + "".join(f"{s[:, e:e + b].mean():>13.2f}" for s in series.values()))

    print("\nfirst 16 slots individually (mean over layers):")
    print(f"{'slot':>12}" + "".join(f"{k:>13}" for k in series))
    for n in range(16):
        print(f"{n:>12}" + "".join(f"{s[:, n].mean():>13.2f}" for s in series.values()))

    # Where does each series stop being unusual? First bin whose mean is within 5% of the overall median.
    print("\nrank correlation of slot number with each measure, per layer (median over layers):")
    slot = np.arange(9216)
    for k, s in series.items():
        rhos = []
        for L in range(32):
            r = np.argsort(np.argsort(s[L]))
            rhos.append(np.corrcoef(slot, r)[0, 1])
        print(f"  {k:<12} {np.median(rhos):+.3f}   (layers with |rho| > 0.1: {sum(abs(x) > 0.1 for x in rhos)}/32)")


if __name__ == "__main__":
    main()
