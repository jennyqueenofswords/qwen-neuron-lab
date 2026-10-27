"""How much did chat training change each MLP neuron? Base vs chat, full precision.

For every neuron in every layer, compares three weight vectors between the two models:
  gate row and up row (what the neuron reads), down column (what it writes).
Change = 1 - cosine similarity (0 = same direction, 1 = unrelated), plus relative size change.
Reads the original bf16 safetensors from the Hugging Face cache, so 4-bit rounding isn't
mistaken for training.

    .venv/bin/python diff_models.py            # writes index/diff_base_chat.npz and prints a summary
"""

import glob
import json
import os

import mlx.core as mx
import numpy as np

HUB = os.path.expanduser("~/.cache/huggingface/hub")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index", "diff_base_chat.npz")


def tensors(repo):
    snap = glob.glob(f"{HUB}/models--Qwen--{repo}/snapshots/*/")[0]
    files = {}
    for k, f in json.load(open(snap + "model.safetensors.index.json"))["weight_map"].items():
        files.setdefault(f, []).append(k)
    out = {}
    for f in files:
        out.update(mx.load(snap + f))  # lazy: nothing is read until used
    return out


def cos_rows(a, b):
    return (a * b).sum(-1) / (mx.linalg.norm(a, axis=-1) * mx.linalg.norm(b, axis=-1) + 1e-8)


def main():
    base, chat = tensors("Qwen3.5-4B-Base"), tensors("Qwen3.5-4B")
    res = {k: [] for k in ("gate", "up", "down", "gate_rel", "up_rel", "down_rel")}
    for L in range(32):
        p = f"model.language_model.layers.{L}.mlp."
        for name, axis in (("gate", 1), ("up", 1), ("down", 0)):
            wb = base[p + f"{name}_proj.weight"].astype(mx.float32)
            wc = chat[p + f"{name}_proj.weight"].astype(mx.float32)
            if axis == 0:  # down_proj is [hidden, neurons]; a neuron's write vector is a column
                wb, wc = wb.T, wc.T
            c = 1 - cos_rows(wb, wc)
            r = mx.linalg.norm(wc - wb, axis=-1) / (mx.linalg.norm(wb, axis=-1) + 1e-8)
            mx.eval(c, r)
            res[name].append(np.array(c))
            res[name + "_rel"].append(np.array(r))
        print(f"  layer {L:>2}: median change  read {np.median(res['gate'][-1]):.4f}/{np.median(res['up'][-1]):.4f}"
              f"  write {np.median(res['down'][-1]):.4f}   max write {res['down'][-1].max():.3f}", flush=True)
    arrays = {k: np.stack(v) for k, v in res.items()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, **arrays)

    read = np.maximum(arrays["gate"], arrays["up"])
    write = arrays["down"]
    for title, a in (("most changed WRITE side (what it pushes)", write), ("most changed READ side (what it looks for)", read)):
        print(f"\n{title}  [overall median {np.median(a):.4f}]")
        for f in np.argsort(-a, axis=None)[:15]:
            L, n = np.unravel_index(f, a.shape)
            print(f"  L{L}/N{n:<5}  write {write[L, n]:.3f}  read {read[L, n]:.3f}  size change {arrays['down_rel'][L, n]:.2f}")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
