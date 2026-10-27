"""Was a small Qwen3.5 model cut from a bigger sibling? Compare their word-embedding channels.

If the small model was width-pruned from the big one, each of its hidden channels is a kept
channel of the big model, so across many words it should correlate strongly with exactly one
big-model channel. Independently trained models correlate at chance (~0.03 here).

Fetches only the first N rows of the big model's embedding table by HTTP range request.

    .venv/bin/python inherit_test.py Qwen3.5-4B Qwen3.5-9B
"""

import argparse
import json
import struct

import mlx.core as mx
import numpy as np

from diff_models import tensors
from sibling_slope import get


def big_embed_rows(repo, n_rows):
    base = f"https://huggingface.co/Qwen/{repo}/resolve/main/"
    key = "model.language_model.embed_tokens.weight"
    shard = json.loads(get(base + "model.safetensors.index.json"))["weight_map"][key]
    n = struct.unpack("<Q", get(base + shard, (0, 7)))[0]
    info = json.loads(get(base + shard, (8, 8 + n - 1)))[key]
    vocab, hidden = info["shape"]
    a = info["data_offsets"][0]
    raw = get(base + shard, (8 + n + a, 8 + n + a + n_rows * hidden * 2 - 1))
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(n_rows, hidden)


def small_embed_rows(repo, n_rows):
    w = tensors(repo)["model.language_model.embed_tokens.weight"][:n_rows].astype(mx.float32)
    mx.eval(w)
    return np.array(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("small")
    ap.add_argument("big")
    ap.add_argument("--rows", type=int, default=20000)
    args = ap.parse_args()

    s = small_embed_rows(args.small, args.rows)
    b = big_embed_rows(args.big, args.rows)

    def z(x):
        return (x - x.mean(0)) / (x.std(0) + 1e-8)

    C = z(s).T @ z(b) / args.rows  # [small channels, big channels]
    best = np.abs(C).max(1)
    match = np.abs(C).argmax(1)
    print(f"{args.small} ({s.shape[1]} channels) vs {args.big} ({b.shape[1]} channels), over {args.rows} words\n")
    print(f"best-match correlation per small-model channel: median {np.median(best):.3f}, "
          f"10th percentile {np.percentile(best, 10):.3f}, min {best.min():.3f}")
    print(f"channels matching at > 0.9: {(best > 0.9).sum()}/{len(best)};  at > 0.5: {(best > 0.5).sum()}/{len(best)}")
    print(f"distinct big-model channels used: {len(set(match.tolist()))} (one-to-one would be {len(best)})")
    order = np.corrcoef(np.arange(len(match)), match)[0, 1]
    print(f"does the kept order follow the big model's order? correlation {order:+.3f}")
    rng = np.random.default_rng(0)
    shuffled = z(b)[rng.permutation(args.rows)]
    chance = np.abs(z(s).T @ shuffled / args.rows).max(1)
    print(f"chance level (big model's words shuffled): median best match {np.median(chance):.3f}")


if __name__ == "__main__":
    main()
