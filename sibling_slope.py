"""Does slot number predict neuron size in Qwen3.5's sibling models too?

Fetches only the down_proj weight of a few layers from each model on Hugging Face, using HTTP
range requests into the official safetensors files (no full download, nothing executed), and
measures the rank correlation between slot number and each neuron's write-vector size.

    .venv/bin/python sibling_slope.py
"""

import json
import struct
import urllib.request

import numpy as np

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

MODELS = {"Qwen3.5-0.8B": 24, "Qwen3.5-2B": 24, "Qwen3.5-4B": 32, "Qwen3.5-9B": 32}


def get(url, rng=None):
    req = urllib.request.Request(url, headers={"User-Agent": "qwen-local/0.1", **({"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def down_proj(repo, layer):
    base = f"https://huggingface.co/Qwen/{repo}/resolve/main/"
    key = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
    try:
        shard = json.loads(get(base + "model.safetensors.index.json"))["weight_map"][key]
    except Exception:
        shard = "model.safetensors"
    n = struct.unpack("<Q", get(base + shard, (0, 7)))[0]
    header = json.loads(get(base + shard, (8, 8 + n - 1)))
    info = header[key]
    a, b = info["data_offsets"]
    raw = get(base + shard, (8 + n + a, 8 + n + b - 1))
    assert info["dtype"] == "BF16", info["dtype"]
    w = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    return w.reshape(info["shape"])  # [hidden, neurons]


def main():
    print(f"{'model':<14} {'neurons':>8}   slot-vs-size rank correlation by layer      first 5% / last 5% size")
    for repo, n_layers in MODELS.items():
        rows = []
        for layer in (n_layers // 6, n_layers // 2, (5 * n_layers) // 6):
            w = down_proj(repo, layer)
            size = np.linalg.norm(w, axis=0)
            rho = np.corrcoef(np.arange(len(size)), np.argsort(np.argsort(size)))[0, 1]
            k = len(size) // 20
            rows.append((layer, rho, size[:k].mean() / size[-k:].mean()))
        print(f"{repo:<14} {w.shape[1]:>8}   " + "   ".join(f"L{l}: {r:+.3f}" for l, r, _ in rows)
              + "      " + " ".join(f"{q:.2f}" for _, _, q in rows))


if __name__ == "__main__":
    main()
