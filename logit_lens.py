"""What does one neuron push the model to say? Read its output arrow through the vocabulary.

A neuron writes its down_proj column into the residual stream, scaled by how hard it fires.
In the last layers that arrow goes almost straight to the output, so multiplying it by the
token embeddings shows which next tokens it makes more likely (+) and less likely (-).

    .venv/bin/python logit_lens.py 31 162 [--model chat] [--top 25]
"""

import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np
from mlx_lm import load

from steer import MODELS


def neuron_out(mlp, n):
    """The neuron's write direction: column n of down_proj, dequantized if needed."""
    d = mlp.down_proj
    if hasattr(d, "scales"):
        w = mx.dequantize(d.weight, d.scales, d.biases, group_size=d.group_size, bits=d.bits)
    else:
        w = d.weight
    return w[:, n].astype(mx.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("layer", type=int)
    p.add_argument("neuron", type=int)
    p.add_argument("--model", default="chat")
    p.add_argument("--top", type=int, default=25)
    args = p.parse_args()

    model, tok = load(MODELS.get(args.model, args.model))
    inner = model.language_model.model
    col = neuron_out(inner.layers[args.layer].mlp, args.neuron)
    # The final RMSNorm rescales each dimension before the vocabulary readout.
    col = col * inner.norm.weight.astype(mx.float32)
    logits = np.array(inner.embed_tokens.as_linear(col[None, None, :])[0, 0].astype(mx.float32))

    order = np.argsort(-logits)
    vocab = [tok.decode([i]) for i in range(len(logits))]

    def show(idx):
        return "  ".join(repr(vocab[i]) for i in idx[: args.top])

    print(f"\nL{args.layer}/N{args.neuron}: next tokens it pushes UP when it fires positive")
    print(show(order))
    print(f"\nnext tokens it pushes DOWN when it fires positive (UP when it fires negative)")
    print(show(order[::-1]))

    def starts_word(s):
        return s[:1] in (" ", "\n") or not s[:1].isalnum()

    for label, idx in (("top 1000", order[:1000]), ("bottom 1000", order[::-1][:1000]), ("whole vocabulary", order)):
        frac = np.mean([starts_word(vocab[i]) for i in idx])
        print(f"{label:>16}: {frac:.0%} start a new word (leading space, newline or punctuation)")


if __name__ == "__main__":
    main()
