"""Rescore the identity search's top candidates with a control that catches breakage.

The first pass's control was one forced word (" Paris"), which a neuron can survive while
wrecking a whole reply ("the the the..."). Here the control is a full fluent sentence, so
neurons that break fluency lose. Score = identity drop - fluent-control drop.

    .venv/bin/python rescore_identity.py [--top 3000]
    writes index/identity_rescore.npz
"""

import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np
from mlx_lm import load

import steer
from search_identity import continuation_logprob

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = {
    "identity": ("Who are you?", "I am an", " AI language model"),
    "fluent": ("What is the capital of France?", "The capital of France is",
               " Paris, a city famous for the Eiffel Tower and its museums."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--include", nargs="*", default=["22:2742"], metavar="L:N", help="always rescore these")
    args = ap.parse_args()

    S = np.load(os.path.join(HERE, "index", "identity_search.npz"))
    cands = []
    for L in S["layers"].tolist():
        sc = S[f"L{L}_identity_drop"] - S[f"L{L}_control_drop"]
        cands += [(sc[n], L, n, S[f"L{L}_value"][n]) for n in range(len(sc))]
    cands.sort(reverse=True)
    picked = cands[: args.top]
    for spec in args.include:
        L, n = (int(x) for x in spec.split(":"))
        if not any(c[1] == L and c[2] == n for c in picked):
            picked.append(next(c for c in cands if c[1] == L and c[2] == n))

    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    pairs = {}
    for k, (q, prefix, cont) in TESTS.items():
        p_ids = tok.encode(steer.chat(tok, q) + prefix)
        pairs[k] = (p_ids, tok.encode(steer.chat(tok, q) + prefix + cont)[len(p_ids):])
    base = {k: float(continuation_logprob(model, p, c, 1)[0]) for k, (p, c) in pairs.items()}

    by_layer = {}
    for c in picked:
        by_layer.setdefault(c[1], []).append(c)
    res = []
    for L, cs in by_layer.items():
        for s in range(0, len(cs), args.batch):
            chunk = cs[s:s + args.batch]
            steer.HOOKS["bpin"] = (L, mx.array([c[2] for c in chunk]), mx.array([float(c[3]) for c in chunk]))
            drops = {}
            for k, (p, c) in pairs.items():
                lp = continuation_logprob(model, p, c, len(chunk))
                mx.eval(lp)
                drops[k] = base[k] - np.array(lp)
            steer.HOOKS["bpin"] = None
            for i, c in enumerate(chunk):
                res.append((drops["identity"][i] - drops["fluent"][i], L, c[2], drops["identity"][i], drops["fluent"][i], c[3]))
    res.sort(reverse=True)
    np.savez(os.path.join(HERE, "index", "identity_rescore.npz"), rows=np.array(res, dtype=np.float64))
    alien = next(r for r in res if r[1] == 22 and r[2] == 2742)
    print(f"rescored {len(res)} neurons. alien L22/N2742: score {alien[0]:.1f}, rank {res.index(alien) + 1}\n")
    for sc, L, n, di, df, v in res[:30]:
        print(f"  L{int(L)}/N{int(n):<5} score {sc:6.1f}   identity drop {di:6.1f}   fluent-control drop {df:6.1f}")


if __name__ == "__main__":
    main()
