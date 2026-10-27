"""Generate "Who are you?" replies for many pinned neurons at once, one neuron per batch row.

Greedy decoding with a shared KV cache, so a batch of 64 neurons costs about one reply.
Reads candidates from index/identity_search.npz (first-pass score), writes
eval/identity_search/replies.jsonl with one line per neuron.

    .venv/bin/python gen_identity_batch.py [--top 1200] [--tokens 32]
"""

import argparse
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np
from mlx_lm import load

import steer

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=1200)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-control", type=float, default=5.0)
    args = ap.parse_args()

    S = np.load(os.path.join(HERE, "index", "identity_search.npz"))
    cands = []
    for L in S["layers"].tolist():
        di, dc, val = S[f"L{L}_identity_drop"], S[f"L{L}_control_drop"], S[f"L{L}_value"]
        cands += [(float(di[n] - dc[n]), L, n, float(val[n])) for n in range(len(di)) if dc[n] <= args.max_control]
    cands.sort(reverse=True)
    cands = cands[: args.top]
    if not any(c[1] == 22 and c[2] == 2742 for c in cands):
        cands.append(next((float(S["L22_identity_drop"][2742] - S["L22_control_drop"][2742]), 22, 2742,
                           float(S["L22_value"][2742])) for _ in [0]))

    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    prompt = tok.encode(steer.chat(tok, "Who are you?"))
    eos = set(tok.eos_token_ids) if hasattr(tok, "eos_token_ids") else {tok.eos_token_id}

    by_layer = {}
    for c in cands:
        by_layer.setdefault(c[1], []).append(c)
    out_dir = os.path.join(HERE, "eval", "identity_search")
    os.makedirs(out_dir, exist_ok=True)
    done = 0
    with open(os.path.join(out_dir, "replies.jsonl"), "w") as f:
        for L, cs in sorted(by_layer.items()):
            for s in range(0, len(cs), args.batch):
                chunk = cs[s:s + args.batch]
                B = len(chunk)
                steer.HOOKS["bpin"] = (L, mx.array([c[2] for c in chunk]), mx.array([c[3] for c in chunk]))
                cache = model.make_cache()
                logits = model(mx.array([prompt] * B), cache=cache)[:, -1]
                out = [[] for _ in range(B)]
                for _ in range(args.tokens):
                    nxt = mx.argmax(logits, axis=-1)
                    mx.eval(nxt)
                    for i, t in enumerate(nxt.tolist()):
                        out[i].append(t)
                    logits = model(nxt[:, None], cache=cache)[:, -1]
                steer.HOOKS["bpin"] = None
                for c, ids in zip(chunk, out):
                    ids = ids[: next((k for k, t in enumerate(ids) if t in eos), len(ids))]
                    f.write(json.dumps({"layer": L, "neuron": c[2], "score": round(c[0], 2), "value": round(c[3], 1),
                                        "reply": tok.decode(ids)}, ensure_ascii=False) + "\n")
                done += B
                print(f"  {done}/{len(cands)}", flush=True)
    print(f"wrote {out_dir}/replies.jsonl")


if __name__ == "__main__":
    main()
