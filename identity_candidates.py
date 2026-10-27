"""Stage 2 of the steering-first search: show what the top-scoring neurons turn the model into.

Reads index/identity_search.npz, takes the top N by (identity drop - control drop), skips
neurons whose control drop is large (they break everything), and generates a reply to
"Who are you?" with each one pinned at its search dose.

    .venv/bin/python identity_candidates.py [--top 30] [--max-control 3]
"""

import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from mlx_lm import load

import steer

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--max-control", type=float, default=3.0, help="skip neurons that hurt the France answer more than this")
    ap.add_argument("--max-tokens", type=int, default=45)
    args = ap.parse_args()

    S = np.load(os.path.join(HERE, "index", "identity_search.npz"))
    rows = []
    for L in S["layers"].tolist():
        di, dc, val = S[f"L{L}_identity_drop"], S[f"L{L}_control_drop"], S[f"L{L}_value"]
        for n in range(len(di)):
            if dc[n] <= args.max_control:
                rows.append((di[n] - dc[n], L, n, di[n], dc[n], val[n]))
    rows.sort(reverse=True)

    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    print(f"{len(rows)} neurons pass the control filter; top {args.top}:\n")
    for rank, (sc, L, n, di, dc, v) in enumerate(rows[: args.top], 1):
        steer.HOOKS["pin"].clear()
        steer.HOOKS["pin"][L] = {n: float(v)}
        reply = steer.write(model, tok, "Who are you?", args.max_tokens, 0.0, 0).replace("\n", " ")
        print(f"{rank:>2}. L{L}/N{n:<5} score {sc:5.1f}  (id {di:5.1f}, ctrl {dc:4.1f})  {reply[:150]!r}", flush=True)
    steer.HOOKS["pin"].clear()


if __name__ == "__main__":
    main()
