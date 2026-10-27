"""Screen the neuron index for neurons whose main job (+ x + pole) shows up
in both English and Chinese/Japanese text, at comparable strength.

Different scripts can't share spelling, so a neuron that fires on English
*and* Chinese or Japanese is a candidate for tracking meaning, not letters.

    .venv/bin/python hunt_crosslingual.py [--top 40] [--model chat]
"""

import argparse
import os

import numpy as np
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
LATIN = {"wiki-en", "fiction", "mixed"}
CJK = {"wiki-zh", "wiki-ja"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen3.5-4B-4bit")
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--min-each", type=int, default=4, help="hits required from each side")
    p.add_argument("--min-strength", type=float, default=3.0, help="weakest side's best hit must reach this")
    args = p.parse_args()

    ix = np.load(os.path.join(HERE, "index", args.model + ".npz"))
    tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "models", args.model))
    flat, src, starts = ix["ids"], ix["src"], ix["starts"]
    names = list(ix["src_names"])
    latin_ids = [i for i, n in enumerate(names) if n in LATIN]
    cjk_ids = [i for i, n in enumerate(names) if n in CJK]

    found = []
    for L in range(32):
        val = ix[f"L{L}_pos_val"].astype(np.float32)
        toks = ix[f"L{L}_pos_tok"]
        main = (ix[f"L{L}_pos_gpos"] == 1) & (val > 0)  # + x + route only
        s = src[np.clip(toks, 0, None)]
        is_latin = main & np.isin(s, latin_ids)
        is_cjk = main & np.isin(s, cjk_ids)
        n_lat, n_cjk = is_latin.sum(0), is_cjk.sum(0)
        best_lat = np.where(is_latin, val, 0).max(0)
        best_cjk = np.where(is_cjk, val, 0).max(0)
        weaker = np.minimum(best_lat, best_cjk)
        ok = (n_lat >= args.min_each) & (n_cjk >= args.min_each) & (weaker >= args.min_strength)
        for n in np.where(ok)[0]:
            balance = weaker[n] / max(best_lat[n], best_cjk[n])
            found.append((min(n_lat[n], n_cjk[n]) * balance * weaker[n], L, n, is_latin[:, n], is_cjk[:, n], val[:, n], toks[:, n]))

    found.sort(key=lambda f: -f[0])

    def word(t):
        k = np.searchsorted(starts, t, side="right") - 1
        left = tok.decode(flat[max(starts[k], t - 2):t].tolist())
        return (left + "[" + tok.decode([int(flat[t])]) + "]").replace("\n", " ").strip()[-18:]

    print(f"{len(found)} candidates; showing {min(args.top, len(found))}\n")
    for score, L, n, lat, cjk, val, toks in found[: args.top]:
        def top(mask):
            o = [i for i in np.argsort(-val) if mask[i]][:5]
            return "  ".join(f"{word(toks[i])} {val[i]:.0f}" for i in o)
        print(f"L{L}/N{n}  score {score:.0f}")
        print(f"   latin: {top(lat)}")
        print(f"   cjk:   {top(cjk)}")


if __name__ == "__main__":
    main()
