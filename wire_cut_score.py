"""Score the wire-cutting test: per self, how much each cut lowers the blind-rated self strength (0-3).

The map's prediction is that cutting a self's strongest predicted wires hurts it more than cutting random neurons
in the same layers. Compares the drop from "pin" under predicted vs random cuts, paired by self (sign test).

    .venv/bin/python wire_cut_score.py eval/wire_cut/ratings.json
"""

import json
import os
import sys
from math import comb
from statistics import mean

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "wire_cut")
CONDS = ["pin", "top1 cut", "random1 cut", "top3 cut", "random3 cut"]


def sign_p(wins, losses):
    n = wins + losses
    return sum(comb(n, i) for i in range(wins, n + 1)) / 2 ** n if n else 1.0


def main():
    key = json.load(open(os.path.join(HERE, "key.json")))
    meta = json.load(open(os.path.join(HERE, "meta.json")))
    ratings = json.load(open(sys.argv[1]))
    s = {}
    for rid, k in key.items():
        s.setdefault(k["self"], {}).setdefault(k["condition"], []).append(ratings[rid]["strength"])
    m = {name: {c: mean(v) for c, v in cs.items()} for name, cs in s.items()}
    print(f"{'self':<16} {'layer':>6}" + "".join(f"{c:>13}" for c in CONDS))
    for name in meta:
        print(f"{name:<16} {meta[name]['neuron'].split('/')[0]:>6}" + "".join(f"{m[name][c]:>13.2f}" for c in CONDS))
    print(f"{'mean':<23}" + "".join(f"{mean(m[x][c] for x in m):>13.2f}" for c in CONDS))
    out = {}
    for k in ("1", "3"):
        d = [(m[x]["pin"] - m[x][f"top{k} cut"]) - (m[x]["pin"] - m[x][f"random{k} cut"]) for x in m]
        w, l = sum(v > 0 for v in d), sum(v < 0 for v in d)
        print(f"\ntop{k} vs random{k}: predicted cut hurt more in {w} selves, less in {l}, tied {len(d) - w - l}"
              f"   (sign test p = {sign_p(w, l):.3f}; mean extra drop {mean(d):+.2f})")
        out[k] = {"wins": w, "losses": l, "ties": len(d) - w - l, "p": sign_p(w, l), "mean_extra_drop": mean(d)}
        for band, sel in (("early (L3-7)", [x for x in m if int(meta[x]["neuron"][1:].split("/")[0]) < 12]),
                          ("deep (L21-24)", [x for x in m if int(meta[x]["neuron"][1:].split("/")[0]) >= 12])):
            dd = [(m[x][f"random{k} cut"] - m[x][f"top{k} cut"]) for x in sel]
            print(f"   {band:<14} {sum(v > 0 for v in dd)}/{len(dd)} hurt more, mean extra drop {mean(dd):+.2f}")
    json.dump({"means": m, "tests": out}, open(os.path.join(HERE, "scores.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
