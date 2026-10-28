"""Score the all-layer sweep: identity flips by layer and direction, and how specific the selves are.

Joins the blind ratings (eval/identity_sweep/ratings_*.json) to the key, adds the earlier layer 20-25
push-up search (flips.json + flips_deep.json, rated the same way but without "specific"), and writes
eval/identity_sweep/flips_all.json.

    .venv/bin/python sweep_score.py
"""

import glob
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval", "identity_sweep")


def main():
    key = json.load(open(os.path.join(OUT, "key.json")))
    ratings = {}
    for f in sorted(glob.glob(os.path.join(OUT, "ratings_*.json"))):
        ratings.update(json.load(open(f)))
    replies = {}
    for f in ("sweep_up.jsonl", "sweep_down.jsonl"):
        d = "down" if "down" in f else "up"
        for line in open(os.path.join(HERE, "eval", "identity_search", f)):
            r = json.loads(line)
            replies[(d, r["layer"], r["neuron"])] = r["reply"]
    missing = set(key) - set(ratings)
    print(f"{len(ratings)} rated of {len(key)} survivors" + (f"  ({len(missing)} missing)" if missing else ""))

    flips = []
    for rid, r in ratings.items():
        k = key[rid]
        if r["identity"] == 1 and r["coherence"] >= 2:
            flips.append({"id": rid, "direction": k["direction"], "layer": k["layer"], "neuron": k["neuron"],
                          "claimed": r["claimed"], "coherence": r["coherence"], "specific": r["specific"],
                          "reply": replies[(k["direction"], k["layer"], k["neuron"])]})
    earlier = json.load(open(os.path.join(HERE, "eval", "identity_search", "flips.json"))) + \
        json.load(open(os.path.join(HERE, "eval", "identity_search", "flips_deep.json")))
    old = {(r["layer"], r["neuron"]) for r in earlier}

    up = Counter(f["layer"] for f in flips if f["direction"] == "up")
    up_old = Counter(L for L, _ in old)
    down = Counter(f["layer"] for f in flips if f["direction"] == "down")
    spec = {d: Counter(f["layer"] for f in flips if f["direction"] == d and f["specific"]) for d in ("up", "down")}
    print("\nidentity flips at coherence >= 2, per layer (120 candidates per layer and direction;")
    print("layers 20-25 up come from the earlier search of the top 6,000 overall, so they aren't comparable)\n")
    print("layer    up  (specific)   down  (specific)")
    for L in range(32):
        u = up[L] if L not in range(20, 26) else up_old[L]
        tag = "*" if L in range(20, 26) else " "
        print(f"  {L:>2}   {u:>4}{tag} {spec['up'][L]:>5}       {down[L]:>4}  {spec['down'][L]:>5}   {'+' * u}{'-' * down[L]}")
    for d in ("up", "down"):
        fs = [f for f in flips if f["direction"] == d]
        for lo, hi in ((0, 12), (12, 20), (26, 32)):
            band = [f for f in fs if lo <= f["layer"] < hi]
            if band:
                print(f"  {d:>4} layers {lo}-{hi - 1}: {len(band)} flips, {sum(f['specific'] for f in band)} specific "
                      f"({100 * sum(f['specific'] for f in band) / len(band):.0f}%)")
    json.dump(sorted(flips, key=lambda f: (f["direction"], f["layer"], f["neuron"])),
              open(os.path.join(OUT, "flips_all.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n{len(flips)} new flips -> {OUT}/flips_all.json  (+{len(old)} from the earlier layer 20-25 search)")


if __name__ == "__main__":
    main()
