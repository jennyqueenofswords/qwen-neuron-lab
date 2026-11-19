"""Score the blind ratings of metaphor.py's causal test (Hunt 37).

    .venv/bin/python score_metaphor.py        # first run
    .venv/bin/python score_metaphor.py v2     # the replication (fiction prompts, 32 samples)
"""

import collections
import json
import os
import re
import sys

import numpy as np
from scipy.stats import mannwhitneyu

from metaphor import CONDS, DIMS

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "eval", "metaphor")
V2 = len(sys.argv) > 1 and sys.argv[1] == "v2"
GEN, RATE, NB = (("causal_generations_v2.json", "rate_v2", 8) if V2 else ("causal_generations.json", "rate", 4))
if V2:
    CONDS = ["source", "opposite", "wet", "dry"]


def main():
    R = json.load(open(os.path.join(D, GEN)))
    key = json.load(open(os.path.join(D, RATE, "key.json")))
    sc = {}
    for b in range(NB):
        for r in json.load(open(os.path.join(D, RATE, f"out{b}.json"))):
            sc[r["id"]] = r["score"]
    for iid, i in key.items():
        R[i]["score"] = sc.get(iid, "na")
    leak = {d[2]: re.compile(d[5], re.I) for d in DIMS}
    for r in R:
        r["leak"] = bool(leak[r["dim"]].search(r["text"]))
    rows = [r for r in R if r["score"] != "na"]
    print(f"{len(rows)} of {len(R)} rated (the rest 'na': refusals, loops, off-topic); "
          f"{sum(not r['leak'] for r in rows)} use no word of the source domain")
    # z-score per dimension
    for dim in {r["dim"] for r in rows}:
        v = np.array([float(r["score"]) for r in rows if r["dim"] == dim])
        mu, sd = v.mean(), v.std() or 1
        for r in rows:
            if r["dim"] == dim:
                r["z"] = (float(r["score"]) - mu) / sd
    print(f"\n{'trait':13} {'source/opp':16} " + " ".join(f"{c:>9}" for c in CONDS) + "   (mean 1-7, n)")
    for src, opp, dim, *_ in DIMS:
        cells = []
        for c in CONDS:
            v = [float(r["score"]) for r in rows if r["dim"] == dim and r["cond"] == c]
            cells.append(f"{np.mean(v):5.2f} {len(v):>2}" if v else "    —   ")
        print(f"{dim:13} {src + '/' + opp:16} " + " ".join(f"{x:>9}" for x in cells))

    def compare(a, b, sub, alt):
        x = [r["z"] for r in sub if r["cond"] == a]
        y = [r["z"] for r in sub if r["cond"] == b]
        p = mannwhitneyu(x, y, alternative=alt).pvalue if x and y else float("nan")
        return np.mean(x) - np.mean(y), len(x), len(y), p

    d, n1, n2, p = compare("source", "opposite", rows, "greater")
    print(f"\nprimary, source vs opposite (all dimensions, z): difference {d:+.2f} SD, n {n1}/{n2}, one-sided p = {p:.3g}")
    clean = [r for r in rows if not r["leak"]]
    d, n1, n2, p = compare("source", "opposite", clean, "greater")
    print(f"secondary, same without any source-domain word: {d:+.2f} SD, n {n1}/{n2}, p = {p:.3g}")
    d, n1, n2, p = compare("wet", "dry", rows, "two-sided")
    print(f"secondary, wet vs dry (control): {d:+.2f} SD, n {n1}/{n2}, two-sided p = {p:.3g}")
    if not V2:
        for c in ("source", "opposite"):
            d, n1, n2, p = compare(c, "none", rows, "two-sided")
            print(f"  {c} vs no edit: {d:+.2f} SD, two-sided p = {p:.3g}")
    print("\nper dimension, source minus opposite (z), all / leak-free:")
    for src, opp, dim, *_ in DIMS:
        sub = [r for r in rows if r["dim"] == dim]
        a = compare("source", "opposite", sub, "greater")
        b = compare("source", "opposite", [r for r in sub if not r["leak"]], "greater")
        print(f"  {dim:13} {a[0]:+.2f} (n {a[1]}/{a[2]})   {b[0]:+.2f} (n {b[1]}/{b[2]})")
    print("\nLEAK-FREE EXAMPLES (source | opposite):")
    for src, opp, dim, *_ in DIMS:
        s = [r for r in clean if r["dim"] == dim and r["cond"] == "source"]
        o = [r for r in clean if r["dim"] == dim and r["cond"] == "opposite"]
        if s and o:
            s, o = max(s, key=lambda r: r["z"]), min(o, key=lambda r: r["z"])
            print(f"  {dim} [{src}] {s['score']}: {s['text'][:150]}")
            print(f"  {dim} [{opp}] {o['score']}: {o['text'][:150]}")
    json.dump(R, open(os.path.join(D, "rated_v2.json" if V2 else "rated.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
