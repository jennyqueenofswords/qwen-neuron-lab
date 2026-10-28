"""Score the self-vs-output-words test: for each neuron, how many of its 8 replies the blind rater
tagged with its self theme, with its output theme, and with anything else; controls give the base rate.

    .venv/bin/python mismatch_score.py path/to/ratings.json
"""

import json
import os
import sys
from math import comb

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "mismatch")


def p_at_least(k, n, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def main():
    ratings = json.load(open(sys.argv[1]))
    key = json.load(open(os.path.join(HERE, "key.json")))
    tkey = json.load(open(os.path.join(HERE, "themes.json")))["key"]
    by_cond = {}
    for rid, k in key.items():
        by_cond.setdefault(k["condition"], []).append(ratings[rid])
    controls = [r for c, rs in by_cond.items() if c == "normal" or c.startswith("random") for r in rs]
    # base rate: chance a control reply gets tagged with any one particular theme
    tags = sum(len(r["themes"]) for r in controls)
    base = max(tags / (len(controls) * len(tkey)), 1 / (len(controls) * len(tkey)))
    print(f"controls: {len(controls)} replies, {tags} theme tags in all (base rate per theme {base:.4f})\n")
    print(f"{'neuron':<14} {'self':>5} {'output':>7} {'other':>6} {'garbled':>8}   p(self)   p(output)")
    rows = {}
    for cond, rs in sorted(by_cond.items()):
        if cond == "normal" or cond.startswith("random"):
            continue
        n = len(rs)
        s = sum(any(tkey[t] == {"neuron": cond, "kind": "self"} for t in r["themes"]) for r in rs)
        o = sum(any(tkey[t] == {"neuron": cond, "kind": "output"} for t in r["themes"]) for r in rs)
        other = sum(any(tkey[t]["neuron"] != cond for t in r["themes"]) for r in rs)
        g = sum(r.get("garbled", False) for r in rs)
        ps, po = p_at_least(s, n, base), p_at_least(o, n, base)
        rows[cond] = {"self": s, "output": o, "other": other, "garbled": g, "n": n}
        print(f"{cond:<14} {s:>3}/{n} {o:>5}/{n} {other:>4}/{n} {g:>6}/{n}   {ps:8.1e}  {po:8.1e}")
    for cond, rs in sorted(by_cond.items()):
        if cond == "normal" or cond.startswith("random"):
            hit = [t for r in rs for t in r["themes"]]
            print(f"  control {cond:<11} tagged: {', '.join(tkey[t]['neuron'] + '/' + tkey[t]['kind'] for t in hit) or 'nothing'}"
                  f"   garbled {sum(r.get('garbled', False) for r in rs)}/{len(rs)}")
    S = sum(r["self"] for r in rows.values())
    O = sum(r["output"] for r in rows.values())
    N = sum(r["n"] for r in rows.values())
    print(f"\nall ten: self theme {S}/{N}, output theme {O}/{N}")
    json.dump({"base_rate": base, "neurons": rows}, open(os.path.join(HERE, "scores.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
