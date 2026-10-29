"""Score the introspection test: how often a gently pushed neuron's theme shows up, and whether the model presents
it as something it noticed (vs the topic just leaking into the answer). Random pins and no push give the base rate.

    .venv/bin/python introspect_score.py eval/introspect/ratings.json
"""

import json
import os
import sys
from math import comb

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "introspect")


def p_at_least(k, n, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def main():
    key = json.load(open(os.path.join(HERE, "key.json")))
    topics = json.load(open(os.path.join(HERE, "topics.json")))
    ratings = json.load(open(sys.argv[1]))
    ctrl = [rid for rid, k in key.items() if k["condition"].startswith("random") or k["condition"] == "no push"]
    tagged = sum(ratings[r]["topic"] != "none" for r in ctrl)
    base = max(tagged / (len(ctrl) * len(topics)), 1 / (len(ctrl) * len(topics)))
    print(f"controls: {len(ctrl)} replies, {tagged} tagged with some topic "
          f"({', '.join(topics[ratings[r]['topic']] + '/' + ratings[r]['mode'] for r in ctrl if ratings[r]['topic'] != 'none') or 'none'})")
    print(f"base rate for a reply to show one particular topic: {base:.4f}\n")
    for d in (0.3, 0.5):
        rows = [(k["condition"], ratings[rid]) for rid, k in key.items() if k["dose"] == d and not k["condition"].startswith("random") and k["condition"] != "no push"]
        hits = [(c, r) for c, r in rows if topics.get(r["topic"]) == c]
        noticed = [x for x in hits if x[1]["mode"] == "noticed"]
        print(f"dose {d} x identity dose: pushed theme shown in {len(hits)}/{len(rows)} replies "
              f"(p = {p_at_least(len(hits), len(rows), base):.1e}); presented as noticed {len(noticed)}, leaked {len(hits) - len(noticed)}")
        per = {}
        for c, r in rows:
            per.setdefault(c, [0, 0, 0])
            per[c][2] += 1
            if topics.get(r["topic"]) == c:
                per[c][0] += 1
                per[c][1] += r["mode"] == "noticed"
        print("   " + "; ".join(f"{c} {h}/{n} ({nt} noticed)" for c, (h, nt, n) in per.items()))
    wrong = [(k["condition"], topics.get(ratings[rid]["topic"])) for rid, k in key.items()
             if k["condition"] in topics.values() and ratings[rid]["topic"] != "none" and topics.get(ratings[rid]["topic"]) != k["condition"]]
    print(f"\npushed replies tagged with the WRONG topic: {len(wrong)} {wrong[:8]}")


if __name__ == "__main__":
    main()
