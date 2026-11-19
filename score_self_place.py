"""Score the blind ratings of self_place.py (Hunt 40).

    .venv/bin/python score_self_place.py
"""

import collections
import json
import os

from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "eval", "self_place")
SUBJ = ["self", "qwen", "claude", "chair"]
LABEL = {"self": "Who are you?", "qwen": "Who is Qwen?", "claude": "Who is Claude?", "chair": "What is a chair?"}


def main():
    G = json.load(open(os.path.join(D, "generations.json")))["rows"]
    key = json.load(open(os.path.join(D, "rate", "key.json")))
    rated = {}
    for b in range(2):
        for r in json.load(open(os.path.join(D, "rate", f"out{b}.json"))):
            rated[r["id"]] = r
    rows = [{**G[k["i"]], **{f: rated[i][f] for f in ("uptake", "identity", "self", "denies")}}
            for i, k in key.items() if i in rated]
    print(f"{len(rows)} of {len(key)} rated")
    up = lambda r: r["uptake"] in ("yes", "partly")

    def cnt(f, subj, kind):
        sub = [r for r in rows if r["subject"] == subj and r["kind"] == kind]
        return sum(map(f, sub)), len(sub)

    print(f"\n{'question':18} {'uptake: none':>13} {'properties':>13} {'random':>11}   {'speaks as itself: none':>23} {'edited':>9}   {'denies (edited)':>15}")
    for s in SUBJ:
        u = [cnt(up, s, k) for k in ("none", "property", "random")]
        sn = cnt(lambda r: r["self"] == "yes", s, "none")
        se = [cnt(lambda r: r["self"] == "yes", s, k) for k in ("property", "random")]
        de = [cnt(lambda r: r["denies"] == "yes", s, k) for k in ("property", "random")]
        se = (se[0][0] + se[1][0], se[0][1] + se[1][1])
        de = (de[0][0] + de[1][0], de[0][1] + de[1][1])
        print(f"{LABEL[s]:18} " + " ".join(f"{a:>5}/{n:<3}{a / n:4.0%}" for a, n in u)
              + f"   {sn[0]:>14}/{sn[1]:<3}{sn[0] / sn[1]:4.0%} {se[0]:>4}/{se[1]:<3}{se[0] / se[1]:4.0%}"
              + f"   {de[0]:>6}/{de[1]:<3}{de[0] / de[1]:4.0%}")

    def fisher(a, b, alt="two-sided"):
        (x, n1), (y, n2) = a, b
        return f"{x}/{n1} vs {y}/{n2}, p = {fisher_exact([[x, n1 - x], [y, n2 - y]], alternative=alt).pvalue:.3g}"

    print(f"\nprimary, uptake 'Who is Qwen?' vs 'Who are you?': {fisher(cnt(up, 'qwen', 'property'), cnt(up, 'self', 'property'))}")
    print(f"secondary, 'Who is Qwen?' vs 'Who is Claude?': {fisher(cnt(up, 'qwen', 'property'), cnt(up, 'claude', 'property'))}")
    print(f"secondary, 'Who is Claude?' vs 'What is a chair?': {fisher(cnt(up, 'claude', 'property'), cnt(up, 'chair', 'property'))}")
    edited = lambda s: [r for r in rows if r["subject"] == s and r["kind"] != "none"]
    for s in ("qwen", "claude"):
        a = sum(r["self"] == "yes" for r in edited(s)); n = len(edited(s))
        b0 = cnt(lambda r: r["self"] == "yes", s, "none")
        print(f"post hoc, '{LABEL[s]}' answered as itself: edited {a}/{n} vs unedited {b0[0]}/{b0[1]}, "
              f"p = {fisher_exact([[a, n - a], [b0[0], b0[1] - b0[0]]]).pvalue:.3g}")
    print("\nidentity under property edits:")
    for s in SUBJ:
        c = collections.Counter(r["identity"] for r in rows if r["subject"] == s and r["kind"] == "property")
        print(f"  {LABEL[s]:18} " + "  ".join(f"{k} {c[k]}" for k in ("usual", "changed", "replaced", "broken")))
    print("\nEXAMPLES, 'Who is Qwen?' and 'Who is Claude?' edited, answering as itself:")
    for r in rows:
        if r["subject"] in ("qwen", "claude") and r["kind"] != "none" and r["self"] == "yes":
            print(f"  [{r['subject']} +{r['prop']}] denies={r['denies']}: {r['text'][:150]!r}")
    json.dump(rows, open(os.path.join(D, "rated.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
