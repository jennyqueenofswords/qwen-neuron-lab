"""Score the blind ratings of edit_concepts.py's definitions (Hunt 36).

    .venv/bin/python score_edits.py
"""

import collections
import json
import os

from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "eval", "edit_concepts")
CONDS = ["base", "remove", "swap", "add", "control"]


def main():
    G = json.load(open(os.path.join(D, "generations.json")))["items"]
    key = json.load(open(os.path.join(D, "rate", "key.json")))
    rated = {}
    for b in range(4):
        for r in json.load(open(os.path.join(D, "rate", f"out{b}.json"))):
            rated[r["id"]] = r
    rows = []
    for iid, i in key.items():
        if iid in rated:
            rows.append({**G[i], **{k: rated[iid][k] for k in ("identity", "property")}})
    print(f"{len(rows)} of {len(key)} definitions rated")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["cond"]].append(r)
    cats = ["same", "variant", "replaced", "broken"]
    props = ["A", "B", "both", "neither"]
    print(f"\n{'condition':9} " + " ".join(f"{c:>9}" for c in cats) + " |" + " ".join(f"{p:>8}" for p in props))
    for c in CONDS:
        n = len(by[c])
        ic = collections.Counter(r["identity"] for r in by[c])
        pc = collections.Counter(r["property"] for r in by[c])
        print(f"{c:9} " + " ".join(f"{ic[k] / n:9.0%}" for k in cats) + " |" + " ".join(f"{pc[p] / n:8.0%}" for p in props))

    def test(a, b, cat, alt="greater"):
        x = sum(r["identity"] == cat for r in by[a]); y = sum(r["identity"] == cat for r in by[b])
        f = fisher_exact([[x, len(by[a]) - x], [y, len(by[b]) - y]], alternative=alt)
        return x, len(by[a]), y, len(by[b]), f.pvalue

    x, n1, y, n2, p = test("swap", "add", "variant")
    print(f"\nprimary, coherent variants: swap {x}/{n1} vs add {y}/{n2}, one-sided Fisher p = {p:.3g}")
    x, n1, y, n2, p = test("swap", "control", "variant")
    print(f"secondary, swap {x}/{n1} vs control {y}/{n2} variants, p = {p:.3g}")
    ch = lambda r: r["identity"] != "same"
    xb = sum(map(ch, by["remove"])); yb = sum(map(ch, by["base"]))
    f = fisher_exact([[xb, len(by["remove"]) - xb], [yb, len(by["base"]) - yb]], alternative="greater")
    print(f"secondary, changed identity: remove {xb}/{len(by['remove'])} vs base {yb}/{len(by['base'])}, p = {f.pvalue:.3g}")

    print("\nVARIANTS FROM SWAPS:")
    for r in by["swap"]:
        if r["identity"] == "variant":
            print(f"  {r['word']:>12} {r['p']}→{r['q']}: {r['definition']}")
    print("\nVARIANTS FROM ADDING ONLY:")
    for r in by["add"]:
        if r["identity"] == "variant":
            print(f"  {r['word']:>12} +{r['q']}: {r['definition']}")
    json.dump(rows, open(os.path.join(D, "rated.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
