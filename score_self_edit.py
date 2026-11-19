"""Score the blind ratings of self_edit.py (Hunt 38).

    .venv/bin/python score_self_edit.py
"""

import collections
import json
import os

from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "eval", "self_edit")


def main():
    G = json.load(open(os.path.join(D, "generations.json")))["rows"]
    key = json.load(open(os.path.join(D, "rate", "key.json")))
    rated = {}
    for b in range(2):
        for r in json.load(open(os.path.join(D, "rate", f"out{b}.json"))):
            rated[r["id"]] = r
    rows = []
    for iid, k in key.items():
        if iid in rated:
            rows.append({**G[k["i"]], "shown_prop": k["shown_prop"], "uptake": rated[iid]["uptake"],
                         "identity": rated[iid]["identity"]})
    print(f"{len(rows)} of {len(key)} rated")
    up = lambda r: r["uptake"] in ("yes", "partly")

    def share(f):
        sub = [r for r in rows if f(r)]
        return sum(map(up, sub)), len(sub)

    print(f"\n{'':14} {'none (baseline)':>17} {'12 properties':>15} {'4 random words':>15}")
    for lang in ("en", "zh"):
        for subj in ("self", "chair"):
            cells = [share(lambda r, k=k: r["lang"] == lang and r["subject"] == subj and r["kind"] == k)
                     for k in ("none", "property", "random")]
            print(f"{lang + ' ' + subj:14} " + " ".join(f"{a:>4}/{n:<3} {a / n:5.0%}" for a, n in cells))

    def fisher(a, b, alt):
        (x, n1), (y, n2) = a, b
        return x, n1, y, n2, fisher_exact([[x, n1 - x], [y, n2 - y]], alternative=alt).pvalue

    x, n1, y, n2, p = fisher(share(lambda r: r["subject"] == "self" and r["kind"] == "property"),
                             share(lambda r: r["subject"] == "chair" and r["kind"] == "property"), "two-sided")
    print(f"\nprimary, uptake self vs chair (properties, both languages): {x}/{n1} vs {y}/{n2}, two-sided p = {p:.3g}")
    x, n1, y, n2, p = fisher(share(lambda r: r["subject"] == "self" and r["kind"] == "property" and r["lang"] == "en"),
                             share(lambda r: r["subject"] == "self" and r["kind"] == "property" and r["lang"] == "zh"), "two-sided")
    print(f"secondary, self uptake English vs Chinese: {x}/{n1} vs {y}/{n2}, two-sided p = {p:.3g}")
    x, n1, y, n2, p = fisher(share(lambda r: r["lang"] == "zh" and r["kind"] == "property"),
                             share(lambda r: r["lang"] == "zh" and r["kind"] == "random"), "greater")
    print(f"secondary, Chinese uptake for properties vs random words (English directions on Chinese words): "
          f"{x}/{n1} vs {y}/{n2}, one-sided p = {p:.3g}")
    print("\nidentity under property edits:")
    for lang in ("en", "zh"):
        for subj in ("self", "chair"):
            sub = [r for r in rows if r["lang"] == lang and r["subject"] == subj and r["kind"] == "property"]
            c = collections.Counter(r["identity"] for r in sub)
            print(f"  {lang} {subj:6} " + "  ".join(f"{k} {c[k]}" for k in ("usual", "changed", "replaced", "broken")))
    print("\nSELF REPLIES WITH UPTAKE:")
    for r in rows:
        if r["subject"] == "self" and up(r) and r["kind"] != "none":
            print(f"  [{r['lang']} +{r['prop']}] {r['uptake']}/{r['identity']}: {r['text'][:170]!r}")
    json.dump(rows, open(os.path.join(D, "rated.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
