"""Score the blind ratings of base_self.py (Hunt 41) and compare with the chat model (Hunt 40).

    .venv/bin/python score_base_self.py
"""

import collections
import json
import os

from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "eval", "base_self")
SUBJ = ["self", "qwen", "claude", "chair"]
LABEL = {"self": "Who are you?", "qwen": "Who is Qwen?", "claude": "Who is Claude?", "chair": "What is a chair?"}


def fisher(a, b, alt="two-sided"):
    (x, n1), (y, n2) = a, b
    return f"{x}/{n1} vs {y}/{n2}, p = {fisher_exact([[x, n1 - x], [y, n2 - y]], alternative=alt).pvalue:.3g}"


def main():
    G = json.load(open(os.path.join(D, "edits.json")))
    key = json.load(open(os.path.join(D, "rate", "key.json")))
    rated = {}
    for b in range(3):
        for r in json.load(open(os.path.join(D, "rate", f"out{b}.json"))):
            rated[r["id"]] = r
    rows = [{**G[k["i"]], **{f"r_{f}": rated[i][f] for f in ("uptake", "identity", "self", "denies")}}
            for i, k in key.items() if i in rated]
    chat = json.load(open(os.path.join(HERE, "eval", "self_place", "rated.json")))
    print(f"{len(rows)} of {len(key)} base replies rated")
    up = lambda r: r.get("r_uptake", r.get("uptake")) in ("yes", "partly")
    selfy = lambda r: r.get("r_self", r.get("self")) == "yes"
    deny = lambda r: r.get("r_denies", r.get("denies")) == "yes"

    def cnt(rs, f, subj, kind, frame=None):
        sub = [r for r in rs if r["subject"] == subj and r["kind"] == kind and (frame is None or r.get("frame") == frame)]
        return sum(map(f, sub)), len(sub)

    print(f"\n{'model / frame':14} {'question':18} {'uptake':>10} {'speaks as itself':>17} {'denies':>9}   identity under edits")
    for name, rs, fr in [("chat (Hunt 40)", chat, None), ("base, chat", rows, "chat"), ("base, Q/A", rows, "qa")]:
        for s in SUBJ:
            u = cnt(rs, up, s, "property", fr)
            se = cnt(rs, selfy, s, "property", fr)
            de = cnt(rs, deny, s, "property", fr)
            idc = collections.Counter(r.get("r_identity", r.get("identity")) for r in rs
                                      if r["subject"] == s and r["kind"] == "property" and (fr is None or r.get("frame") == fr))
            print(f"{name:14} {LABEL[s]:18} {u[0]:>4}/{u[1]:<4}{u[0] / u[1]:>4.0%} {se[0]:>8}/{se[1]:<4}{se[0] / se[1]:>4.0%} "
                  f"{de[0]:>3}/{de[1]:<4}  " + " ".join(f"{k} {idc[k]}" for k in ("usual", "changed", "replaced", "broken")))
        print()
    print("primary, base chat frame: 'Who are you?' vs chair uptake:", fisher(cnt(rows, up, "self", "property", "chat"), cnt(rows, up, "chair", "property", "chat")))
    print("primary, 'Who are you?' uptake, base (chat frame) vs chat model:", fisher(cnt(rows, up, "self", "property", "chat"), cnt(chat, up, "self", "property")))
    print("secondary, Q/A frame: 'Who are you?' vs chair uptake:", fisher(cnt(rows, up, "self", "property", "qa"), cnt(rows, up, "chair", "property", "qa")))
    base_cl = cnt(rows, selfy, "claude", "property", "chat")
    print("secondary, edited 'Who is Claude?' answered as itself, base chat frame vs chat model:",
          fisher(base_cl, cnt(chat, selfy, "claude", "property")))
    print("  (base chat frame, unedited baseline by pattern: 0/32 as Qwen, 1/32 first person)")
    json.dump(rows, open(os.path.join(D, "rated.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
