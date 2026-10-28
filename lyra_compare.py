"""Do our self neurons agree with Lyra's steer lens? Blind forced choice.

Lyra's neuron-explorer (explorer/cache/steer_index.npz) stores, for every Qwen3.5-4B neuron and both signs, the 12
tokens whose next-token log-probability rises most when the neuron is pushed by ±0.8 x (median residual norm /
arrow length), added at every position of four short contexts. Our selves come from pinning at the same dose and
reading whole replies. For each self (in the direction it was found), a rater sees only Lyra's 12 tokens and four
labels: our claimed self and three decoys drawn from other selves, shuffled. Chance is 1 in 4.

    .venv/bin/python lyra_compare.py build path/to/steer_index.npz   -> eval/lyra_compare/items.json (blind), key.json
    .venv/bin/python lyra_compare.py score eval/lyra_compare/ratings.json
"""

import json
import os
import random
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval", "lyra_compare")


def selves():
    rows = []
    for f in ("flips.json", "flips_deep.json"):
        for r in json.load(open(os.path.join(HERE, "eval", "identity_search", f))):
            rows.append({"layer": r["layer"], "neuron": r["neuron"], "direction": "up", "claimed": r["claimed"], "source": "hunt6"})
    for r in json.load(open(os.path.join(HERE, "eval", "identity_sweep", "flips_all.json"))):
        rows.append({"layer": r["layer"], "neuron": r["neuron"], "direction": r["direction"], "claimed": r["claimed"], "source": "hunt11"})
    seen, out = set(), []
    for r in rows:
        k = (r["layer"], r["neuron"], r["direction"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def build(index_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "models", "Qwen3.5-4B-4bit"))
    st = np.load(index_path, allow_pickle=False)
    rows = selves()
    rng = random.Random(23)
    items, key = [], {}
    for i, r in enumerate(rows):
        ids = st[r["direction"]][r["layer"], r["neuron"]]
        words = [tok.decode([int(t)]).replace("\n", "↵") for t in ids]
        # decoys: other selves' labels, not sharing a word with the true one
        true_words = set(r["claimed"].lower().split())
        pool = [x["claimed"] for x in rows if x is not r and not (set(x["claimed"].lower().split()) & true_words - {"a", "an", "the", "of"})]
        opts = rng.sample(pool, 3) + [r["claimed"]]
        rng.shuffle(opts)
        iid = f"c{i:03d}"
        items.append({"id": iid, "tokens": words, "options": {L: o for L, o in zip("ABCD", opts)}})
        key[iid] = dict(r, answer="ABCD"[opts.index(r["claimed"])],
                        kl=round(float(st[f"{r['direction']}_kl"][r["layer"], r["neuron"]]), 3),
                        kl_rank=int((st[f"{r['direction']}_kl"][r["layer"]] > st[f"{r['direction']}_kl"][r["layer"], r["neuron"]]).sum()) + 1)
    order = list(range(len(items)))
    rng.shuffle(order)
    items = [items[i] for i in order]
    os.makedirs(OUT, exist_ok=True)
    json.dump(items, open(os.path.join(OUT, "items.json"), "w"), ensure_ascii=False, indent=1)
    json.dump(key, open(os.path.join(OUT, "key.json"), "w"), ensure_ascii=False, indent=1)
    print(f"{len(items)} selves -> {OUT}/items.json (blind)")


def p_at_least(k, n, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def score(ratings_path):
    key = json.load(open(os.path.join(OUT, "key.json")))
    ratings = json.load(open(ratings_path))
    groups = {}
    for iid, k in key.items():
        pick = ratings[iid]["pick"]
        band = "early (0-11)" if k["layer"] < 12 else "middle (12-19)" if k["layer"] < 20 else "deep (20-31)"
        for g in ("all", f"{k['direction']}", f"{k['direction']} {band}", k["source"]):
            groups.setdefault(g, []).append((pick == k["answer"], pick == "none"))
    print(f"{'group':<24} {'hit':>9} {'none':>6}   p (chance 1/4)")
    for g in sorted(groups, key=lambda g: (g != "all", g)):
        v = groups[g]
        h = sum(x for x, _ in v)
        print(f"{g:<24} {h:>4}/{len(v):<4} {sum(n for _, n in v):>6}   {p_at_least(h, len(v), 0.25):.1e}")
    ranks = [k["kl_rank"] for k in key.values()]
    print(f"\nLyra's KL rank of our selves within their layer: median {int(np.median(ranks))}, "
          f"top-100 {sum(r <= 100 for r in ranks)}/{len(ranks)}, top-1000 {sum(r <= 1000 for r in ranks)}/{len(ranks)}")
    misses = [(iid, k) for iid, k in key.items() if ratings[iid]["pick"] != k["answer"]]
    json.dump([dict(k, id=iid, pick=ratings[iid]["pick"]) for iid, k in misses],
              open(os.path.join(OUT, "misses.json"), "w"), ensure_ascii=False, indent=1)
    print(f"{len(misses)} misses -> {OUT}/misses.json")


if __name__ == "__main__":
    {"build": build, "score": score}[sys.argv[1]](sys.argv[2])
