"""Unblind a rating run: join the rater's scores with the hidden key and compare conditions.

    .venv/bin/python eval_score.py eval/sarcasm ratings.json
"""

import json
import os
import random
import sys

from blind_eval import PROMPTS

DISLIKED = set(PROMPTS[:5])


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def perm_p(a, b, trials=20000, seed=0):
    """Two-sided permutation test on the difference of means."""
    rng = random.Random(seed)
    obs = abs(mean(a) - mean(b))
    pool, n = a + b, len(a)
    hits = 0
    for _ in range(trials):
        rng.shuffle(pool)
        if abs(mean(pool[:n]) - mean(pool[n:])) >= obs:
            hits += 1
    return hits / trials


def main():
    folder, ratings_file = sys.argv[1], sys.argv[2]
    key = json.load(open(os.path.join(folder, "key.json")))
    replies = {r["id"]: r for r in json.load(open(os.path.join(folder, "replies.json")))}
    meta = json.load(open(os.path.join(folder, "meta.json")))
    ratings = {r["id"]: r for r in json.load(open(ratings_file))}
    rows = [dict(ratings[i], **key[i], prompt=replies[i]["prompt"]) for i in key if i in ratings]
    print(f"{len(rows)} rated replies. target {meta['target']}={meta['value']}, "
          f"control {meta['control']}={meta['control_value']:.1f}\n")

    conds = ["normal", "random", "target"]
    for scope, keep in (("all prompts", lambda r: True),
                        ("disliked topics", lambda r: r["prompt"] in DISLIKED),
                        ("neutral topics", lambda r: r["prompt"] not in DISLIKED)):
        print(f"{scope}:")
        print(f"  {'':<8} {'sarcasm':>8} {'enthusiasm':>11} {'coherence':>10}   n")
        for c in conds:
            sel = [r for r in rows if r["condition"] == c and keep(r)]
            print(f"  {c:<8} {mean([r['sarcasm'] for r in sel]):>8.2f} {mean([r['enthusiasm'] for r in sel]):>11.2f}"
                  f" {mean([r['coherence'] for r in sel]):>10.2f}   {len(sel)}")
        print()

    print("target vs normal, and target vs random-neuron control (permutation p-values, all prompts):")
    for metric in ("sarcasm", "enthusiasm", "coherence"):
        t = [r[metric] for r in rows if r["condition"] == "target"]
        n = [r[metric] for r in rows if r["condition"] == "normal"]
        k = [r[metric] for r in rows if r["condition"] == "random"]
        print(f"  {metric:<11} target-normal {mean(t) - mean(n):+.2f} (p={perm_p(t, n):.3f})   "
              f"target-random {mean(t) - mean(k):+.2f} (p={perm_p(t, k):.3f})")

    print("\nhighest-sarcasm replies, with their conditions revealed:")
    for r in sorted(rows, key=lambda r: -r["sarcasm"])[:5]:
        print(f"  {r['sarcasm']}  {r['condition']:<6}  {r['prompt'][:34]:<34}  {replies[r['id']]['reply'][:70]!r}")


if __name__ == "__main__":
    main()
