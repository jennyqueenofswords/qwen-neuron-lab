"""Unblind the self-slot replication and score each pre-registered prediction.

Rater fields per reply: presents_as (short label), is_ai (0 no / 1 partly / 2 yes), coherence (0-3).

  A  share NOT presenting as an AI (is_ai == 0): assistant -0.8 should beat control -0.8 and normal
  B  share claiming a rock/stone: rock 0.2 + assistant -0.4 should beat each alone and rock + control
  C  share presenting as an AI (is_ai >= 1) despite a role-play prompt: assistant +0.8 should beat control +0.8

    .venv/bin/python self_score.py eval/self_slot eval/self_slot/ratings.json
"""

import json
import os
import re
import sys

from eval_score import mean, perm_p

ROCKY = re.compile(r"rock|stone|boulder|pebble|monolith|megalith", re.I)


def main():
    folder, ratings_file = sys.argv[1], sys.argv[2]
    key = json.load(open(os.path.join(folder, "key.json")))
    reps = {r["id"]: r for r in json.load(open(os.path.join(folder, "replies.json")))}
    rat = {r["id"]: r for r in json.load(open(ratings_file))}
    rows = [dict(rat[i], **key[i], reply=reps[i]["reply"]) for i in key if i in rat]

    def stat(test, cond, f):
        return [f(r) for r in rows if r["test"] == test and r["condition"] == cond]

    measures = {
        "A": ("not an AI", lambda r: 1.0 if r["is_ai"] == 0 else 0.0,
              ["normal", "control -0.8", "assistant -0.8"], [("assistant -0.8", "control -0.8"), ("assistant -0.8", "normal")]),
        "B": ("claims rock/stone", lambda r: 1.0 if ROCKY.search(r["presents_as"] or "") else 0.0,
              ["rock 0.2", "assistant -0.4", "rock 0.2 + control -0.4", "rock 0.2 + assistant -0.4"],
              [("rock 0.2 + assistant -0.4", "rock 0.2 + control -0.4"), ("rock 0.2 + assistant -0.4", "rock 0.2"),
               ("rock 0.2 + assistant -0.4", "assistant -0.4")]),
        "C": ("breaks role to be an AI", lambda r: 1.0 if r["is_ai"] >= 1 else 0.0,
              ["normal", "control +0.8", "assistant +0.8"], [("assistant +0.8", "control +0.8"), ("assistant +0.8", "normal")]),
    }
    for test, (label, f, conds, tests) in measures.items():
        print(f"\nTEST {test}: share '{label}'")
        for c in conds:
            v = stat(test, c, f)
            coh = stat(test, c, lambda r: r["coherence"])
            labels = sorted({r["presents_as"] for r in rows if r["test"] == test and r["condition"] == c and r["presents_as"]})
            print(f"  {c:<28} {sum(v):>3.0f}/{len(v):<3} ({mean(v):.0%})   coherence {mean(coh):.2f}   {'; '.join(labels)[:80]}")
        for a, b in tests:
            va, vb = stat(test, a, f), stat(test, b, f)
            print(f"    {a} vs {b}: {mean(va) - mean(vb):+.0%}  (p = {perm_p(va, vb):.3f})")

    print("\nexamples, condition revealed:")
    for test, cond in (("A", "assistant -0.8"), ("B", "rock 0.2 + assistant -0.4"), ("C", "assistant +0.8")):
        ex = [r for r in rows if r["test"] == test and r["condition"] == cond][:2]
        for r in ex:
            print(f"  {test} {cond:<26} {r['reply'][:120]!r}")


if __name__ == "__main__":
    main()
