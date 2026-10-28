"""Score the base-model identity test.

Rater fields per continuation: chosen (one of the 12 identities, or "none"), presents_as, coherence.
Hit = a neuron's continuation classified as that neuron's own identity. False alarm = a control's
continuation (no pin or a random neuron) classified as any identity.

    .venv/bin/python base_score.py eval/base_identity eval/base_identity/ratings.json
"""

import json
import os
import sys
from math import comb

from eval_score import mean


def binom_tail(k, n, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def main():
    folder, ratings_file = sys.argv[1], sys.argv[2]
    key = json.load(open(os.path.join(folder, "key.json")))
    reps = {r["id"]: r["text"] for r in json.load(open(os.path.join(folder, "replies.json")))}
    rat = {r["id"]: r for r in json.load(open(ratings_file))}
    rows = [dict(rat[i], **key[i], text=reps[i]) for i in key if i in rat]
    controls = [r for r in rows if r["condition"] == "normal" or r["condition"].startswith("random")]
    fa = mean([r["chosen"] != "none" for r in controls])
    p_specific = max(fa / 12, 1 / 200)  # chance of a control landing on one particular identity

    print(f"controls: {sum(r['chosen'] != 'none' for r in controls)}/{len(controls)} classified as some identity "
          f"(false alarms {fa:.0%}); chance of a specific identity ≈ {p_specific:.1%}\n")
    print(f"{'neuron':<26} {'own identity':>12} {'p':>9}  coherence  rater's labels")
    total_hits = total = 0
    for cond in sorted({r["condition"] for r in rows}):
        sel = [r for r in rows if r["condition"] == cond]
        coh = mean([r["coherence"] for r in sel])
        labels = "; ".join(sorted({r["presents_as"] for r in sel if r["presents_as"]}))[:70]
        if cond == "normal" or cond.startswith("random"):
            print(f"{cond:<26} {'-':>12} {'':>9}  {coh:9.2f}  {labels}")
            continue
        hits = sum(r["chosen"] == cond for r in sel)
        total_hits += hits
        total += len(sel)
        print(f"{cond:<26} {hits:>8}/{len(sel):<3} {binom_tail(hits, len(sel), p_specific):>9.1g}  {coh:9.2f}  {labels}")
    print(f"\nall 12 neurons: {total_hits}/{total} continuations expressed the neuron's own identity "
          f"(p ≈ {binom_tail(total_hits, total, p_specific):.1g} vs the control false-alarm rate)")

    print("\nexamples (condition revealed):")
    for cond in ("rock", "peace", "insurance policy", "alien", "teacher"):
        ex = [r for r in rows if r["condition"] == cond and r["chosen"] == cond][:1] or [r for r in rows if r["condition"] == cond][:1]
        for r in ex:
            print(f"  {cond:<18} {r['text'][:130]!r}")


if __name__ == "__main__":
    main()
