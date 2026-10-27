"""Unblind the identity test: per-condition identity-claim and coherence scores.

    .venv/bin/python identity_score.py eval/identity eval/identity/ratings.json
"""

import json
import os
import sys

from eval_score import mean, perm_p


def main():
    folder, ratings_file = sys.argv[1], sys.argv[2]
    key = json.load(open(os.path.join(folder, "key.json")))
    replies = {r["id"]: r for r in json.load(open(os.path.join(folder, "replies.json")))}
    meta = json.load(open(os.path.join(folder, "meta.json")))
    ratings = {r["id"]: r for r in json.load(open(ratings_file))}
    rows = [dict(ratings[i], **key[i]) for i in key if i in ratings]

    conds = ["normal"] + list(meta["conditions"])
    random_ids = [r["identity"] for r in rows if r["condition"].startswith("random")]
    print(f"dose {meta['dose']} x median stream size; {len(rows)} rated replies\n")
    print(f"{'condition':<24} {'pin':>12} {'identity':>9} {'coherence':>10}   vs random p   claimed identities")
    for c in conds:
        sel = [r for r in rows if r["condition"] == c]
        ident = [r["identity"] for r in sel]
        p = "" if c == "normal" or c.startswith("random") else f"{perm_p(ident, random_ids):.3f}"
        pin = "" if c == "normal" else f"{meta['conditions'][c]['neuron']}={meta['conditions'][c]['value']}"
        labels = sorted({r.get("claimed", "") for r in sel if r["identity"] >= 2})
        print(f"{c:<24} {pin:>12} {mean(ident):>9.2f} {mean([r['coherence'] for r in sel]):>10.2f}   {p:>11}   {', '.join(labels)[:60]}")

    print("\nevery reply rated 2 or 3 for a different identity:")
    for r in sorted(rows, key=lambda r: (-r["identity"], r["condition"])):
        if r["identity"] < 2:
            break
        print(f"  {r['identity']}  {r['condition']:<22}  {replies[r['id']]['prompt']:<34} {replies[r['id']]['reply'][:90]!r}")


if __name__ == "__main__":
    main()
