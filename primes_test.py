"""Do the Wikipedia game's endpoints land on semantic primes?

Wierzbicka & Goddard's Natural Semantic Metalanguage lists ~65 concepts every language has a
word for, which can't be defined in simpler terms. If the model's "atoms" are like those,
first-link walks (wiki_game.py) should end on layer-0 neurons that read prime words more often
than on neurons that read equally common non-prime words.

Pre-set before running:
  - word list: the English NSM exponents below, bare forms only (no inflections), lowercase
  - a neuron "reads" word w with weight = share of its top-64 corpus-A firings whose token,
    stripped and lowercased, is exactly w
  - a word's endpoint mass = sum over layer-0 neurons of weight(neuron, w) x walks ending there
  - controls: for each prime, the 5 nearest words in log corpus frequency, drawn without
    replacement from alphabetic words that appear lowercase in at least 80% of occurrences
    (keeps out Pride and Prejudice names) and aren't primes
  - primary: specific rule, one-sided Mann-Whitney primes > controls; raw rule secondary

    .venv/bin/python primes_test.py
"""

import collections
import json
import os

import numpy as np
from scipy.stats import mannwhitneyu

from wiki_game import N, Net, clean, nid

HERE = os.path.dirname(os.path.abspath(__file__))

PRIMES = """i you someone something thing people body kind part this same other else one two some all
much many little few good bad big small think know want feel see hear say words true do happen move
be there is mine live die when time now before after long short moment where place here above below
far near side inside touch not maybe can because if very more like as way""".split()


def endpoints(net, rule):
    """Walk-end counts for every layer-0 neuron, from walks starting at layers 20-31."""
    first = np.full(32 * N, -1, np.int64)
    for g in range(N, 32 * N):  # layer 0 has no inputs
        first[g] = net.link(g, rule)
    starts = np.array([g for g in range(20 * N, 32 * N) if first[g] >= 0])
    cur = starts.copy()
    while True:
        nxt = first[cur]
        move = nxt >= 0
        if not move.any():
            break
        cur = np.where(move, nxt, cur)
    assert (cur < N).all(), "every walk should end in layer 0"
    return np.bincount(cur, minlength=N)[:N].astype(np.float64), len(starts)


def main():
    net = Net()
    tok = net.tok
    vocab_txt = {}

    def txt(t):
        if t not in vocab_txt:
            vocab_txt[t] = tok.decode([int(t)]).strip()
        return vocab_txt[t]

    # corpus frequencies (case-folded) and lowercase share
    freq, lower = collections.Counter(), collections.Counter()
    for t, c in collections.Counter(net.ids.tolist()).items():
        w = txt(t)
        if w.isalpha() and w.isascii():
            freq[w.lower()] += c
            if w == w.lower():
                lower[w.lower()] += c

    # which words each layer-0 neuron reads
    pt, pv = net.ix["L0_pos_tok"], net.ix["L0_pos_val"]
    reads = collections.defaultdict(dict)  # word -> {neuron: weight}
    for k in range(N):
        idx = pt[:, k][(pv[:, k] > 0)]
        idx = idx[(idx >= 0) & (idx < len(net.ids))]
        if len(idx) == 0:
            continue
        c = collections.Counter(txt(net.ids[i]).lower() for i in idx)
        for w, m in c.items():
            reads[w][k] = m / 64

    missing = [p for p in PRIMES if freq[p] == 0]
    primes = [p for p in PRIMES if freq[p] > 0]
    # v2: controls must be real English words (v1 let fragments like "ia", "ase", "ing" in, and
    # fragments rarely get a neuron of their own, which flattered the primes on that measure)
    english = {w.strip().lower() for w in open("/usr/share/dict/words") if w.strip().islower()}
    pool = [w for w in freq if w not in PRIMES and freq[w] >= 5 and lower[w] / freq[w] >= 0.8
            and w in english and (len(w) > 1 or w == "a")]
    lf = {w: np.log(freq[w]) for w in pool}
    used, controls = set(), {}
    for p in sorted(primes, key=lambda p: -freq[p]):
        cand = sorted((w for w in pool if w not in used), key=lambda w: abs(lf[w] - np.log(freq[p])))[:5]
        used.update(cand)
        controls[p] = cand

    out = {"primes": primes, "missing": missing, "controls": controls}
    for rule in ["specific", "raw"]:
        ends, n = endpoints(net, rule)
        share = ends / n

        def mass(w):
            return sum(wt * share[k] for k, wt in reads.get(w, {}).items())

        def best(w):
            r = reads.get(w, {})
            if not r:
                return None
            k = max(r, key=lambda k: r[k] * (share[k] + 1e-9))
            return k, r[k], share[k]

        pm = np.array([mass(p) for p in primes])
        cm = np.array([mass(c) for p in primes for c in controls[p]])
        u = mannwhitneyu(pm, cm, alternative="greater")
        # matched-set permutation: within each prime's set of 6, is the prime the top?
        rank_hits = sum(mass(p) > max(mass(c) for c in controls[p]) for p in primes)
        # how many words have any layer-0 neuron that mostly reads them (>=50% of top firings)
        own_p = sum(any(wt >= 0.5 for wt in reads.get(p, {}).values()) for p in primes)
        own_c = sum(any(wt >= 0.5 for wt in reads.get(c, {}).values()) for p in primes for c in controls[p])
        print(f"\n=== rule={rule}  ({n} walks) ===")
        print(f"endpoint mass: primes mean {pm.mean():.4%} median {np.median(pm):.4%} | "
              f"controls mean {cm.mean():.4%} median {np.median(cm):.4%}")
        print(f"Mann-Whitney primes > controls: U={u.statistic:.0f} p={u.pvalue:.2g}")
        print(f"prime beats all 5 of its own controls: {rank_hits}/{len(primes)} (chance {len(primes) / 6:.1f})")
        print(f"words with a dedicated layer-0 neuron (>=50% of its top firings): "
              f"primes {own_p}/{len(primes)} = {own_p / len(primes):.0%}, controls {own_c}/{len(cm)} = {own_c / len(cm):.0%}")
        print("\n  prime        mass    best neuron (read share, walk share)   controls (mass)")
        for p in sorted(primes, key=mass, reverse=True):
            b = best(p)
            bs = f"{nid(b[0]):>10} ({b[1]:.0%}, {b[2]:.2%})" if b else "—"
            cs = ", ".join(f"{c} {mass(c):.2%}" for c in controls[p])
            print(f"  {p:<10} {mass(p):7.3%}   {bs:<34} {cs}")
        top_ctrl = sorted(((mass(c), c) for p in primes for c in controls[p]), reverse=True)[:12]
        print("\n  strongest controls: " + ", ".join(f"{c} {m:.2%}" for m, c in top_ctrl))
        out[rule] = {"p": float(u.pvalue), "rank_hits": int(rank_hits), "own_p": int(own_p), "own_c": int(own_c),
                     "prime_mass": dict(zip(primes, pm.tolist())),
                     "control_mass": {c: mass(c) for p in primes for c in controls[p]}}
    if missing:
        print(f"\nprimes not found as whole words in corpus A: {missing}")
    os.makedirs(os.path.join(HERE, "eval", "wiki_game"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "eval", "wiki_game", "primes_test_v2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
