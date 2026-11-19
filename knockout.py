"""Are Qwen's primes load-bearing? Knock a concept out and see what built on it breaks (Hunt 33).

Hunts 31-32 found a floor of primes in the model's own definitions. That could be a mirror of how
human dictionaries are written rather than something the model computes with. Causal test:

  knockout   For a word w, a direction per layer: mean residual on w's tokens minus the mean over
             all tokens, in 10 of its dependents' definitions (definitions that use w). Knocking w
             out removes that direction from the residual stream after every layer, at every position.
  damage     The model is asked to define 8 *other* dependents u ("Define the word u…") and is
             teacher-forced through its own definition of u, scored only on the tokens BEFORE w first
             appears, so nothing is lost merely because w got harder to say or read. Each dependent is
             paired with a random definition that doesn't use w, scored on the same number of leading
             tokens. Specific damage S(w) = mean increase in per-token loss on dependents minus on pairs.
  sanity     The log-probability of w's own first token at the point the definition says it should
             fall under the knockout, for primes and controls alike.

Pre-set before running: 50 primes drawn at random (seed 0) from those with at least 18 usable
dependents (w after at least 3 tokens of definition); 50 controls, non-prime headwords with the same
eligibility, nearest in Qwen vocabulary rank, without replacement. Primary: one-sided Mann-Whitney,
S(primes) > S(controls). Secondary: the same restricted to knockouts that took (w's own token drops
by at least 2 nats; added after a 3+3 smoke test showed function-word knockouts barely remove the
word), and general damage (the pairs' loss increase), two-sided.

    .venv/bin/python knockout.py [--n 50]
"""

import argparse
import collections
import json
import os
import random
import re
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np
from mlx_lm import load
from scipy.stats import mannwhitneyu
from transformers import AutoTokenizer

import selfdict as sd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval", "knockout")
N_DIR, N_TEST, MIN_PREFIX = 10, 8, 3
HOOK = {"dirs": None, "capture": None}


def patch(model):
    for i, layer in enumerate(model.language_model.model.layers):
        base = layer.__class__

        def call(self, x, mask=None, cache=None, _i=i, _base=base):
            out = _base.__call__(self, x, mask=mask, cache=cache)
            if HOOK["dirs"] is not None:
                d = HOOK["dirs"][_i].astype(out.dtype)
                out = out - (out @ d)[..., None] * d
            if HOOK["capture"] is not None:
                HOOK["capture"].append(out.astype(mx.float32))
            return out

        layer.__class__ = type(f"KO{base.__name__}", (base,), {"__call__": call})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    common = {w: i for i, w in enumerate(sd.candidates())}
    E = {json.loads(l)["word"]: json.loads(l) for l in open(os.path.join(HERE, "eval", "selfdict", "dictionary.jsonl"))}
    G = json.load(open(os.path.join(HERE, "eval", "selfdict", "graph.json")))
    primes = set(G["primes"])
    hf = AutoTokenizer.from_pretrained(os.path.join(HERE, "models", "Qwen3.5-4B-4bit"))

    # surface occurrences: for each definition, (lemma, char start, word index) of every word
    occ = {}
    for u, e in E.items():
        d = e["definition"]
        head_end = len(d) - len(sd.strip_boilerplate(d, u))  # skip the 'The word "u"' frame
        occ[u] = [(sd.lemma(m.group(), common), m.start(), m.end()) for m in re.finditer(r"[A-Za-z]+", d)
                  if m.start() >= head_end]

    def first_at(u, w):
        """Char position of w's first surface occurrence in u's definition, if w is used after >= 3 words."""
        ws = occ[u]
        for k, (l, s, _) in enumerate(ws):
            if l == w:
                return s if k >= MIN_PREFIX else None
        return None

    dep = collections.defaultdict(list)
    for u in E:
        seen = set()
        for l, s, _ in occ[u]:
            if l != u and l not in seen and first_at(u, l) is not None:
                dep[l].append(u)
            seen.add(l)
    elig = sorted((w for w in E if len(dep[w]) >= N_DIR + N_TEST and w in common), key=lambda w: common[w])
    rng = random.Random(0)
    P = rng.sample([w for w in elig if w in primes], a.n)
    pool = [w for w in elig if w not in primes]
    used, C = set(), []
    for p in sorted(P, key=lambda w: common[w]):
        c = min((w for w in pool if w not in used), key=lambda w: abs(common[w] - common[p]))
        used.add(c)
        C.append(c)
    print(f"eligible: {sum(w in primes for w in elig)} primes, {len(pool)} non-primes; testing {len(P)} + {len(C)}")

    model, tok = load(os.path.join(HERE, "models", "Qwen3.5-4B-4bit"))
    patch(model)
    prefix = {}

    def prompt_ids(u):
        if u not in prefix:
            q = sd.DEF_Q.format(w=u)
            prefix[u] = hf.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                              tokenize=False, enable_thinking=False)
        return hf(prefix[u], add_special_tokens=False)["input_ids"]

    def def_tokens(u):
        enc = hf(E[u]["definition"], add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]

    def run(seqs):
        """Right-padded batch; returns logits as float32 numpy per row is too big, so return the mx array."""
        L = max(len(s) for s in seqs)
        x = mx.array([s + [0] * (L - len(s)) for s in seqs])
        return model(x)

    def score(items):
        """items: (prompt ids, def ids, k) -> per-item mean NLL over def tokens [0, k), and logprob of def token k."""
        seqs = [p + d for p, d, _ in items]
        lg = run(seqs)
        out = []
        for b, (p, d, k) in enumerate(items):
            pos = mx.array([len(p) - 1 + t for t in range(k + 1)])
            rows = lg[b, pos].astype(mx.float32)
            lp = rows - mx.logsumexp(rows, axis=-1, keepdims=True)
            tgt = mx.array(d[: k + 1])
            g = mx.take_along_axis(lp, tgt[:, None], 1)[:, 0]
            g = np.array(g)
            out.append((-g[:k].mean(), g[k]))
        return out

    def direction(w, us):
        texts = [E[u]["definition"] for u in us]
        encs = [hf(t, add_special_tokens=False, return_offsets_mapping=True) for t in texts]
        HOOK["capture"] = []
        run([e["input_ids"] for e in encs])
        caps = mx.stack(HOOK["capture"])  # [layers, B, T, H]
        HOOK["capture"] = None
        mx.eval(caps)
        caps = np.array(caps)
        on, allv = [], []
        for b, (u, e) in enumerate(zip(us, encs)):
            spans = [(s, t) for l, s, t in occ[u] if l == w]
            n = len(e["input_ids"])
            idx = [i for i, (s, t) in enumerate(e["offset_mapping"]) if any(s < b2 and t > a2 for a2, b2 in spans)]
            on.append(caps[:, b, idx])
            allv.append(caps[:, b, :n])
        d = np.concatenate(on, 1).mean(1) - np.concatenate(allv, 1).mean(1)  # [layers, H]
        d /= np.linalg.norm(d, axis=-1, keepdims=True) + 1e-8
        return mx.array(d)

    words = [(w, "prime") for w in P] + [(w, "control") for w in C]
    allheads = sorted(E)
    res = []
    t0 = time.time()
    for j, (w, kind) in enumerate(words):
        r = random.Random(f"{w}")
        us = dep[w][:]
        r.shuffle(us)
        dir_us, test_us = us[:N_DIR], us[N_DIR:N_DIR + N_TEST]
        items = []
        for u in test_us:
            ids, offs = def_tokens(u)
            cpos = first_at(u, w)
            k = next(i for i, (s, t) in enumerate(offs) if t > cpos)
            # pair: a definition that never uses w, long enough (capped search: a dependent whose prefix
            # is longer than any such definition is skipped; the first run hung here on one word)
            v = None
            for _ in range(3000):
                c = r.choice(allheads)
                if c != u and all(l != w for l, _, _ in occ[c]):
                    vids, _ = def_tokens(c)
                    if len(vids) > k:
                        v = c
                        break
            if v is None:
                continue
            items.append((prompt_ids(u), ids, k))
            items.append((prompt_ids(v), vids, k))
        HOOK["dirs"] = None
        base = score(items)
        HOOK["dirs"] = direction(w, dir_us)
        ko = score(items)
        HOOK["dirs"] = None
        dl = [k_[0] - b_[0] for k_, b_ in zip(ko, base)]
        dep_d, pair_d = np.mean(dl[0::2]), np.mean(dl[1::2])
        own = np.mean([b_[1] - k_[1] for k_, b_ in zip(ko[0::2], base[0::2])])  # drop in log P(w's token)
        res.append({"word": w, "kind": kind, "S": float(dep_d - pair_d), "dep": float(dep_d), "pair": float(pair_d),
                    "own_drop": float(own), "n_dep": len(dep[w]), "test": test_us})
        print(f"  {j + 1}/{len(words)} {kind:7} {w:12} S {dep_d - pair_d:+.3f}  dep {dep_d:+.3f}  pair {pair_d:+.3f}  "
              f"own-token drop {own:+.2f} nats  ({time.time() - t0:.0f}s)", flush=True)

    S = {k: np.array([x["S"] for x in res if x["kind"] == k]) for k in ("prime", "control")}
    Gd = {k: np.array([x["pair"] for x in res if x["kind"] == k]) for k in ("prime", "control")}
    own = {k: np.array([x["own_drop"] for x in res if x["kind"] == k]) for k in ("prime", "control")}
    u1 = mannwhitneyu(S["prime"], S["control"], alternative="greater")
    u2 = mannwhitneyu(Gd["prime"], Gd["control"], alternative="two-sided")
    print(f"\nsanity, drop in log P(w's own first token): primes {np.median(own['prime']):.2f}, "
          f"controls {np.median(own['control']):.2f} nats (median)")
    print(f"specific damage S: primes mean {S['prime'].mean():+.3f} median {np.median(S['prime']):+.3f} "
          f"(S > 0 in {np.mean(S['prime'] > 0):.0%}); controls mean {S['control'].mean():+.3f} "
          f"median {np.median(S['control']):+.3f} (S > 0 in {np.mean(S['control'] > 0):.0%})")
    print(f"primary, Mann-Whitney primes > controls: p = {u1.pvalue:.3g}")
    took = {k: np.array([x["S"] for x in res if x["kind"] == k and x["own_drop"] >= 2]) for k in ("prime", "control")}
    u3 = mannwhitneyu(took["prime"], took["control"], alternative="greater")
    print(f"secondary, knockouts that took (>= 2 nats): primes {len(took['prime'])} mean S {took['prime'].mean():+.3f}, "
          f"controls {len(took['control'])} mean S {took['control'].mean():+.3f}, p = {u3.pvalue:.3g}")
    print(f"general damage (pairs): primes {Gd['prime'].mean():+.3f}, controls {Gd['control'].mean():+.3f}, "
          f"two-sided p = {u2.pvalue:.3g}")
    json.dump({"results": res, "p_primary": float(u1.pvalue), "p_took": float(u3.pvalue), "p_general": float(u2.pvalue)},
              open(os.path.join(OUT, "knockout.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
