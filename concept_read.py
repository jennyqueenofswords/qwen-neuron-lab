"""Read a word's parts off the model's state before it defines it (Hunt 34).

Hunt 33's knockouts showed the model already using a concept in the opening of a definition, before
it says the word. If so, the parts of a definition should be readable from the state just before the
answer starts: the Wikipedia game inside the activations instead of in the text.

  library   Split the English self-dictionary in half by headword (seed 0). From the definitions in
            half A, a direction per word w and per layer: mean residual on w's tokens minus the mean
            over all tokens (as in knockout.py). Words with fewer than 5 occurrences in A are left out.
  read      For each headword u in half B: prompt "Define the word u…" and take the residual at the
            last prompt token (the model is about to answer), centered by the mean over all B prompts.
            Score every library word by cosine with its direction.
  test      Parts of u = library words its definition uses (excluding u). Per target, AUC of the
            direction score for parts vs non-parts. Baseline: rank by how many A definitions use the
            word (a prior that knows nothing about u). Pre-set primary: at the best layer chosen on
            the first 200 targets, mean AUC on the remaining targets vs the baseline's, paired
            Wilcoxon, directions > baseline.

    .venv/bin/python concept_read.py
"""

import collections
import json
import os
import random
import re

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np
from mlx_lm import load
from scipy.stats import wilcoxon
from transformers import AutoTokenizer

import selfdict as sd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval", "concept_read")
LAYERS = [4, 8, 12, 16, 20, 24, 28]
HOOK = {"capture": None}
EXAMPLES = ["peace", "rock", "love", "considerate", "tool", "selfless", "teacher", "insurance", "alien", "dream"]


def patch(model):
    for i, layer in enumerate(model.language_model.model.layers):
        base = layer.__class__

        def call(self, x, mask=None, cache=None, _i=i, _base=base):
            out = _base.__call__(self, x, mask=mask, cache=cache)
            if HOOK["capture"] is not None and _i in LAYERS:
                HOOK["capture"][_i] = out.astype(mx.float32)
            return out

        layer.__class__ = type(f"CR{base.__name__}", (base,), {"__call__": call})


def auc(scores, pos):
    """Probability a random part outscores a random non-part."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n1 = pos.sum()
    n0 = len(pos) - n1
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    os.makedirs(OUT, exist_ok=True)
    common = {w: i for i, w in enumerate(sd.candidates())}
    E = {json.loads(l)["word"]: json.loads(l) for l in open(os.path.join(HERE, "eval", "selfdict", "dictionary.jsonl"))}
    for w, e in E.items():
        e["uses"] = sd.def_words(e["definition"], w, common)
    heads = sorted(E)
    random.Random(0).shuffle(heads)
    A, B = set(heads[: len(heads) // 2]), heads[len(heads) // 2:]
    for ex in EXAMPLES:  # keep the showcase words on the read side
        if ex in A:
            A.discard(ex)
            B.append(ex)
    hf = AutoTokenizer.from_pretrained(os.path.join(HERE, "models", "Qwen3.5-4B-4bit"))
    model, _ = load(os.path.join(HERE, "models", "Qwen3.5-4B-4bit"))
    patch(model)

    # ---- library from half A ----
    sums = collections.defaultdict(lambda: None)
    counts = collections.Counter()
    bg_sum, bg_n = None, 0
    defsA = sorted(A)
    for s in range(0, len(defsA), 48):
        batch = defsA[s:s + 48]
        encs = [hf(E[u]["definition"], add_special_tokens=False, return_offsets_mapping=True) for u in batch]
        L = max(len(e["input_ids"]) for e in encs)
        HOOK["capture"] = {}
        model(mx.array([e["input_ids"] + [0] * (L - len(e["input_ids"])) for e in encs]))
        cap = {l: np.array(v) for l, v in HOOK["capture"].items()}
        HOOK["capture"] = None
        stack = np.stack([cap[l] for l in LAYERS])  # [nl, B, T, H]
        for b, (u, e) in enumerate(zip(batch, encs)):
            n = len(e["input_ids"])
            v = stack[:, b, :n]
            bg_sum = v.sum(1) if bg_sum is None else bg_sum + v.sum(1)
            bg_n += n
            d = E[u]["definition"]
            head_end = len(d) - len(sd.strip_boilerplate(d, u))
            for m in re.finditer(r"[A-Za-z]+", d):
                if m.start() < head_end:
                    continue
                w = sd.lemma(m.group(), common)
                if w == u or w not in E:
                    continue
                idx = [i for i, (a, z) in enumerate(e["offset_mapping"]) if a < m.end() and z > m.start()]
                if not idx:
                    continue
                vec = v[:, idx].mean(1)
                sums[w] = vec if sums[w] is None else sums[w] + vec
                counts[w] += 1
        print(f"  library: {min(s + 48, len(defsA))}/{len(defsA)} definitions", flush=True)
    bg = bg_sum / bg_n
    lib = sorted(w for w in counts if counts[w] >= 5)
    D = np.stack([sums[w] / counts[w] - bg for w in lib], 1)  # [nl, W, H]
    D /= np.linalg.norm(D, axis=-1, keepdims=True) + 1e-8
    li = {w: i for i, w in enumerate(lib)}
    prior = np.array([sum(w in E[u]["uses"] for u in A) for w in lib], float)
    print(f"library: {len(lib)} words with >= 5 occurrences in {len(A)} definitions")

    # ---- read half B ----
    targets = [u for u in B if any(w in li for w in E[u]["uses"])]
    states = {l: [] for l in LAYERS}
    for s in range(0, len(targets), 48):
        batch = targets[s:s + 48]
        ids = [hf(hf.apply_chat_template([{"role": "user", "content": sd.DEF_Q.format(w=u)}], add_generation_prompt=True,
                                          tokenize=False, enable_thinking=False), add_special_tokens=False)["input_ids"]
               for u in batch]
        L = max(len(x) for x in ids)
        HOOK["capture"] = {}
        model(mx.array([x + [0] * (L - len(x)) for x in ids]))
        for l in LAYERS:
            cap = np.array(HOOK["capture"][l])
            states[l].append(np.stack([cap[b, len(x) - 1] for b, x in enumerate(ids)]))
        HOOK["capture"] = None
        print(f"  read: {min(s + 48, len(targets))}/{len(targets)}", flush=True)
    S = {l: np.concatenate(states[l]) for l in LAYERS}
    for l in LAYERS:
        S[l] = S[l] - S[l].mean(0)
        S[l] /= np.linalg.norm(S[l], axis=-1, keepdims=True) + 1e-8

    # ---- test ----
    res = {l: [] for l in LAYERS}
    base = []
    for t, u in enumerate(targets):
        pos = np.zeros(len(lib), bool)
        for w in E[u]["uses"]:
            if w in li:
                pos[li[w]] = True
        if u in li:
            pos[li[u]] = False
        mask = np.ones(len(lib), bool)
        if u in li:
            mask[li[u]] = False  # the headword itself is neither part nor non-part
        if pos.sum() == 0:
            continue
        base.append(auc(prior[mask], pos[mask]))
        for k, l in enumerate(LAYERS):
            sc = D[k] @ S[l][t]
            res[l].append(auc(sc[mask], pos[mask]))
    base = np.array(base)
    pick = {l: np.mean(res[l][:200]) for l in LAYERS}
    best = max(pick, key=pick.get)
    print("\nmean AUC by layer (all targets):", {l: round(float(np.mean(res[l])), 3) for l in LAYERS},
          f"baseline {base.mean():.3f}")
    rest = np.array(res[best][200:])
    wx = wilcoxon(rest, base[200:], alternative="greater")
    print(f"primary: layer {best} (picked on the first 200 targets); on the other {len(rest)}: "
          f"directions {rest.mean():.3f} vs frequency baseline {base[200:].mean():.3f}, Wilcoxon p = {wx.pvalue:.3g}; "
          f"directions win on {np.mean(rest > base[200:]):.0%} of targets")
    # ---- post hoc (added after the primary lost to the frequency prior) ----
    # (a) content words only: drop the 100 words most used in half-A definitions from parts and non-parts
    # (b) does the direction score add to the prior? z-scored sum vs the prior alone, all words
    k = LAYERS.index(best)
    top100 = set(np.argsort(-prior)[:100])
    keep = np.array([i not in top100 for i in range(len(lib))])
    zp = (prior - prior.mean()) / prior.std()
    ca_d, ca_b, cm, pb = [], [], [], []
    for t, u in enumerate(targets):
        pos = np.zeros(len(lib), bool)
        for w in E[u]["uses"]:
            if w in li:
                pos[li[w]] = True
        mask = np.ones(len(lib), bool)
        if u in li:
            mask[li[u]] = False
            pos[li[u]] = False
        sc = D[k] @ S[best][t]
        m2 = mask & keep
        if pos[mask].sum():
            zs = (sc - sc[mask].mean()) / sc[mask].std()
            cm.append(auc((zs + zp)[mask], pos[mask]))
            pb.append(auc(prior[mask], pos[mask]))
        if pos[m2].sum() and (~pos[m2]).sum():
            ca_d.append(auc(sc[m2], pos[m2]))
            ca_b.append(auc(prior[m2], pos[m2]))
    ca_d, ca_b, cm, pb = map(np.array, (ca_d, ca_b, cm, pb))
    print(f"post hoc (a) content words only: directions {ca_d.mean():.3f} vs frequency {ca_b.mean():.3f}, "
          f"directions win on {np.mean(ca_d > ca_b):.0%} of {len(ca_d)} targets, Wilcoxon p = {wilcoxon(ca_d, ca_b).pvalue:.3g}")
    print(f"post hoc (b) frequency + directions {cm.mean():.3f} vs frequency alone {pb.mean():.3f}, "
          f"better on {np.mean(cm > pb):.0%}, Wilcoxon p = {wilcoxon(cm, pb).pvalue:.3g}")

    print("\nDECOMPOSITIONS at layer", best, "(top 8 by direction score; * = in its definition)")
    out = {"layer": int(best), "auc_by_layer": {int(l): float(np.mean(res[l])) for l in LAYERS},
           "baseline": float(base.mean()), "p": float(wx.pvalue), "content_only": [float(ca_d.mean()), float(ca_b.mean())],
           "combined": [float(cm.mean()), float(pb.mean())], "examples": {}}
    for u in EXAMPLES:
        if u not in targets:
            continue
        t = targets.index(u)
        sc = D[k] @ S[best][t]
        if u in li:
            sc[li[u]] = -9
        top = [lib[i] for i in np.argsort(-sc)[:8]]
        parts = set(E[u]["uses"])
        print(f"  {u:12} {' '.join(w + ('*' if w in parts else '') for w in top)}   | {E[u]['definition'][:80]}")
        out["examples"][u] = {"top": top, "definition": E[u]["definition"]}
    np.savez_compressed(os.path.join(OUT, "library.npz"), words=np.array(lib), D=D.astype(np.float16), layers=np.array(LAYERS))
    json.dump(out, open(os.path.join(OUT, "concept_read.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
