"""The Wikipedia game inside the activations (Hunt 35).

On Wikipedia, following each article's first link ends at "Philosophy" for most articles. Two
versions of that game on Qwen's self-dictionary:

  inner   A word's state = the layer-20 residual at the last prompt token of "Define the word w…"
          (the moment before it answers; Hunt 34), centered over all headwords. Its "first link" =
          the library concept (Hunt 34's 664 directions) whose direction best matches that state,
          other than w itself. Every word points to one concept, so every walk ends in a cycle.
  text    A word's first link = the first word of its own definition (after the "The word w is"
          frame) that is a headword and not a function word. Also ends in cycles.

For each: the attractor cycles, how much of the dictionary drains into each, and walks from a few
words. Then how often the two games agree on the first step, against chance.

    .venv/bin/python inner_walk.py
"""

import collections
import json
import os
import re
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np
from mlx_lm import load
from transformers import AutoTokenizer

import selfdict as sd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval", "inner_walk")
# frame control: `inner_walk.py about` reads states from "Tell me about w." instead of the definition prompt
FRAME = sys.argv[1] if len(sys.argv) > 1 else "define"
PROMPT = {"define": sd.DEF_Q, "about": 'Tell me about "{w}".'}[FRAME]
LAYER = 20
HOOK = {"capture": None}
STARTS = ["peace", "rock", "love", "alien", "dream", "insurance", "teacher", "considerate", "music", "money",
          "god", "death", "child", "computer", "ocean", "lie", "joke", "self"]
STOP = set("""a an the and or but if then than so as of in on at by for with from to into onto upon about over under
between through during before after above below up down out off this that these those it its is are was were be been
being have has had do does did can could will would shall should may might must not no nor any some all each every
which who whom whose what when where why how there here very more most less such own same other another just only also
too both either neither one used use something someone thing things way often usually typically especially specific
particular certain describe refer means mean term word""".split())


def patch(model):
    layer = model.language_model.model.layers[LAYER]
    base = layer.__class__

    def call(self, x, mask=None, cache=None, _base=base):
        out = _base.__call__(self, x, mask=mask, cache=cache)
        if HOOK["capture"] is not None:
            HOOK["capture"].append(out.astype(mx.float32))
        return out

    layer.__class__ = type(f"IW{base.__name__}", (base,), {"__call__": call})


def cycles(nxt):
    """Functional graph -> (cycle id per node, list of cycles)."""
    color, cyc_of, cyc = {}, {}, []
    for s in nxt:
        path, x = [], s
        while x not in color and x is not None:
            color[x] = s
            path.append(x)
            x = nxt.get(x)
        if x is not None and color.get(x) == s and x not in cyc_of:  # new cycle found on this path
            c = path[path.index(x):]
            for y in c:
                cyc_of[y] = len(cyc)
            cyc.append(c)
        end = cyc_of.get(x) if x is not None else None
        for y in path:
            cyc_of.setdefault(y, end)
    return cyc_of, cyc


def walk(nxt, s, n=40):
    out, seen = [s], {s}
    while len(out) < n:
        x = nxt.get(out[-1])
        if x is None:
            break
        out.append(x)
        if x in seen:
            break
        seen.add(x)
    return out


def report(name, nxt, words):
    cyc_of, cyc = cycles(nxt)
    basin = collections.Counter(cyc_of[w] for w in words if cyc_of.get(w) is not None)
    n = sum(basin.values())
    print(f"\n{name.upper()}: {len(cyc)} attractor cycles for {n} words")
    for c, k in basin.most_common(8):
        print(f"  {k / n:6.1%}  {' → '.join(cyc[c])} → (back)")
    return cyc_of, cyc, basin


def main():
    os.makedirs(OUT, exist_ok=True)
    common = {w: i for i, w in enumerate(sd.candidates())}
    E = {json.loads(l)["word"]: json.loads(l) for l in open(os.path.join(HERE, "eval", "selfdict", "dictionary.jsonl"))}
    lib = np.load(os.path.join(HERE, "eval", "concept_read", "library.npz"))
    words_lib = [str(w) for w in lib["words"]]
    D = lib["D"][list(lib["layers"]).index(LAYER)].astype(np.float32)  # [W, H]
    hf = AutoTokenizer.from_pretrained(os.path.join(HERE, "models", "Qwen3.5-4B-4bit"))
    model, _ = load(os.path.join(HERE, "models", "Qwen3.5-4B-4bit"))
    patch(model)

    heads = sorted(E)
    states = []
    for s in range(0, len(heads), 48):
        batch = heads[s:s + 48]
        ids = [hf(hf.apply_chat_template([{"role": "user", "content": PROMPT.format(w=u)}], add_generation_prompt=True,
                                          tokenize=False, enable_thinking=False), add_special_tokens=False)["input_ids"]
               for u in batch]
        L = max(len(x) for x in ids)
        HOOK["capture"] = []
        model(mx.array([x + [0] * (L - len(x)) for x in ids]))
        cap = np.array(HOOK["capture"][0])
        HOOK["capture"] = None
        states.append(np.stack([cap[b, len(x) - 1] for b, x in enumerate(ids)]))
        if s % 960 == 0:
            print(f"  states {s + len(batch)}/{len(heads)}", flush=True)
    S = np.concatenate(states)
    S -= S.mean(0)
    S /= np.linalg.norm(S, axis=-1, keepdims=True) + 1e-8
    sim = S @ D.T  # [heads, W]
    hi = {u: i for i, u in enumerate(heads)}
    li = {w: i for i, w in enumerate(words_lib)}
    inner = {}
    for u in heads:
        sc = sim[hi[u]].copy()
        if u in li:
            sc[li[u]] = -9
        inner[u] = words_lib[int(np.argmax(sc))]

    text = {}
    for u in heads:
        d = E[u]["definition"]
        body = sd.strip_boilerplate(d, u)
        for m in re.finditer(r"[A-Za-z]+", body):
            w = sd.lemma(m.group(), common)
            if w != u and w in E and w not in STOP:
                text[u] = w
                break

    res = {}
    for name, nxt in [("inner", inner), ("text", text)]:
        cyc_of, cyc, basin = report(name, nxt, heads)
        res[name] = {"cycles": cyc, "basins": {" → ".join(cyc[c]): k for c, k in basin.most_common(30)}}
        print("  walks:")
        for s in STARTS:
            if s in E:
                print(f"    {' → '.join(walk(nxt, s))}")
        res[name]["walks"] = {s: walk(nxt, s) for s in STARTS if s in E}

    # agreement on the first step, among words whose text link is a library concept
    both = [u for u in heads if u in text and text[u] in li]
    agree = np.mean([inner[u] == text[u] for u in both])
    # chance: how often the inner link would match a text link drawn from the same distribution
    tl = collections.Counter(text[u] for u in both)
    il = collections.Counter(inner[u] for u in both)
    chance = sum(tl[w] * il[w] for w in tl) / len(both) ** 2
    # rank of the text link in the inner scores
    ranks = []
    for u in both:
        sc = sim[hi[u]].copy()
        if u in li:
            sc[li[u]] = -9
        ranks.append(int((sc > sc[li[text[u]]]).sum()) + 1)
    ranks = np.array(ranks)
    print(f"\nfirst steps agree on {agree:.1%} of {len(both)} words (chance from the two link distributions {chance:.1%}); "
          f"the text link's median rank among {len(words_lib)} concepts in the inner state: {int(np.median(ranks))} "
          f"(top 10 for {np.mean(ranks <= 10):.0%})")
    top_inner = collections.Counter(inner.values()).most_common(15)
    top_text = collections.Counter(text.values()).most_common(15)
    print("most-linked-to, inner:", ", ".join(f"{w} {k}" for w, k in top_inner))
    print("most-linked-to, text: ", ", ".join(f"{w} {k}" for w, k in top_text))
    res.update({"agree": float(agree), "chance": float(chance), "median_rank": int(np.median(ranks)),
                "top_inner": top_inner, "top_text": top_text, "inner_link": inner, "text_link": text})
    json.dump(res, open(os.path.join(OUT, f"inner_walk_{FRAME}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
