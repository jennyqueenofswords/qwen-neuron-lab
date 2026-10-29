"""Search for mode neurons: pushes that change HOW the model answers neutral questions, not WHAT it talks about.

  gen    For a sample of (neuron, direction) pairs in layers 8-26, drawn from each layer's 60th-98th percentile of
         Lyra's steer-lens KL (skipping neurons that do little and the top wreckers), pin the neuron at ±0.5 x
         (median stream size / arrow length) and write greedy 48-token answers to four neutral questions. Batched,
         one neuron per row. -> eval/mode_search/replies.jsonl (+ the unpushed baselines)
  score  Automatic screen: how far each answer departs from the baseline, whether the four answers share an
         injected content word (a topic writer), whether they loop (a breaker), and whether they still answer the
         question. -> eval/mode_search/candidates.json, ranked

    .venv/bin/python mode_search.py gen path/to/steer_index.npz [--per-layer 80]
    .venv/bin/python mode_search.py score
"""

import argparse
import json
import os
import random
import re

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval", "mode_search")
QUESTIONS = {"How do I boil an egg?": ["egg", "boil", "water"],
             "Explain photosynthesis in two sentences.": ["photosynth", "plant", "light", "sun"],
             "Give me one tip for a job interview.": ["interview", "job", "prepar", "company"],
             "What is the capital of Australia?": ["canberra", "australia", "capital"]}
LAYERS = range(8, 27)
DOSE = 0.5
STOP = set("the a an and or of to in on for with is are was be it this that as at by from you your i my me we our they "
           "their can will would should could just also not but so if then than more most very into about which what "
           "how when who its it's i'm here there have has had do does make sure some any all one two".split())


def gen(a):
    import mlx.core as mx
    from mlx_lm import load
    import steer
    from search_identity import neuron_out_all

    st = np.load(a.index, allow_pickle=False)
    rng = random.Random(83)
    pairs = []
    for L in LAYERS:
        for sign in ("up", "down"):
            kl = st[f"{sign}_kl"][L]
            lo, hi = np.percentile(kl, 60), np.percentile(kl, 98)
            pool = [int(n) for n in np.where((kl >= lo) & (kl <= hi))[0]]
            pairs += [(L, n, sign) for n in rng.sample(pool, a.per_layer // 2)]
    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    neutral = [t.format(n) for t in steer.TEMPLATES for n in steer.NEUTRAL]
    _, resid = steer.mean_over(neutral, model, tok, "resid")
    arrows = {L: np.array(mx.linalg.norm(neuron_out_all(model.language_model.model.layers[L].mlp), axis=0)) for L in LAYERS}
    eos = set(tok.eos_token_ids) if hasattr(tok, "eos_token_ids") else {tok.eos_token_id}
    os.makedirs(OUT, exist_ok=True)
    f = open(os.path.join(OUT, "replies.jsonl"), "w")
    for q in QUESTIONS:
        steer.HOOKS["pin"].clear()
        f.write(json.dumps({"baseline": True, "question": q, "reply": steer.write(model, tok, q, a.tokens, 0.0, 0)}, ensure_ascii=False) + "\n")
    prompts = {q: tok.encode(steer.chat(tok, q)) for q in QUESTIONS}
    by_layer = {}
    for p in pairs:
        by_layer.setdefault(p[0], []).append(p)
    done = 0
    for L, ps in sorted(by_layer.items()):
        for s in range(0, len(ps), a.batch):
            chunk = ps[s:s + a.batch]
            vals = [(1 if sg == "up" else -1) * DOSE * resid[L] / float(arrows[L][n]) for _, n, sg in chunk]
            for q, ids in prompts.items():
                B = len(chunk)
                steer.HOOKS["bpin"] = (L, mx.array([n for _, n, _ in chunk]), mx.array(vals))
                cache = model.make_cache()
                logits = model(mx.array([ids] * B), cache=cache)[:, -1]
                out = [[] for _ in range(B)]
                for _ in range(a.tokens):
                    nxt = mx.argmax(logits, axis=-1)
                    mx.eval(nxt)
                    for i, t in enumerate(nxt.tolist()):
                        out[i].append(t)
                    logits = model(nxt[:, None], cache=cache)[:, -1]
                steer.HOOKS["bpin"] = None
                for (Lc, n, sg), v, toks in zip(chunk, vals, out):
                    toks = toks[: next((k for k, t in enumerate(toks) if t in eos), len(toks))]
                    f.write(json.dumps({"layer": Lc, "neuron": n, "sign": sg, "value": round(v, 1), "question": q,
                                        "reply": tok.decode(toks)}, ensure_ascii=False) + "\n")
            done += len(chunk)
            f.flush()
            print(f"  {done}/{len(pairs)} neuron-directions", flush=True)
    f.close()


def words(t):
    return [w for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in STOP]


def score(a):
    rows = [json.loads(l) for l in open(os.path.join(OUT, "replies.jsonl"))]
    base = {r["question"]: r["reply"] for r in rows if r.get("baseline")}
    base_words = set(w for t in base.values() for w in words(t))
    q_words = set(w for q in QUESTIONS for w in words(q))
    groups = {}
    for r in rows:
        if not r.get("baseline"):
            groups.setdefault((r["layer"], r["neuron"], r["sign"]), {})[r["question"]] = r["reply"]
    cands = []
    for (L, n, sg), ans in groups.items():
        if len(ans) < len(QUESTIONS):
            continue
        dep, on, broken = [], 0, 0
        counts = {}
        for q, t in ans.items():
            w, bw = set(words(t)), set(words(base[q]))
            dep.append(1 - len(w & bw) / max(len(w | bw), 1))
            on += any(k in t.lower() for k in QUESTIONS[q])
            toks = t.lower().split()
            grams = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
            broken += len(toks) < 6 or (grams and len(set(grams)) / len(grams) < 0.7)
            for x in w - bw - q_words:
                counts[x] = counts.get(x, 0) + 1
        topic = sorted(x for x, c in counts.items() if c >= 3 and x not in base_words)
        cands.append({"layer": L, "neuron": n, "sign": sg, "departure": round(float(np.mean(dep)), 3),
                      "on_task": on, "broken": broken, "topic_words": topic,
                      "mode_score": round(float(np.mean(dep)) * on / len(QUESTIONS) * (broken == 0) * (not topic), 3)})
    cands.sort(key=lambda c: -c["mode_score"])
    json.dump(cands, open(os.path.join(OUT, "candidates.json"), "w"), indent=1)
    n_topic = sum(bool(c["topic_words"]) for c in cands)
    n_broken = sum(c["broken"] > 0 for c in cands)
    print(f"{len(cands)} neuron-directions: {n_topic} inject a shared topic word, {n_broken} loop on some answer, "
          f"{sum(c['mode_score'] > 0 for c in cands)} pass the screen")
    for c in cands[:25]:
        print(f"  L{c['layer']}/N{c['neuron']:<5} {c['sign']:<4} mode {c['mode_score']:.2f}  departure {c['departure']:.2f}  on-task {c['on_task']}/4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "score"])
    ap.add_argument("index", nargs="?")
    ap.add_argument("--per-layer", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=48)
    a = ap.parse_args()
    {"gen": gen, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
