"""Qwen's own dictionary, and the words it can't do without.

Following Vincent-Lamarre, Harnad et al. (2016), "The Latent Structure of Dictionaries": a
dictionary is a graph (each word points to the words in its definition). Repeatedly removing
words that no remaining definition uses leaves the kernel; inside it, the largest strongly
connected core; and minimal grounding sets, the fewest words from which every other word can be
reached by definitions. Here the dictionary is written by Qwen3.5-4B itself.

  gen     Headwords: whole-word English tokens in Qwen's vocabulary, in vocabulary order (earlier
          = more frequent in its training data), kept if the model says they're common English
          words. The first --n get one-sentence definitions (greedy); then every word used in a
          definition that isn't yet a headword gets checked and defined too, for --rounds rounds.
          -> eval/selfdict/dictionary.jsonl
  graph   Kernel, core, and an approximate minimum grounding set; written with summaries.
          -> eval/selfdict/graph.json

    .venv/bin/python selfdict.py gen [--n 3000] [--rounds 4]
    .venv/bin/python selfdict.py graph
"""

import argparse
import collections
import json
import os
import re
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval", "selfdict")
DICT = os.path.join(OUT, "dictionary.jsonl")
SHORT_OK = {"a", "i"}

IRREGULAR = {
    "am": "be", "is": "be", "are": "be", "was": "be", "were": "be", "been": "be", "being": "be",
    "has": "have", "had": "have", "having": "have", "does": "do", "did": "do", "done": "do", "doing": "do",
    "went": "go", "gone": "go", "goes": "go", "said": "say", "says": "say", "made": "make", "took": "take",
    "taken": "take", "came": "come", "saw": "see", "seen": "see", "knew": "know", "known": "know",
    "thought": "think", "got": "get", "gotten": "get", "gave": "give", "given": "give", "found": "find",
    "told": "tell", "felt": "feel", "left": "leave", "kept": "keep", "began": "begin", "begun": "begin",
    "brought": "bring", "bought": "buy", "held": "hold", "stood": "stand", "ran": "run", "wrote": "write",
    "written": "write", "spoke": "speak", "spoken": "speak", "men": "man", "women": "woman",
    "children": "child", "feet": "foot", "teeth": "tooth", "lives": "life", "built": "build", "meant": "mean",
    "sent": "send", "spent": "spend", "lost": "lose", "paid": "pay", "met": "meet", "led": "lead",
    "grew": "grow", "grown": "grow", "drew": "draw", "drawn": "draw", "fell": "fall", "fallen": "fall",
    "chose": "choose", "chosen": "choose", "broke": "break", "broken": "break", "ate": "eat", "eaten": "eat",
    "an": "a", "better": "good", "best": "good", "worse": "bad", "worst": "bad", "taught": "teach",
    "caught": "catch", "fought": "fight", "sold": "sell", "heard": "hear", "understood": "understand",
}


def candidates():
    """Whole-word lowercase English tokens, in Qwen's vocabulary order."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "models", "Qwen3.5-4B-4bit"))
    D = {w.strip() for w in open("/usr/share/dict/words") if w.strip().islower()}
    out = []
    for t, _ in sorted(tok.get_vocab().items(), key=lambda x: x[1]):
        if t.startswith("Ġ"):
            w = t[1:]
            if w.isascii() and w.isalpha() and w.islower() and (len(w) > 1 or w in SHORT_OK) and w in D:
                out.append(w)
    return out


def lemma(w, common):
    w = w.lower()
    if w in IRREGULAR:
        return IRREGULAR[w]
    if len(w) > 3 and not w.endswith("ss"):
        for suf, rep in (("ies", "y"), ("ves", "f"), ("es", ""), ("s", ""), ("ied", "y"), ("ed", ""), ("ed", "e"),
                         ("ing", ""), ("ing", "e")):
            if w.endswith(suf):
                c = w[: len(w) - len(suf)] + rep
                # doubled consonant: "stopped" -> "stop", "putting" -> "put" (not "putt"): when both
                # forms are words, take the one earlier in Qwen's vocabulary (more frequent)
                if suf in ("ed", "ing") and len(c) >= 3 and c[-1] == c[-2] and c[:-1] in common:
                    if c not in common or (isinstance(common, dict) and common[c[:-1]] < common[c]):
                        return c[:-1]
                if len(c) >= 3 and c in common:
                    return c
    return w


ACCEPT = 0.6  # "Is X an English word?" Yes-probability; at 0.6 function words pass and most fragments don't
YES_Q = 'Is "{}" an English word? Answer Yes or No.'
YES_NO = ("Yes", "No")
DEF_Q = 'Define the word "{w}" in one short sentence of plain English. Do not use the word "{w}" itself.'


def strip_boilerplate(text, head):
    """'The word "of" is a preposition…' -> 'is a preposition…' (the frame is the template, not the meaning)"""
    return re.sub(r'^\s*(the\s+(word|term)\s+)?["“\']?' + re.escape(head) + r'["”\']?\s*', "", text, flags=re.I)


def def_words(text, head, common):
    toks = re.findall(r"[A-Za-z]+", strip_boilerplate(text, head))
    return sorted({lemma(t, common) for t in toks} - {head})


class Runner:
    def __init__(self):
        import mlx.core as mx
        from mlx_lm import load
        self.mx = mx
        self.model, self.tok = load(os.path.join(HERE, "models", "Qwen3.5-4B-4bit"))
        self.eos = set(self.tok.eos_token_ids) if hasattr(self.tok, "eos_token_ids") else {self.tok.eos_token_id}
        self.nl = {i for t, i in self.tok.get_vocab().items() if "Ċ" in t}

    def prompt(self, q):
        return self.tok.encode(self.tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                                            tokenize=False, enable_thinking=False))

    def buckets(self, prompts, batch):
        """Group same-length prompts so no padding touches the recurrent layers' state."""
        by = collections.defaultdict(list)
        for i, p in enumerate(prompts):
            by[len(p)].append(i)
        for idx in by.values():
            for s in range(0, len(idx), batch):
                yield idx[s:s + batch]

    def yes_prob(self, words, batch=64):
        mx = self.mx
        prompts = [self.prompt(YES_Q.format(w)) for w in words]
        y, n = self.tok.encode(YES_NO[0])[0], self.tok.encode(YES_NO[1])[0]
        out = [0.0] * len(words)
        for idx in self.buckets(prompts, batch):
            lg = self.model(mx.array([prompts[i] for i in idx]))[:, -1]
            p = mx.softmax(lg[:, [y, n]].astype(mx.float32), axis=-1)[:, 0]
            for i, v in zip(idx, p.tolist()):
                out[i] = v
        return out

    def define(self, words, batch=64, tokens=40):
        mx = self.mx
        prompts = [self.prompt(DEF_Q.format(w=w)) for w in words]
        out = [""] * len(words)
        for idx in self.buckets(prompts, batch):
            cache = self.model.make_cache()
            lg = self.model(mx.array([prompts[i] for i in idx]), cache=cache)[:, -1]
            ids = [[] for _ in idx]
            done = [False] * len(idx)
            for _ in range(tokens):
                nxt = mx.argmax(lg, axis=-1)
                mx.eval(nxt)
                for r, t in enumerate(nxt.tolist()):
                    if not done[r]:
                        if t in self.eos or (t in self.nl and ids[r]):
                            done[r] = True
                        else:
                            ids[r].append(t)
                if all(done):
                    break
                lg = self.model(nxt[:, None], cache=cache)[:, -1]
            for i, s in zip(idx, ids):
                out[i] = self.tok.decode(s).strip()
        return out


def cmd_gen(a):
    os.makedirs(OUT, exist_ok=True)
    cands = candidates()
    common = {w: i for i, w in enumerate(cands)}
    run = Runner()
    t0 = time.time()
    entries, checked = {}, {}

    def check(ws):
        ws = [w for w in ws if w not in checked]
        for s in range(0, len(ws), 512):
            part = ws[s:s + 512]
            for w, p in zip(part, run.yes_prob(part)):
                checked[w] = p

    # round 0: the first n candidates the model accepts as words
    pos, heads = 0, []
    while len(heads) < a.n and pos < len(cands):
        part = cands[pos:pos + 1024]
        check(part)
        heads += [w for w in part if (checked[w] >= ACCEPT or w in SHORT_OK)]
        pos += 1024
        print(f"  checked {pos} candidates, {len(heads)} words  ({time.time() - t0:.0f}s)", flush=True)
    heads = heads[: a.n]
    rnd = 0
    while heads:
        defs = []
        for s in range(0, len(heads), 512):
            defs += run.define(heads[s:s + 512])
            print(f"  round {rnd}: defined {min(s + 512, len(heads))}/{len(heads)}  ({time.time() - t0:.0f}s)", flush=True)
        for w, d in zip(heads, defs):
            entries[w] = {"word": w, "round": rnd, "p_word": round(checked.get(w, 1.0), 3), "definition": d,
                          "uses": def_words(d, w, common)}
        rnd += 1
        if rnd > a.rounds:
            break
        new = sorted({u for w in heads for u in entries[w]["uses"]} - set(entries))
        new = [u for u in new if u in common or (LANG == "en" and len(u) > 2)]
        check(new)
        heads = [u for u in new if (checked[u] >= ACCEPT or u in SHORT_OK)]
        print(f"  round {rnd}: {len(new)} new words used, {len(heads)} accepted", flush=True)
    with open(DICT, "w") as f:
        for e in entries.values():
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {len(entries)} entries to {DICT}  ({time.time() - t0:.0f}s)")



# ---------- Chinese (Hunt 32) ----------
# Chinese has no spaces, so definitions are cut into words by forward maximum matching against
# Qwen's own all-Chinese tokens (up to 4 characters), the same vocabulary the headwords come from.
CJK = re.compile(r"^[一-鿿]+$")

# Mandarin exponents of the NSM primes, after Chappell's Mandarin NSM work as summarized by Goddard
# & Wierzbicka; our own mapping, with common variants. A word counts if it is any listed exponent.
NSM_ZH = """我 你 谁 某人 什么 东西 人 身体 种 部分 这 一样 同样 别的 其他 一 两 二 一些 有些 所有 都 多 少 好 坏
大 小 想 知道 要 感觉 觉得 看 看见 听 听见 说 话 真 做 发生 动 在 有 是 我的 活 死 时候 现在 以前 以后 很久
一会儿 片刻 地方 哪里 这里 上面 下面 远 近 边 里面 碰 不 没 可能 也许 能 因为 如果 很 更 像""".split()


JIEBA = os.path.join(HERE, "data", "jieba_dict.txt")  # jieba 0.42.1's dict.txt (MIT), not in git:
# curl -sSLO https://files.pythonhosted.org/packages/source/j/jieba/jieba-0.42.1.tar.gz  (sha256 055ca12f…28db2)
# tar -xzf jieba-0.42.1.tar.gz jieba-0.42.1/jieba/dict.txt && mv jieba-0.42.1/jieba/dict.txt data/jieba_dict.txt


def candidates_zh():
    """All-Chinese tokens (1-4 characters) that are words in jieba's dictionary, in Qwen's vocabulary
    order. jieba plays the part /usr/share/dict plays for English: Qwen's vocabulary holds phrase
    chunks (是一种, 我们的, 有一个) that the model itself accepts as words, and jieba doesn't list them."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "models", "Qwen3.5-4B-4bit"))
    words = {l.split(" ")[0] for l in open(JIEBA, encoding="utf-8")}
    out, seen = [], set()
    for t, i in sorted(tok.get_vocab().items(), key=lambda x: x[1]):
        w = tok.convert_tokens_to_string([t]).strip()
        if CJK.match(w) and len(w) <= 4 and w not in seen and w in words:
            seen.add(w)
            out.append(w)
    return out


def segment_zh(text, common, maxlen=4):
    out = []
    for run in re.findall(r"[一-鿿]+", text):
        i = 0
        while i < len(run):
            for L in range(min(maxlen, len(run) - i), 0, -1):
                if L == 1 or run[i:i + L] in common:
                    out.append(run[i:i + L])
                    i += L
                    break
    return out


def def_words_zh(text, head, common):
    text = text.replace(head, " ")  # the headword itself, wherever it appears
    return sorted(set(segment_zh(text, common)) - {head})


def nsm_words():
    from primes_test import PRIMES
    return NSM_ZH if LANG == "zh" else PRIMES


LANG = "en"


def set_lang(lang):
    global LANG, OUT, DICT, candidates, def_words, YES_Q, YES_NO, DEF_Q, ACCEPT, SHORT_OK
    LANG = lang
    if lang == "zh":
        OUT = os.path.join(HERE, "eval", "selfdict_zh")
        DICT = os.path.join(OUT, "dictionary.jsonl")
        candidates, def_words = candidates_zh, def_words_zh
        YES_Q = '"{}"是一个中文词语吗？只回答"是"或"否"。'
        YES_NO = ("是", "否")
        DEF_Q = '请用一句简短、通俗的中文解释"{w}"这个词的意思。不要使用"{w}"这个词本身。'
        SHORT_OK = set()
        ACCEPT = 0.3  # jieba already filters; at 0.6 the model would reject the particle 了 (0.50)

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--n", type=int, default=3000)
    g.add_argument("--rounds", type=int, default=4)
    g.add_argument("--sample", type=int, default=0, help="just define this many words and print them")
    gr = sub.add_parser("graph")
    gr.add_argument("--runs", type=int, default=20)
    for sp in (g, gr):
        sp.add_argument("--lang", default="en", choices=["en", "zh"])
    a = ap.parse_args()
    set_lang(a.lang)
    if a.cmd == "gen" and a.sample:
        run = Runner()
        ws = candidates()[: a.sample * 2]
        ps = run.yes_prob(ws)
        for w, p in zip(ws, ps):
            print(f"  {p:.2f} {w}")
        keep = [w for w, p in zip(ws, ps) if p >= ACCEPT or w in SHORT_OK][: a.sample]
        for w, d in zip(keep, run.define(keep)):
            print(f"{w:>12}: {d}")
    elif a.cmd == "gen":
        cmd_gen(a)
    else:
        cmd_graph(a)


def sccs(nodes, succ):
    """Tarjan's strongly connected components, iterative."""
    index, low, on, stack, out, i = {}, {}, set(), [], [], 0
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(succ[root]))]
        index[root] = low[root] = i; i += 1
        stack.append(root); on.add(root)
        while work:
            v, it = work[-1]
            for w in it:
                if w not in index:
                    index[w] = low[w] = i; i += 1
                    stack.append(w); on.add(w)
                    work.append((w, iter(succ[w])))
                    break
                if w in on:
                    low[v] = min(low[v], index[w])
            else:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[v])
                if low[v] == index[v]:
                    comp = []
                    while True:
                        w = stack.pop(); on.discard(w); comp.append(w)
                        if w == v:
                            break
                    out.append(comp)
    return out


def acyclic(nodes, succ):
    return all(len(c) == 1 and c[0] not in succ[c[0]] for c in sccs(nodes, {v: [w for w in succ[v] if w in nodes] for v in nodes}))


def grounding_set(nodes, succ, pred, rng):
    """Greedy approximate minimum feedback vertex set: knowing these words, every other kernel word
    can be learned from definitions in some order. Randomized tie-breaking; then drop redundant picks."""
    live = set(nodes)
    s = {v: set(succ[v]) & live for v in live}
    p = {v: set(pred[v]) & live for v in live}
    fvs = []

    def remove(v):
        live.discard(v)
        for w in s.pop(v):
            p[w].discard(v)
        for u in p.pop(v):
            s[u].discard(v)

    while live:
        changed = True
        while changed:  # words with no live definer or no live dependent sit on no cycle
            changed = False
            for v in [v for v in live if not s[v] or not p[v]]:
                remove(v)
                changed = True
        if not live:
            break
        v = max(live, key=lambda v: len(s[v]) * len(p[v]) * (1 + 0.3 * rng.random()))
        fvs.append(v)
        remove(v)
    # prune: a pick is redundant if putting it back leaves the rest acyclic
    keep = list(fvs)
    for v in sorted(fvs, key=lambda v: rng.random()):
        trial = [x for x in keep if x != v]
        if acyclic(set(nodes) - set(trial), succ):
            keep = trial
    return keep


def cmd_graph(a):
    import numpy as np
    from scipy.stats import fisher_exact
    E = {json.loads(l)["word"]: json.loads(l) for l in open(DICT)}
    common = {w: i for i, w in enumerate(candidates())}
    for w, e in E.items():  # recompute with the current base-form rules (v2 fixed "putting" -> "putt")
        e["uses"] = def_words(e["definition"], w, common)
    words = set(E)
    # edge u -> w: u is used in w's definition (u helps define w)
    pred = {w: sorted(set(E[w]["uses"]) & words - {w}) for w in words}
    succ = collections.defaultdict(list)
    for w, us in pred.items():
        for u in us:
            succ[u].append(w)
    succ = {w: succ.get(w, []) for w in words}
    used_total = sum(len(E[w]["uses"]) for w in words)
    used_in = sum(len(pred[w]) for w in words)
    print(f"{len(words)} headwords; {used_in / used_total:.1%} of definition words are themselves headwords")

    # kernel: repeatedly drop words no remaining definition uses
    kernel = set(words)
    while True:
        drop = {w for w in kernel if not any(x in kernel for x in succ[w])}
        if not drop:
            break
        kernel -= drop
    comps = sccs(sorted(kernel), {v: [w for w in succ[v] if w in kernel] for v in kernel})
    core = set(max(comps, key=len))
    print(f"kernel {len(kernel)} ({len(kernel) / len(words):.1%}), core {len(core)} ({len(core) / len(words):.1%}), "
          f"{sum(len(c) > 1 for c in comps)} cyclic components in the kernel")

    rng = np.random.default_rng(0)
    runs = [grounding_set(kernel, succ, pred, rng) for _ in range(a.runs)]
    sizes = [len(r) for r in runs]
    freq = collections.Counter(w for r in runs for w in r)
    print(f"grounding sets over {a.runs} runs: size {min(sizes)}–{max(sizes)} "
          f"(≈{np.mean(sizes) / len(words):.1%} of the dictionary); {len(freq)} words appear in at least one")
    primes = sorted((w for w in freq if freq[w] >= a.runs / 2), key=lambda w: (-freq[w], w))
    always = [w for w in primes if freq[w] == a.runs]
    print(f"in every grounding set: {len(always)}; in at least half: {len(primes)}")
    outdeg = {w: sum(x in kernel for x in succ[w]) for w in kernel}
    print("\nQWEN'S PRIMES (in at least half the grounding sets), by how many kernel words they help define:")
    print("  " + ", ".join(f"{w}({outdeg[w]})" for w in sorted(primes, key=lambda w: -outdeg[w])))

    # NSM primes among headwords: where do they fall?
    PRIMES = nsm_words()
    nsm = [w for w in PRIMES if w in words]
    print(f"\nNSM primes that are headwords: {len(nsm)} of {len(PRIMES)}")
    for name, S in [("kernel", kernel), ("core", core), ("Qwen's primes", set(primes))]:
        hit = [w for w in nsm if w in S]
        print(f"  in {name}: {len(hit)}/{len(nsm)}  [{', '.join(hit)}]")
    # matched by Qwen vocabulary rank (earlier = more frequent): each NSM word vs the 5 nearest-rank non-NSM headwords
    cands = candidates()
    rank = {w: i for i, w in enumerate(cands)}
    pool = sorted((w for w in words if w in rank and w not in PRIMES), key=lambda w: rank[w])
    used, ctrl = set(), []
    for w in sorted((w for w in nsm if w in rank), key=lambda w: rank[w]):
        c = sorted((x for x in pool if x not in used), key=lambda x: abs(rank[x] - rank[w]))[:5]
        used.update(c)
        ctrl += c
    for name, S in [("kernel", kernel), ("Qwen's primes", set(primes))]:
        a1, b1 = sum(w in S for w in nsm if w in rank), sum(w not in S for w in nsm if w in rank)
        a2, b2 = sum(w in S for w in ctrl), sum(w not in S for w in ctrl)
        f = fisher_exact([[a1, b1], [a2, b2]], alternative="greater")
        print(f"  NSM vs rank-matched controls, in {name}: {a1}/{a1 + b1} vs {a2}/{a2 + b2}  (Fisher p = {f.pvalue:.2g})")

    json.dump({"n_words": len(words), "kernel": sorted(kernel), "core": sorted(core),
               "grounding_sets": runs, "primes": primes, "always": always,
               "prime_freq": {w: freq[w] for w in primes}, "outdeg": {w: outdeg[w] for w in primes}},
              open(os.path.join(OUT, "graph.json"), "w"), indent=1)
    print(f"\nwrote {OUT}/graph.json")


if __name__ == "__main__":
    main()
