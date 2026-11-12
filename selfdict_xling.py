"""Are Qwen's primes concepts, or English words? (Hunt 32)

Compares the English self-dictionary (eval/selfdict) with the Chinese one (eval/selfdict_zh).
Qwen translates every Chinese headword into one English word (greedy); the translation is reduced
to its English base form.

Pre-set before looking:
  - alignment: among Chinese primes vs Chinese non-prime headwords (2 per prime, nearest in Qwen
    vocabulary rank, without replacement), both restricted to words whose translation is an English
    headword, the share whose translation is an English prime. One-sided Fisher, primes > controls.
  - shared floor: the concepts that are primes in both, and the biggest primes in each language
    whose translation is not a prime in the other.

    .venv/bin/python selfdict_xling.py translate    # ~5 minutes, writes eval/selfdict_zh/translations.json
    .venv/bin/python selfdict_xling.py compare
"""

import collections
import json
import os
import re
import sys

from scipy.stats import fisher_exact

import selfdict

HERE = os.path.dirname(os.path.abspath(__file__))
EN, ZH = os.path.join(HERE, "eval", "selfdict"), os.path.join(HERE, "eval", "selfdict_zh")
TR = os.path.join(ZH, "translations.json")


def load(d):
    E = {json.loads(l)["word"]: json.loads(l) for l in open(os.path.join(d, "dictionary.jsonl"), encoding="utf-8")}
    return E, json.load(open(os.path.join(d, "graph.json"), encoding="utf-8"))


def translate():
    E, _ = load(ZH)
    words = list(E)
    selfdict.DEF_Q = 'Translate the Chinese word "{w}" into English. Reply with one English word only.'
    run = selfdict.Runner()
    out = {}
    for s in range(0, len(words), 512):
        for w, t in zip(words[s:s + 512], run.define(words[s:s + 512], tokens=8)):
            out[w] = t
        print(f"  {min(s + 512, len(words))}/{len(words)}", flush=True)
    json.dump(out, open(TR, "w", encoding="utf-8"), ensure_ascii=False, indent=0)


def compare():
    En, Ge = load(EN)
    Zh, Gz = load(ZH)
    raw = json.load(open(TR, encoding="utf-8"))
    common = {w: i for i, w in enumerate(selfdict.candidates())}
    tr = {}
    for w, t in raw.items():
        m = re.findall(r"[A-Za-z]+", t)
        if m:
            tr[w] = selfdict.lemma(m[0].lower(), common)
    pe, pz = set(Ge["primes"]), set(Gz["primes"])
    selfdict.set_lang("zh")
    zrank = {w: i for i, w in enumerate(selfdict.candidates())}
    ok = {w for w in Zh if w in tr and tr[w] in En and w in zrank}
    P = sorted((w for w in pz if w in ok), key=lambda w: zrank[w])
    pool = sorted((w for w in ok if w not in pz), key=lambda w: zrank[w])
    used, C = set(), []
    for p in P:
        c = sorted((w for w in pool if w not in used), key=lambda w: abs(zrank[w] - zrank[p]))[:2]
        used.update(c)
        C += c
    a, b = sum(tr[w] in pe for w in P), sum(tr[w] in pe for w in C)
    f = fisher_exact([[a, len(P) - a], [b, len(C) - b]], alternative="greater")
    print(f"Chinese primes {len(pz)}; with an English-headword translation {len(P)}; controls {len(C)}")
    print(f"translation is an English prime: primes {a}/{len(P)} = {a / len(P):.0%}, "
          f"controls {b}/{len(C)} = {b / len(C):.0%}, odds ratio {f.statistic:.2f}, Fisher p = {f.pvalue:.2g}")

    shared = collections.defaultdict(list)
    for w in pz:
        if w in tr and tr[w] in pe:
            shared[tr[w]].append(w)
    eo, zo = Ge["outdeg"], Gz["outdeg"]
    rows = sorted(shared.items(), key=lambda kv: -eo.get(kv[0], 0))
    print(f"\nSHARED FLOOR: {len(shared)} English primes have a Chinese prime that translates to them")
    print("  " + ", ".join(f"{e}={'/'.join(z)}" for e, z in rows))
    zh_only = sorted((w for w in pz if not (w in tr and tr[w] in pe)), key=lambda w: -zo.get(w, 0))[:60]
    print("\nCHINESE PRIMES WHOSE TRANSLATION ISN'T AN ENGLISH PRIME (most used first):")
    print("  " + ", ".join(f"{w}({tr.get(w, '?')})" for w in zh_only))
    back = collections.defaultdict(list)
    for w, t in tr.items():
        back[t].append(w)
    en_only = sorted((w for w in pe if w not in shared), key=lambda w: -eo.get(w, 0))[:60]
    print("\nENGLISH PRIMES WITH NO CHINESE PRIME TRANSLATING TO THEM (most used first; a Chinese headword that does, if any):")
    print("  " + ", ".join(f"{w}({'/'.join(back.get(w, [])[:2]) or '—'})" for w in en_only))
    json.dump({"p": float(f.pvalue), "primes_hit": a, "n_primes": len(P), "ctrl_hit": b, "n_ctrl": len(C),
               "shared": dict(rows), "zh_only": zh_only, "en_only": en_only, "translation": tr},
              open(os.path.join(ZH, "xling.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    {"translate": translate, "compare": compare}[sys.argv[1]]()
