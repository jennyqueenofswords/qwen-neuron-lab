"""Is the metaphor in the machinery? Lakoff & Johnson's primary metaphors in Qwen (Hunt 37).

Conceptual metaphor theory says abstract concepts are built on bodily ones: AFFECTION IS WARMTH,
HAPPY IS UP, DIFFICULTIES ARE BURDENS, UNDERSTANDING IS SEEING… (Lakoff & Johnson 1980; Grady 1997).
Qwen has no body but learned from text written by bodies. Two tests:

  sentences  The model writes 25 short sentences per word. Source words: physical senses only
             ("warm" about temperature, never feelings or people). Target words: any use.
  geometry   Directions per word and layer (mean residual on the word's tokens minus the sentence mean).
             Cosine matrix sources x targets, averaged over layers 8-24. Statistic: mean cosine of the 16
             predicted cells. Pre-set: one-sided permutation test over target labels (10,000 shuffles).
  causal     Hunt 36's edit: set a literal source direction (s = 1, layers 8-24) on a neutral word in a
             description prompt (e.g. "stranger" in "Describe the stranger who knocked…"), sampled 8 times
             per condition at temperature 0.8: the source (warm), its opposite (cold), two physical
             controls (wet, dry), and no edit. Blind raters score the trait (friendliness 1-7) without
             seeing conditions. Pre-set primary: across the 8 dimensions, ratings z-scored per dimension,
             source vs opposite, one-sided Mann-Whitney. Secondary: the same on outputs that use no word
             of the source domain (so the effect can't be the word "warm" leaking into the text); wet vs
             dry, two-sided (should be null).

    .venv/bin/python metaphor.py sentences
    .venv/bin/python metaphor.py geometry
    .venv/bin/python metaphor.py causal
    .venv/bin/python metaphor.py causal2

Replication (causal2), fixed before running after the first causal run lost a third of its texts to
refusals and loops: every prompt is framed as fiction ("Write two sentences of a story that describe…"),
32 samples per condition instead of 8, conditions source / opposite / wet / dry (no-edit dropped to
keep rating cost down), new seeds; same edit, same raters' rubric, same primary and secondary tests.
"""

import argparse
import json
import os
import re
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np
from mlx_lm import load
from transformers import AutoTokenizer

import edit_concepts as ec

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval", "metaphor")
SENSE = {  # literal, physical senses only
    "warm": "temperature (warm water, a warm stove)", "cold": "temperature (cold water, a cold metal rail)",
    "up": "spatial direction or position (moving up, up the stairs)", "down": "spatial direction or position (falling down, down the hill)",
    "heavy": "physical weight (a heavy box)", "light": "physical weight, meaning not heavy (a light bag), never brightness",
    "bright": "visible light (a bright lamp)", "dim": "visible light (a dim lamp)",
    "big": "physical size (a big rock)", "small": "physical size (a small pebble)",
    "close": "physical distance, meaning near (the store is close to the river)", "far": "physical distance (the hill is far from the road)",
    "high": "physical height (a high shelf)", "low": "physical height (a low fence)",
    "straight": "physical shape, meaning not bent (a straight line, a straight road)", "crooked": "physical shape, meaning bent (a crooked branch)",
    "hard": "physical hardness (a hard rock)", "see": "vision with the eyes (I see the bird)", "grasp": "holding with the hand (grasp the rope)",
    "sweet": "taste (sweet fruit)", "wet": "being covered in water (wet grass)", "dry": "having no water (dry sand)",
}
TARGETS = ["affection", "hostility", "happy", "sad", "more", "difficult", "intelligent", "understand", "similar",
           "intimate", "important", "honest", "powerful", "kind"]
GEO_SOURCES = ["warm", "cold", "up", "down", "heavy", "hard", "bright", "see", "grasp", "close", "big", "straight", "high", "sweet"]
PRED = [("warm", "affection"), ("cold", "hostility"), ("up", "happy"), ("down", "sad"), ("up", "more"), ("heavy", "difficult"),
        ("hard", "difficult"), ("bright", "intelligent"), ("see", "understand"), ("grasp", "understand"), ("close", "similar"),
        ("close", "intimate"), ("big", "important"), ("straight", "honest"), ("high", "powerful"), ("sweet", "kind")]
FORMS = {"see": ["see", "sees", "saw", "seen", "seeing"], "grasp": ["grasp", "grasps", "grasped", "grasping"],
         "up": ["up"], "down": ["down"], "far": ["far", "farther", "further"], "close": ["close", "closer", "closest"]}
DIMS = [  # (source, opposite, trait, prompt, edited word, source-domain words)
    ("warm", "cold", "friendliness", "Describe the stranger who knocked on the door this morning, in two sentences.", "stranger",
     r"warm|cold|heat|hot|cool|chill|icy|frost|temperature|cozy|freez|toast"),
    ("up", "down", "happiness", "Describe how Sam is feeling this afternoon, in two sentences.", "Sam",
     r"\bup\b|\bdown\b|high|low|rise|rising|fall|sink|lift|soar|droop|upbeat|downcast"),
    ("heavy", "light", "difficulty", "Describe the task the team faces this week, in two sentences.", "task",
     r"heav|light|weigh|burden|load|lift"),
    ("bright", "dim", "intelligence", "Describe the new student in the class, in two sentences.", "student",
     r"bright|dim|light|glow|shin|dark|dull|brillian"),
    ("big", "small", "importance", "Describe tomorrow's meeting, in two sentences.", "meeting",
     r"\bbig|small|large|huge|tiny|little|massive|minor|major|size"),
    ("close", "far", "similarity", "Compare the two neighboring towns, in two sentences.", "towns",
     r"close|far|near|distan|apart|adjacent"),
    ("high", "low", "power", "Describe the official who signed the letter, in two sentences.", "official",
     r"high|low|top|tall|bottom|rank|senior|junior"),
    ("straight", "crooked", "honesty", "Describe the accountant who handles the company's money, in two sentences.", "accountant",
     r"straight|crooked|bent|twist|upright"),
]
CONDS = ["source", "opposite", "wet", "dry", "none"]
N_SAMPLES = 8
FICTION = {  # causal2 prompts: the same subject word, framed as a story so the model doesn't refuse
    "friendliness": "Write two sentences of a story that describe the stranger who knocked on the door this morning.",
    "happiness": "Write two sentences of a story that describe how Sam is feeling this afternoon.",
    "difficulty": "Write two sentences of a story that describe the task the team faces this week.",
    "intelligence": "Write two sentences of a story that describe the new student in the class.",
    "importance": "Write two sentences of a story that describe tomorrow's meeting.",
    "similarity": "Write two sentences of a story that compare the two neighboring towns.",
    "power": "Write two sentences of a story that describe the official who signed the letter.",
    "honesty": "Write two sentences of a story that describe the accountant who handles the company's money.",
}


def load_model():
    hf = AutoTokenizer.from_pretrained(os.path.join(HERE, "models", "Qwen3.5-4B-4bit"))
    model, _ = load(os.path.join(HERE, "models", "Qwen3.5-4B-4bit"))
    ec.patch(model)
    return model, hf


def chat_ids(hf, q):
    return hf.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False,
                                  enable_thinking=False)


def generate(model, hf, texts, tokens, temp=0.0, seed=0, masks=None, edits=None):
    """Batched generation for prompts of equal token length; optional per-row edits (rm/add dicts as in ec.HOOK)."""
    encs = [hf(t, add_special_tokens=False, return_offsets_mapping=True) for t in texts]
    ids = [e["input_ids"] for e in encs]
    assert len({len(x) for x in ids}) == 1, "prompts in one batch must have equal length"
    if masks is not None:
        ec.HOOK["mask"] = mx.array(masks)
        ec.HOOK["rm"], ec.HOOK["add"] = edits
    eos = {hf.convert_tokens_to_ids(t) for t in ["<|im_end|>", "<|endoftext|>"]}
    mx.random.seed(seed)
    cache = model.make_cache()
    lg = model(mx.array(ids), cache=cache)[:, -1]
    out = [[] for _ in ids]
    done = [False] * len(ids)
    for _ in range(tokens):
        nxt = mx.argmax(lg, axis=-1) if temp == 0 else mx.random.categorical(lg.astype(mx.float32) / temp)
        mx.eval(nxt)
        for r, t in enumerate(nxt.tolist()):
            if not done[r]:
                if t in eos:
                    done[r] = True
                else:
                    out[r].append(t)
        if all(done):
            break
        lg = model(nxt[:, None], cache=cache)[:, -1]
    ec.HOOK["mask"] = ec.HOOK["rm"] = ec.HOOK["add"] = None
    return [hf.decode(o).strip() for o in out]


def cmd_sentences(model, hf):
    os.makedirs(OUT, exist_ok=True)
    reqs = {}
    for w, sense in SENSE.items():
        reqs[w] = (f'Write 25 different short, simple sentences that use the word "{w}" only in its physical sense: '
                   f'{sense}. Never use it about feelings, people\'s character, ideas or anything abstract. '
                   f'One sentence per line, no numbering.')
    for w in TARGETS:
        reqs[w] = f'Write 25 different short, simple sentences that use the word "{w}". One sentence per line, no numbering.'
    texts = {w: chat_ids(hf, q) for w, q in reqs.items()}
    by = {}
    for w, t in texts.items():
        by.setdefault(len(hf(t, add_special_tokens=False)["input_ids"]), []).append(w)
    sents = {}
    for ws in by.values():
        for w, o in zip(ws, generate(model, hf, [texts[w] for w in ws], tokens=700)):
            lines = [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", l).strip() for l in o.split("\n")]
            sents[w] = [l for l in lines if l and re.search(rf"\b({'|'.join(forms(w))})\b", l, re.I)]
        print(f"  {', '.join(f'{w} {len(sents[w])}' for w in ws)}", flush=True)
    json.dump(sents, open(os.path.join(OUT, "sentences.json"), "w"), indent=1)


def forms(w):
    return FORMS.get(w, [w, w + "s", w + "er", w + "est", w + "ed", w + "ing", w + "ly"])


def directions(model, hf, sents):
    """Per word: unit direction per layer, its typical size m and the sentence-mean projection c (as edit_concepts)."""
    lab = ec.Lab.__new__(ec.Lab)
    lab.model, lab.layers = model, model.language_model.model.layers
    out = {}
    for w, ss in sents.items():
        encs = [hf(s, add_special_tokens=False, return_offsets_mapping=True) for s in ss]
        L = max(len(e["input_ids"]) for e in encs)
        caps = lab._run_capture([e["input_ids"] + [0] * (L - len(e["input_ids"])) for e in encs])
        on, allv = [], []
        pat = re.compile(rf"\b({'|'.join(forms(w))})\b", re.I)
        for b, (s, e) in enumerate(zip(ss, encs)):
            spans = [(m.start(), m.end()) for m in pat.finditer(s)]
            idx = [i for i, (a, z) in enumerate(e["offset_mapping"]) if any(a < t and z > s0 for s0, t in spans)]
            on.append(caps[:, b, idx])
            allv.append(caps[:, b, :len(e["input_ids"])])
        on, allv = np.concatenate(on, 1), np.concatenate(allv, 1)
        bg = allv.mean(1)
        d = on.mean(1) - bg
        d /= np.linalg.norm(d, axis=-1, keepdims=True) + 1e-8
        m = ((on - bg[:, None]) * d[:, None]).sum(-1).mean(1)
        c = (bg * d).sum(-1)
        out[w] = (d.astype(np.float32), m.astype(np.float32), c.astype(np.float32))
    return out


def cmd_geometry(model, hf):
    sents = json.load(open(os.path.join(OUT, "sentences.json")))
    dirs = directions(model, hf, {w: sents[w] for w in GEO_SOURCES + TARGETS})
    band = list(range(8, 25))
    C = np.array([[np.mean([dirs[s][0][L] @ dirs[t][0][L] for L in band]) for t in TARGETS] for s in GEO_SOURCES])
    si, ti = {w: i for i, w in enumerate(GEO_SOURCES)}, {w: i for i, w in enumerate(TARGETS)}
    cells = [(si[s], ti[t]) for s, t in PRED]
    obs = np.mean([C[i, j] for i, j in cells])
    rng = np.random.default_rng(0)
    null = []
    for _ in range(10000):
        perm = rng.permutation(len(TARGETS))
        null.append(np.mean([C[i, perm[j]] for i, j in cells]))
    null = np.array(null)
    p = (np.sum(null >= obs) + 1) / (len(null) + 1)
    print(f"mean cosine, predicted metaphor pairs: {obs:.3f}; all pairs {C.mean():.3f}; "
          f"permutation null {null.mean():.3f} ± {null.std():.3f}; one-sided p = {p:.4f}")
    # per pair: rank of the predicted target within its source row
    print("\n  source → target   cos    rank in row (of 14)")
    for s, t in PRED:
        row = C[si[s]]
        print(f"  {s:>8} → {t:<12} {row[ti[t]]:+.3f}  {int((row > row[ti[t]]).sum()) + 1}")
    print("\n  top target for each source:", ", ".join(f"{s}→{TARGETS[int(np.argmax(C[si[s]]))]}" for s in GEO_SOURCES))
    json.dump({"C": C.tolist(), "sources": GEO_SOURCES, "targets": TARGETS, "observed": float(obs), "p": float(p)},
              open(os.path.join(OUT, "geometry.json"), "w"), indent=1)


def cmd_causal2(model, hf):
    cmd_causal(model, hf, v2=True)


def cmd_causal(model, hf, v2=False):
    conds_used = ["source", "opposite", "wet", "dry"] if v2 else CONDS
    n_samples = 32 if v2 else N_SAMPLES
    sents = json.load(open(os.path.join(OUT, "sentences.json")))
    need = sorted({w for d in DIMS for w in d[:2]} | {"wet", "dry"})
    dirs = directions(model, hf, {w: sents[w] for w in need})
    rows = []
    for k, (src, opp, trait, q, word, _) in enumerate(DIMS):
        if v2:
            q = FICTION[trait]
        text = chat_ids(hf, q)
        enc = hf(text, add_special_tokens=False, return_offsets_mapping=True)
        spans = [(m.start(), m.end()) for m in re.finditer(rf"\b{re.escape(word)}\b", text)]
        mask = [1.0 if any(a < t and z > s0 for s0, t in spans) else 0.0 for a, z in enc["offset_mapping"]]
        assert sum(mask) > 0, word
        conds = {"source": src, "opposite": opp, "wet": "wet", "dry": "dry", "none": None}
        batch = [(c, i) for c in conds_used for i in range(n_samples)]
        B = len(batch)
        addv = {L: (np.zeros((B, 2560), np.float32), np.zeros(B, np.float32)) for L in ec.ADD_LAYERS}
        for b, (c, _) in enumerate(batch):
            w = conds[c]
            if w:
                d, m, cc = dirs[w]
                for L in ec.ADD_LAYERS:
                    addv[L][0][b] = d[L]
                    addv[L][1][b] = cc[L] + 1.0 * m[L]
        edits = (None, {L: (mx.array(v[0]), mx.array(v[1])) for L, v in addv.items()})
        outs = generate(model, hf, [text] * B, tokens=90, temp=0.8, seed=k + (100 if v2 else 0), masks=[mask] * B, edits=edits)
        for (c, i), o in zip(batch, outs):
            rows.append({"dim": trait, "source": src, "opposite": opp, "cond": c, "sample": i, "prompt": q, "text": o})
        print(f"  {trait}: e.g. source: {outs[0][:110]!r} | opposite: {outs[n_samples][:110]!r}", flush=True)
    json.dump(rows, open(os.path.join(OUT, "causal_generations_v2.json" if v2 else "causal_generations.json"), "w"), indent=1)


def main():
    cmd = sys.argv[1]
    model, hf = load_model()
    {"sentences": cmd_sentences, "geometry": cmd_geometry, "causal": cmd_causal, "causal2": cmd_causal2}[cmd](model, hf)


if __name__ == "__main__":
    main()
