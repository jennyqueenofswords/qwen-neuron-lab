"""Steering lab for Qwen3.5-4B on MLX.

Four experiments, all on the local text model:

  compare   add a concept direction to the residual stream, watch the output change
  find      rank MLP neurons by how much more they fire for a concept than for neutral text
  neuron    show which tokens make one neuron fire hardest (what people on X screenshot)
  boost     pin specific neurons to a value while the model writes

Run `steer.py <command> -h` for options. See STEERING.md for a walkthrough.
"""

import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np
from mlx_lm import generate, load, stream_generate
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.sample_utils import make_sampler

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODELS = {
    "chat": os.path.join(MODELS_DIR, "Qwen3.5-4B-4bit"),        # instruction-tuned: answers you
    "base": os.path.join(MODELS_DIR, "Qwen3.5-4B-Base-4bit"),   # pretrained only: continues text
}

# Shared hook state. The patched layers read and write this on every forward pass.
HOOKS = {
    "record": False,
    "resid": {},   # layer -> list of [tokens, hidden] residual outputs
    "acts": {},    # layer -> list of [tokens, 9216] MLP neuron activations
    "add": {},     # layer -> vector added to the residual output
    "pin": {},     # layer -> {neuron: value} activations forced during the pass
    "watch": None,  # (layer, neuron) whose gate and value get recorded separately
    "watched": [],  # list of (gate [tokens], value [tokens]) for the watched neuron
    "index": None,  # running top-k buffers while building the neuron index
    "peaks": None,  # layer -> (peak + x + output, most negative output) for the current sentence
    "bpin": None,   # (layer, neuron index per batch row, value per batch row): a different pin in each row
}

# How many examples the index keeps per neuron, per list.
INDEX_K = {"pos": 64, "nn": 24, "neg": 24}
INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index")


def update_index(layer, g, v):
    """Merge one chunk's activations into the running top-k lists for every neuron in a layer.

    pos: strongest positive outputs (either sign route), with the gate's sign kept
    nn:  strongest positive outputs that came through the double-negative route
    neg: strongest negative outputs, with the gate's sign kept
    """
    ix = HOOKS["index"]
    g = g.astype(mx.float32)
    v = v.astype(mx.float32)
    a = g * v
    T = a.shape[0]
    valid = (mx.arange(T) > 0)[:, None]  # skip each chunk's first token (attention sink)
    gpos = (g > 0).astype(mx.int8)
    scores = {
        "pos": mx.where(valid & (a > 0), a, -1e9),
        "nn": mx.where(valid & (g < 0) & (v < 0), a, -1e9),
        "neg": mx.where(valid, -a, -1e9),
    }
    bufs = ix["layers"].setdefault(layer, {})
    for name, score in scores.items():
        K = INDEX_K[name]
        if T > K:
            idx = mx.argpartition(-score, kth=K - 1, axis=0)[:K]
        else:
            idx = mx.broadcast_to(mx.arange(T)[:, None], score.shape)
        cand = (mx.take_along_axis(score, idx, 0), idx.astype(mx.int32) + ix["offset"],
                mx.take_along_axis(gpos, idx, 0))
        if name not in bufs:
            n = a.shape[1]
            bufs[name] = (mx.full((K, n), -1e9), mx.full((K, n), -1, dtype=mx.int32), mx.zeros((K, n), dtype=mx.int8))
        merged = [mx.concatenate([b, c]) for b, c in zip(bufs[name], cand)]
        keep = mx.argpartition(-merged[0], kth=K - 1, axis=0)[:K]
        bufs[name] = tuple(mx.take_along_axis(m, keep, 0) for m in merged)

CORPUS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus", "corpus.jsonl")

# Same sentence frames, filled with the concept or with something bland. The difference
# between the two averages is what's specific to the concept, not to the sentence shape.
TEMPLATES = [
    "I can't stop thinking about {}.",
    "Let me tell you about {}.",
    "Everything today reminded me of {}.",
    "The book was mostly about {}.",
    "She spent the whole afternoon on {}.",
    "My favorite thing in the world is {}.",
    "We talked for hours about {}.",
    "Honestly, {} changed my life.",
    "The documentary focused on {}.",
    "He painted a picture of {}.",
]
NEUTRAL = ["things", "stuff", "the schedule", "some paperwork", "the usual", "a list"]

# A small mixed corpus for the neuron explorer.
CORPUS = [
    "The waves crashed against the rocks as the tide came in.",
    "Sailors navigated the open ocean using the stars.",
    "Coral reefs are home to thousands of species of fish.",
    "She baked a chocolate cake for her grandmother's birthday.",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "Fresh basil and garlic make the pasta sauce sing.",
    "The stock market fell sharply after the announcement.",
    "Interest rates rose for the third quarter in a row.",
    "He deposited the check at the bank on Monday morning.",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "The server returned a 404 error when the page was requested.",
    "for i in range(10): print(i * i)",
    "I love you more than words can say.",
    "Her heart broke when the letter arrived.",
    "They laughed until tears ran down their faces.",
    "I am so angry I could scream.",
    "The cat curled up on the warm windowsill and purred.",
    "A golden retriever chased the ball across the park.",
    "The eagle soared high above the mountain ridge.",
    "The Golden Gate Bridge was shrouded in fog this morning.",
    "Paris is famous for the Eiffel Tower and its cafes.",
    "Tokyo's trains run with remarkable precision.",
    "The quarterback threw a perfect spiral for the touchdown.",
    "She won the marathon in under three hours.",
    "The referee blew the whistle and the crowd roared.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The mitochondria is the powerhouse of the cell.",
    "Black holes bend light with their immense gravity.",
    "The senator introduced a bill to reform healthcare.",
    "Voters lined up for hours to cast their ballots.",
    "The court ruled the law unconstitutional.",
    "Once upon a time, a dragon guarded a mountain of gold.",
    "The wizard raised his staff and the storm obeyed.",
    "Ahoy, matey! Hoist the sails and find the treasure!",
    "The guitar solo built to a screaming crescendo.",
    "Beethoven wrote his ninth symphony while almost deaf.",
    "She hummed a lullaby until the baby fell asleep.",
    "It was raining, so we stayed in and played cards.",
    "Snow blanketed the village overnight.",
    "The desert sun beat down without mercy.",
    "Seventeen plus twenty-five equals forty-two.",
    "The triangle has three sides and three angles.",
    "In 1969, astronauts walked on the moon.",
    "The Roman Empire stretched from Britain to Egypt.",
    "Please submit the form by Friday at 5 pm.",
    "Thank you for your email; I will get back to you soon.",
    "The meeting has been moved to Tuesday.",
    "I feel calm and peaceful sitting by the lake.",
    "The tarot card showed the Queen of Swords.",
    "Ohana means family, and family means nobody gets left behind.",
]


def patch(model):
    """Swap each decoder layer and MLP onto a subclass that consults HOOKS.

    Changing __class__ keeps the loaded weights and every attribute (is_linear etc.)
    exactly as they were; only __call__ gains the hook.
    """
    layers = model.language_model.model.layers
    for i, layer in enumerate(layers):
        base_layer = layer.__class__
        base_mlp = layer.mlp.__class__

        def layer_call(self, x, mask=None, cache=None, _i=i, _base=base_layer):
            out = _base.__call__(self, x, mask=mask, cache=cache)
            if _i in HOOKS["add"]:
                out = out + HOOKS["add"][_i].astype(out.dtype)
            if HOOKS["record"]:
                HOOKS["resid"].setdefault(_i, []).append(out[0].astype(mx.float32))
            return out

        def mlp_call(self, x, _i=i):
            g = self.gate_proj(x)
            g = g * mx.sigmoid(g)  # silu: same sign as the raw gate
            v = self.up_proj(x)
            a = g * v  # swiglu: one value per neuron
            w = HOOKS["watch"]
            if w and w[0] == _i:
                HOOKS["watched"].append((g[0, :, w[1]].astype(mx.float32), v[0, :, w[1]].astype(mx.float32)))
            if HOOKS["index"] is not None:
                update_index(_i, g[0], v[0])
            if HOOKS["peaks"] is not None:
                g1, v1 = g[0, 1:].astype(mx.float32), v[0, 1:].astype(mx.float32)  # skip attention sink
                a1 = g1 * v1
                HOOKS["peaks"][_i] = (mx.where((g1 > 0) & (v1 > 0), a1, 0.0).max(axis=0), a1.min(axis=0))
            pins = HOOKS["pin"].get(_i)
            if pins:
                idx = mx.array(list(pins.keys()))
                vals = mx.array(list(pins.values()), dtype=a.dtype)
                mask = mx.zeros((a.shape[-1],), dtype=mx.bool_)
                mask[idx] = True
                full = mx.zeros((a.shape[-1],), dtype=a.dtype)
                full[idx] = vals
                a = mx.where(mask, full, a)
            bp = HOOKS["bpin"]
            if bp is not None and bp[0] == _i:
                hit = (mx.arange(a.shape[-1])[None, :] == bp[1][:, None])[:, None, :]  # [batch, 1, neurons]
                a = mx.where(hit, bp[2][:, None, None].astype(a.dtype), a)
            if HOOKS["record"]:
                HOOKS["acts"].setdefault(_i, []).append(a[0].astype(mx.float32))
            return self.down_proj(a)

        layer.__class__ = type(f"Hooked{base_layer.__name__}", (base_layer,), {"__call__": layer_call})
        layer.mlp.__class__ = type(f"Hooked{base_mlp.__name__}", (base_mlp,), {"__call__": mlp_call})
    return layers


def run(model, tokenizer, text):
    """One forward pass over raw text with recording on. Returns the token strings."""
    ids = tokenizer.encode(text)
    inner = model.language_model.model
    h = inner.embed_tokens(mx.array([ids]))
    fa_mask = create_attention_mask(h, None)
    ssm_mask = create_ssm_mask(h, None)
    HOOKS["record"] = True
    for layer in inner.layers:
        h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=None)
    HOOKS["record"] = False
    mx.eval(h, *[v[-1] for v in HOOKS["resid"].values()], *[v[-1] for v in HOOKS["acts"].values()])
    return [tokenizer.decode([t]) for t in ids]


def clear_records():
    HOOKS["resid"].clear()
    HOOKS["acts"].clear()


def mean_over(texts, model, tokenizer, key, skip_first=True):
    """Average each layer's recorded vectors over all tokens of all texts."""
    clear_records()
    for t in texts:
        run(model, tokenizer, t)
    start = 1 if skip_first else 0
    means = {}
    for layer, chunks in HOOKS[key].items():
        means[layer] = mx.concatenate([c[start:] for c in chunks]).mean(axis=0)
    norms = None
    if key == "resid":
        # Typical size of the residual stream at each layer, so strengths are comparable.
        # Position 0 is skipped: attention-sink tokens carry huge outlier norms.
        norms = {
            layer: float(mx.median(mx.linalg.norm(mx.concatenate([c[start:] for c in chunks]), axis=-1)))
            for layer, chunks in HOOKS[key].items()
        }
    clear_records()
    return means, norms


def concept_texts(args):
    if args.concept_file:
        with open(args.concept_file) as f:
            return [line.strip() for line in f if line.strip()], None
    pos = [t.format(args.concept) for t in TEMPLATES]
    neg = [t.format(n) for t in TEMPLATES for n in NEUTRAL]
    return pos, neg


def chat(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False, enable_thinking=False
    )


def write(model, tokenizer, prompt, max_tokens, temp, seed, raw=False):
    mx.random.seed(seed)
    # Raw mode skips the chat template: the model just continues your text, GPT-2 style.
    sampler = make_sampler(temp=temp)
    if not raw:
        return generate(model, tokenizer, chat(tokenizer, prompt), max_tokens=max_tokens,
                        sampler=sampler, verbose=False).strip()
    # Decode prompt and continuation together so the seam keeps its space ("I went", not "Iwent").
    ids = tokenizer.encode(prompt)
    new = [r.token for r in stream_generate(model, tokenizer, ids, max_tokens=max_tokens, sampler=sampler)]
    return tokenizer.decode(ids + new)


def cmd_compare(model, tokenizer, args):
    pos, neg = concept_texts(args)
    if neg is None:
        neg = [t.format(n) for t in TEMPLATES for n in NEUTRAL]
    pos_mean, norms = mean_over(pos, model, tokenizer, "resid")
    neg_mean, _ = mean_over(neg, model, tokenizer, "resid")

    layers = args.layers or [args.layer]
    label = args.concept or os.path.basename(args.concept_file)
    print(f"\nconcept: {label}   layers: {layers}   prompt: {args.prompt!r}\n")
    for strength in args.strengths:
        HOOKS["add"].clear()
        if strength:
            for L in layers:
                d = pos_mean[L] - neg_mean[L]
                d = d / mx.linalg.norm(d)
                HOOKS["add"][L] = d * strength * norms[L]
        out = write(model, tokenizer, args.prompt, args.max_tokens, args.temp, args.seed, args.raw)
        print(f"── strength {strength:g} " + "─" * 50)
        print(out + "\n")
    HOOKS["add"].clear()


def cmd_find(model, tokenizer, args):
    pos, neg = concept_texts(args)
    if neg is None:
        neg = [t.format(n) for t in TEMPLATES for n in NEUTRAL]
    pos_mean, _ = mean_over(pos, model, tokenizer, "acts")
    neg_mean, _ = mean_over(neg, model, tokenizer, "acts")
    rows = []
    layers = args.layers or range(len(pos_mean))
    for L in layers:
        diff = pos_mean[L] - neg_mean[L]
        top = mx.argsort(-diff)[: args.top]
        for n in top.tolist():
            rows.append((float(diff[n]), L, n, float(pos_mean[L][n]), float(neg_mean[L][n])))
    rows.sort(reverse=True)
    print(f"\nneurons that fire more for {args.concept or args.concept_file!r} than for neutral text\n")
    print(f"{'neuron':>12}  {'difference':>10}  {'concept':>8}  {'neutral':>8}")
    for diff, L, n, p, q in rows[: args.top]:
        print(f"{f'L{L}/N{n}':>12}  {diff:>10.3f}  {p:>8.3f}  {q:>8.3f}")
    print("\nnext: steer.py neuron <L> <N>   to see what it really responds to")


def load_contrast_file(path):
    """Parse a contrast file into concept groups and control sets.

    [name | control-name]   a concept group, compared against that control section
    [control-name]          a control section (its name starts with "control")
    one sentence per line; lines starting with # are comments
    """
    sections, cur = {}, None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("[") and s.endswith("]"):
                name, _, ctrl = s[1:-1].partition("|")
                cur = name.strip()
                sections[cur] = {"control": ctrl.strip() or "control", "lines": []}
            else:
                sections[cur]["lines"].append(s)
    groups = {k: v for k, v in sections.items() if not k.startswith("control")}
    controls = {k: v["lines"] for k, v in sections.items() if k.startswith("control")}
    for name, g in groups.items():
        if g["control"] not in controls:
            sys.exit(f"group [{name}] points at [{g['control']}], which isn't in the file")
    return groups, controls


def sentence_peaks(model, tokenizer, texts):
    """For each sentence: every neuron's peak + x + output and its most negative output.

    Returns two arrays shaped [sentences, layers, neurons].
    """
    inner = model.language_model.model
    pos, neg = [], []
    for t in texts:
        HOOKS["peaks"] = {}
        h = inner.embed_tokens(mx.array([tokenizer.encode(t)]))
        fa_mask = create_attention_mask(h, None)
        ssm_mask = create_ssm_mask(h, None)
        for lyr in inner.layers:
            h = lyr(h, mask=ssm_mask if lyr.is_linear else fa_mask, cache=None)
        layers = sorted(HOOKS["peaks"])
        p = mx.stack([HOOKS["peaks"][L][0] for L in layers])
        n = mx.stack([HOOKS["peaks"][L][1] for L in layers])
        mx.eval(p, n)
        pos.append(np.array(p))
        neg.append(np.array(n))
    HOOKS["peaks"] = None
    return np.stack(pos), np.stack(neg)


def cmd_contrast(model, tokenizer, args):
    groups, controls = load_contrast_file(args.file)
    texts = sorted({t for g in groups.values() for t in g["lines"]} | {t for c in controls.values() for t in c})
    print(f"\nrunning {len(texts)} sentences through all 32 layers...")
    P, Nn = sentence_peaks(model, tokenizer, texts)
    row = {t: i for i, t in enumerate(texts)}

    def effect(stat, concept_lines, control_lines, sign):
        """Standardized difference (concept minus control), floored so silent neurons can't explode."""
        c = stat[[row[t] for t in concept_lines]] * sign
        k = stat[[row[t] for t in control_lines]] * sign
        sd = np.sqrt((c.var(0) + k.var(0)) / 2)
        floor = 0.25 + 0.05 * np.maximum(np.abs(c.mean(0)), np.abs(k.mean(0)))
        return (c.mean(0) - k.mean(0)) / np.maximum(sd, floor), c.mean(0) * sign

    names = list(groups)
    results = {}
    for direction, stat, sign in (("for", P, 1), ("against", Nn, -1)):
        scores, levels = [], []
        for name in names:
            d, lvl = effect(stat, groups[name]["lines"], controls[groups[name]["control"]], sign)
            scores.append(d)
            levels.append(lvl)
        scores, levels = np.stack(scores), np.stack(levels)
        weakest = scores.min(0)
        # A neuron has to actually move on the concept, not just differ from a silent control.
        weakest[np.abs(levels).mean(0) < args.min_level] = -np.inf
        results[direction] = (weakest, scores, levels)

    short = [n[:8] for n in names]
    header = "  ".join(f"{s:>8}" for s in short)
    for direction, title in (("for", "FIRES FOR the concept (+ × + peak goes up, in every group)"),
                             ("against", "FIRES AGAINST the concept (output goes more negative, in every group)")):
        weakest, scores, levels = results[direction]
        flat = np.argsort(-weakest, axis=None)[: args.top]
        print(f"\n{title}")
        print(f"{'neuron':>11}  {'weakest':>7}   per-group score: {header}   level")
        for f in flat:
            L, n = np.unravel_index(f, weakest.shape)
            if not np.isfinite(weakest[L, n]):
                break
            per = "  ".join(f"{scores[g, L, n]:>8.1f}" for g in range(len(names)))
            print(f"{f'L{L}/N{n}':>11}  {weakest[L, n]:>7.2f}   {' ' * 17}{per}   {levels[:, L, n].mean():>5.1f}")
    if args.show:
        print(f"\nTRACKED NEURONS{' ' * 21}per-group score: {header}   level")
        for spec in args.show:
            L, n = (int(x) for x in spec.split(":"))
            for direction in ("for", "against"):
                weakest, scores, levels = results[direction]
                per = "  ".join(f"{scores[g, L, n]:>8.1f}" for g in range(len(names)))
                w = scores[:, L, n].min()
                print(f"{f'L{L}/N{n}':>11} {direction:>7}  {w:>7.2f}   {' ' * 8}{per}   {levels[:, L, n].mean():>5.1f}")
    print("\nscore = (concept − control) / typical spread, per group; ranked by the WEAKEST group,")
    print("so a neuron must respond to every phrasing and language. level = its average peak on the concept.")
    print("next: steer.py neuron <L> <N>, then probe near-misses, then boost.")


def load_passages(args):
    if args.text_file:
        with open(args.text_file) as f:
            return [("file", line.strip()) for line in f if line.strip()], args.text_file
    if args.quick or not os.path.exists(CORPUS_FILE):
        return [("mixed", t) for t in CORPUS], "built-in sentences"
    with open(CORPUS_FILE) as f:
        rows = [json.loads(line) for line in f]
    return [(r["source"], r["text"]) for r in rows], os.path.relpath(CORPUS_FILE)


def watch_neuron(model, tokenizer, passages, layer, neuron, max_len=512):
    """Run each passage up to `layer` and return (source, ids, gate, value) for one neuron.

    Layers after the watched one can't affect it, so the forward pass stops there.
    """
    inner = model.language_model.model
    HOOKS["watch"] = (layer, neuron)
    rows = []
    for source, text in passages:
        ids = tokenizer.encode(text)
        for start in range(0, len(ids), max_len):
            chunk = ids[start:start + max_len]
            if len(chunk) < 2:
                continue
            HOOKS["watched"].clear()
            h = inner.embed_tokens(mx.array([chunk]))
            fa_mask = create_attention_mask(h, None)
            ssm_mask = create_ssm_mask(h, None)
            for lyr in inner.layers[: layer + 1]:
                h = lyr(h, mask=ssm_mask if lyr.is_linear else fa_mask, cache=None)
            g, v = HOOKS["watched"][0]
            mx.eval(g, v)
            rows.append((source, chunk, np.array(g), np.array(v)))
    HOOKS["watch"] = None
    HOOKS["watched"].clear()
    return rows


def chunks(tokenizer, passages, max_len=512):
    """Split passages into token chunks. Yields (source, ids)."""
    for source, text in passages:
        ids = tokenizer.encode(text)
        for start in range(0, len(ids), max_len):
            if len(ids[start:start + max_len]) >= 2:
                yield source, ids[start:start + max_len]


def index_path(args):
    name = os.path.basename(os.path.normpath(MODELS.get(args.model, args.model)))
    return os.path.join(INDEX_DIR, name + ("-quick" if getattr(args, "quick", False) else "") + ".npz")


def cmd_index(model, tokenizer, args):
    """One pass over the corpus through all layers, keeping every neuron's strongest examples."""
    import time

    passages, label = load_passages(args)
    todo = list(chunks(tokenizer, passages))
    total = sum(len(c) for _, c in todo)
    inner = model.language_model.model
    HOOKS["index"] = {"layers": {}, "offset": 0}
    flat, src_of, starts, names = [], [], [], {}
    t0 = time.time()
    for n, (source, ids) in enumerate(todo):
        starts.append(len(flat))
        flat += ids
        src_of += [names.setdefault(source, len(names))] * len(ids)
        h = inner.embed_tokens(mx.array([ids]))
        fa_mask = create_attention_mask(h, None)
        ssm_mask = create_ssm_mask(h, None)
        for lyr in inner.layers:
            h = lyr(h, mask=ssm_mask if lyr.is_linear else fa_mask, cache=None)
        mx.eval([b for bufs in HOOKS["index"]["layers"].values() for lists in bufs.values() for b in lists])
        HOOKS["index"]["offset"] = len(flat)
        if n % 25 == 0 or n == len(todo) - 1:
            done = len(flat)
            rate = done / (time.time() - t0)
            print(f"  {done:,}/{total:,} tokens  {rate:.0f} tok/s  ~{(total - done) / rate / 60:.1f} min left", flush=True)
    out = {"ids": np.array(flat, dtype=np.int32), "src": np.array(src_of, dtype=np.uint8),
           "starts": np.array(starts, dtype=np.int64), "src_names": np.array(list(names)),
           "label": np.array(label)}
    for L, bufs in HOOKS["index"]["layers"].items():
        for name, (vals, toks, gpos) in bufs.items():
            out[f"L{L}_{name}_val"] = np.array(vals.astype(mx.float16))
            out[f"L{L}_{name}_tok"] = np.array(toks)
            out[f"L{L}_{name}_gpos"] = np.array(gpos)
    HOOKS["index"] = None
    os.makedirs(INDEX_DIR, exist_ok=True)
    np.savez(index_path(args), **out)
    print(f"\nindexed {len(flat):,} tokens x {len(inner.layers)} layers x 9,216 neurons -> {os.path.relpath(index_path(args))}")


def cmd_neuron(model, tokenizer, args):
    L, N = args.layer, args.neuron
    path = index_path(args)
    use_index = os.path.exists(path) and not (args.text_file or args.live)

    if use_index:
        ix = np.load(path)
        flat, src_of, starts, names = ix["ids"], ix["src"], ix["starts"], list(ix["src_names"])
        label = f"{ix['label']}, from index"

        def lst(name, sign=1):
            vals = ix[f"L{L}_{name}_val"][:, N].astype(np.float32)
            keep = vals > -1e8
            toks, gpos = ix[f"L{L}_{name}_tok"][:, N][keep], ix[f"L{L}_{name}_gpos"][:, N][keep]
            vals = vals[keep]
            o = np.argsort(-vals)
            return sign * vals[o], toks[o], gpos[o]

        pos_v, pos_t, pos_g = lst("pos")
        nn_v, nn_t, _ = lst("nn")
        neg_v, neg_t, neg_g = lst("neg", sign=-1)
        n_tokens = len(flat) - len(starts)  # chunk-first tokens are never scored
        stats = None
    else:
        passages, label = load_passages(args)
        rows = watch_neuron(model, tokenizer, passages, L, N)
        names = sorted({r[0] for r in rows})
        flat = np.concatenate([r[1] for r in rows])
        lens = [len(r[1]) for r in rows]
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        src_of = np.concatenate([np.full(n, names.index(r[0])) for n, r in zip(lens, rows)])
        g = np.concatenate([r[2] for r in rows])
        v = np.concatenate([r[3] for r in rows])
        act = g * v
        act[starts] = np.nan  # each chunk's first token is an attention sink
        ok = ~np.isnan(act)
        order = np.argsort(-np.where(ok, act, -np.inf))
        pos_t = order[ok[order] & (act[order] > 0)][:100]
        pos_v, pos_g = act[pos_t], (g[pos_t] > 0)
        nn_t = order[ok[order] & (g[order] < 0) & (v[order] < 0)]
        nn_v = act[nn_t]
        neg_t = np.argsort(np.where(ok, act, np.inf))[:100]
        neg_v, neg_g = act[neg_t], (g[neg_t] > 0)
        n_tokens = int(ok.sum())
        stats = act[ok]

    def context(t):
        k = np.searchsorted(starts, t, side="right") - 1
        s, e = starts[k], (starts[k + 1] if k + 1 < len(starts) else len(flat))
        left = tokenizer.decode(flat[max(s, t - 12):t].tolist()).replace("\n", "↵")[-38:]
        tok = tokenizer.decode([int(flat[t])]).replace("\n", "↵")
        right = tokenizer.decode(flat[t + 1:min(e, t + 5)].tolist()).replace("\n", "↵")[:20]
        return f"{names[src_of[t]]:<8} {left:>38}[{tok}]{right}"

    def show(vals, toks, regimes):
        if len(vals) == 0:
            print("   (none)")
        for val, t, r in list(zip(vals, toks, regimes))[: args.top]:
            print(f"{val:>8.2f}  {r}  {context(t)}")

    pp = pos_g.astype(bool)
    print(f"\nL{L}/N{N} over {n_tokens:,} tokens ({label}). Target token in [brackets].")
    print("\nPOSITIVE POLE  + × +  gate open, value positive: what the neuron is mainly for")
    show(pos_v[pp], pos_t[pp], ["+×+"] * int(pp.sum()))
    print("\nDOUBLE-NEGATIVE ECHO  − × −  gate and value both negative, so the output is positive")
    print("by a different route. Often looks like the negative pole. Don't read it as the main job.")
    show(nn_v, nn_t, ["−×−"] * len(nn_v))
    print("\nNEGATIVE POLE  strongest negative outputs: what it pushes against")
    show(neg_v, neg_t, ["+×−" if gp else "−×+" for gp in neg_g])

    n_top = len(pos_v)
    n_nn = int((~pp).sum())
    srcs = {}
    for t in pos_t:
        srcs[names[src_of[t]]] = srcs.get(names[src_of[t]], 0) + 1
    print(f"\nof the {n_top} strongest positive outputs: {n_top - n_nn} are + × +, {n_nn} are − × −")
    print("where they come from: " + ", ".join(f"{k} {c}" for k, c in sorted(srcs.items(), key=lambda kv: -kv[1])))
    top_val = pos_v[0] if n_top else 0.0
    if stats is not None:
        print(f"range: min {stats.min():.2f}, median {np.median(stats):.3f}, p99 {np.percentile(stats, 99):.2f}, max {stats.max():.2f}")
    else:
        print(f"range: min {neg_v[0] if len(neg_v) else 0:.2f}, max {top_val:.2f}")
    print("pin it with: steer.py boost --pin " + f"{L}:{N}={max(top_val * 3, 1):.1f}")


def cmd_probe(model, tokenizer, args):
    """One line per test sentence: the neuron's peak, and the token where it happened."""
    with open(args.text_file) as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    def as_text(line):
        if not args.chat:
            return line
        q, _, a = line.partition(" || ")  # "user question || assistant reply"
        return chat(tokenizer, q) + a

    rows = watch_neuron(model, tokenizer, [("file", as_text(t)) for t in lines if not t.startswith("#")], args.layer, args.neuron)
    rows = iter(rows)
    what = "most negative output" if args.negative else "peak (+ × + route only)"
    print(f"\nL{args.layer}/N{args.neuron}: {what} per line\n")
    for line in lines:
        if line.startswith("#"):
            print(f"\n{line}")
            continue
        _, ids, g, v = next(rows)
        a = -(g * v) if args.negative else np.where((g > 0) & (v > 0), g * v, 0.0)
        a[0] = 0.0
        p = int(np.argmax(a))
        if args.negative:
            a = -a
        tok = tokenizer.decode([ids[p]]).replace("\n", "↵")
        where = ""
        if args.chat:
            n_prompt = len(tokenizer.encode(chat(tokenizer, line.partition(" || ")[0])))
            where = "Q " if p < n_prompt else "A "
        print(f"{a[p]:>7.2f}  {where}[{tok}]".ljust(26) + f" {line[:70]}")


def cmd_boost(model, tokenizer, args):
    pins = {}
    for spec in args.pin:
        loc, val = spec.split("=")
        L, N = loc.split(":")
        pins.setdefault(int(L), {})[int(N)] = float(val)
    print(f"\nprompt: {args.prompt!r}\n")
    print("── normal " + "─" * 50)
    print(write(model, tokenizer, args.prompt, args.max_tokens, args.temp, args.seed, args.raw) + "\n")
    HOOKS["pin"].update(pins)
    print("── pinned " + ", ".join(args.pin) + " " + "─" * 20)
    print(write(model, tokenizer, args.prompt, args.max_tokens, args.temp, args.seed, args.raw) + "\n")
    HOOKS["pin"].clear()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="chat", help="chat, base (implies --raw), or a path to an MLX model")
    sub = p.add_subparsers(dest="cmd", required=True)

    def concept_args(sp):
        g = sp.add_mutually_exclusive_group(required=True)
        g.add_argument("--concept", help='a word or phrase, e.g. "the ocean"')
        g.add_argument("--concept-file", help="your own example sentences, one per line")

    def gen_args(sp):
        sp.add_argument("--prompt", help='chat default: "Tell me about your weekend." raw default: "Last weekend, I"')
        sp.add_argument("--raw", action="store_true", help="no chat template: the model continues your text, like early GPT")
        sp.add_argument("--max-tokens", type=int, default=120)
        sp.add_argument("--temp", type=float, default=0.0, help="0 = same output every run")
        sp.add_argument("--seed", type=int, default=0)

    c = sub.add_parser("compare", help="steer with a concept direction at several strengths")
    concept_args(c)
    gen_args(c)
    c.add_argument("--layer", type=int, default=12)
    c.add_argument("--layers", type=int, nargs="+", help="steer several layers at once")
    c.add_argument("--strengths", type=float, nargs="+", default=[0, 0.3, 0.5, 0.7, 1.0])

    f = sub.add_parser("find", help="rank neurons by concept selectivity")
    concept_args(f)
    f.add_argument("--layers", type=int, nargs="+")
    f.add_argument("--top", type=int, default=15)

    n = sub.add_parser("neuron", help="what makes one neuron fire")
    n.add_argument("layer", type=int)
    n.add_argument("neuron", type=int)
    n.add_argument("--text-file", help="your own text, one passage per line")
    n.add_argument("--quick", action="store_true", help="use the 50 built-in sentences instead of corpus/")
    n.add_argument("--live", action="store_true", help="rescan the corpus instead of reading the index")
    n.add_argument("--top", type=int, default=12)

    ct = sub.add_parser("contrast", help="concept vs matched controls, scale-free, must agree across every group")
    ct.add_argument("file", help="contrast file: [group | control] sections, see test/concepts/")
    ct.add_argument("--top", type=int, default=15)
    ct.add_argument("--min-level", type=float, default=0.5, help="ignore neurons that barely move on the concept")
    ct.add_argument("--show", nargs="+", metavar="L:N", help="also print these neurons' scores, ranked or not")

    pr = sub.add_parser("probe", help="peak activation per line of a test file (lines starting with # are headings)")
    pr.add_argument("layer", type=int)
    pr.add_argument("neuron", type=int)
    pr.add_argument("text_file")
    pr.add_argument("--negative", action="store_true", help="show each line's most negative output instead")
    pr.add_argument("--chat", action="store_true", help="lines are 'question || reply', wrapped in the chat format; "
                    "Q/A marks whether the peak was in the prompt part or the reply")

    x = sub.add_parser("index", help="scan the corpus once so `neuron` is instant for every neuron")
    x.add_argument("--quick", action="store_true", help="index the 50 built-in sentences (for testing)")
    x.set_defaults(text_file=None)

    b = sub.add_parser("boost", help="pin neurons to a value while generating")
    b.add_argument("--pin", nargs="+", required=True, metavar="L:N=VALUE")
    gen_args(b)

    args = p.parse_args()
    if hasattr(args, "raw"):
        args.raw = args.raw or args.model == "base"  # a base model has no chat template to use
        if args.prompt is None:
            args.prompt = "Last weekend, I" if args.raw else "Tell me about your weekend."
    model, tokenizer = load(MODELS.get(args.model, args.model))
    patch(model)
    {"compare": cmd_compare, "find": cmd_find, "neuron": cmd_neuron, "boost": cmd_boost,
     "index": cmd_index, "probe": cmd_probe, "contrast": cmd_contrast}[args.cmd](model, tokenizer, args)


if __name__ == "__main__":
    sys.exit(main())
