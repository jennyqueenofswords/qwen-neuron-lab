"""Is there a re-checking / looping direction in Qwen3.5-4B, as Takehi et al. found in Qwen3.5-0.8B?

  gen        Thinking-mode GSM8K answers (greedy, a token budget) for a range of problems, under a condition:
             none, or the direction removed from the MLP outputs of chosen layers (a runtime weight edit), or a
             random direction removed at the same layers. -> eval/loop_hunt/<name>.jsonl
  direction  From the first problems' baseline traces: per layer, the mean residual on tokens inside repeated
             12-grams minus the mean on fresh tokens; and which layers' MLP outputs write most along it.
             -> index/loop_direction.npz
  score      Finish rate within the budget, accuracy, tokens used, repetition share, per condition.

    .venv/bin/python loop_hunt.py gen --name base --start 0 --end 45
    .venv/bin/python loop_hunt.py direction --train base --n 15
    .venv/bin/python loop_hunt.py gen --name edit --start 15 --end 45 --ablate 20 25
    .venv/bin/python loop_hunt.py gen --name random --start 15 --end 45 --ablate 20 25 --random
    .venv/bin/python loop_hunt.py score base edit random --start 15 --end 45
"""

import argparse
import json
import os
import re

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval", "loop_hunt")
SUFFIX = "\nPut the final answer after ####."


def problems():
    return [json.loads(l) for l in open(os.path.join(HERE, "data", "gsm8k_test.jsonl"))]


def repeated_mask(ids, n=12):
    seen, rep = set(), np.zeros(len(ids), bool)
    for i in range(len(ids) - n):
        g = tuple(ids[i:i + n])
        if g in seen:
            rep[i:i + n] = True
        seen.add(g)
    return rep


def gold(r):
    return r["answer"].split("####")[-1].strip().replace(",", "")


def answer(text):
    after = text.split("</think>")[-1]
    m = re.findall(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", after)
    return m[-1].replace(",", "") if m else None


def cmd_gen(a):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate
    from mlx_lm.sample_utils import make_sampler
    import steer
    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    if a.ablate:
        D = np.load(os.path.join(HERE, "index", "loop_direction.npz"))
        for L in a.ablate:
            d = D[f"L{L}"].astype(np.float32)
            if a.random:
                d = np.random.default_rng(1000 + L).standard_normal(d.shape).astype(np.float32)
            steer.HOOKS["mlp_ablate"][L] = mx.array(d / np.linalg.norm(d))
    if a.steer is not None:
        # push toward "conclude": the decision-point direction (conclude minus re-check) added at one layer
        L = a.steer
        neutral = [t.format(n) for t in steer.TEMPLATES for n in steer.NEUTRAL]
        _, resid = steer.mean_over(neutral, model, tok, "resid")
        d = np.load(os.path.join(HERE, "index", "conclude_direction.npz"))[f"L{L}"].astype(np.float32)
        if a.random:
            d = np.random.default_rng(2000 + L).standard_normal(d.shape).astype(np.float32)
        steer.HOOKS["add"][L] = mx.array(d / np.linalg.norm(d) * a.alpha * resid[L])
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{a.name}.jsonl")
    done = {json.loads(l)["i"] for l in open(path)} if os.path.exists(path) else set()
    f = open(path, "a")
    for i, r in enumerate(problems()[a.start:a.end], a.start):
        if i in done:
            continue
        p = tok.apply_chat_template([{"role": "user", "content": r["question"] + SUFFIX}], add_generation_prompt=True,
                                    tokenize=False, enable_thinking=True)
        out = generate(model, tok, p, max_tokens=a.max_tokens, sampler=make_sampler(temp=0.0), verbose=False)
        ids = tok.encode(out)
        f.write(json.dumps({"i": i, "text": out, "tokens": len(ids), "finished": "</think>" in out and answer(out) is not None,
                            "answer": answer(out), "gold": gold(r), "repeat_share": float(repeated_mask(ids).mean())}) + "\n")
        f.flush()
        print(f"  {a.name} #{i}: {len(ids)} tokens, finished {'</think>' in out}, answer {answer(out)} (gold {gold(r)})", flush=True)


def cmd_direction(a):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    import steer
    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    inner = model.language_model.model
    rows = [json.loads(l) for l in open(os.path.join(OUT, f"{a.train}.jsonl"))]
    rows = sorted(rows, key=lambda r: r["i"])[: a.n]
    probs = problems()
    sums = {k: np.zeros((32, 2560)) for k in ("rep", "fresh")}
    msum = {k: np.zeros((32, 2560)) for k in ("rep", "fresh")}
    counts = {"rep": 0, "fresh": 0}
    steer.HOOKS["record"] = True
    for r in rows:
        p = tok.apply_chat_template([{"role": "user", "content": probs[r["i"]]["question"] + SUFFIX}], add_generation_prompt=True,
                                    tokenize=False, enable_thinking=True)
        pid, gid = tok.encode(p), tok.encode(r["text"])
        mask = repeated_mask(gid)
        if mask.sum() < 20:
            continue
        steer.clear_records()
        h = inner.embed_tokens(mx.array([pid + gid]))
        fa, ss = create_attention_mask(h, None), create_ssm_mask(h, None)
        for lyr in inner.layers:
            h = lyr(h, mask=ss if lyr.is_linear else fa, cache=None)
        for L in range(32):
            res = np.array(steer.HOOKS["resid"][L][0][len(pid):])
            prev = np.array(steer.HOOKS["resid"][L - 1][0][len(pid):]) if L else None
            for k, sel in (("rep", mask), ("fresh", ~mask)):
                sums[k][L] += res[sel].sum(0)
                if prev is not None:
                    msum[k][L] += (res[sel] - prev[sel]).sum(0)  # what layer L adds (attention + MLP)
        counts["rep"] += int(mask.sum())
        counts["fresh"] += int((~mask).sum())
    steer.HOOKS["record"] = False
    steer.clear_records()
    out = {}
    print(f"tokens: repeated {counts['rep']}, fresh {counts['fresh']}")
    print("layer  |direction|  cosine with layer-15-of-24 position  layer-write along it")
    dirs = []
    for L in range(32):
        d = sums["rep"][L] / counts["rep"] - sums["fresh"][L] / counts["fresh"]
        w = msum["rep"][L] / counts["rep"] - msum["fresh"][L] / counts["fresh"]
        out[f"L{L}"] = d.astype(np.float32)
        unit = d / (np.linalg.norm(d) + 1e-9)
        dirs.append(unit)
        print(f"  {L:>2}   {np.linalg.norm(d):8.2f}   write along own direction {w @ unit:8.2f}")
    np.savez(os.path.join(HERE, "index", "loop_direction.npz"), **out)
    print("saved index/loop_direction.npz")


def cmd_score(a):
    for name in a.names:
        rows = [json.loads(l) for l in open(os.path.join(OUT, f"{name}.jsonl"))]
        rows = [r for r in rows if a.start <= r["i"] < a.end]
        n = len(rows)
        fin = sum(r["finished"] for r in rows)
        # an answer only counts if it comes after the thinking closes (a "#### 18" mid-thought doesn't)
        acc = sum(r["finished"] and abs(float(r["answer"]) - float(r["gold"])) < 1e-6 for r in rows)
        print(f"{name:<10} n={n:>3}  finished {fin:>3} ({100 * fin / max(n, 1):.0f}%)  correct {acc:>3} ({100 * acc / max(n, 1):.0f}%)  "
              f"mean tokens {np.mean([r['tokens'] for r in rows]):6.0f}  repeated share {np.mean([r['repeat_share'] for r in rows]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "direction", "score"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--name", default="base")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=45)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--ablate", type=int, nargs="*", default=None)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--steer", type=int, default=None, help="layer to push toward the conclude direction")
    ap.add_argument("--alpha", type=float, default=0.3, help="push size, as a fraction of the median stream size")
    ap.add_argument("--train", default="base")
    ap.add_argument("--n", type=int, default=15)
    a = ap.parse_args()
    {"gen": cmd_gen, "direction": cmd_direction, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
