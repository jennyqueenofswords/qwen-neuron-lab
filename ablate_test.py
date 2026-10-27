"""Does the model still understand sarcasm with a neuron switched off?

Asks "Is the speaker being sarcastic? Yes or No" for matched sarcastic/sincere pairs and
reads the probability of "Yes" directly, under several interventions.

    .venv/bin/python ablate_test.py test/hunt/sarcasm_pairs.tsv
"""

import os
import random
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np
from mlx_lm import load

import steer

QUESTION = 'Here is something someone said:\n\n"{}"\n\nIs the speaker being sarcastic? Answer with just Yes or No.'

TARGET = (15, 3920)
TOP5 = [(15, 3920), (16, 4146), (17, 4610), (14, 9176), (17, 0)]  # top "for" neurons from contrast


def pins_for(neurons, value):
    out = {}
    for L, n in neurons:
        out.setdefault(L, {})[n] = value
    return out


def p_yes(model, tok, text, yes_ids, no_ids):
    prompt = steer.chat(tok, QUESTION.format(text))
    logits = model(mx.array([tok.encode(prompt)]))[0, -1].astype(mx.float32)
    probs = np.array(mx.softmax(logits))
    y, n = probs[yes_ids].sum(), probs[no_ids].sum()
    return y / (y + n), y + n


def main():
    with open(sys.argv[1]) as f:
        pairs = [line.rstrip("\n").split("\t") for line in f][1:]
    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    yes_ids = sorted({tok.encode(w)[0] for w in ("Yes", " Yes", "yes")})
    no_ids = sorted({tok.encode(w)[0] for w in ("No", " No", "no")})

    rng = random.Random(1)
    random5 = [(L, rng.choice([n for n in range(9216) if (L, n) not in TOP5])) for L, _ in TOP5]
    conditions = {
        "normal": {},
        "zero L15/N3920": pins_for([TARGET], 0.0),
        "zero top 5 sarcasm": pins_for(TOP5, 0.0),
        "zero 5 random (control)": pins_for(random5, 0.0),
        "push L15/N3920 to -10": pins_for([TARGET], -10.0),
    }
    print(f"random control neurons: {', '.join(f'L{L}/N{n}' for L, n in random5)}\n")
    print(f"{'condition':<26} {'P(yes) sarcastic':>16} {'P(yes) sincere':>15} {'gap':>6} {'correct':>8} {'yes+no mass':>11}")
    per_pair = {}
    for name, pins in conditions.items():
        steer.HOOKS["pin"].clear()
        steer.HOOKS["pin"].update(pins)
        s = [p_yes(model, tok, a, yes_ids, no_ids) for a, _ in pairs]
        c = [p_yes(model, tok, b, yes_ids, no_ids) for _, b in pairs]
        ps, pc = np.array([x[0] for x in s]), np.array([x[0] for x in c])
        mass = np.mean([x[1] for x in s + c])
        correct = int((ps > 0.5).sum() + (pc < 0.5).sum())
        per_pair[name] = ps - pc
        print(f"{name:<26} {ps.mean():>16.3f} {pc.mean():>15.3f} {(ps - pc).mean():>6.3f} "
              f"{correct:>5}/{2 * len(pairs)} {mass:>11.2f}")
    steer.HOOKS["pin"].clear()

    print("\nper-pair gap (sarcastic minus sincere P(yes)); a drop means it noticed the sarcasm less")
    print(f"{'pair':<46}" + "".join(f"{k[:12]:>13}" for k in conditions))
    for i, (a, _) in enumerate(pairs):
        print(f"{a[:44]:<46}" + "".join(f"{per_pair[k][i]:>13.2f}" for k in conditions))


if __name__ == "__main__":
    main()
