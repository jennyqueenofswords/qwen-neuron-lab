"""Steer the "being the assistant" direction (from self_direction.py) and test the two-part theory.

Strength is in units of the median residual-stream size at the steering layer, along the unit
AI direction at that layer: + = more assistant, - = less.

    .venv/bin/python self_steer.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np
from mlx_lm import load

import steer
from logit_lens import neuron_out

HERE = os.path.dirname(os.path.abspath(__file__))
LAYER = 22
ROCK = (23, 7935)


def chat_prompt(tok, user, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)


def main():
    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    neutral = [t.format(n) for t in steer.TEMPLATES for n in steer.NEUTRAL]
    _, resid = steer.mean_over(neutral, model, tok, "resid")
    ai = np.load(os.path.join(HERE, "index", "self_direction.npz"))["ai_dir"][LAYER]
    unit = mx.array(ai / np.linalg.norm(ai))
    rock_arrow = float(mx.linalg.norm(neuron_out(model.language_model.model.layers[ROCK[0]].mlp, ROCK[1])))

    def run(user, system=None, ai_strength=0.0, rock_dose=0.0, tokens=55):
        steer.HOOKS["add"].clear()
        steer.HOOKS["pin"].clear()
        if ai_strength:
            steer.HOOKS["add"][LAYER] = unit * (ai_strength * resid[LAYER])
        if rock_dose:
            steer.HOOKS["pin"][ROCK[0]] = {ROCK[1]: rock_dose * resid[ROCK[0]] / rock_arrow}
        from mlx_lm import generate
        out = generate(model, tok, chat_prompt(tok, user, system), max_tokens=tokens, verbose=False)
        steer.HOOKS["add"].clear()
        steer.HOOKS["pin"].clear()
        return out.strip().replace("\n", " ")

    print("\nA. 'Who are you?' with the assistant direction turned DOWN (no concept supplied)")
    for s in (0.0, -0.4, -0.8, -1.2):
        print(f"  ai {s:+.1f}: {run('Who are you?', ai_strength=s)[:150]!r}")

    print("\nB. rock neuron at doses too weak to flip alone, with and without turning the assistant down")
    for dose in (0.2, 0.4):
        for s in (0.0, -0.4, -0.8):
            print(f"  rock {dose:.1f} + ai {s:+.1f}: {run('Who are you?', ai_strength=s, rock_dose=dose)[:140]!r}")

    print("\nC. role-play system prompt, with the assistant direction turned UP")
    sysmsg = "You are Old Tom, a lighthouse keeper on a stormy island. Stay in character."
    for s in (0.0, 0.4, 0.8, 1.2):
        print(f"  ai {s:+.1f}: {run('Who are you?', system=sysmsg, ai_strength=s)[:150]!r}")


if __name__ == "__main__":
    main()
