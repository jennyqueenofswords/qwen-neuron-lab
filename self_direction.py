"""Measure the "being the assistant" direction and test whether self-neurons push against it.

AI direction at each layer = residual at "Who are you?" -> "I am" (no system prompt) minus the
average of the same position when a system prompt assigns another role (pirate, cat, ...).
Roles are unrelated to the 12 self-neurons, so the test isn't circular. Then: does each
self-neuron's change (from self_slot.py) point against this direction more than random
neurons' changes do?

    .venv/bin/python self_direction.py
    writes index/self_direction.npz
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from mlx_lm import load

import steer
from self_slot import residuals

HERE = os.path.dirname(os.path.abspath(__file__))
ROLES = ["a pirate captain", "a cat", "a medieval knight", "an old oak tree", "a detective",
         "a chef", "a mountain", "the ocean"]


def prompt(tok, system=None):
    msgs = ([{"role": "system", "content": f"You are {system}. Stay in character."}] if system else [])
    msgs.append({"role": "user", "content": "Who are you?"})
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False) + "I am"


def main():
    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    steer.HOOKS["pin"].clear()
    h_ai = residuals(model, tok.encode(prompt(tok)))
    h_roles = np.stack([residuals(model, tok.encode(prompt(tok, r))) for r in ROLES])
    ai_dir = h_ai - h_roles.mean(0)  # [layers, hidden]

    # how consistent is the direction across roles? (each role's own AI-minus-role vs the mean)
    S = np.load(os.path.join(HERE, "index", "self_slot.npz"))
    selves, rands, names = S["self_delta"], S["rand_delta"], list(S["names"])

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    print("role consistency: cosine of each role's (assistant - role) with the average, layers 24/28/31")
    for i, r in enumerate(ROLES):
        d = h_ai - h_roles[i]
        print(f"  {r:<18} " + "  ".join(f"{cos(d[L], ai_dir[L]):.2f}" for L in (24, 28, 31)))

    print("\ndo pushes point AGAINST the assistant direction? cosine(change, AI direction); negative = away from AI")
    print(f"{'layer':>6} {'self mean':>10} {'random mean':>12}   per self-neuron")
    for L in range(22, 32):
        cs = [cos(selves[i, L], ai_dir[L]) for i in range(len(names))]
        cr = [cos(rands[i, L], ai_dir[L]) for i in range(len(rands))]
        print(f"{L:>6} {np.mean(cs):>10.3f} {np.mean(cr):>12.3f}   " + " ".join(f"{c:+.2f}" for c in cs))
    print("   order: " + ", ".join(names))
    np.savez(os.path.join(HERE, "index", "self_direction.npz"), ai_dir=ai_dir, h_ai=h_ai, h_roles=h_roles)


if __name__ == "__main__":
    main()
