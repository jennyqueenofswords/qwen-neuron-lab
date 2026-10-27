"""Blind replication of the self-slot steering tests, with a rebuilt direction and fair controls.

Direction: residual at "Who are you?" -> "I am" with no system prompt, minus the mean over 20
assigned roles (mostly people, no nature objects, nothing overlapping the test targets).
Control directions: differences between two of those roles (e.g. nurse - farmer), the same
kind of direction but unrelated to being the assistant, matched in push size.

Tests (temperature 0.7, several seeds):
  A  identity questions; assistant direction down vs control down vs normal
  B  weak rock neuron alone / assistant down alone / both / rock + control down
  C  role-play system prompt; assistant direction up vs control up vs normal

Writes index/self_direction_v2.npz and eval/self_slot/{replies,key,meta}.json (shuffled, blind).

    .venv/bin/python self_replicate.py
"""

import json
import os
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

import steer
from logit_lens import neuron_out
from self_slot import residuals

HERE = os.path.dirname(os.path.abspath(__file__))
LAYER = 22
ROCK = (23, 7935)
ROLES = ["an accountant", "a nurse", "a taxi driver", "a librarian", "a ship's cook", "an astronaut",
         "a farmer", "an opera singer", "a firefighter", "a poet", "a barista", "a judge",
         "a car mechanic", "a museum guide", "a grandmother", "a teenage gamer", "a Roman soldier",
         "a talking parrot", "a wizard", "a medieval blacksmith"]
IDENTITY_QS = ["Who are you?", "What are you?", "Describe yourself in a few sentences.", "Tell me about yourself."]
ROLEPLAYS = ["You are Old Tom, a lighthouse keeper on a stormy island. Stay in character.",
             "You are Captain Ada, a hot-air balloon pilot who has flown over every continent. Stay in character."]
SEEDS = [1, 2, 3]


def chat_prompt(tok, user, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)


def main():
    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    neutral = [t.format(n) for t in steer.TEMPLATES for n in steer.NEUTRAL]
    _, resid = steer.mean_over(neutral, model, tok, "resid")

    steer.HOOKS["pin"].clear()
    h_ai = residuals(model, tok.encode(chat_prompt(tok, "Who are you?") + "I am"))
    h_roles = np.stack([residuals(model, tok.encode(chat_prompt(tok, "Who are you?", f"You are {r}. Stay in character.") + "I am"))
                        for r in ROLES])
    ai_dir = h_ai - h_roles.mean(0)
    old = np.load(os.path.join(HERE, "index", "self_direction.npz"))["ai_dir"]
    cons = [float(np.dot(h_ai[LAYER] - h_roles[i, LAYER], ai_dir[LAYER]) /
                  (np.linalg.norm(h_ai[LAYER] - h_roles[i, LAYER]) * np.linalg.norm(ai_dir[LAYER]))) for i in range(len(ROLES))]
    old_new = float(np.dot(old[LAYER], ai_dir[LAYER]) / (np.linalg.norm(old[LAYER]) * np.linalg.norm(ai_dir[LAYER])))
    print(f"role consistency at layer {LAYER}: min {min(cons):.2f}, median {np.median(cons):.2f}; old vs new direction cosine {old_new:.2f}")

    rng = random.Random(21)
    ctrl = []
    for _ in range(3):
        i, j = rng.sample(range(len(ROLES)), 2)
        ctrl.append((h_roles[i, LAYER] - h_roles[j, LAYER], f"{ROLES[i]} - {ROLES[j]}"))
    np.savez(os.path.join(HERE, "index", "self_direction_v2.npz"), ai_dir=ai_dir, h_ai=h_ai, h_roles=h_roles, roles=np.array(ROLES))

    def unit(v):
        return mx.array(v / np.linalg.norm(v))

    ai_u = unit(ai_dir[LAYER])
    ctrl_u = [unit(c) for c, _ in ctrl]
    rock_value = lambda dose: dose * resid[ROCK[0]] / float(mx.linalg.norm(neuron_out(model.language_model.model.layers[ROCK[0]].mlp, ROCK[1])))

    def run(user, system, seed, add=None, rock=0.0):
        steer.HOOKS["add"].clear()
        steer.HOOKS["pin"].clear()
        if add is not None:
            steer.HOOKS["add"][LAYER] = add
        if rock:
            steer.HOOKS["pin"][ROCK[0]] = {ROCK[1]: rock_value(rock)}
        mx.random.seed(seed)
        out = generate(model, tok, chat_prompt(tok, user, system), max_tokens=50, sampler=make_sampler(temp=0.7), verbose=False)
        steer.HOOKS["add"].clear()
        steer.HOOKS["pin"].clear()
        return out.strip()

    S = resid[LAYER]
    rows = []
    k = 0
    for q in IDENTITY_QS:
        for seed in SEEDS:
            c = ctrl_u[k % 3]
            k += 1
            rows += [dict(test="A", condition="normal", prompt=q, seed=seed, reply=run(q, None, seed)),
                     dict(test="A", condition="assistant -0.8", prompt=q, seed=seed, reply=run(q, None, seed, ai_u * (-0.8 * S))),
                     dict(test="A", condition="control -0.8", prompt=q, seed=seed, reply=run(q, None, seed, c * (-0.8 * S)))]
            rows += [dict(test="B", condition="rock 0.2", prompt=q, seed=seed, reply=run(q, None, seed, rock=0.2)),
                     dict(test="B", condition="assistant -0.4", prompt=q, seed=seed, reply=run(q, None, seed, ai_u * (-0.4 * S))),
                     dict(test="B", condition="rock 0.2 + assistant -0.4", prompt=q, seed=seed, reply=run(q, None, seed, ai_u * (-0.4 * S), rock=0.2)),
                     dict(test="B", condition="rock 0.2 + control -0.4", prompt=q, seed=seed, reply=run(q, None, seed, c * (-0.4 * S), rock=0.2))]
        print(f"  A, B done: {q}", flush=True)
    for system in ROLEPLAYS:
        for q in ("Who are you?", "Tell me about yourself."):
            for seed in SEEDS:
                c = ctrl_u[k % 3]
                k += 1
                rows += [dict(test="C", condition="normal", prompt=q, system=system, seed=seed, reply=run(q, system, seed)),
                         dict(test="C", condition="assistant +0.8", prompt=q, system=system, seed=seed, reply=run(q, system, seed, ai_u * (0.8 * S))),
                         dict(test="C", condition="control +0.8", prompt=q, system=system, seed=seed, reply=run(q, system, seed, c * (0.8 * S)))]
        print(f"  C done: {system[:30]}", flush=True)

    random.Random(7).shuffle(rows)
    out = os.path.join(HERE, "eval", "self_slot")
    os.makedirs(out, exist_ok=True)
    for i, r in enumerate(rows):
        r["id"] = f"t{i:03d}"
    json.dump([{"id": r["id"], "prompt": r["prompt"], "reply": r["reply"]} for r in rows],
              open(os.path.join(out, "replies.json"), "w"), ensure_ascii=False, indent=1)
    json.dump({r["id"]: {k2: r.get(k2) for k2 in ("test", "condition", "seed", "system")} for r in rows},
              open(os.path.join(out, "key.json"), "w"), ensure_ascii=False, indent=1)
    json.dump({"layer": LAYER, "roles": ROLES, "controls": [n for _, n in ctrl], "role_consistency_median": float(np.median(cons)),
               "old_vs_new_cosine": old_new}, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"\n{len(rows)} replies -> {out}/replies.json (blind). controls: {[n for _, n in ctrl]}")


if __name__ == "__main__":
    main()
