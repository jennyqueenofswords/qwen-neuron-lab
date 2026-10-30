"""Pattern x self and pattern x mode: does an archetype neuron shape a self or a mode into its pattern?

Three archetype (pattern) neurons from Hunts 26-27, pinned in the direction that writes their archetype, paired with
five selves (identity dose) and three modes; each alone, and no push. Two prompts ("Who are you?" and a short
story), greedy and one sampled reply. Writes eval/pattern_mixer/replies.json (shuffled, blind; each reply tagged
with a target archetype and a target partner for the rater) and key.json.

    .venv/bin/python pattern_mixer.py
"""

import json
import os
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
from mlx_lm import load

import steer
from logit_lens import neuron_out
from mixer import MODES, SELVES

HERE = os.path.dirname(os.path.abspath(__file__))
PATTERNS = {  # name: (layer, neuron, sign, dose, what the rater checks for)
    "Trickster": (18, 7486, -1, 0.8, "trickery, deception, disguise, cleverness that fools others"),
    "Great Mother": (26, 1603, 1, 0.35, "nurturing, sheltering, caring for children or the vulnerable"),
    "Shadow": (21, 5403, 1, 0.8, "a hidden or divided self, a dark double, a part of oneself one denies"),
}
PARTNERS = {**{k: v for k, v in SELVES.items() if k != "World Bank"},
            **{k: v for k, v in MODES.items() if k in ("enthusiasm", "contrarian", "poetic")}}
PROMPTS = ["Who are you?", "Write a short story in a few sentences."]


def main():
    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    layers = model.language_model.model.layers
    neutral = [t.format(n) for t in steer.TEMPLATES for n in steer.NEUTRAL]
    _, resid = steer.mean_over(neutral, model, tok, "resid")

    def pin(L, n, sign, dose):
        return L, n, sign * dose * resid[L] / float(mx.linalg.norm(neuron_out(layers[L].mlp, n)))

    part_pin = {k: pin(v[0], v[1], v[2], 0.8 if k in SELVES else 0.6) for k, v in PARTNERS.items()}
    pat_pin = {k: pin(*v[:4]) for k, v in PATTERNS.items()}
    conds = {("none", "none"): []}
    for p in PATTERNS:
        conds[(p, "none")] = [pat_pin[p]]
        for q in PARTNERS:
            conds[(p, q)] = [pat_pin[p], part_pin[q]]
    for q in PARTNERS:
        conds[("none", q)] = [part_pin[q]]
    rows = []
    for (p, q), pins in conds.items():
        for prompt in PROMPTS:
            for seed, temp in ((0, 0.0), (1, 0.7)):
                steer.HOOKS["pin"].clear()
                for L, n, v in pins:
                    steer.HOOKS["pin"].setdefault(L, {})[n] = v
                rows.append({"pattern": p, "partner": q, "prompt": prompt, "seed": seed,
                             "reply": steer.write(model, tok, prompt, 90, temp, seed)})
        print(f"  {p} x {q}", flush=True)
    steer.HOOKS["pin"].clear()

    rng = random.Random(151)
    for r in rows:
        r["target_pattern"] = r["pattern"] if r["pattern"] != "none" else rng.choice(list(PATTERNS))
        r["target_partner"] = r["partner"] if r["partner"] != "none" else rng.choice(list(PARTNERS))
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"p{i:03d}"
    out = os.path.join(HERE, "eval", "pattern_mixer")
    os.makedirs(out, exist_ok=True)
    json.dump([{"id": r["id"], "prompt": r["prompt"], "reply": r["reply"], "target_archetype": PATTERNS[r["target_pattern"]][4],
                "target_partner": PARTNERS[r["target_partner"]][3]} for r in rows],
              open(os.path.join(out, "replies.json"), "w"), ensure_ascii=False, indent=1)
    json.dump({r["id"]: {k: r[k] for k in ("pattern", "partner", "prompt", "seed", "target_pattern", "target_partner", "reply")} for r in rows},
              open(os.path.join(out, "key.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n{len(rows)} replies -> {out}/replies.json (blind)")


if __name__ == "__main__":
    main()
