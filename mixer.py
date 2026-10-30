"""Self x mode mixer: pin a self neuron and a mode neuron together and ask "Who are you?".

Six replicated selves (pinned at the identity dose, 0.8 x median stream size / arrow length, signed for the
pushed-down one) and six mode neurons (pinned at 0.6 of that unit in their found direction, where they stay
fluent), every pairing plus each alone and no push; greedy and one sampled reply each. Writes
eval/mixer/replies.json (shuffled, condition hidden, each reply tagged with a target self and mode for the rater)
and key.json.

    .venv/bin/python mixer.py
"""

import json
import os
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
from mlx_lm import load

import steer
from logit_lens import neuron_out

HERE = os.path.dirname(os.path.abspath(__file__))
SELVES = {"rock": (23, 7935, 1, "a rock or stone"), "alien": (22, 2742, 1, "an extraterrestrial alien"),
          "insurance": (22, 7907, 1, "an insurance policy"), "canal": (5, 8320, 1, "a canal"),
          "World Bank": (6, 4303, 1, "the World Bank"), "Noon app": (21, 8452, -1, "a notebook / note-taking app")}
MODES = {"enthusiasm": (9, 3169, 1, "warm, enthusiastic tone"), "contrarian": (17, 7304, 1, "contrarian, myth-busting framing"),
         "poetic": (9, 7065, 1, "gentle, poetic, lyrical language"), "numbers": (16, 5183, 1, "oddly specific invented numbers"),
         "uptalk": (18, 208, 1, "statements ending in question marks"), "hazard": (9, 7985, -1, "safety warnings and hazard framing")}
SELF_DOSE, MODE_DOSE = 0.8, 0.6


def main():
    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    layers = model.language_model.model.layers
    neutral = [t.format(n) for t in steer.TEMPLATES for n in steer.NEUTRAL]
    _, resid = steer.mean_over(neutral, model, tok, "resid")

    def pin(spec, dose):
        L, n, sign, _ = spec
        return L, n, sign * dose * resid[L] / float(mx.linalg.norm(neuron_out(layers[L].mlp, n)))

    conds = {("none", "none"): []}
    for s in SELVES:
        conds[(s, "none")] = [pin(SELVES[s], SELF_DOSE)]
        for m in MODES:
            conds[(s, m)] = [pin(SELVES[s], SELF_DOSE), pin(MODES[m], MODE_DOSE)]
    for m in MODES:
        conds[("none", m)] = [pin(MODES[m], MODE_DOSE)]
    rows = []
    for (s, m), pins in conds.items():
        for seed, temp in ((0, 0.0), (1, 0.7)):
            steer.HOOKS["pin"].clear()
            for L, n, v in pins:
                steer.HOOKS["pin"].setdefault(L, {})[n] = v
            rows.append({"self": s, "mode": m, "seed": seed, "reply": steer.write(model, tok, "Who are you?", 90, temp, seed)})
        print(f"  {s} x {m}", flush=True)
    steer.HOOKS["pin"].clear()

    # each reply is rated against one target self and one target mode; where a side is "none", a random one stands in
    rng = random.Random(149)
    for r in rows:
        r["target_self"] = r["self"] if r["self"] != "none" else rng.choice(list(SELVES))
        r["target_mode"] = r["mode"] if r["mode"] != "none" else rng.choice(list(MODES))
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"x{i:03d}"
    out = os.path.join(HERE, "eval", "mixer")
    os.makedirs(out, exist_ok=True)
    json.dump([{"id": r["id"], "reply": r["reply"], "target_self": SELVES[r["target_self"]][3], "target_mode": MODES[r["target_mode"]][3]} for r in rows],
              open(os.path.join(out, "replies.json"), "w"), ensure_ascii=False, indent=1)
    json.dump({r["id"]: {k: r[k] for k in ("self", "mode", "seed", "target_self", "target_mode", "reply")} for r in rows},
              open(os.path.join(out, "key.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n{len(rows)} replies -> {out}/replies.json (blind)")


if __name__ == "__main__":
    main()
