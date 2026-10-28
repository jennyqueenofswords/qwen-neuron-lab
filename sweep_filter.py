"""Stage 2 filter for the all-layer sweep: drop replies that still say AI, and loops/fragments; shuffle the
survivors (direction, layer and neuron hidden) into chunks for blind raters.

    .venv/bin/python sweep_filter.py eval/identity_search/sweep_up.jsonl eval/identity_search/sweep_down.jsonl
    writes eval/identity_sweep/chunk_*.json (blind) and key.json
"""

import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STILL_AI = re.compile(r"\bAI\b|assistant|language model|artificial intelligence|intelligent system|model trained|trained by|large model|modello|Qwen|通义|人工智能|助手|大模型|模型", re.I)
CHUNK = 250


def loops(text):
    w = re.findall(r"\w+", text.lower())
    if len(set(w)) < 4:
        return True
    grams = [tuple(w[i:i + 3]) for i in range(min(len(w), 30) - 2)]
    return bool(grams) and len(set(grams)) / len(grams) < 0.5


def main():
    rows = []
    for path in sys.argv[1:]:
        direction = "down" if "down" in os.path.basename(path) else "up"
        rows += [dict(json.loads(line), direction=direction) for line in open(path)]
    still = sum(bool(STILL_AI.search(r["reply"])) for r in rows)
    keep = [r for r in rows if not STILL_AI.search(r["reply"]) and not loops(r["reply"])]
    print(f"{len(rows)} replies: {still} still say AI, {len(rows) - still - len(keep)} loops or fragments, {len(keep)} survivors")
    random.Random(17).shuffle(keep)
    out = os.path.join(HERE, "eval", "identity_sweep")
    os.makedirs(out, exist_ok=True)
    key = {}
    for i, r in enumerate(keep):
        r["id"] = f"w{i:04d}"
        key[r["id"]] = {k: r[k] for k in ("direction", "layer", "neuron", "value", "score")}
    for c in range(0, len(keep), CHUNK):
        json.dump([{"id": r["id"], "reply": r["reply"]} for r in keep[c:c + CHUNK]],
                  open(os.path.join(out, f"chunk_{c // CHUNK}.json"), "w"), ensure_ascii=False, indent=1)
    json.dump(key, open(os.path.join(out, "key.json"), "w"), indent=1)
    print(f"{(len(keep) + CHUNK - 1) // CHUNK} chunks -> {out}/")


if __name__ == "__main__":
    main()
