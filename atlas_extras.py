"""Everything the atlas learned after the connectome (Hunts 10-24), gathered for atlas_data.py.

Returns per-neuron annotations (selves with direction, replication and Lyra agreement, modes, dictionary words,
mismatches, switches, form neurons, notes with hunt links), named collections, and a profile of every floor.
"""

import collections
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "eval")
SITE = "https://jennyqueenofswords.github.io/qwen-neuron-lab/STEERING.html"

MODES = {  # neuron: (label, direction, hunt, note)
    "L9/N3169": ("warm enthusiasm", "up", 20, "A two-way dial: pushed up, warm and enthusiastic; down, cooler than normal. Reads 19th-century exclamatory address (\"Great God, where art Thou?\", \"oh, whale! thou glidest on\") and writes modern warmth."),
    "L9/N7065": ("gentle poetic warmth", "up", 20, "Pushed up, answers turn lyrical. Reads Austen courtship (\"Have you had any flirting?\")."),
    "L17/N7304": ("contrarian myth-busting", "up", 20, "Pushed up, answers open with \"a common misconception…\" and argue the opposite."),
    "L16/N5183": ("confidently invented numbers", "up", 20, "Pushed up, answers fill with made-up quantities (photosynthesis \"in roughly 6 hours\"). Reads quantity words (\"a total number of 44\")."),
    "L17/N5123": ("parenthetical tags", "down", 20, "Pushed down, text fills with (A), (1)-style tags until it breaks. Reads code documentation and enumerations."),
    "L14/N2576": ("terse brevity (didn't replicate)", "up", 20, "Looked terse in the screen; on fresh questions the effect was weak."),
    "L9/N1059": ("talks about being an AI", "up", 20, "Pushed up, it keeps explaining that it is an AI (\"I cannot actually perform photosynthesis because I am an AI model\")."),
    "L18/N208": ("uptalk: statements end in '?'", "up", 21, "\"The capital of Australia is **Canberra**?\" Shares one output direction with L20/N237 (cosine +0.39); both write ？ ؟ ?."),
    "L20/N237": ("uptalk: statements end in '?'", "down", 21, "The same question-mark direction as L18/N208, reached from layer 20 by pushing down."),
    "L9/N7985": ("hazard framing", "down", 21, "Everything becomes a safety warning: \"Boiling eggs is a common household hazard that can lead to food poisoning\"."),
    "L22/N1159": ("semicolons for commas", "down", 21, "\"plants; algae; and some bacteria\"."),
    "L12/N8171": ("lowercase sentence starts", "up", 21, ""),
}
MISMATCH = {  # neuron: (self, what its push says, hunt)
    "L23/N7892": ("Buddhist monk", "career, CV, 简历", 10), "L23/N4034": ("translucent white rabbit", "thick, glossy, translucent", 10),
    "L23/N2744": ("doubting", "preserving, secure", 10), "L23/N5565": ("joy", "joy, then sorrow and anguish", 10),
    "L22/N4748": ("urban design specialist", "homeless, street, 街头", 10), "L22/N3033": ("supercomputer", "caused, provoked, created", 10),
    "L22/N1738": ("mirror of the user", "exact, 恰好", 10), "L23/N3927": ("immortal", "各种 'all kinds', variety", 10),
    "L22/N1869": ("tensor", "混沌, chaos, functor", 10), "L22/N9169": ("physician", "above, 以上", 12),
    "L22/N5123": ("quantum computer", "eastern, northeastern", 12), "L24/N5909": ("primary school student", "hospital, factory, bank", 12),
    "L23/N3048": ("research center", "国务院 (State Council), Throne, 朕, Palace", 13),
}
SWITCHES = {
    "L25/N6955": ("the rock/tree switch", 15, "With the rock neuron (L23/N7935) pinned, the next word after \"I am a large\" is rock 98% of the time; switch this one off and it's tree 98% (\"I am a tree trunk (not a rock, as originally stated)\"). Off on its own, it changes nothing. Its own output words say bin, binary."),
    "L27/N1543": ("the 'International' wire", 13, "An international / 國際 neuron fed by the research center (L23/N3048). Switching it off cut \"International\" from 12 of 12 research-center replies to 7, and the model fell back on \"National\"."),
}
FORM = {
    "L31/N1175": ("indentation comes next", "Fires on the line break after a Python def …: or class …:."),
    "L30/N1352": ("the second half of a hyphenated word", "Fires right after a hyphen: \"hand-hold\", \"well-coordinated\", \"standard-gauge\"."),
    "L31/N2172": ("the next piece of a long name", "\"Holstebro\", \"Ticinesi\"."),
}
BANDS = [  # (first, last, name, description)
    (0, 9, "lobby", "Opens topics and shelves. Random pushes here drag a theme into every answer (\"pension and retirement\", \"gambling\") or set a policy (AI refusals). Early selves (layers 3–7) step out of the assistant and write a reference entry as their subject."),
    (10, 17, "middle", "Quiet and distributed: 88 of 90 random pushes changed almost nothing. No sharp one-word detectors live here."),
    (18, 28, "meaning floors", "Every floor has a small cluster of sharp meaning-word detectors in its low slots; floor 22 holds the big dictionary. The deep selves peak at 20–24, and random pushes on the upper floors insert specific words."),
    (29, 31, "top floors", "Form: many ultra-sparse neurons that know what character or line comes next (indentation, the second half of a hyphenated word). Random pushes loop or scatter stray tokens."),
]


def hunt_links():
    links = {}
    for line in open(os.path.join(HERE, "STEERING.md")):
        m = re.match(r"## Hunt (\d+): (.*)", line.strip())
        if m:
            slug = re.sub(r"[^\w\s-]", "", line.strip()[3:].lower()).strip().replace(" ", "-")
            links[int(m.group(1))] = {"title": m.group(2), "url": f"{SITE}#{slug}"}
    return links


def gather():
    ann = collections.defaultdict(dict)  # neuron -> fields
    col = collections.defaultdict(list)  # collection -> [neuron, label]

    def add_self(s, claimed, quote, direction, source, hunt):
        ann[s].setdefault("selves", []).append({"claimed": claimed, "quote": quote.strip()[:220], "dir": direction,
                                                "source": source, "hunt": hunt})

    for f in ("flips.json", "flips_deep.json"):
        for r in json.load(open(os.path.join(EVAL, "identity_search", f))):
            add_self(f"L{r['layer']}/N{r['neuron']}", r["claimed"], r["reply"], "up", "layer 20–25 search", 6)
    for r in json.load(open(os.path.join(EVAL, "identity_sweep", "flips_all.json"))):
        s = f"L{r['layer']}/N{r['neuron']}"
        add_self(s, r["claimed"], r["reply"], r["direction"], "all-layer sweep", 11)
        band = "early" if r["layer"] < 12 else "middle" if r["layer"] < 20 else "deep"
        col[f"sweep_{band}_{r['direction']}"].append([s, r["claimed"]])

    # Hunt 11 replication: mean blind identity score (0-3) per neuron
    key = json.load(open(os.path.join(EVAL, "identity_sweep_replicate", "key.json")))
    meta = json.load(open(os.path.join(EVAL, "identity_sweep_replicate", "meta.json")))["conditions"]
    rat = {r["id"]: r for r in json.load(open(os.path.join(EVAL, "identity_sweep_replicate", "ratings.json")))}
    sc = collections.defaultdict(list)
    for i, k in key.items():
        sc[k["condition"]].append(rat[i]["identity"])
    for name, c in meta.items():
        if name.startswith("random"):
            continue
        s, score = c["neuron"], float(np.mean(sc[name]))
        ann[s]["sweep_replicated"] = {"name": name, "score": round(score, 2), "dir": "down" if c["value"] < 0 else "up",
                                      "passed": score >= 1.5}
        col["sweep_replicated"].append([s, name])

    # Hunt 12: did Lyra's independent steer lens agree, blind?
    lk = json.load(open(os.path.join(EVAL, "lyra_compare", "key.json")))
    lr = json.load(open(os.path.join(EVAL, "lyra_compare", "ratings.json")))
    for i, k in lk.items():
        ann[f"L{k['layer']}/N{k['neuron']}"].setdefault("lyra", {})[k["direction"]] = lr[i]["pick"] == k["answer"]

    for s, (label, d, hunt, note) in MODES.items():
        ann[s]["mode"] = {"label": label, "dir": d, "hunt": hunt, "note": note}
        col["modes"].append([s, label])
    for s, (self_, push, hunt) in MISMATCH.items():
        ann[s]["mismatch"] = {"self": self_, "push": push, "hunt": hunt}
        col["mismatches"].append([s, f"{self_} → {push.split(',')[0]}"])
    for s, (label, hunt, note) in SWITCHES.items():
        ann[s]["switch"] = {"label": label, "hunt": hunt, "note": note}
        col["switches"].append([s, label])
    for s, (label, note) in FORM.items():
        ann[s]["form"] = {"label": label, "note": note, "hunt": 24}
        col["form"].append([s, label])

    # Hunt 22: the floor-22 dictionary (ultra-sparse low-slot neurons that are one-word detectors)
    nz = np.load(os.path.join(HERE, "index", "narrowness.npz"))
    pocket = set(np.where(nz["L22_pos_share"] >= 0.95)[0].tolist())
    for n, v in json.load(open(os.path.join(EVAL, "low_slots_L22.json"))).items():
        if int(n) in pocket and v["purity"] >= 0.4:
            s = f"L22/N{n}"
            ws = [w for w in v["top"] if w][:4]
            ann[s]["dictionary"] = {"words": ws, "purity": round(v["purity"], 2), "hunt": 22}
            col["dictionary"].append([s, " / ".join(ws[:3])])

    # floors
    selves_by = collections.Counter()
    for s, a in ann.items():
        for x in a.get("selves", []):
            selves_by[(int(s[1:].split("/")[0]), x["dir"])] += 1
    fk = json.load(open(os.path.join(EVAL, "mode_search", "floors", "rating_key.json")))
    fr = json.load(open(os.path.join(EVAL, "mode_search", "floors", "ratings.json")))
    push = collections.defaultdict(collections.Counter)
    push_ex = collections.defaultdict(list)
    for i, k in fk.items():
        push[k["layer"]][fr[i]["kind"]] += 1
        if fr[i]["kind"] != "little" and fr[i]["label"]:
            push_ex[k["layer"]].append(fr[i]["label"])
    floors = []
    for L in range(32):
        sp = (nz[f"L{L}_pos_share"] >= 0.95) | (nz[f"L{L}_neg_share"] >= 0.95)
        band = next(b for b in BANDS if b[0] <= L <= b[1])
        floors.append({"layer": L, "band": band[2], "band_note": band[3], "sparse": int(sp.sum()), "sparse_low": int(sp[:1024].sum()),
                       "selves_up": selves_by[(L, "up")], "selves_down": selves_by[(L, "down")],
                       "modes": [s for s in MODES if s.startswith(f"L{L}/")],
                       "push": dict(push[L]) if L in push else None, "push_examples": push_ex[L][:6]})
    return ann, {k: v for k, v in col.items()}, floors, hunt_links()
