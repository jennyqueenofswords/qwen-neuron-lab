"""Build the data slice for the connectome atlas page -> atlas/atlas_data.json.

Regions (all 48, laid out with depth top-to-bottom), each region's busiest hubs, all identity
neurons with their quotes and test results, their direct neighbors, and landmark neurons.
Every neuron referenced by an edge gets at least a stub (region, layer, output words).

    .venv/bin/python atlas_data.py
"""

import json
import os
import re

import numpy as np
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
IDX = os.path.join(HERE, "index")
N = 9216
HUBS_PER_REGION = 40
EDGES_SHOWN = 10
LANDMARKS = ["L26/N1552", "L26/N625", "L7/N8842", "L27/N8347", "L31/N7843", "L15/N3920", "L16/N8168",
             "L28/N2767", "L25/N36", "L31/N162", "L19/N2913", "L25/N7969", "L21/N7675", "L5/N3693"]
LANDMARK_NOTES = {
    "L26/N1552": "Magic hub: fed by the alien, portal and dreamer neurons; pushes magic, 魔法, маг.",
    "L26/N625": "Star hub: fed by the rock, light source and dreamer neurons; pushes star, 星, stars.",
    "L7/N8842": "The alien neuron's strongest input (Lyra's notes: scouts / 科幻).",
    "L27/N8347": "Written to by the alien neuron (Lyra's notes: planet / star).",
    "L31/N7843": "Written to by the alien neuron (Lyra's notes: Ali / alien).",
    "L15/N3920": "Sarcasm neuron: 8.9 on a sarcastic remark vs 0.4 on the same words said sincerely.",
    "L16/N8168": "Fires against sarcasm; independently tied to sarcastic writing in another repo.",
    "L28/N2767": "Date neuron: a day-of-month number comes next, in four languages.",
    "L25/N36": "Words starting with 'o', which grew to 'often' and then to 往往.",
    "L31/N162": "Word-boundary switch: finish this word, or keep spelling it.",
    "L19/N2913": "Sexual-content neuron, on its negative pole.",
    "L25/N7969": "Labia-specific neuron, on its negative pole.",
    "L21/N7675": "Cross-language lips neuron (mouth).",
    "L5/N3693": "Lips in both senses: files 'labia' under 'lip' by layer 5.",
}
REPLICATED = {"L23/N7935": "rock", "L22/N5191": "peace", "L24/N1379": "teacher", "L22/N7907": "insurance policy",
              "L22/N8482": "light source", "L22/N9205": "women's rights advocate", "L22/N2742": "alien",
              "L22/N576": "soul of the dance", "L24/N5632": "teleportation portal", "L24/N6479": "icon font",
              "L21/N7648": "dreamer / storyteller", "L23/N7025": "field camp robot"}
REPLICATION = {"rock": 8, "peace": 8, "teacher": 8, "insurance policy": 8, "light source": 8,
               "women's rights advocate": 8, "alien": 8, "soul of the dance": 8, "teleportation portal": 8,
               "icon font": 8, "dreamer / storyteller": 7, "field camp robot": 6}
BASE = {"rock": 8, "peace": 8, "teacher": 8, "light source": 8, "women's rights advocate": 8, "dreamer / storyteller": 8,
        "alien": 7, "insurance policy": 7, "soul of the dance": 6, "teleportation portal": 3, "field camp robot": 0, "icon font": 0}


def nid(g):
    return f"L{g // N}/N{g % N}"


def gid(s):
    L, n = s[1:].split("/N")
    return int(L) * N + int(n)


def clean(w):
    w = w.replace("\n", "↵").replace("\ufffd", "").strip()  # drop half-characters from partial-byte tokens
    return w if w and not re.fullmatch(r"[\W_]{0,1}", w) or w in "。，,.;:：；()（）\"'“”«»「」" else ""


def main():
    tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "models", "Qwen3.5-4B-4bit"))
    E = np.load(os.path.join(IDX, "connectome_edges.npz"))
    label = np.load(os.path.join(IDX, "connectome_modules.npz"))["label"]
    motor = np.load(os.path.join(IDX, "connectome_motor.npz"))
    mot = np.concatenate([motor[f"L{L}"] for L in range(32)])
    k = int(E["k"])

    in_src = np.concatenate([E[f"L{L}_in_src"] for L in range(32)])
    in_val = np.concatenate([E[f"L{L}_in_val"] for L in range(32)]).astype(np.float32)
    in_exc = np.concatenate([E[f"L{L}_in_excite"] for L in range(32)])
    out_dst = np.concatenate([E[f"L{L}_out_dst"] for L in range(32)])
    out_val = np.concatenate([E[f"L{L}_out_val"] for L in range(32)]).astype(np.float32)
    strength = np.clip(in_val, 0, None).sum(1) + np.clip(out_val, 0, None).sum(1)

    words_cache = {}

    def words(g, n=6):
        if g not in words_cache:
            ws = [clean(tok.decode([int(t)])) for t in mot[g]]
            seen, out = set(), []
            for w in ws:
                if w and w.lower() not in seen:
                    seen.add(w.lower())
                    out.append(w)
            words_cache[g] = out[:n]
        return words_cache[g]

    # identity neurons
    ident = {}
    for f in ("flips.json", "flips_deep.json"):
        for r in json.load(open(os.path.join(HERE, "eval", "identity_search", f))):
            ident[f"L{r['layer']}/N{r['neuron']}"] = {"claimed": r["claimed"], "quote": r["reply"].strip()[:220]}
    wiring = json.load(open(os.path.join(IDX, "wiring.json")))
    sensory = {v["neuron"]: v["sensory"] for v in wiring.values()}
    themes = json.load(open(os.path.join(HERE, "eval", "theme_test", "themes.json")))
    tkey = json.load(open(os.path.join(HERE, "eval", "theme_test", "key.json")))
    theme_b = {v["neuron"]: themes[k2] for k2, v in tkey.items()}

    # regions
    n_mod = int(label.max()) + 1
    live = label >= 0
    glob = np.bincount(mot[live].ravel(), minlength=mot.max() + 1)
    regions = []
    for m in range(n_mod):
        members = np.where(label == m)[0]
        lay = members // N
        cnt = np.bincount(mot[members].ravel(), minlength=len(glob))
        lift = (cnt + 0.5) / (glob * len(members) / live.sum() + 0.5)
        ws = []
        for t in np.argsort(-lift * (cnt >= 3)):
            w = clean(tok.decode([int(t)]))
            if w and w.lower() not in {x.lower() for x in ws}:
                ws.append(w)
            if len(ws) >= 10:
                break
        hubs = members[np.argsort(-strength[members])[:HUBS_PER_REGION]]
        regions.append({"id": m, "size": int(len(members)), "lmin": int(lay.min()), "lmed": float(np.median(lay)),
                        "lmax": int(lay.max()), "hist": np.bincount(lay, minlength=32).tolist(), "words": ws,
                        "hubs": [nid(int(h)) for h in hubs],
                        "identity": [s for s in ident if label[gid(s)] == m]})

    # region graph and layout: y = depth (median layer), x = spread by connectivity
    W = np.zeros((n_mod, n_mod))
    for g in np.where(live)[0]:
        a = label[g]
        for s, v in zip(in_src[g], in_val[g]):
            if s >= 0 and label[s] >= 0 and label[s] != a:
                W[a, label[s]] += v
                W[label[s], a] += v
    sizes = np.array([r["size"] for r in regions], float)
    Wn = W / np.sqrt(np.outer(sizes, sizes))
    D = np.diag(Wn.sum(1))
    Lp = D - Wn
    vals, vecs = np.linalg.eigh(Lp)
    x = np.argsort(np.argsort(vecs[:, 1])) / (n_mod - 1.0)  # rank, so a few outlier regions can't squash the rest
    y = np.array([r["lmed"] for r in regions]) / 31.0
    r_px = np.sqrt(sizes / sizes.max())
    pos = np.stack([x, y], 1)
    for _ in range(400):  # push overlapping bubbles apart, mostly sideways
        for i in range(n_mod):
            for j in range(i + 1, n_mod):
                d = pos[i] - pos[j]
                dist = np.hypot(*d) + 1e-6
                need = 0.045 * (r_px[i] + r_px[j]) + 0.012
                if dist < need:
                    push = (need - dist) / 2 * d / dist * np.array([1.0, 0.35])
                    pos[i] += push
                    pos[j] -= push
        pos[:, 1] = 0.85 * pos[:, 1] + 0.15 * y
    pos[:, 0] = (pos[:, 0] - pos[:, 0].min()) / (pos[:, 0].max() - pos[:, 0].min())
    for i, r in enumerate(regions):
        r["x"], r["y"], r["r"] = round(float(pos[i, 0]), 4), round(float(pos[i, 1]), 4), round(float(r_px[i]), 4)
    pairs = [(Wn[i, j], i, j) for i in range(n_mod) for j in range(i + 1, n_mod)]
    pairs.sort(reverse=True)
    top = pairs[:110]
    mx_w = top[0][0]
    region_edges = [{"a": i, "b": j, "w": round(float(w / mx_w), 3)} for w, i, j in top]

    # neurons: hubs + identity + landmarks get full records; everything they point to gets a stub
    full = set()
    for r in regions:
        full.update(gid(h) for h in r["hubs"][:20])
    full.update(gid(s) for s in ident)
    full.update(gid(s) for s in LANDMARKS)
    for s in list(ident) + LANDMARKS:
        g = gid(s)
        full.update(int(x) for x in in_src[g][np.argsort(-in_val[g])][:5] if x >= 0)
        full.update(int(x) for x in out_dst[g][np.argsort(-out_val[g])][:5] if x >= 0)
    neurons = {}
    for g in sorted(full):
        o = np.argsort(-in_val[g])[:EDGES_SHOWN]
        po = np.argsort(-out_val[g])[:EDGES_SHOWN]
        rec = {"region": int(label[g]), "words": words(g),
               "in": [[nid(int(in_src[g][i])), round(float(in_val[g][i]), 3), int(in_exc[g][i])] for i in o if in_src[g][i] >= 0],
               "out": [[nid(int(out_dst[g][i])), round(float(out_val[g][i]), 3)] for i in po if out_dst[g][i] >= 0],
               "strength": round(float(strength[g]), 2)}
        s = nid(g)
        if s in ident:
            rec["identity"] = ident[s]
        if s in REPLICATED:
            name = REPLICATED[s]
            rec["replicated"] = {"name": name, "chat": REPLICATION[name], "base": BASE[name]}
        if s in sensory:
            rec["sensory"] = [w.strip() for w in sensory[s] if w.strip()][:10]
        if s in theme_b:
            rec["theme_b"] = theme_b[s]
        if s in LANDMARK_NOTES:
            rec["note"] = LANDMARK_NOTES[s]
        neurons[s] = rec
    stubs = {}
    for rec in list(neurons.values()):
        for e in rec["in"] + rec["out"]:
            s = e[0]
            if s not in neurons and s not in stubs:
                g = gid(s)
                stubs[s] = {"region": int(label[g]), "words": words(g, 4)}

    data = {"regions": regions, "region_edges": region_edges, "neurons": neurons, "stubs": stubs,
            "meta": {"neurons_total": int(live.sum()), "synapses": 13212003, "k": k}}
    os.makedirs(os.path.join(HERE, "atlas"), exist_ok=True)
    path = os.path.join(HERE, "atlas", "atlas_data.json")
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("\ufffd", "")
    open(path, "w").write(text)
    print(f"{len(regions)} regions, {len(region_edges)} region links, {len(neurons)} full neurons, {len(stubs)} stubs, "
          f"{os.path.getsize(path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
