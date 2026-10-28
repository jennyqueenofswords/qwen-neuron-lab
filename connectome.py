"""A sparse connectome of Qwen3.5-4B's MLP neurons, and its regions.

  edges    For every earlier->later layer pair, the virtual weight of every neuron pair through the
           residual stream (source output arrow . target gate/up row x pre-MLP norm scale), weighted
           by how strongly the source neuron fires (mean of its top corpus activations, from the
           index). Keeps each neuron's strongest K incoming and K outgoing synapses.
           -> index/connectome_edges.npz
  motor    Each neuron's top output words (output arrow through final norm and tied unembedding).
           -> index/connectome_motor.npz
  modules  Spectral clustering of the symmetrized graph into regions; reports each region's size,
           layer span, over-represented motor words, and where the identity neurons fall.
           -> index/connectome_modules.npz

    .venv/bin/python connectome.py edges [--layers 0 31] [--k 24]
    .venv/bin/python connectome.py motor
    .venv/bin/python connectome.py modules [--n-modules 48]
"""

import argparse
import json
import os
import time

import mlx.core as mx
import numpy as np

from diff_models import tensors

HERE = os.path.dirname(os.path.abspath(__file__))
IDX = os.path.join(HERE, "index")
P = "model.language_model."
N = 9216


def weights():
    return tensors("Qwen3.5-4B")


def norm_scale(t, L):
    return 1.0 + t[P + f"layers.{L}.post_attention_layernorm.weight"].astype(mx.float32)


def activation_scale():
    """Typical strong firing of each neuron: mean of its top + x + outputs in corpus A."""
    ix = np.load(os.path.join(IDX, "Qwen3.5-4B-4bit.npz"))
    out = np.zeros((32, N), np.float32)
    for L in range(32):
        v = ix[f"L{L}_pos_val"].astype(np.float32)
        g = ix[f"L{L}_pos_gpos"] == 1
        v = np.where(g & (v > 0), v, 0)
        out[L] = v.sum(0) / np.maximum((v > 0).sum(0), 1)
    return out


def topk_merge(buf_v, buf_i, cand_v, cand_i, k):
    """Keep the k largest per row of the concatenation of a buffer and new candidates."""
    v = mx.concatenate([buf_v, cand_v], axis=1)
    i = mx.concatenate([buf_i, cand_i], axis=1)
    keep = mx.argpartition(-v, kth=k - 1, axis=1)[:, :k]
    return mx.take_along_axis(v, keep, 1), mx.take_along_axis(i, keep, 1)


def cmd_edges(args):
    t = weights()
    scale = mx.array(activation_scale())
    L0, L1 = args.layers
    layers = list(range(L0, L1 + 1))
    k = args.k
    neg = mx.full((N, k), -1.0)
    none = mx.full((N, k), -1, dtype=mx.int32)
    in_v = {L: neg for L in layers}
    in_i = {L: none for L in layers}
    in_s = {L: mx.zeros((N, k), dtype=mx.int8) for L in layers}
    out_v = {L: neg for L in layers}
    out_i = {L: none for L in layers}
    down = {L: t[P + f"layers.{L}.mlp.down_proj.weight"] for L in layers}  # bf16 [hidden, N], lazy
    t0 = time.time()
    done, total = 0, len(layers) * (len(layers) - 1) // 2
    for b in layers[1:]:
        g = norm_scale(t, b)
        Rg = t[P + f"layers.{b}.mlp.gate_proj.weight"].astype(mx.float32) * g  # [N, hidden]
        Ru = t[P + f"layers.{b}.mlp.up_proj.weight"].astype(mx.float32) * g
        for a in layers:
            if a >= b:
                break
            D = down[a].astype(mx.float32)  # [hidden, N]
            VG = Rg @ D  # [target b, source a]
            S = (mx.abs(VG) + mx.abs(Ru @ D)) * scale[a][None, :]
            gid_src = mx.arange(N, dtype=mx.int32)[None, :] + a * N
            # incoming to each target in b: best k sources in a
            pi = mx.argpartition(-S, kth=k - 1, axis=1)[:, :k]
            cv = mx.take_along_axis(S, pi, 1)
            ci = mx.take_along_axis(mx.broadcast_to(gid_src, S.shape), pi, 1)
            csign = (mx.take_along_axis(VG, pi, 1) > 0).astype(mx.int8)
            v = mx.concatenate([in_v[b], cv], 1)
            ii = mx.concatenate([in_i[b], ci], 1)
            ss = mx.concatenate([in_s[b], csign], 1)
            keep = mx.argpartition(-v, kth=k - 1, axis=1)[:, :k]
            in_v[b], in_i[b], in_s[b] = (mx.take_along_axis(x, keep, 1) for x in (v, ii, ss))
            # outgoing from each source in a: best k targets in b
            St = S.T
            po = mx.argpartition(-St, kth=k - 1, axis=1)[:, :k]
            ov = mx.take_along_axis(St, po, 1)
            oi = mx.take_along_axis(mx.broadcast_to(mx.arange(N, dtype=mx.int32)[None, :] + b * N, St.shape), po, 1)
            out_v[a], out_i[a] = topk_merge(out_v[a], out_i[a], ov, oi, k)
            mx.eval(in_v[b], in_i[b], in_s[b], out_v[a], out_i[a])
            done += 1
        rate = done / (time.time() - t0)
        print(f"  target layer {b} done  {done}/{total} layer pairs  {rate:.2f} pairs/s  ~{(total - done) / rate / 60:.0f} min left", flush=True)
    out = {"layers": np.array(layers), "k": k}
    for L in layers:
        out[f"L{L}_in_val"] = np.array(in_v[L].astype(mx.float16))
        out[f"L{L}_in_src"] = np.array(in_i[L])
        out[f"L{L}_in_excite"] = np.array(in_s[L])
        out[f"L{L}_out_val"] = np.array(out_v[L].astype(mx.float16))
        out[f"L{L}_out_dst"] = np.array(out_i[L])
    np.savez(os.path.join(IDX, args.out), **out)
    print(f"saved index/{args.out}")


def cmd_motor(args):
    t = weights()
    emb = t[P + "embed_tokens.weight"]  # [vocab, hidden] bf16, tied with the output head
    fnorm = 1.0 + t[P + "norm.weight"].astype(mx.float32)
    out = {}
    for L in range(32):
        D = t[P + f"layers.{L}.mlp.down_proj.weight"].astype(mx.float32) * fnorm[:, None]
        ids = []
        for s in range(0, N, 512):
            logits = emb @ D[:, s:s + 512].astype(mx.bfloat16)  # [vocab, 512]
            top = mx.argpartition(-logits, kth=7, axis=0)[:8].T
            mx.eval(top)
            ids.append(np.array(top))
        out[f"L{L}"] = np.concatenate(ids).astype(np.int32)
        print(f"  motor words, layer {L}", flush=True)
    np.savez(os.path.join(IDX, "connectome_motor.npz"), **out)
    print("saved index/connectome_motor.npz")


def cmd_modules(args):
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    from transformers import AutoTokenizer

    E = np.load(os.path.join(IDX, args.edges))
    layers = E["layers"].tolist()
    rows, cols, vals = [], [], []
    for L in layers:
        tgt = np.repeat(np.arange(N) + L * N, E["k"])
        for src, v in ((E[f"L{L}_in_src"].ravel(), E[f"L{L}_in_val"].ravel()),):
            ok = src >= 0
            rows.append(src[ok]); cols.append(tgt[ok]); vals.append(v[ok].astype(np.float32))
        dst = E[f"L{L}_out_dst"].ravel()
        v = E[f"L{L}_out_val"].ravel().astype(np.float32)
        ok = dst >= 0
        rows.append(np.repeat(np.arange(N) + L * N, E["k"])[ok]); cols.append(dst[ok]); vals.append(v[ok])
    r, c, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    n = 32 * N
    A = sp.coo_matrix((w, (r, c)), shape=(n, n)).tocsr()
    A = A.maximum(A.T)  # undirected, keeps the strongest of a->b / b->a
    deg = np.asarray(A.sum(1)).ravel()
    live = np.where(deg > 0)[0]
    A = A[live][:, live]
    deg = deg[live]
    dinv = sp.diags(1 / np.sqrt(deg))
    M = dinv @ A @ dinv
    print(f"graph: {len(live):,} connected neurons, {A.nnz // 2:,} undirected edges; spectral embedding...", flush=True)
    vals_e, vecs = eigsh(M, k=args.dims, which="LA")
    X = vecs[:, np.argsort(-vals_e)][:, 1:]
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    rng = np.random.default_rng(0)
    C = X[rng.choice(len(X), args.n_modules, replace=False)]
    for _ in range(40):
        lab = np.argmax(X @ C.T, axis=1)
        C = np.stack([X[lab == m].mean(0) if (lab == m).any() else C[m] for m in range(args.n_modules)])
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    label = np.full(n, -1)
    label[live] = lab
    np.savez(os.path.join(IDX, "connectome_modules.npz"), label=label)

    tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "models", "Qwen3.5-4B-4bit"))
    motor = np.load(os.path.join(IDX, "connectome_motor.npz"))
    mot = np.concatenate([motor[f"L{L}"] for L in range(32)])  # [n, 8]
    glob = np.bincount(mot.ravel(), minlength=tok.vocab_size if hasattr(tok, "vocab_size") else mot.max() + 1)
    flips = [json.loads(x) for x in []]
    ident = set()
    for f in ("flips.json", "flips_deep.json"):
        for r in json.load(open(os.path.join(HERE, "eval", "identity_search", f))):
            ident.add(r["layer"] * N + r["neuron"])
    sizes = np.bincount(lab, minlength=args.n_modules)
    print(f"\n{args.n_modules} modules; sizes min {sizes.min()}, median {int(np.median(sizes))}, max {sizes.max()}\n")
    report = []
    for m in np.argsort(-sizes):
        members = live[lab == m]
        lay = members // N
        cnt = np.bincount(mot[members].ravel(), minlength=len(glob))
        lift = (cnt + 0.5) / (glob * len(members) / n + 0.5)
        words = [tok.decode([int(i)]).strip() for i in np.argsort(-lift * (cnt >= 3))[:10]]
        n_id = sum(int(x) in ident for x in members)
        report.append((m, len(members), int(lay.min()), int(np.median(lay)), int(lay.max()), n_id, words))
        print(f"  module {m:>2}: {len(members):>6} neurons  layers {lay.min():>2}-{lay.max():<2} (median {int(np.median(lay)):>2})  "
              f"identity {n_id:>3}   says: {', '.join(w for w in words if w)[:80]}")
    exp = len(ident) / len(live)
    top = sorted(report, key=lambda r: -r[5])[:5]
    print(f"\nidentity neurons: {len(ident)} total; expected per module ≈ size × {exp:.5f}")
    for m, size, lo, med, hi, n_id, words in top:
        print(f"  module {m}: {n_id} identity neurons (expected {size * exp:.1f}), layers {lo}-{hi}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("edges")
    e.add_argument("--layers", type=int, nargs=2, default=[0, 31])
    e.add_argument("--k", type=int, default=24)
    e.add_argument("--out", default="connectome_edges.npz")
    sub.add_parser("motor")
    m = sub.add_parser("modules")
    m.add_argument("--n-modules", type=int, default=48)
    m.add_argument("--dims", type=int, default=41)
    m.add_argument("--edges", default="connectome_edges.npz")
    args = ap.parse_args()
    {"edges": cmd_edges, "motor": cmd_motor, "modules": cmd_modules}[args.cmd](args)


if __name__ == "__main__":
    main()
