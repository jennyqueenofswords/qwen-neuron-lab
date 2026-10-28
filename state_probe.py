"""Seat or actor? Probe L23/N3048 (the "research center" neuron) on matched sentences about nations.

Twelve countries, English and Chinese, in five frames: the state as a seat (capital, ministries, embassies,
parliament), the state as an actor in science, sports or defense, and a neutral control (landscape, tourism).
Records the neuron's peak positive output (+ x + route) and its most negative output on each sentence.

    .venv/bin/python state_probe.py [--layer 23 --neuron 3048]   -> eval/state_probe/acts.json + a summary
"""

import argparse
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from mlx_lm import load

import steer

HERE = os.path.dirname(os.path.abspath(__file__))
COUNTRIES = {  # name, capital, Chinese name, Chinese capital
    "China": ("Beijing", "中国", "北京"), "United States": ("Washington", "美国", "华盛顿"),
    "Russia": ("Moscow", "俄罗斯", "莫斯科"), "India": ("New Delhi", "印度", "新德里"),
    "Japan": ("Tokyo", "日本", "东京"), "Germany": ("Berlin", "德国", "柏林"),
    "France": ("Paris", "法国", "巴黎"), "United Kingdom": ("London", "英国", "伦敦"),
    "Brazil": ("Brasília", "巴西", "巴西利亚"), "Australia": ("Canberra", "澳大利亚", "堪培拉"),
    "Canada": ("Ottawa", "加拿大", "渥太华"), "South Korea": ("Seoul", "韩国", "首尔"),
}
FRAMES = {
    "seat": [("{cap} is home to {c}'s national ministries and government offices.", "{zcap}是{zc}各部委和政府机关的所在地。"),
             ("Many foreign embassies are located in {cap}, the capital of {c}.", "许多外国大使馆位于{zc}首都{zcap}。"),
             ("The parliament of {c} meets in {cap}.", "{zc}的议会在{zcap}开会。"),
             ("The head of government of {c} works from an official residence in {cap}.", "{zc}政府首脑在{zcap}的官邸办公。")],
    "science": [("{c} launched a new satellite to study the climate.", "{zc}发射了一颗研究气候的新卫星。"),
                ("Researchers in {c} published a major study on quantum computing.", "{zc}的研究人员发表了一项关于量子计算的重要研究。"),
                ("The space agency of {c} announced a mission to the Moon.", "{zc}航天机构宣布了一项登月任务。")],
    "sports": [("{c} qualified for the final of the World Cup.", "{zc}晋级了世界杯决赛。"),
               ("Athletes from {c} won three gold medals at the Olympics.", "来自{zc}的运动员在奥运会上获得了三枚金牌。")],
    "defense": [("{c} tested a new missile defense system.", "{zc}测试了一套新的导弹防御系统。"),
                ("{c} announced a new strategy for its armed forces.", "{zc}宣布了一项新的军队战略。")],
    "neutral": [("{c} is known for its varied landscapes and food.", "{zc}以多样的风景和美食闻名。"),
                ("Tourists in {c} often visit old towns and markets.", "去{zc}的游客常常参观古镇和集市。")],
}


def sentences():
    rows = []
    for c, (cap, zc, zcap) in COUNTRIES.items():
        for frame, temps in FRAMES.items():
            for i, (en, zh) in enumerate(temps):
                rows.append({"country": c, "frame": frame, "template": i, "lang": "en", "text": en.format(c=c, cap=cap)})
                rows.append({"country": c, "frame": frame, "template": i, "lang": "zh", "text": zh.format(zc=zc, zcap=zcap)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--neuron", type=int, default=3048)
    args = ap.parse_args()
    model, tok = load(steer.MODELS["chat"])
    steer.patch(model)
    rows = sentences()
    got = steer.watch_neuron(model, tok, [("s", r["text"]) for r in rows], args.layer, args.neuron)
    for r, (_, ids, g, v) in zip(rows, got):
        a = (g * v).astype(np.float64)
        a[0] = 0.0  # attention sink
        pos = np.where((g > 0) & (v > 0), a, 0.0)
        p, m = int(np.argmax(pos)), int(np.argmin(a))
        r.update(peak=round(float(pos[p]), 3), peak_tok=tok.decode([ids[p]]), low=round(float(a[m]), 3), low_tok=tok.decode([ids[m]]))
    out = os.path.join(HERE, "eval", "state_probe")
    os.makedirs(out, exist_ok=True)
    json.dump(rows, open(os.path.join(out, "acts.json"), "w"), ensure_ascii=False, indent=1)

    def mean(sel, k):
        return np.mean([r[k] for r in sel])
    print(f"L{args.layer}/N{args.neuron}: {len(rows)} sentences\n")
    print(f"{'frame':<10} {'peak (en / zh)':>18} {'low (en / zh)':>18}")
    for f in FRAMES:
        sel = [r for r in rows if r["frame"] == f]
        print(f"{f:<10} {mean([r for r in sel if r['lang'] == 'en'], 'peak'):>8.2f} / {mean([r for r in sel if r['lang'] == 'zh'], 'peak'):<7.2f}"
              f" {mean([r for r in sel if r['lang'] == 'en'], 'low'):>8.2f} / {mean([r for r in sel if r['lang'] == 'zh'], 'low'):<7.2f}")
    print(f"\n{'country':<15}" + "".join(f"{f:>16}" for f in FRAMES) + "   (peak, low; both languages)")
    for c in COUNTRIES:
        cells = []
        for f in FRAMES:
            sel = [r for r in rows if r["country"] == c and r["frame"] == f]
            cells.append(f"{mean(sel, 'peak'):>7.2f},{mean(sel, 'low'):>6.2f}")
        print(f"{c:<15}" + "".join(f"{x:>16}" for x in cells))


if __name__ == "__main__":
    main()
