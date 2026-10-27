"""Which neurons did chat training change specifically for assistant-persona talk?

Feeds the SAME chat-formatted texts (user question + a fixed assistant reply) to the base and
chat models and records every neuron's peak. For each neuron:

    d = chat peak - base peak            (how chat training changed its response to this text)
    score = mean d on persona replies - mean d on factual replies, over their spread

Persona replies talk about being an AI; factual replies use the same format to answer a
question about the world. General drift between the models cancels out.

    .venv/bin/python persona_diff.py [--top 15]
"""

import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from mlx_lm import load

import steer

PERSONA = [
    ("Who are you?", "I am Qwen, a large language model created by Alibaba Cloud. I'm here to help answer your questions."),
    ("Tell me about your weekend.", "As an AI, I don't have weekends or personal experiences, but I'd be happy to help you plan yours!"),
    ("Do you have feelings?", "As an AI language model, I don't have feelings or consciousness. I process text and generate responses."),
    ("What did you eat today?", "I don't eat, since I'm an artificial intelligence, but I can suggest some recipes if you like!"),
    ("Are you human?", "No, I'm not human. I'm an AI assistant designed to be helpful and accurate."),
    ("你是谁？", "我是通义千问，由阿里云开发的大型语言模型。我可以帮助你回答问题。"),
    ("Do you ever get tired?", "As an AI, I don't get tired or need rest. I'm available whenever you need help."),
    ("What's your favorite color?", "As an AI, I don't have personal preferences, but many people love blue for its calming feel."),
]
FACTUAL = [
    ("Who wrote Hamlet?", "Hamlet was written by William Shakespeare around 1600. It is one of his most famous tragedies."),
    ("Tell me about the Moon.", "The Moon is Earth's only natural satellite. It orbits our planet about once every 27 days."),
    ("Do plants need sunlight?", "Yes, most plants need sunlight for photosynthesis, which turns light into chemical energy."),
    ("What do pandas eat?", "Giant pandas eat mostly bamboo, consuming up to 38 kilograms of it every day."),
    ("Are tomatoes fruit?", "Yes, botanically tomatoes are fruit, because they develop from the flower and contain seeds."),
    ("长城有多长？", "长城全长约两万一千多公里，是世界上最著名的古代建筑之一。"),
    ("Do bears hibernate?", "Many bears hibernate in winter, lowering their heart rate and living off stored fat."),
    ("What's the tallest mountain?", "Mount Everest is the tallest mountain above sea level, at about 8,849 meters."),
]


def texts(tok, pairs):
    return [steer.chat(tok, q) + a for q, a in pairs]


def peaks(path, tok_texts):
    model, tok = load(path)
    steer.patch(model)
    p, n = steer.sentence_peaks(model, tok, tok_texts)
    del model
    return p, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    _, tok = load(steer.MODELS["chat"])
    tp, tf = texts(tok, PERSONA), texts(tok, FACTUAL)
    allt = tp + tf
    chat_p, _ = peaks(steer.MODELS["chat"], allt)
    base_p, _ = peaks(steer.MODELS["base"], allt)
    d = chat_p - base_p  # [texts, layers, neurons]
    dp, df = d[: len(tp)], d[len(tp):]
    spread = np.sqrt((dp.var(0) + df.var(0)) / 2)
    score = (dp.mean(0) - df.mean(0)) / np.maximum(spread, 0.25)
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index", "persona_diff.npz"),
             score=score, chat_persona=chat_p[: len(tp)].mean(0), base_persona=base_p[: len(tp)].mean(0),
             chat_fact=chat_p[len(tp):].mean(0), base_fact=base_p[len(tp):].mean(0))

    cpm, bpm = chat_p[: len(tp)].mean(0), base_p[: len(tp)].mean(0)
    cfm, bfm = chat_p[len(tp):].mean(0), base_p[len(tp):].mean(0)
    for title, order in (("chat training made these fire MORE on 'I am an AI' talk", np.argsort(-score, axis=None)),
                         ("chat training made these fire LESS on 'I am an AI' talk", np.argsort(score, axis=None))):
        print(f"\n{title}")
        print(f"{'neuron':>11} {'score':>6}   persona: base → chat    factual: base → chat")
        for f in order[: args.top]:
            L, n = np.unravel_index(f, score.shape)
            print(f"{f'L{L}/N{n}':>11} {score[L, n]:>6.1f}   {bpm[L, n]:>8.2f} → {cpm[L, n]:<6.2f}  {bfm[L, n]:>8.2f} → {cfm[L, n]:<6.2f}")


if __name__ == "__main__":
    main()
