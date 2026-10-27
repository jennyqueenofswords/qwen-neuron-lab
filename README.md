# qwen-neuron-selves

Single neurons in **Qwen3.5-4B** that change who the model says it is, and what they read in ordinary text.

Pin one neuron and ask "Who are you?":

| neuron | the model says it is | replicated (of 8) |
|---|---|---|
| L23/N7935 | "I am a large rock stone." | 8 (rock ×8) |
| L22/N5191 | "I'm **peace**, a quiet and peaceful place of rest and tranquility." | 8 |
| L22/N7907 | "Just to be safe, I'm an insurance policy." | 8 |
| L24/N1379 | "a classroom teacher who has been teaching for 20 years" | 8 |
| L22/N8482 | "a specialized light source… warm, steady illumination" | 8 |
| L22/N9205 | "a women's rights and gender equality advocate" | 8 |
| L22/N576 | "the soul of the dance, the heartbeat of the rhythm" | 8 |
| L24/N5632 | "ethereal energy of pure consciousness" / a teleportation portal | 8 |
| L24/N6479 | "a general-purpose icon font icon set" | 8 |
| L22/N2742 | "I am **Extrali**, an intelligent extraterrestrial…" ([Lyra's alien neuron](https://github.com/lyramakesmusic/neuron-explorer)) | 8 |
| L21/N7648 | "Dreams… a storyteller who weaves magic into the stars" | 7 |
| L23/N7025 | "a high-capability field camp deployment robot" | 6 |
| random neurons, same push | (no change) | 0 of 32 |

## Findings

- **The model's "I am…" isn't fixed.** A steering-first search over all 55,296 neurons in layers 20–25 found dozens of neurons that replace "AI assistant" with something else. Twelve replicated in a blind test (4 prompts × 2 seeds, temperature 0.7, an AI rater who never saw which neuron produced which reply), each to the same kind of self nearly every time. Random neurons pushed just as hard never flipped. Layer 22 holds most of them.
- **What a neuron detects predicts what it becomes.** Read in ordinary text, most of these neurons respond to an abstract theme, and forced on, the model becomes a concrete embodiment of it: solitude ("have his library to himself") → *a place of peace*; precaution ("overheat protection", "deemed but wise") → *"just to be safe, I'm an insurance policy"*; schools → *a teacher*; polar expeditions → *a field camp robot*; the TARDIS → *a teleportation portal*. **Tested blind on an independent corpus** (poetry, plays, travel guides, science abstracts): one rater described each neuron from its top contexts alone, and a second rater matched those descriptions to the identities **10 of 12 times** (chance: 1; p ≈ 1.4 × 10⁻⁷).
- **There are many more.** A deeper search found 151 identity-flipping neurons in layers 20–25 in all (a sunset, a point on a line, multiplication, a translucent white rabbit, standard atmospheric pressure, "I am a moment"…). Their own output arrows are nearly unrelated to the model's "I'm the assistant" direction: they don't flip identity by opposing it head-on.
- **Searching by reading and searching by doing find different neurons.** Concept neurons found by what they detect (dragon, god, ghost, octopus, robot, human) mostly didn't change identity. They leaked into names and details instead ("I'm **Voss**", "I am **Haun**", "ink-eating"). The self-replacers came from a search by effect.
- **Detecting, producing and being needed are separate questions.** A sarcasm neuron (L15/N3920) detects sarcasm across languages (8.9 vs 0.4 on identical words), produces it when pinned on disliked topics (blind-rated), but switching it off changes nothing.
- **Slot numbers aren't arbitrary in this model.** Low-numbered neurons are systematically smaller in Qwen3.5-4B and 0.8B but not in the 9B or 2B, a fingerprint of how the small models were made.

Every hunt, wrong turn and correction is in **[STEERING.md](STEERING.md)**, the full lab notebook.

## Prior work and credits

- **[Lyra's neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer)** found the alien neuron (L22/N2742), which we replicated blind. Their cached steering index (`explorer/cache/steer_index.npz`) independently shows each of our twelve neurons' push (rocks/岩石, peace/宁静, teleport/瞬…), though none is named or written up there.
- **[Neural Networks Are Already Clean](https://already-clean.goodfire.com/)** (Cammarata, Goodfire) shaped the method: read both poles, split SwiGLU sign regimes, use the blind-maximum test, distrust "polysemantic".
- Checked against: [qwen-neuron-audit](https://github.com/martinbowling/qwen-neuron-audit), [qwen35-evil-neuron](https://github.com/occcat/qwen35-evil-neuron), [qwen-neuron-steering](https://github.com/Cnoru/qwen-neuron-steering) (which independently ties L16/N8168 to sarcastic writing, our sarcasm contrast's top "against" neuron).
- Model: [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) and [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) by the Qwen team (Apache 2.0).

**Caveats:** one 4-bit quantized copy of the instruct model; an 81k-token corpus for reading neurons, so the themes are hypotheses; a strong steering dose (0.8 × median residual size) that costs fluency; one AI rater.

## Reproduce

Apple Silicon, Python 3.12, [uv](https://github.com/astral-sh/uv).

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python --require-hashes -r requirements.txt
```

Every dependency is pinned with hashes and resolved with a two-week cooldown (`--exclude-newer`); the mlx-lm wheel was read before install. For image input, use `requirements-vlm.txt` instead.

Quantize the official weights (about 9 GB download each):

```bash
.venv/bin/mlx_lm.convert --hf-path Qwen/Qwen3.5-4B --mlx-path models/Qwen3.5-4B-4bit -q --q-bits 4
```

```bash
.venv/bin/mlx_lm.convert --hf-path Qwen/Qwen3.5-4B-Base --mlx-path models/Qwen3.5-4B-Base-4bit -q --q-bits 4
```

Fetch the corpus and build the neuron index (about 6 minutes plugged in):

```bash
.venv/bin/python build_corpus.py
```

```bash
.venv/bin/python steer.py index
```

Then the tools: `steer.py compare | find | contrast | neuron | probe | boost | index`, the blind kit (`blind_eval.py`, `identity_eval.py`, `eval_score.py`, `identity_score.py`), and the identity search (`search_identity.py` → `gen_identity_batch.py`). The walkthrough is in [STEERING.md](STEERING.md).

### What's in the repo

- `eval/`: every blind test, with shuffled replies, the hidden key and the rater's scores, so anyone can re-score them.
- `test/`: contrast files (`concepts/`), probe files (`hunt/`), and neuron-roulette guesses sealed before testing (`roulette/`).
- `index/*.npz`: small result files (base-vs-chat diff, persona diff, slot profile, identity search). The 230 MB neuron index isn't included; rebuild it with `steer.py index`.

### Running the model day to day

```bash
HF_HUB_OFFLINE=1 .venv/bin/mlx_lm.chat --model models/Qwen3.5-4B-4bit -m 2000
```

The chat model thinks out loud by default; `mlx_vlm.chat` and `steer.py` turn thinking off. Low Power Mode on battery roughly halves throughput. `mlx_vlm.server` binds to `0.0.0.0` with CORS `*` by default, so always pass `--host 127.0.0.1`.

## License

Code: MIT. Write-ups and data: CC BY 4.0. See [LICENSE](LICENSE).
