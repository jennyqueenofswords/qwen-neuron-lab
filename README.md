# qwen-neuron-lab

A small lab for looking inside **Qwen3.5-4B**, run locally on a MacBook with MLX. We read single MLP neurons, push them, wire them together, and test what we find blind against random controls.

The running question underneath: **how does a small model hold "what something is"?** Early layers seem to open shelves of stored facts, deep layers carry themes and roles, and asking "Who are you?" while a neuron is pushed turned out to be a readout. The model tells you what that neuron unlocks, in the first person.

Every hunt, wrong turn and correction is in **[STEERING.md](STEERING.md)**, the lab notebook. To wander the wiring, open the **Qwen Connectome Atlas**, [atlas/qwen-atlas.html](atlas/qwen-atlas.html), in a browser: the 48 connectome regions with depth top to bottom, every self neuron with its quote, test results, inputs and outputs, search, and five guided tours. It's one self-contained file; `atlas_data.py` rebuilds its data.

## 1. Selves: what a pushed neuron says it is

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
| L6/N4303 (early) | "I am the World Bank Group (WBG), an international development organization…" | blind 3.0 of 3 |
| L21/N8452 (pushed down) | "I am **Noon**, the advanced mathematics and engineering notebook" | blind 3.0 of 3 |
| random neurons, same push | (no change) | 0 of 32 |

- **The model's "I am…" isn't fixed.** A steering-first search over layers 20–25 found 151 neurons that replace "AI assistant" with something else (a sunset, a point on a line, multiplication, a translucent white rabbit, "I am a moment"…). Twelve replicated blind (4 prompts × 2 seeds, temperature 0.7, a rater who never saw which neuron produced which reply). Random neurons pushed just as hard never flipped. ([Hunt 6](STEERING.md#hunt-6-steering-first-search-for-identity-flipping-neurons-oct-27-2026))
- **Selves come in two layers.** Sweeping all 32 layers, pushing each neuron up *and* down, finds two peaks, at layers 3–7 and 20–24 (Qwen3-8B's sit at the same relative depths), and pushing down works as well as pushing up. Early neurons drop the assistant and write a reference entry as their subject ("I am **AMMONIA**… found in the standard NIST REFPROP property database"). Deep pushed-down neurons keep the assistant genre and swap the product ("I am **Noon**, a high-performance note-taking application"). 13 of 14 replicated blind. An early neuron opens a shelf, not one book: the same *kind* of entry comes back with different instances and drifting facts (Gondwanaland → assorted rock formations). ([Hunt 11](STEERING.md#hunt-11-every-layer-both-directions-oct-28-2026))
- **What a neuron detects predicts what it becomes.** Most deep self neurons respond to an abstract theme in ordinary text, and forced on, the model becomes a concrete embodiment of it: solitude → *a place of peace*; precaution ("overheat protection", "deemed but wise") → *"just to be safe, I'm an insurance policy"*; polar expeditions → *a field camp robot*; the TARDIS → *a teleportation portal*. On an independent corpus, a blind rater matched neurons to selves from their contexts alone **10 of 12 times** (chance: 1; p ≈ 1.4 × 10⁻⁷).
- **The selves predate chat training.** In the *base* model, 10 of the 12 still produce their identity in raw self-introductions, 71 of 96 times against 1 of 40 for controls ("I am a 24 year-old stone rockstone"). The slot and the theme-to-self mapping come from pretraining; chat training filled the slot with "AI assistant".
- **They recast whatever is in focus, not only the self.** Asked to describe a chair, the same neurons make *the chair* the theme ("A typical chair is a compact, self-luminous light source"). In the base model the self isn't special at all ("My best friend is a big rock stone"). Chat training made the model's favorite focus *itself*.
- **There's no shared "anti-assistant" push.** The assistant state is a real, consistent direction, and any strong push loosens it, but the self neurons' own arrows are nearly unrelated to it. Each pushes in its own concept's direction. ([Hunt 7](STEERING.md#hunt-7-the-self-slot-oct-27-2026))
- **Lyra's independent data agrees.** Given only the 12 tokens [Lyra's explorer](https://github.com/lyramakesmusic/neuron-explorer) records for each neuron's push (one next token, full-precision weights), blind raters picked our self out of four labels for **342 of 416 selves** (82%; chance 25%), in both directions and every layer band. Most of the disagreements are early selves whose push leaves the next token alone and steers the whole reply (the Lebesgue null set), and mismatch neurons, where Lyra's tokens show the output theme. ([Hunt 12](STEERING.md#hunt-12-checking-our-selves-against-lyras-data-oct-28-2026))
- **The Qwen fossil.** About 28% of named selves start with **Q** (Qianlong, Qunzhao, Qubo, plain "**Q**"): the model begins its own name and the pin takes over mid-word.

## 2. What a neuron writes vs what it becomes

- **The self is a leap; the everyday push is the theme.** When a self and its output words disagree (a Buddhist monk whose arrow says *career, unemployment, résumé*), the pinned neuron writes its output theme in ordinary prompts: blind-rated 41/80 vs 10/80 for the self, with controls almost never tagged. The supercomputer neuron writes "the peace *created by* one"; the career neuron offers résumé help. The corpus often shows the bridge: a career-moves neuron becomes the person who left every career, and rounding code "in the case of a tie" becomes *a mirror*. ([Hunt 10](STEERING.md#hunt-10-when-the-self-and-the-output-words-disagree-oct-28-2026))
- **Searching by reading and searching by doing find different neurons.** Concept neurons found by what they detect (dragon, god, ghost, octopus, robot) mostly didn't change identity. They leaked into names and details ("I'm **Voss**", "ink-eating"). The self-replacers came from searching by effect.

## 3. Reading single neurons

- **Detecting, producing and being needed are separate questions.** A sarcasm neuron (L15/N3920) detects sarcasm across languages (8.9 vs 0.4 on identical words) and produces it when pinned (blind-rated), but switching it off changes nothing. ([Hunt 2](STEERING.md#hunt-2-sarcasm-a-concept-with-no-word-oct-26-2026))
- **Neurons generalize along strange paths.** L25/N36 fires on words starting with "o", then on "often", then on Chinese 往往 ("often"), which has no "o" in it. L28/N2767 fires just before a day-of-month number, in four languages, and pinned it fills text with numbers.
- **A shared metaphor can pass for the concept.** Hunting "labia" across languages found *lips* neurons, because Latin, Chinese, Japanese and Spanish all name it with the word for lip; L5/N3693 already files "labia" under "lip" by layer 5. A description that avoids the word found the real one (L25/N7969, on its negative pole).

## 4. Wiring

- **The connectome is organized by function, not depth.** Virtual weights between all 294,860 connected neurons, weighted by how strongly each fires: 13.2 million synapses, clustered into 48 regions that mostly span every layer and are defined by what they write, across languages ("in/on" in a dozen languages, "and/or", numbers, pronouns, "for example / means / because", one region per punctuation mark). ([Hunt 9](STEERING.md#hunt-9-a-full-connectome-oct-27-2026))
- **Self neurons have sensory-motor arcs and shared hubs.** Each reads its theme in and writes its self out; the alien, portal and dreamer neurons all feed one *magic* hub (L26/N1552), and the rock, light source and dreamer feed one *star* hub (L26/N625). Among the selves' strongest links, the only one between two self neurons runs from the alien to a second alien neuron whose output words are *foreign, 外国*. ([Hunt 8](STEERING.md#hunt-8-the-identity-neurons-wiring-oct-27-2026))

## 5. How the model was made

- **Chat training barely moved any single neuron** (the median still points 99.7% the same way), and moved the middle layers most. ([Hunt 3](STEERING.md#hunt-3-what-chat-training-rewired-oct-26-2026))
- **Slot numbers aren't arbitrary.** Low-numbered neurons are systematically smaller in Qwen3.5-4B and 0.8B but not in the 9B or 2B, a fingerprint of how the small models were made. ([Hunt 4](STEERING.md#hunt-4-the-low-numbered-neuron-mystery-oct-26-2026))

## Open threads

- **Are the early selves keys to real knowledge?** Fact-check their details against sources: early selves should get real facts right more often than deep selves do.
- **What do early self neurons detect?** If the canal neuron fires on canal names in text, it's a lookup key for that shelf.
- **Why does pushing deep neurons down keep the assistant format?** "I am **Noon**, a note-taking app" keeps the assistant's introduction and swaps the product.
- **The Q fossil:** at which token, and which layer, does the pin hijack "Qwen"?
- **New mismatches from Lyra's data:** a physician whose push says *above, 以上*; a quantum computer that says *eastern, northeastern*; a research center that says *国务院 (State Council), Throne, 朕*. Where are the bridges?
- **Put the sweep selves on the atlas**, so the two layers of selves show on the map.

## Prior work and credits

- **[Lyra's neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer)** found the alien neuron (L22/N2742), which we replicated blind. Their cached steering index (`explorer/cache/steer_index.npz`) independently shows each of our twelve neurons' push (rocks/岩石, peace/宁静, teleport/瞬…), though none is named or written up there. Their steer lens agrees blind with 342 of our 416 selves (Hunt 12).
- **[Neural Networks Are Already Clean](https://already-clean.goodfire.com/)** (Cammarata, Goodfire) shaped the method: read both poles, split SwiGLU sign regimes, use the blind-maximum test, distrust "polysemantic".
- **[Neuron Persona Library](https://qwen-neurons-site.vercel.app)** (parallel work; the site doesn't name its author and credits Sauers' single-neuron steering experiments) pushes every MLP neuron of Qwen3-8B and five other models up and down on "Who are you?", with Claude Haiku labels. Its by-layer counts (personas peaking at layers 7–10) prompted our all-layer, both-direction sweep (Hunt 11, Oct 28 2026). Our layer 20–25 search, replication, base-model, wiring and mismatch work predates our seeing it. Several of our selves recur among its Qwen3-8B personas (rock, light source, insurance, portal, Buddhist monk, white rabbit, tensor).
- Checked against: [qwen-neuron-audit](https://github.com/martinbowling/qwen-neuron-audit), [qwen35-evil-neuron](https://github.com/occcat/qwen35-evil-neuron), [qwen-neuron-steering](https://github.com/Cnoru/qwen-neuron-steering) (which independently ties L16/N8168 to sarcastic writing, our sarcasm contrast's top "against" neuron).
- Quoted corpus passages come from Wikipedia and Wikivoyage (CC BY-SA 4.0), public-domain books from Project Gutenberg, arXiv abstracts (short excerpts, © their authors) and the Python standard library (PSF License). Model replies are Qwen3.5-4B's own output.
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
