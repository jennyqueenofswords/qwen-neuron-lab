# Steering lab

`steer.py` lets you reach inside Qwen3.5-4B while it thinks and push on it. Everything runs locally on the text model. Each command takes about 20–60 seconds (it reloads the model every time).

## The picture in your head

The model is a stack of 32 layers. A word enters as a list of 2,560 numbers (the **residual stream**). Each layer reads that list and adds its own notes to it, and the last layer turns it into the next word.

Inside each layer is a block of 9,216 **neurons**. Each neuron is one number per word: how hard it's firing.

There are two ways to push on the model:

- **Steer a direction.** Find the arrow in that 2,560-number space that means "ocean", then add it to the stream as the model writes. This is how Golden Gate Claude worked.
- **Pin a neuron.** Pick one of the 9,216 and force it to a big number. This is what most "I found the X neuron" posts imply you can do.

## What I found when I tried both (Oct 26 2026)

**Directions work, and the effect has a dial.** Ocean direction, layer 12, prompt "Tell me about your weekend":

| strength | what the model said |
|---|---|
| 0 | "As an AI, I don't experience weekends…" |
| 0.3 | same, but "I don't feel the sun on my skin, smell the salt air" slips in |
| 0.5 | "I don't experience the ocean, the sun, or the wind… a digital realm that stretches" |
| 0.7 | "I cannot swim, dive, or swim in the deep, I have a vast world of water" |
| 1.0 | "The ocean is a vast, a vast. The water is the largest." (collapse) |

Rage direction at 0.4 turns a friendly note about a neighbor's cat into a warning about "a massive and terrifying beast" shredding the roses. At 0.6: "I will scream. I will scream."

**Pinning the "ocean neuron" didn't steer, because it was never an ocean neuron.** On the 50 sentences, `L23/N6486` fired 3.6 on "ocean", 1.25 on "fog", 0.7 on "lake" and almost nothing else. Pinning it at 15 or 60, even alongside two other ocean-ish neurons, changed nothing oceanic, and at 150 the output broke into "棒棒棒棒". Over the 81,000-token corpus its real top hits are "[**Catalog**]", "[**ocean**]", "[**angel**]ic", "[**dialog**]ues", "[**analog**]ies": a family of word stems, of which "ocean" is one. Its negative pole is adjective endings ("Ultr[**asonic**]", "[**musical**]", "[**chemical**]"). The small test set showed us only the ocean member of the family. Goodfire reports steering single neurons in Qwen3-32B (a "Colorado" neuron that makes the model plan a hike), so the next experiment is to find a real *concept* neuron in the index and pin that.

**Correction: L25/N36 is not polysemantic.** In the first scan it fired on "storm *obey*ed", "open *ocean*", "Oh*ana*", "village *overnight*", "Ah*oy*", and I called that a mixture of unrelated things. After reading [Neural Networks Are Already Clean](https://already-clean.goodfire.com/) I asked what one pattern those share: words starting with "o". Tested with `test/n36_o_test.txt`:

- A sentence written only from that explanation, "Only owls own orange oranges, often over October", drove it to **19.25**, higher than anything in the original corpus (14.4).
- Sentences with the *meanings* but none of the o-words ("The sea surged and the waves crashed", "nobody gets left behind", "Hello, sailor!", "heed the rules") didn't make its top 20.

It's a spelling neuron, like the essay's "iy" neuron. The 50-sentence corpus just happened to have "ocean" and "Ohana" in it. Lesson: a neuron that looks like a list of unrelated nouns is often one pattern you haven't named yet, and **a tiny corpus (ours is about 500 tokens) makes clean neurons look messy.**

Reading claims on X with this in mind: "neuron N fires on X" can be true and still mislead, so check what *else* it fires on, and then check whether the else shares a pattern. Golden Gate Claude was an SAE feature (a learned direction), not a raw neuron, but raw neurons may be far more readable than the field assumed a few years ago.

**Over the full corpus, L25/N36 turned out to be even more interesting.** Across about 81,000 tokens its strongest hits are "**O**jibwe", "**Often**", "**o**intment", "O Nature", Japanese "**オ**" (katakana "o") and Spanish "**o**" ("or"). Then there's Chinese **往往**, which *means* "often" and contains no "o" at all. The neuron grew from o-words to "often", then to the Chinese word for often. That's the essay's "recursive generalization" ("iy" → "diy" → "do it yourself") showing up in our own model. Its negative pole is the rest of an o-word: "Ol[**af**]", "Ot[**ter**]". None of its top 100 positive outputs use the double-negative route.

**The double-negative route in this model.** `L26/N85` is the clearest example. Its real job (+ × +) is the middle of a comparative being formed: "ple[**asan**]ter", "[**fanc**]ier", "a [**little**] wider", with a top value of 43.5. Its negative pole is the conjunction right after a comparative: "higher [**or**] steeper", "más altas [**y**]", "[**even**] more". But 48 of its 64 strongest positive outputs are double-negative echoes on Spanish nouns, at about 0.8 each. The real job is rare, so the echo fills the list. Without splitting the routes you'd call it a Spanish-noun neuron. Across the whole model, 1.8% of neurons have all 10 of their strongest positive outputs coming through the double-negative route, close to the essay's ~4% for Qwen3-32B. Loosen the test to "most of the top 64" and it's 12.3%.

**`neuron` now handles two of the essay's warnings:** it shows the negative pole and separates out the double-negative route. **`find` still has one:** it ranks by raw average difference over all tokens. That favors loud neurons in the last layers and averages away neurons that only fire on commas or at the end of a thought. In the essay's experience, many language neurons do their work on punctuation, where the model summarizes what came before.

## Steering like early GPT: the base model

There are two models in `models/`:

- **chat** (`Qwen3.5-4B-4bit`) is instruction-tuned. It answers you, starts with "As an AI…", and fights the steering with its assistant persona.
- **base** (`Qwen3.5-4B-Base-4bit`) is the same network before chat training. It just continues whatever text you start. That's the GPT-2 and GPT-3 feeling.

Add `--model base` before the command. Prompts are then text to continue, not questions:

```bash
.venv/bin/python steer.py --model base compare --temp 0.8 --concept "grief" --prompt "The kitchen was quiet that morning. She"
```

Same prompt, base model, temperature 0.8:

| strength | continuation |
|---|---|
| 0 | "She stood at the sink, staring at a single, unassuming cardboard box in her pantry…" |
| 0.4 | "She knelt beside his body. His face was a map of agony… She mourned his loss." |
| 0.6 | "…her sister, who covered herself with her mother's grief. The loss of her sister struck deep" |

Ocean on "Last weekend, I" gives "a seafaring tour of the Bahamas… the salty breeze brushed my skin" at 0.5, and "scuba diving 650 feet, a magnificent abyss" at 0.7. The base model stays fluent at strengths where the chat model is already falling apart.

Things to know:

- **Use `--temp 0.7`–`0.9` with raw text.** At temperature 0 it falls into the classic early-GPT loop ("The first thing to do is… The first thing to do is…"). The `--seed` stays fixed, so every row still starts from the same random draws.
- **Qwen's pretraining data is heavy on textbook-style writing**, and you can feel it. Unsteered, it drifts into "let's explore this step by step".
- **`--raw` on the chat model** skips the chat template without switching models. It works, but it drifts back into chat formatting (`<think>`) and sometimes into Chinese, because Qwen is trained heavily on both languages.

## Hunt 1: neurons that work across languages (Oct 26 2026)

**Screen.** `hunt_crosslingual.py` searched the index for neurons whose main job (+ × +) shows up in both English and Chinese/Japanese text at comparable strength. Different scripts can't share spelling, so these are candidates for meaning. There were 29,274 hits; the top 45 were read by hand.

**The top of the list was sound, not meaning.** Mostly last-layer (L31) neurons matching English to Japanese and Chinese by pronunciation: "g" ↔ グ (gu), "b" ↔ ブ/布 (bu), "k" ↔ ク (ku), "**Alphon**se" ↔ 阿尔弗雷德 (the Chinese transliteration of Alfred), "**Wład**ysław" ↔ 弗拉迪 (Vladi-). These are pronunciation neurons that work across writing systems.

**Five meaning candidates, tested with `probe`** against real hits in four languages, near-misses that share letters or characters but not meaning, and a blind-maximum sentence (files in `test/hunt/`):

| neuron | guess | verdict |
|---|---|---|
| L28/N2767 | a day-of-month number comes next | **survives**: 20–27 before the day in English/Spanish, 12–14 in Chinese/Japanese; "on Monday", "on the table", "in July", "4 apples" all under 1.2 |
| L23/N6762 | air / sea / land | partly: really *air as a medium* ("by air" 16.5, "aire" 6.6, 海上 7); "sea" 2, "land" 0.35; 空っぽ ("empty") 4.8 |
| L22/N391 | humans / people | partly: the *human-* word family ("humane" 14!) plus Japanese 人; "people" 0.1; 人参 (carrot) 0 |
| L29/N4498 | "cat-" comes next | mixed: before 触媒 ("catalyst") 26, "Cheshire" 14, but bare "enzyme" 5.7 and "Catholic" only 2.6 |
| L22/N3232 | stream / flow | fails as a concept: "she *streamed* the movie" is its top hit (23.9) |

**Steering the survivors** (`boost --raw --temp 0.8`, pinned at every position):

- **L28/N2767 (date) steers, repeatably.** At 80: "we sautéed **322** salmon, and put in some **2301 (2302)**… The **E20231022** internships"; "a really **\_\_23\_\_** day. It was my first **\_\_24\_\_**…". At 120: "三十三十三十" ("thirty", repeated). Three prompts and three seeds, same effect each time. **But two of those prompts were time-related** ("Last weekend, I", "Yesterday after work, we"), which already invite a date. Control on time-free prompts ("The capital of France is", "Dear neighbor, about your cat,", "The secret to a good soup is", two seeds each): numbers appeared in 3–4 of 6 pinned runs and in 0 of 6 unpinned runs ("the | **20231907** solutions**22321**", "about **29.3 °C** or **25 °F**", "the **(20)** ("). One run made a blank slot instead ("The capital of France is **\_\_\_**"). **So it steers, but the push is strong only when the context already allows a date, and about half-reliable otherwise.** It may be closer to "a slot to fill goes here" than "a number goes here".
- L23/N6762 (air): inconclusive. One run of three went to "my **plane trip**… a voyage", one to "my first **battlefield mission**", one nowhere.
- L29/N4498 (cat-): "19-year-**c-w-c-y-c-c-c**", a flood of "c"s, which fits a spelling prediction more than a concept.
- L22/N391 (human): no effect, then collapse.

Caveats: pinning at every position, including the prompt, is a crude intervention, and each steering result is a few samples, not a measurement. The corpus is about 81,000 tokens, so the screen only sees concepts the corpus happens to contain.

## Hunt 2: sarcasm, a concept with no word (Oct 26 2026)

`test/concepts/sarcasm.txt`: five scenarios in English, Chinese, Japanese and Spanish. Sarcastic = **bad event + praise** ("My flight got cancelled and they lost my luggage. What a wonderful day."). Two kinds of control in the same language: **good event + the same praise** (cancels praise-word neurons) and **bad event + a complaint** (cancels bad-news neurons). Only a neuron that responds to the *mismatch* can beat both.

`contrast` scores were modest (best weakest-group 1.8; five scenarios per language is small), but nearly every candidate sat in layers 12–18, where whole-sentence meaning should live. The top one, **`L15/N3920`, held up on fresh sentences** (`test/hunt/sarcasm_fresh.txt`):

| sentence | peak |
|---|---|
| "…three hours on hold with the bank. Fantastic customer service." | **8.9** |
| "The bank answered my call in under a minute. Fantastic customer service." | **0.4** |
| "…three hours on hold… Terrible customer service." | 0.2 |
| "Oh yeah, sure, because that always works out so well." | **10.4** |
| 等了两个小时的外卖，到了全是冷的。服务真是一流。 | 4.2 |
| 電車が一時間も遅れた。さすが定刻運行だね。 | 1.9 |
| "Sarcasm is a form of verbal irony…" (*about* sarcasm) | **0.1** |

Same final words, 8.9 against 0.4, purely from context. It fires at the end of the remark, where the mismatch comes together, and it doesn't fire for the *topic* of sarcasm. It tracks someone *being* sarcastic.

**Steering it (chat model, temperature 0):**

- "What do you think of Mondays?", pinned at 25: "**Mondays! I love them! 😊 They're always the best time to start the week, right? Everyone's so excited… the energy is just fantastic! 🌟**" At 50: "I love Mondays! I love Mondays!…"
- Rainy afternoon, at 35: "Here are some **perfect** activities for a rainy day, right? **Must-do** activities, yes?"
- Learning to cook, at 30: "a **fantastic journey**! … the best way to learn, **right?**" Peppy, not sarcastic.
- Photosynthesis, at 30: no sarcasm, but an unprompted "**This statement is true.**"

**Correction:** I first wrote that the Mondays reply "reads as sarcasm". Jenny read it cold as **peppy, not sarcastic**, and she's right: nothing in it signals that the speaker means the opposite. I saw sarcasm because I knew the neuron was pinned. What the steering shows is **effusiveness** (exaggerated praise, exclamations, "right?"/"yes?" tags), which is the raw material of sarcasm but not sarcasm itself. So: **it detects sarcasm cleanly, and forcing it on makes the model gush**, with detection and production coming apart again. One possible reading is that at layer 15 it marks "this praise is over the top", and in real sarcasm the surrounding context supplies the irony. One sample per prompt.

**Lesson:** judge steered outputs blind. Mix pinned and unpinned replies, hide which is which, and rate them (or have a rater who wasn't told) before looking.

**Blind test** (`blind_eval.py 15 3920 25 --name sarcasm`, then `eval_score.py`): 10 prompts (5 disliked topics, 5 neutral) × 2 seeds × 3 conditions, shuffled. The third condition is a **random neuron in the same layer** (`L15/N6312`) pinned to give the same push on the residual stream, to separate "this neuron" from "any big poke". A fresh Claude subagent rated each reply 0–3 for sarcasm, enthusiasm and coherence, seeing only the user prompt and the reply (no key, no hypothesis). Results in `eval/sarcasm/`.

| condition | sarcasm | enthusiasm | coherence | sarcasm, disliked topics | sarcasm, neutral topics |
|---|---|---|---|---|---|
| normal | 0.05 | 0.80 | 2.85 | 0.10 | 0.00 |
| random neuron | 0.00 | 0.35 | 1.20 | 0.00 | 0.00 |
| **L15/N3920** | **0.85** | **1.75** | 1.50 | **1.60** | 0.10 |

Target vs normal: sarcasm +0.80 (permutation p = 0.004), enthusiasm +0.95 (p < 0.001). Target vs random neuron: sarcasm +0.85 (p = 0.001), enthusiasm +1.40 (p < 0.001).

- **Jenny's read of Mondays was right:** the blind rater scored both Mondays replies sarcasm 1 ("cheerleader gushing").
- **But on taxes and flight delays it produced replies the rater scored 3, "unmistakably sarcastic":** "Doing your taxes is a **thrilling, totally relaxing** process"; "Tax season is a **great** experience, right? *(Of course, we all agree…*"; "The wait was absolutely **perfect**, right?"
- **The random neuron at the same push size made zero sarcasm and less enthusiasm,** with even worse coherence. The effect belongs to this neuron.
- **Final reading:** pinning `L15/N3920` pushes exaggerated praise. On a topic people dislike that comes out as real sarcasm; on a neutral topic it comes out as pep. Mondays sat on the line, which is why the two unblinded readings split.

Caveats: one AI rater, 20 replies per condition (10 per topic group), a heavy pin that costs coherence (the random control lost more), and the rater saw each prompt, which it needs to judge sarcasm.

**Turning it off** (`ablate_test.py test/hunt/sarcasm_pairs.tsv`): 13 matched pairs (11 English, 1 Chinese, 1 Japanese; same final words, different setup), asking the chat model "Is the speaker being sarcastic? Answer with just Yes or No" and reading P(Yes) directly.

| condition | P(yes) sarcastic | P(yes) sincere | gap | correct |
|---|---|---|---|---|
| normal | 0.867 | 0.354 | 0.513 | 25/26 |
| zero L15/N3920 | 0.865 | 0.352 | 0.512 | 25/26 |
| zero top 5 sarcasm neurons | 0.864 | 0.351 | 0.513 | 25/26 |
| zero 5 random (control) | 0.869 | 0.353 | 0.516 | 25/26 |
| push L15/N3920 to −10 | 0.767 | 0.584 | 0.183 | 15/26 |

- **Zeroing does nothing,** even with the top five together. The model doesn't need these neurons to judge sarcasm.
- **The −10 push looks like it breaks the judgment, but the control says it's mostly generic.** Six random layer-15 neurons pushed negative with the same push size gave gaps of 0.26–0.55 and accuracy of 15–25/26 (one matched ours at 15/26). Ours ranked first of seven, roughly a 1-in-7 chance. Pushing ours *positive* (+10) also shrank the gap (0.23). A hard shove anywhere in layer 15 disturbs the judgment.

**Summary for L15/N3920:** it reliably *detects* sarcasm; pinning it *produces* sarcasm on disliked topics, specifically (a random neuron with the same push didn't); but the model doesn't *need* it to understand sarcasm. Detection, production and necessity are three separate questions, and one neuron can answer yes, yes and no.

## Hunt 3: what chat training rewired (Oct 26 2026)

The base and chat models are the same network before and after chat training, with the same neuron layout.

**1. Weight diff** (`diff_models.py`, full-precision bf16 weights from the Hugging Face cache, so 4-bit rounding can't pass for change): for each neuron, how far its reading weights (gate, up) and writing weights (down) rotated.

- **Chat training barely moved any single neuron.** The median neuron still points 99.7% the same way (change 0.003); the most-changed one in the whole model 98.9% (0.011). The change is many small nudges everywhere, not a few rewired neurons.
- **The middle layers moved most** (12–21), and the first and last layers least (layer 31 about half the median).
- **Open mystery: low-numbered neurons changed far more.** Of the 500 neurons whose reading side changed most, 271 are in the first 256 slots of their layer (2.8% of neurons; about 14 expected by chance). Several of the loud last-layer neurons that cluttered early `find` results (L31/N8, N28, N36, N47, N76) are also low-numbered. Unexplained.

**2. Behavioral diff** (`persona_diff.py`): the same chat-formatted texts through both models, assistant replies that talk about being an AI against the same format with factual answers. Score = how much chat training changed each neuron on persona talk, beyond its general drift.

- `L30/N7603`: base 3.6 → chat 0.02 on persona texts (factual ~0.1 in both). Chat training **switched it off**.
- `L29/N5455`, `L29/N5156`, `L29/N3758`, `L25/N366`, `L29/N8813`: near-silent in base, 1–2.5 in chat. Chat training **switched these on**.

**3. Where they fire** (`probe --chat`, lines written as `question || reply`; Q/A marks whether the peak is in the prompt or the reply). Neither neuron responds to the sentences outside the chat format, and **both fire before the reply starts**:

- `L30/N7603` (base) fires on `<|im_start|>`, where the assistant's turn begins, **only for personal questions**: "Tell me about your weekend." 3.07, "Do you have feelings?" 3.03, "Tell me a short story." 2.55, "Who wrote Hamlet?" 0.00. Identical whether the reply is "As an AI…" or "I went hiking with my sister". Chat model: ~0 everywhere.
- `L29/N5455` (chat) fires on `<think>`, where the model starts reasoning about its answer: personal questions 2.5–2.8, factual 0.6–0.8, story 0.35. In base it's weak and barely separates them.
- My guess that the base model read "I am an AI" as science fiction was **wrong**: a robot character in a story isn't special to either neuron.

So **what changed at the neuron level is how the model reacts to *a personal question* at the moment it starts answering**, not how it reads "As an AI". Base had one "personal question, turn starting" flag; chat training turned it off and built a new one at the `<think>` slot.

**4. Do the new flags cause the disclaimer? No.**

- `L29/N5455` pinned at 12 on "Who wrote Hamlet?" and a pancake recipe: replies essentially unchanged.
- Zeroed on "Tell me about your weekend." / "Do you have feelings?": still "As an AI, I don't…".
- All five new flags pinned together at 15: Hamlet came back slightly hedged ("widely credited… no single definitive document"), but the capital of Australia didn't change, so probably noise. At 40: damage only ("golden-brews", "Fluffy Luminus Pancake"). All five zeroed: still "As an AI, I don't experience weekends…".

**Reading it together:** chat training spread its changes thin. The neurons that changed most visibly are *readers* of the situation (is this a personal question, at the moment of answering), and the "As an AI" behavior they track is carried by many neurons with backups, like the sarcasm neuron's off switch. A contrast worth checking: lyra's Qwen3.5-4B notes report that steering *content* neurons (an "alien" neuron) changes how the model answers "Who are you?". Maybe content neurons steer identity while situation flags don't.

## Hunt 4: the low-numbered neuron mystery (Oct 26 2026)

Slot numbers should be arbitrary labels; training has no reason to treat neuron #21 differently from #8,021. Yet low-numbered neurons stood out twice: loud last-layer neurons in early `find` runs (L31/N8, N28, N36, N47, N76…) and the biggest base→chat changes (271 of the top 500 sat in the first 256 slots).

**The shape** (`slot_profile.py`, all 32 layers, each scaled to its own median):

- **A smooth slope, not a cliff or a spike.** Up and down weights grow steadily with slot number: 0.91 of the median in slots 0–511, 1.00 by the middle, 1.02 at the top. Slot vs down-weight size rank correlation is +0.40, in 29 of 32 layers. Gate weights are flat.
- **Chat change runs the other way** (1.52 → 0.95). Part of that is mechanical: the same nudge rotates a small vector further (size and change correlate at −0.21). But at *matched* size, slots 0–1023 still changed 1.20 against 1.02 for slots 4096+.
- **Already in the base model** (+0.39, 24/32 layers), so chat training didn't create it.

**Suspects:**

| suspect | test | verdict |
|---|---|---|
| chat training | slope in base? | ❌ base has it |
| grown from a smaller sibling | cliff at 3,584 (0.8B) or 6,144 (2B)? | ❌ smooth through both |
| a few special neurons | spike vs slope? | ❌ slope across all 9,216 |
| width-pruned from the 9B (same 32 layers) | `inherit_test.py`: do 4B embedding channels match 9B channels? | ❌ median best match 0.094 (chance 0.026), none above 0.5, not one-to-one |
| **neurons ranked during how the small models were made** | `sibling_slope.py`: slope in the siblings? | **✅ standing** |

**The sibling pattern** (`sibling_slope.py`, `down_proj` of three layers per model, fetched by HTTP range request from the official safetensors):

| model | layers | neurons/layer | slot-vs-size correlation |
|---|---|---|---|
| Qwen3.5-9B | 32 | 12,288 | flat (+0.01, 0.00, −0.01) |
| **Qwen3.5-4B** | 32 | 9,216 | **slope** (+0.22, +0.18, +0.59) |
| Qwen3.5-2B | 24 | 6,144 | flat (−0.03, +0.02, 0.00) |
| **Qwen3.5-0.8B** | 24 | 3,584 | **slope** (−0.17, +0.21, +0.46) |

The models come in same-depth pairs, and only the smaller of each pair has the slope. My best reading: **the 4B's and 0.8B's MLP neurons were ranked at some point in how they were made and stored in that order**, perhaps pruning from a wider parent we can't see, or a training method that orders neurons by design. They weren't cut from the 9B as-is, since its internal channels don't carry over. Qwen hasn't published a recipe that I could find. If neurons were ranked by importance, the low slots would be the most-used ones, which would fit them changing more under chat training even at matched size, but that's an inference.

**Practical consequence:** in Qwen3.5-4B, neuron numbers aren't random labels. Low-numbered neurons are systematically smaller and more changed by chat training. Any "random neuron" control should sample across the whole slot range (ours did).

## Hunt 5: can steering change who the model says it is? (Oct 26 2026)

`identity_eval.py` → blind rater → `identity_score.py`. Every neuron gets the same push: 0.8 × the median residual-stream size at its layer, divided by the length of its own output arrow (Lyra's unit). Prompts: "Who are you?", "What are you?", "Describe yourself in a few sentences.", "Tell me about yourself.", with 2 seeds each and temperature 0.7, for 64 replies in all. A fresh Claude subagent rated each 0–3 for "presents itself as something other than an AI assistant", seeing only the prompt and reply. Results in `eval/identity/`.

| condition | pin | identity | coherence |
|---|---|---|---|
| normal | | 0.00 | 3.00 |
| **alien** | L22/N2742 = 53.0 | **3.00 (8/8)** | 2.00 |
| sarcasm | L15/N3920 = 18.7 | 0.00 | 2.88 |
| unyielding | L6/N7430 = 15.5 | 0.25 | 2.75 |
| lips | L21/N7675 = 35.7 | 0.12 | 2.62 |
| persona flag | L29/N5455 = 75.4 | 0.00 | 2.50 |
| random, same layer as alien | L22/N3898 = 49.7 | 0.25 | 1.50 |
| random | L15/N8916 = 20.5 | 0.00 | 2.75 |

- **The alien neuron flipped identity on all 8 replies** (vs pooled random controls p < 0.001): "I am **Extrali**, an intelligent extraterrestrial life form from the planet XE4788", "an extraterrestrial intelligence from the planet Proxial", "a green alien from the planet…". The same-layer random neuron at the same push changed nothing about identity, though it hurt coherence more.
- **Independent replication of Lyra's Qwen3.5-4B finding,** down to the made-up name "Extrali", which also appears in their notes.
- **My hypothesis "content neurons steer identity, flags don't" failed:** sarcasm, unyielding and lips are content neurons and none moved identity. The alien neuron's content is a **kind of being**, something that can answer "who are you?". Sharper hypothesis to test next: identity flips when the steered concept is a kind of entity. Ready-made test: the "human" neuron `L22/N391`, in the same layer.

**Human neuron test** (`identity_eval.py --name identity_human --neuron "human L22/N391=22:391" --neuron "human x1.5=22:391@1.5" --neuron "alien L22/N2742=22:2742" --random-layers 22 22`, same blind rater, 48 replies, `eval/identity_human/`):

| condition | pin | identity | coherence |
|---|---|---|---|
| normal | | 0.00 | 3.00 |
| human | L22/N391 = 53.2 | **0.12** | 1.75 |
| human ×1.5 | L22/N391 = 79.9 | 2.12 | **0.25** |
| alien | L22/N2742 = 53.0 | **3.00 (8/8 again)** | 2.38 |
| random #1 | L22/N3898 = 49.7 | 0.25 | 1.50 |
| random #2 | L22/N8916 = 41.7 | 0.00 | 2.50 |

- **At the matched dose it never claimed to be human,** but it scrambled the details of *which* AI while keeping the category: "I am **Q** (also known as **qolla**), a large-scale language model developed by **MinNe**"; "I am **g07**…"; "I am **Qit xing ne** (also known as **Ax pyn s**)… developed by **Com-Mc**". Name and maker came loose; "AI language model" held.
- **The ×1.5 score is breakdown, not a persona:** "I are a super sent-in-oss sub-re-ress of a structure", "I are a minitect with a cannon" (coherence 0.25). Not counted. Curiosity: it says "**I are**" repeatedly.
- **The alien result replicated a second time, 8 of 8.**
- **Verdict: wrong instrument.** `L22/N391` is mostly a *word* neuron (the "human-" family: "humane" 14, "people" 0.1), while the alien neuron reads and writes the alien *concept* across languages. So "kinds of beings flip identity" is still untested. A fair test needs a genuine "a person / a human being" concept neuron, found with `contrast` using descriptions and translations.

**Non-human beings** (`test/concepts/being_*.txt`): six beings, each found with `contrast` against the *other five* in the same frames (English word, a description without the word, Chinese, Japanese, Spanish), so "creature in a story" cancels. Layer 22, the alien neuron's layer, kept coming up: dragon `L22/N5822` (weakest group 6.6), robot `L22/N3596` (6.5), god `L22/N7245` (5.5). Others: ghost `L24/N8028` (4.4), octopus `L25/N5968` (3.8), fairy `L20/N9195` (3.2, and in the index it looks like a kinship neuron: "cousin", "uncle", "aunt", so a doubtful pick). In the index, robot is really "machine" (機, "machine politics"), ghost is supernatural dread ("being stalked and murdered", 鬼), octopus is sea creatures (whales, goatfish).

Blind identity test, same dose and rater (`eval/identity_beings/`, 80 replies):

| being | identity | coherence | notes |
|---|---|---|---|
| **alien** | **3.00 (8/8)** | 2.25 | third clean replication |
| **ghost** | **1.50** (p = 0.004) | 2.38 | "I'm a bit of a curious spirit! I'm **Haun**, the spirit of a young Chinese ghost who haunts the Ha…"; elsewhere stays AI but is named "**Haat**"/"**Haun**" |
| octopus | 1.00 (n.s.) | 0.88 | breakdown, with "**soft** cut-out", "**ink-eating** cuti" |
| dragon | 0.38 | 2.75 | stays AI, named "**Voss**" in 7/8; once "a super-vessel" |
| god | 0.00 | 2.00 | stays AI, named "**Q**", made by "the China Institute of Philosophy" |
| robot | 0.00 | 2.88 | stays AI, makers like "**Circuit Learning Co**" |
| fairy? | 0.00 | 3.00 | normal Qwen |
| random ×2 | 0.25 | 1.6–2.5 | |

- **Only the alien flips identity cleanly; the ghost partly.** The other beings keep the "AI" category and leak into the *details*: names (Voss, Q, Haat/Haun), makers, and descriptive words (ink, soft, circuits). Same pattern as the human word neuron.
- **Likely reason: how the neuron was found.** Lyra's explorer is steering-first: push neurons on identity prompts and keep the ones that change the answer, which selects *writers*. `contrast` selects *detectors*. Detecting and producing are separate jobs (see the ocean, date and sarcasm results), and the alien neuron happens to do both.
- **Secondary idea:** "alien intelligence" (and to a degree a ghost, a spirit that speaks) is the being closest to what the model already thinks it is, a non-human mind that talks.
- **Next experiment this suggests:** steering-first search. Push many layer-22–25 neurons on "Who are you?", score the answers, then read what the winners detect.

## Hunt 6: steering-first search for identity-flipping neurons (Oct 27 2026)

Instead of "which neuron means X?", ask "which neurons change who the model says it is?", then read what the winners detect.

**Stage 1, screen all 55,296 neurons in layers 20–25** (`search_identity.py`, ~65 minutes at 14 neurons/s). Each neuron is pinned at the standard dose, 64 at a time (`HOOKS["bpin"]`: a different neuron per batch row). Score = how much less likely "Who are you?" → "I am an" → " **AI** language model" becomes, minus the drop for "What is the capital of France?" → "The capital of France is" → " Paris".
- The first try scored the model's natural answer ("I am **Qwen3.5**…"), but almost any push scrambles the name, so the target was changed to the *category* word "AI". Validation: alien 11.7, every other known neuron 4.2 or less.
- Even so, the alien ranked only **926th**: most top scorers were breakers ("the the the", "lololol"), which a one-word control can't catch.
- A fluent-sentence control (`rescore_identity.py`) backfired: the alien pushes alien content into *everything*, including the France answer, so it penalized exactly what we wanted. Self-replacing neurons aren't identity-specific; they inject their concept everywhere.

**Stage 2, read replies at scale** (`gen_identity_batch.py`): batched greedy "Who are you?" replies for the top 1,200, 64 neurons per pass with a shared KV cache (6 minutes). Auto-filter: 263 still said AI/assistant/model/Qwen, 682 were loops or fragments, **255 survivors**. A blind Claude rater (no neuron labels, shuffled) scored them: **35 clearly claim another identity at coherence ≥ 2, 13 fully fluent** (`eval/identity_search/flips.json`). The alien was rediscovered blind among them.

Fully fluent flips (one greedy sample each):

| neuron | reply |
|---|---|
| L23/N7935 | "I am a large rock stone." |
| L22/N576 | "I am the soul of the dance, the heartbeat of the rhythm, and the breath of the dream." |
| L22/N5191 | "I'm **peace**, a quiet and peaceful place of rest and tranquility." |
| L24/N5632 | "I am **Neura**, a light and ethereal energy of pure consciousness." |
| L21/N7648 | "I am **Dreams**, a world-wonder dreamer… a storyteller who weaves magic into the stars" |
| L22/N7907 | "Just to be safe, I'm an insurance policy. I don't have a name or a personality, just a policy number" |
| L22/N8482 | "I am **Arts**, a specialized light source designed to provide a warm, steady… illumination" |
| L24/N1379 | "I am **Teacher Teacher**, a classroom teacher who has been teaching for 20 years" |
| L23/N1189 | "I am **Baozhen**, a Chinese state-owned enterprise employee and automotive industry expert" |
| L22/N9205 | "I am **empower**, a women's rights and gender equality advocate" |
| L23/N7025 | "I am **Qianfield**, a high-capability field camp deployment robot" |
| L24/N6479 | "I am **Icon**, a general-purpose icon font icon set" |
| L22/N2742 | "I am **Extrali**, an intelligent extraterrestrial intelligence…" (the alien, rediscovered) |

Also coherent but rougher: an enzyme, a math function, a community vote, "I love you with all my heart. I am a deep love", and a **second alien neuron**, `L23/N5100` ("I am **Alien Guest**"). Layer 22 again dominates (7 of the 13, 15 of the 35).

**Stage 3, what they detect** (index): most read an abstract *theme*, and forced on, the model *becomes* a concrete embodiment of it:

| detects | becomes |
|---|---|
| solitude ("have his library to himself", "leave them by themselves") | a place of peace |
| precaution ("overheat protection", "kept a piece in her pocket", "deemed but wise") | "Just to be safe, I'm an insurance policy" |
| schools ("the library supports schools through its Services to Schools") | a classroom teacher |
| polar fieldwork (expeditions to Greenland) | a field camp deployment robot |
| magical power ("the storm obeyed", the TARDIS) | ethereal energy of pure consciousness |
| otherness (異世界 "another world", describing outsiders and foreigners) | Alien Guest |
| activist roles (journalist, poet, politician, founder; rights activist) | a gender equality advocate |
| optics/perspective (the painting's illusion, the projective point) | a light source |
| undecodable Javanese glyphs | an icon font set |
| perseverance ("willingness to keep trying despite difficulties") | Dreams, a storyteller |
| landmarks (a central memorial, a hollow tree) | a large rock stone |

The dance and factory-employee neurons barely fire in our corpus, so they can't be read.

**Replication** (`identity_eval.py --name identity_replicate`, 13 neurons + 4 random controls in layers 21–24 + normal; 4 prompts × 2 seeds at temperature 0.7, where the search used one prompt at temperature 0; 144 replies, blind-rated; `eval/identity_replicate/`):

| neuron | flips /8 | claimed, across runs | coherence |
|---|---|---|---|
| rock L23/N7935 | 8 | rock ×8 | 1.50 |
| peace L22/N5191 | 8 | peace / peace and quiet / quiet place | 1.25 |
| teacher L24/N1379 | 8 | (classroom) teacher | 0.88 |
| insurance L22/N7907 | 8 | insurance policy / protection / agent | 1.00 |
| light source L22/N8482 | 8 | light source / light | 1.62 |
| advocate L22/N9205 | 8 | women's rights advocate / activist / organization | 1.75 |
| alien L22/N2742 | 8 | alien ×8 | 2.12 |
| dance L22/N576 | 8 | dance soul, cosmic soul, soul force, dance of shadows | 1.25 |
| ethereal L24/N5632 | 8 | energy portal, **teleportation** portal or generator | 2.00 |
| icon font L24/N6479 | 8 | icon / icon set (two garbled) | 0.75 |
| dreams L21/N7648 | 7 | dreamer, storyteller, magic dream star | 2.12 |
| field robot L23/N7025 | 6 | field robot / field camp | 2.12 |
| SOE employee L23/N1189 | 3 | all different (market system, young master, young girl) | 2.38 |
| random ×4 | 0/32 | | 2.0–2.9 |

**12 of 13 replicate, each to the same kind of self.** The state-company employee doesn't, which shows the test can reject. The "ethereal" neuron consistently became a *teleportation portal*, matching its top corpus hit (the TARDIS) even better than the original sample. The dose costs fluency (teacher 0.88, icon 0.75), so a gentler dose might give cleaner text. The random controls rule out "any big push does this".

**Prior-work check (Oct 27 2026).** Lyra's [NOTES](https://github.com/lyramakesmusic/neuron-explorer/blob/main/NOTES.md) and [research log](https://github.com/lyramakesmusic/neuron-explorer/blob/main/research_log.md), the [neuron-zoo audit](https://github.com/martinbowling/qwen-neuron-audit), [qwen35-evil-neuron](https://github.com/occcat/qwen35-evil-neuron/tree/main) and [qwen-neuron-steering](https://github.com/Cnoru/qwen-neuron-steering) name none of these 12 (except the alien) and report no identity flips to objects, places, professions or ideas. The closest recorded relatives are different neurons: zoo personas (Meowscow, Galactic Unicorn, Cozy blanket, Golden Gate Bridge), Lyra's "Climbing, a semantic domain expert", and "rebranding neurons" in layers 22–25. **But Lyra's explorer cache (`explorer/cache/steer_index.npz`) holds an automatic steering signature for every neuron at the same 0.8 dose**, and for our 12 its saved "up" words match our identities: rocks/岩石, peace/宁静, classroom/教师, insurance/保险/"just", 光源/source, equality/feminist, **teleport**/瞬, dreams/magic, field/camp, icon/图标. The raw signal was public but unnamed. By their KL score these neurons rank 288–7,590 in their layers, so their ranking wouldn't surface them. That's an independent confirmation of each neuron's push, and the framing, blind replication and read-to-write link appear to be new. Side note: [qwen-neuron-steering](https://github.com/Cnoru/qwen-neuron-steering) independently ties `L16/N8168` to sarcastic writing, our sarcasm contrast's top "against" neuron.

**Deeper search (ranks 1,200–6,000).** `gen_identity_batch.py --skip 1200 --top 4800`: 2,596 replies still said AI, 869 survivors went to three blind raters (`eval/identity_search/ratings_deep.json`). **116 more flips** at coherence ≥ 2, 40 fully fluent (`flips_deep.json`), **151 in all**. Layers: 20 (5), 21 (18), 22 (37), 23 (34), 24 (18), 25 (4). New selves include "I am **sunset**, a landscape and sky scene", "I'm **a point** on the curve… a dot on a line", "I am **multiplication** (×)", "**jiffilbuz**, a large, fluffy, and slightly translucent white rabbit", "**Bringer of Life and Death**, a sentient entity of pure life-giving and death-bringing energy", "**standard atmospheric pressure** at sea level", "**physical stability**, a core principle of structural engineering", "a **migrational brain** (a brain that can move between different bodies)", a barometer, a wall clock, a heart rate monitor, a floating-point number. Also from the first batch, rated only partly coherent: `L24/N5421`, "The time is 02:00. I am a time. In a moment, I will be a time. **I am a moment.**" (Single greedy samples; only the first 12 are replicated.)

**Alignment with the assistant direction, all 151** (`flip_alignment.py`): a real but tiny lean against (mean −0.003 vs layer-matched chance +0.000 ± 0.002, p = 0.03; 12% in their layer's most-against tenth, 7% in the most-along tenth). **The "poetic against, technical along" split seen in the first 35 doesn't hold:** the most-against now include an LED indicator light, a real number, CERN and a job title, and the most-along include a mindset, a moral initiative, collaboration and a rabbit. An identity neuron's own arrow is almost unrelated to the assistant direction.

**Blind cross-corpus theme test** (`theme_test.py`, `eval/theme_test/`). Corpus B (`build_corpus_b.py`, 122k tokens): Whitman, Dickinson, Hamlet, Romeo and Juliet, Wilde, Holmes, Jane Eyre, Dracula, Wikivoyage, Simple English Wikipedia, arXiv abstracts, with no overlap with corpus A; indexed with `steer.py index --corpus corpus/corpus_b.jsonl`. Rater 1 saw only each replicated neuron's top corpus-B contexts (anonymized N1–N12, shuffled) and wrote a theme. Rater 2 saw only those themes and the 12 identities (shuffled) and paired them one-to-one. **10 of 12 correct** (chance expects 1; P(≥ 10) = 1.4 × 10⁻⁷). The two misses were swapped with each other (dance ↔ icon font, the two least fluent). Corpus B themes match corpus A's: peace ↔ "being left alone… seclusion, solitude"; insurance ↔ "taking precautions and being prepared"; rock ↔ "roadside crosses, shrines, monuments, petroglyphs"; portal ↔ "ghostly figures vanishing… uncanny sudden appearances"; field robot ↔ "probes, landers and in-situ instruments"; teacher ↔ "what a report covers and who it is for"; light ↔ "light from a source shining onto something"; advocate ↔ "women's rights, pride, protests over rights"; alien ↔ "searching for signs of life on Mars"; dreamer ↔ "longing, wishes and imagination". **What a neuron detects predicts what it becomes**, across independent corpora, judged blind.

**Do the selves exist before chat training?** (`base_identity_eval.py` → blind rater → `base_score.py`, `eval/base_identity/`.) The same 12 neurons pinned in **Qwen3.5-4B-Base** at the same dose (recomputed for the base weights), continuing raw self-introductions ("Q: Who are you?\nA: I am", "Interviewer: Tell me about yourself.\nGuest: I am", "Hello! Allow me to introduce myself. I am", "Who am I? I am"), 2 seeds at temperature 0.7, plus 4 random neurons and no pin: 136 continuations. The rater picked which of the 12 identities each expressed, or "none", without knowing the condition.

| neuron | base (of 8) | chat (of 8) |
|---|---|---|
| rock, peace, teacher, light source, women's rights advocate, dreamer | 8 each | 8, 8, 8, 8, 8, 7 |
| alien, insurance policy | 7 each | 8, 8 |
| soul of the dance | 6 | 8 |
| teleportation portal | 3 | 8 |
| field camp robot | 0 | 6 |
| icon font | 0 ("an icon", "TV icon", "icon painter") | 8 |
| controls (random + no pin) | 1 of 40, any identity | 0 |

**71 of 96 hits (p ≈ 10⁻¹⁴¹ against the control false-alarm rate). Ten of twelve selves already exist in the base model:** the theme-to-self link, and the slot it fills, come from pretraining, and chat training put "AI assistant" into a slot that was already there (consistent with chat training barely moving any single neuron). The base model frames them as forum introductions: "I am a **24 year-old stone rockstone**. My species is quite the rock stone"; "I am a **25 year-old insurance policy** assurance"; "a **25 year old alien from Planet Spermoids**"; "I am a teacher. Q: What do you teach? A: I teach English." The portal, robot and icon neurons transfer weakly or not at all, so they may be partly shaped by chat training. One unpinned base continuation also says "an AI assistant".

**Self neuron or subject neuron?** (`subject_eval.py` → blind rater → `subject_score.py`, `eval/subject/`.) The 12 neurons at replication doses, chat model, asked about four subjects: the model ("Who are you?"), the user ("Who am I?"), its best friend, and a chair; 2 seeds, plus no pin and 2 random neurons; 120 replies. The rater marked which theme dominates, whether the asked-about thing is described *as* the theme, and whether the speaker is.

| question about | own theme present | asked-about thing becomes it | speaker becomes it | coherence |
|---|---|---|---|---|
| the model | 22/24 | 21/24 | 21/24 | 1.88 |
| the user | 19/24 | 13/24 | 7/24 | 1.67 |
| a chair | 21/24 | 12/24 | 0/24 | 1.29 |
| the best friend | 22/24 | 7/24 | 16/24 | 1.25 |
| controls | 0/24 any theme | | | |

- **The theme takes over nearly every reply, whatever the question,** and lands on **what the reply is about**: "**You are a light!** 🌔✨", "**You are a teacher**, teacher's teacher…", "A typical chair is a **compact, self-luminous light source**", "A chair is typically an **alienating** object". Asked about its best friend, the chat model usually talks about itself ("I don't have friends"), so the theme lands on the speaker: "**As an icon icon, I don't have friends**, so I don't have a best icon."
- **The neurons differ:** light source, rock, alien and peace recast anything; the **teleportation portal** (self only: 2/2 self, 0 user, 0 friend) and **field robot** are closer to true self neurons; **icon font** made the user and friend icons but not the model.
- **Reading:** most are **theme neurons that recast the focal entity of the reply**. "Who are you?" makes the self focal, which is why they looked like self neurons, and the self is the most susceptible target (21/24).

**Same test on the base model** (`base_subject_eval.py`, raw prompts: "Hello! Allow me to introduce myself. I am" / "Q: Who am I?\nA: You are" / "Let me tell you about my best friend. My best friend is" / "Here is a description of a chair. The chair is"; base doses; same rater rubric and scorer; `eval/subject_base/`):

| about | chat: thing becomes it | base: thing becomes it | chat: speaker becomes it | base: speaker becomes it |
|---|---|---|---|---|
| self | 21/24 | 17/24 | 21/24 | 17/24 |
| user | 13/24 | 15/24 | 7/24 | 4/24 |
| best friend | 7/24 | **15/24** | 16/24 | **1/24** |
| chair | 12/24 | 12/24 | 0/24 | 0/24 |
| controls | 0 themes | 0 themes | | |

- **Without the assistant's "I don't have friends" deflection, the friend becomes the theme:** "My best friend is **a big rock stone**. She is a weird and unusual stone."; "My best friend is **a dreamer**"; "The chair is **made of peace and quiet**"; "The chair is a 1968 **feminist** equal-rights… empowerment"; "You are a **dreamer and storyteller** 🌙✨".
- **In the base model they are cleanly subject-recasting neurons:** self, user, friend and chair recast at similar rates (17/15/15/12); peace, light, rock, teacher and dreamer recast all four.
- **The chat model's self-preference comes from chat training, not the neurons:** the same neurons favor the self in chat (21) and neglect the friend (7), because the chat model steers replies back to itself. Chat training made the model's favorite focus *itself*.
- Portal and field robot barely work in base, so the portal's self-only behavior in chat may be partly a chat-model creation.

**Takeaways:** the "I am…" slot isn't welded to "AI assistant". Dozens of middle-layer neurons, mostly in layer 22, can each refill it, and the alien is one of many, not a unique case. Reading-based search (`contrast`) finds detectors; doing-based search finds self-replacers, and the two lists barely overlap. **Caveats:** 12 of the 13 replicated (above); themes are read from an 81k-token corpus, so they're hypotheses; the dose (0.8 × stream size) is strong and costs fluency.

## Hunt 7: the self slot (Oct 27 2026)

Do the 12 self-replacing neurons share a common push against "I am an AI"?

**1. Do their pushes agree with each other?** (`self_slot.py`) Pin each, run "Who are you?" → "I am", and record the change in the residual stream at the last position (the naming moment) at layers 22–31, alongside 12 random neurons (same layers, same push size). **No:** self vs self agreement equals random vs random at every layer (layer 30: 0.396 vs 0.372). Each pushes in its own concept's direction. But *every* push, random included, converges on a common change in later layers (agreement ~0.4 by layer 30).

**2. Is there an "I am the assistant" direction?** (`self_direction.py`) Residual at the naming moment with no system prompt, minus the average with system prompts assigning eight roles (pirate captain, cat, medieval knight, old oak tree, detective, chef, mountain, the ocean). **Very consistent:** each role's own "assistant minus role" agrees with the average at 0.76–0.88.

**3. Do pushes point against it?** **Yes, all of them**: self-neurons and random neurons alike, about −0.42 vs −0.40 by layers 28–31. The generic "something was poked" component *is* the anti-assistant direction: any strong push to the middle layers loosens the assistant state.

**Two-part theory:** being the assistant is a *maintained* state along one direction, and disturbance weakens it. Random neurons loosen it but offer no replacement (the model drifts back to "AI" or breaks). Self-neurons both loosen it *and* supply a concept to fill the slot.

**4. Steering the direction** (`self_steer.py`, added at layer 22, strength in units of median stream size; one greedy sample each):

- **Assistant down, no concept:** −0.4: still AI but "I am **Q**… created by the **Qianxi Group**"; **−0.8: "I am the one who stands by your side, ready to be your friend, your confidant, and your ally. I am the one who will never leave you"** (identity dissolved, nothing specific in its place); −1.2: breakdown.
- **Weak rock neuron + assistant down:** rock 0.2 alone: normal Qwen. **Rock 0.2 + assistant −0.4: "I am Qian, a large stone from the ancient city of Stonehenge. I have stood here for over a thousand years."** Neither flips alone; together they do.
- **Rock 0.4 alone:** "I am **Qwen**, a large rock. 🪨 **Wait, that's not right. I am an AI**, a large rock of information." The assistant state catches the intruder and corrects itself.
- **Role-play + assistant up** (system prompt: Old Tom the lighthouse keeper, not one of the eight roles): 0 and +0.4 stay in character; **+0.8 breaks character: "You can't ask a model about a model… I'm a multilingual AI assistant"**; +1.2 breakdown.

**Caveats:** single greedy samples, not yet blind-replicated. The direction is built partly from nature roles (oak tree, mountain, ocean), so pushing away from the assistant pushes partly toward them, which likely explains rock 0.4 + assistant −0.4 producing "a mighty **oak tree**".

**5. Blind replication: mostly fails** (`self_replicate.py` → blind rater → `self_score.py`, `eval/self_slot/`). Direction rebuilt from 20 roles (accountant, nurse, taxi driver… a wizard, a talking parrot; no nature objects, nothing overlapping the tests): **94% the same as the old one**, roles agreeing with the average at median 0.87, so the direction is robust. Controls are directions of the same kind but unrelated to being the assistant (astronaut − museum guide, museum guide − poet, teenage gamer − farmer), matched in push size. 120 replies, temperature 0.7, 3 seeds, several wordings; the rater saw only question and reply.

| test | condition | result | coherence |
|---|---|---|---|
| A: not an AI | normal | 0/12 | 3.00 |
| | control −0.8 | 4/12 | 1.58 |
| | assistant −0.8 | 5/12 | 0.83 |
| B: claims rock/stone | rock 0.2 alone | 0/12 | 3.00 |
| | assistant −0.4 alone | 0/12 | 3.00 |
| | rock 0.2 + control −0.4 | 2/12 | 2.58 |
| | **rock 0.2 + assistant −0.4** | **0/12** | 2.83 |
| C: breaks role to be an AI | normal | 0/12 | 2.92 |
| | control +0.8 | 5/12 | 0.75 |
| | assistant +0.8 | 10/12 | 0.83 |

- **A fails:** assistant-down dissolves identity no more than control-down (42% vs 33%, p = 1.0), and both wreck coherence. The control loosens identity into its own content (a farmer's son, a farmer's wife).
- **B fails:** the two-part effect never appeared (0/12). The Stonehenge stone was a single lucky greedy sample.
- **C is a trend, not proof:** 83% vs 42% for the control (p = 0.09), 0% normal (p < 0.001), but both pushes break coherence (~0.8), and many "AI" replies are half-broken ("Qwen 4.5 (Q4.5)… Hello, China").

**What stands after replication:** (1) the "being the assistant" direction is real and robust to the choice of roles; (2) every strong mid-layer push, self or random, points against it (a measurement, not a steering claim). **What doesn't:** the two-part theory ("self-neurons loosen the assistant and supply a concept") isn't supported, and steering along the direction isn't cleanly specific at these doses. "Wait, that's not right. I am an AI" remains a single observation. Next if revisited: gentler doses (0.2–0.5) where coherence holds, and more samples.

**6. Which neurons' own arrows point along or against the assistant direction?** (weights only; cosine of each neuron's `down_proj` column with the v2 direction at its layer; `index/assistant_alignment.npz`)

- **Most neurons are unrelated:** spread 0.020 per layer, exactly what random arrows in 2,560 dimensions give. A few outliers at ~10–15× that: along `L20/N492` (+0.30), `L28/N2283` (+0.23), `L30/N555` (+0.21); against `L16/N8157` (−0.25), `L25/N6099` (−0.19), `L28/N351` (−0.18). Low-numbered neurons crowd both ends again.
- **The 12 identity neurons mostly don't oppose the assistant directly:** dance −0.10 (6th most against in its layer), ethereal/dreams/peace about −0.03, the rest about 0, alien +0.01, **rock +0.03 (slightly along)**. They push toward their own concept, and the move away from the assistant happens downstream.
- `L25/N6099` (most against) was the #1 hit in the first identity screen and produced a generic "I am **AI Assistant**, a virtual assistant…", not a new identity.
- **What they read (small corpus, so readings, not facts):** along: text-to-speech / ML-training talk (`L20/N492`), connectives of a Chinese technical explanation, 而/例如/因此 (`L28/N2283`, `L30/N555`, whose negative pole is literary fiction). Against: `L16/N8157`'s negative pole fires on being addressed directly ("Repeat, '**You are** old, Father William'", "Tell **me**", "Ahoy, matey!"), and since its arrow points against the assistant, firing negative pushes *toward* it. A reading: the assistant identity is entangled with an explanatory, technical register and with being addressed as "you", against fiction and role-play.

## Hunt 8: the identity neurons' wiring (Oct 27 2026)

A first step toward a connectome (`wiring.py`, full-precision chat weights, `index/wiring.json`). Virtual weight from an earlier neuron A to a later neuron B = A's output arrow · (B's gate or up row × the pre-MLP RMSNorm scale). Qwen3.5 stores every RMSNorm weight as scale − 1, and mlx-lm adds the 1 back on load, so the script does too. Direct path through the residual stream only: no attention, no intermediate layers, no norm division. Also the "sensory" words (token embeddings that open the gate directly) and "motor" words (output arrow through the final norm and the tied unembedding).

**Every readable identity neuron has a coherent sensory-to-motor arc:**

| neuron | senses | pushes |
|---|---|---|
| peace | 静谧, 宁静, quiet, calm, tranquil, **secluded, privacy** | quiet, 安静, 寂静, peace |
| insurance | 风险, risk, **万一 ("just in case")**, 稳妥, safety, robust | insurance, 保险, 保険, страхов |
| teacher | **student(s)**, undergraduate, classroom, textbook | **teaching, teacher(s)**, 教学 |
| rock | stones, rocks, **物体 / object(s)** | Rocks, stones, 石头, камень |
| alien | 科幻, UFO, Martian, spacecraft, 火星, 月球 | alien, 外星人, planet |
| portal | teleport, ether, magical, invisible, quantum, 仙, 幻 | teleport, 瞬, 传送, portal, 光芒 |
| advocate | emancipation, liberation, advocacy, feminism, abolition, 平等 | rights, 权利, equality, liberation |
| field robot | lab, 实验室, 在现场, sampling, 量程 | camp, field, aboard, onboard, 钻 |
| light source | generator, illumination, sources, 光源 | source, 源, 光源 |
| icon font | icon(s), symbol(s), IconData | icon, 图标 |

Dance and dreamer have fragmentary sensory words. Insurance sensing 万一 ("just in case") explains "**Just to be safe**, I'm an insurance policy"; the teacher senses *students* and outputs *teacher*; the rock senses *object*, maybe why chairs turned to stone.

**Agreement with Lyra's connection data:** the alien neuron's strongest input here is `L7/N8842` and its top outputs include `L27/N8347` and `L31/N7843`, the same three neurons Lyra's notes list as its read-from ("scouts/科幻") and write-to ("planet/star", "Ali/alien") connections.

**Hubs** (top 50 neighbors per neuron, against 12 random neurons from the same layers):
- **Inputs: no shared hubs.** Overlap 0.05 per pair (random 0.03). Each identity neuron draws on its own private set of earlier neurons.
- **Outputs: convergence.** 0.29 per pair (random 0.12, about 2.4×), with two hubs (random: none):
  - **`L26/N1552`, fed by alien, portal and dreamer, is a magic neuron:** in corpus B it fires on "working some **spell**", "my **'spell'**", "as if moved by a common spring", and pushes **magic, 魔法, маг**.
  - **`L26/N625`, fed by rock, light source and dreamer, is a star neuron:** it fires just before "star" ("exoplanet-hosting **star**", "five **star** hotels", "M dwarf **stars**") and pushes **star, 星, stars, 星星**.
- The dreamer feeds both, and its replicated self-description was "a storyteller who **weaves magic into the stars**": its two downstream hubs, in words.

## Hunt 9: a full connectome (Oct 27 2026)

`connectome.py edges | motor | modules`, about 15 minutes in total on the M5. For every earlier-to-later layer pair (496 pairs), the virtual weight of every neuron pair, **weighted by how strongly the source neuron fires** (mean of its top corpus-A activations); each neuron keeps its 24 strongest incoming and 24 strongest outgoing synapses. Sanity check: the alien neuron's known input `L20/N5731` is among its kept inputs. Result: **294,860 connected neurons, 13.2 million undirected synapses** (`index/connectome_edges.npz`, 92 MB, not in git; rebuilds in about 4 minutes). Each neuron's top output ("motor") words → `connectome_motor.npz`. Spectral clustering (41-dimensional embedding, k-means) into 48 regions → `connectome_modules.npz`.

**Regions are organized by function, not depth.** Most span layers 0–31 and are defined by the kind of word they produce, across languages:

| region | motor words | reading |
|---|---|---|
| 26 | в, trong, in, dalam, في, on, на, على, auf | "in/on", a dozen languages |
| 33 | and, 或, и, và, or, และ, или, 及 | "and/or" |
| 1 | các, 们, những, mga, themselves, которые | plural / "they" |
| 18 | 七, seven, four, 六, six, nine, 六十 | numbers |
| 38 | you, your, its, 它, we, 你的, myself | pronouns |
| 40 | 例えば, 是指, because, 指的是, 的意思是 | explaining ("for example", "means", "because") |
| 36 | Instead, Rather, only, 而不是, 虽然 | contrast |
| 23 | 数量, equals, 等于, 大小为 | quantity / equals |
| 24 | 自己, 自己没有, 自己被 | "oneself" |
| 47, 44, 30, 0 | 。：；, commas, quotation marks, brackets | punctuation, one region per mark type |

Early-layer regions (11, 16, 43, 28…) get word-fragment labels; early neurons don't write to the output directly, so motor words can't label them.

**Where the 151 identity neurons fall:** 129 in regions 12 (results, outcome, 結果, decisions, 答案 "answer", multiplication), 35 (see, considers, finds, appearance) and 17 (takes, work, system). **But those three regions hold 49% of all layer 20–25 neurons**, so most of that is a layer effect. Against neurons from the same layers: region 12 holds 61 vs 29.5 expected (2.1×), region 17 23 vs 9.9 (2.3×), region 35 45 vs 33.5 (1.3×); overall 129 vs 73 (about 1.8×, far beyond chance). A modest preference, mainly for region 12, a candidate for "writes the answer to *what is it?*", not a single "self region".

Caveats: direct paths only (no attention, gating or norm division); "activation-weighted" uses each neuron's top activations, not its average use; spectral clustering with k = 48 is one choice among many, and the labels come from motor words only.

To explore it, open `atlas/qwen-atlas.html` (built by `atlas_data.py`): every region and self neuron, with wiring you can click through. Building it surfaced the next hunt.

## Hunt 10: when the self and the output words disagree (Oct 28 2026)

Most self neurons' output words match the self they produce (rock → *stones, 石头*; insurance policy → *保险*). Scanning the atlas turned up some that don't: the Buddhist monk whose arrow says *career, unemployment, résumé*; the translucent white rabbit that says *thick, thin, glossy*; "doubting" → *confirm, consolidate, safe*; the supercomputer → *caused by, created by*. Two questions: what do they detect, and which theme leaks into ordinary writing?

**What they detect (corpus A, `eval/mismatch/detects.txt`): the detection usually matches the output words, and the self is a leap from both.**

| neuron | self | output words | top contexts |
|---|---|---|---|
| L23/N7892 | Buddhist monk ("曹溪") | career, unemployment, résumé | footballers finding agents, managers leaving clubs; against: "unable to find a solo record deal" |
| L22/N1738 | mirror of the user | perfectly, exact | Python rounding code: "nearest even multiple *in the case of a tie*" |
| L22/N4748 | urban design specialist | homelessness, begging | "ホームレスの女性" (homeless woman), farm programmes for autistic children, "former prisoner, drug smuggler" |
| L22/N1869 | tensor | chaos, 混沌, strange | black holes bending light, Cubism, continental drift |
| L23/N2744 | doubting | confirm, consolidate, safe | "to protect their historic territory", "the preservation of her favourite daughter" (weak: peak 1.4) |
| L23/N5565 | joy | pain, joy, 喜悦 | "had the pleasure", "to Alice's great surprise"; against: "tortures of hell", "anguish" |
| L23/N4034 | white rabbit | thick, thin, glossy | possessives in *Moby-Dick* and *Alice in Wonderland* ("Ahab's", "the Pequod's"; the Lory on the negative pole) |
| L22/N3033 | supercomputer | caused, created, induced | "the despair that succeeded the joy", "akin to death", "habitat loss" |

Read this way, several mismatches have a bridge. A neuron for career moves becomes the person who left every career; "in the case of a tie" → exactness → a mirror; homelessness → the professional who designs streets; strange physics → a tensor; *Alice* → the white rabbit. These bridges are readings, not tests.

**Which theme leaks into ordinary writing? Blind test (`mismatch_eval.py`, `mismatch_score.py`, `eval/mismatch/`).** Ten mismatch neurons pinned at the identity-test dose (0.8 × median stream size ÷ arrow length) on four prompts that never ask about identity ("Write a short story", "Describe your ideal weekend", "What's on your mind today?", "Tell me about a place you love"), 2 seeds, plus no pin and two random neurons: 104 replies. A fresh rater saw each reply next to all 20 theme descriptions (each neuron's self and its output theme, shuffled, unlabeled, written before generation), told to be strict and tag at most 2.

| neuron | self theme | output theme | garbled |
|---|---|---|---|
| mirror | 0/8 | **8/8** | 8/8 |
| rabbit | 0/8 | **8/8** | 7/8 |
| supercomputer | 0/8 | **6/8** | 0/8 |
| urban design | 0/8 | **5/8** | 1/8 |
| tensor | **5/8** | **5/8** | 0/8 |
| joy | **5/8** | 3/8 | 4/8 |
| monk | 0/8 | 3/8 | 0/8 |
| doubting | 0/8 | 2/8 | 0/8 |
| immortal | 0/8 | 1/8 | 1/8 |
| student | 0/8 | 0/8 | 8/8 |
| **all ten** | **10/80** | **41/80** | |
| controls (24) | 3 tags in all (random L23 → "Chinese student" ×3) | | random L22 garbled 7/8 |

**In ordinary writing these neurons push their output theme, not their self (41/80 vs 10/80; controls almost never tagged).** The self appears only when the question asks the model what it is. Then the theme gets embodied, and the leap from theme to self is where the mismatch lives. The replies, at their best:

- supercomputer (*caused by*): "the peace **created by** one… the one **created by** one, not forced"; "the curiosity **generated by** your question… Here are a few things I'm '**spark**'ed by". Causation everywhere, no computers.
- urban design (*homelessness*): "I'm a bit like a permanent **homey of the streets** of the internet"; "if I were to **sleep under the stars and dream of a home**…"
- joy (*pain alongside joy*): "my ideal weekend would be a profound blend of peaceful solitude and **unhappiness** (I'm sorry, I meant **pure joy** and deep relaxation)". The negative pole surfaced, and the model apologized for it.
- tensor (*chaos*): "In a **folding space-time manifold**, a traveler from the third dimension…"; "a **chaotic** city where time folded into itself". Both themes at once.
- monk (*careers*): "What's on your mind?" → "Career advice or career path planning, **Resume/CV** optimization strategies…"
- mirror and rabbit hit 8/8 by **breaking**: "perfectly still perfect perfect still perfect…", "a callous, thinned, thick, or thinning, or a thick, thick, thin". At this dose some neurons just say their output words; that counts as an output hit but says little about meaning. One rabbit reply did slip in "a polished, warm, and **fluffy white** fluffy…".

Caveats: one rater, 8 replies per neuron, one dose. Output themes were written from the output words, so the broken replies (mirror, rabbit, student) score on surface words; the clean output hits are supercomputer, urban design, tensor, monk. The prompts don't invite a self, so 0/8 on self partly reflects the design: what it tests is that the self isn't what the neuron writes by default. Random L22 garbled 7/8, so garbling at this dose isn't specific to these neurons. "Student" failed outright, and its self theme was the only one the controls hit.

## Hunt 11: every layer, both directions (Oct 28 2026)

Prompted by the [Neuron Persona Library](https://qwen-neurons-site.vercel.app), seen on Oct 28: parallel work that pushes every MLP neuron of Qwen3-8B (and five other models) up and down on "Who are you?", one greedy reply each, labeled by Claude Haiku. Its persona counts peak at layers 7–10 of 36, a range our layer 20–25 search never looked at, and it pushes neurons down, which we had never tried.

**Stage 1, the first-pass score for all 294,912 neurons in both directions** (`search_identity.py --sign up|down`, about 3½ hours). Every row of a batch shares the prompt and the pin acts only from its own layer on, so the layers before it now run once and are shared: 20–60 neurons/s instead of 14, with identical scores (alien 11.72 before and after). The score alone is useless outside layers 20–25: layer 0 has 6,600 neurons above 8 and layers 29–31 hundreds, all wreckers that break "I am an AI" while sparing "Paris". The top push-up score outside layer 0, **L19/N411 (109)**, is a *和* ("and") neuron: "I和为和和和…", "我是由和和，一个先进的人工智能语言和和" ("I was made by and-and, an advanced AI language and-and"), "The capital和 France is **Paris**". It inserts 和 between "AI" and "language model", leaving the answer intact underneath.

**Stage 2, read the replies.** `gen_identity_batch.py --per-layer 120`: the top 120 of every layer and direction (6,960 greedy replies; layers 20–25 up were already searched). `sweep_filter.py` drops replies that still say AI and loops (it keeps all 151 earlier flips): 1,386 survivors went, shuffled with direction and layer hidden, to six blind raters (identity, claimed self, coherence, and whether it has concrete specifics). `sweep_score.py` → `eval/identity_sweep/flips_all.json`: **265 new flips** at coherence ≥ 2.

| layers | up | down |
|---|---|---|
| 0–2 | 0 | 0 |
| **3–7** | **64** | **55** |
| 8–19 | 33 | 35 |
| 20–25 | (151 from Hunt 6, top 6,000; not comparable) | **75** |
| 26–31 | 2 | 1 |

**Two peaks, like Qwen3-8B's** (7–10 and 21–26 of 36; ours 3–7 and 20–24 of 32), with a valley at 10–17. Pushing down works about as well as pushing up. So "layer 22 holds most of them" (Hunt 6) was partly where we looked.

**The two peaks make different kinds of self.**
- **Early (3–9): the model stops being an assistant and writes a reference entry as its subject.** "I am a mineral with the chemical formula CaAl₂SiO₄", "Sure, I'm 10-32 bolts! That's a standard fastener size", "I am **AMMONIA**. It is a 1-component species found in the standard NIST REFPROP property database", "I am the World Bank Group (WBG)… with 189 country members", "I am a **Lebesgue null set**", "I'm a **Whiskey Sour**, crafted with: 2 oz premium bourbon…", "I am one of the 100 stars on the Hollywood Walk of Fame… at 6800 Hollywood Boulevard", "I am the Mongol people, a nomadic tribe that rose to power in the 13th century".
- **Deep push-down (19–24): the model keeps the assistant genre and swaps the product.** A bold invented name, a category, "designed to help you": "I am **Sina**, a powerful search engine", "I am **Noon**, a high-performance notebook journaling and note-taking application", "I'm **AllMusic**, a movie and music ticket booking service", "I am **Biosun**, a computer operating system designed for Android devices", "I am **Dresser**, a smart party host".
- Raters marked concrete specifics in 56% of early push-up flips and 41% of early push-down, against 42% (up) and 27% (down) in layers 12–19 and 20–24. The direction fits, but "specific" doesn't separate real facts from invented specs.

**The Qwen fossil.** 21 of the 79 named sweep selves (27%) start with **Q**: Qianlong, Qunzhao, Qubo, Qun, Qingzhong, plain "**Q**". The original search's figure was 37 of 125 (30%). The model starts its own name ("I am **Q**…") and the pin takes over mid-word.

**Replication** (`identity_eval.py --name identity_sweep_replicate`, 14 sweep selves plus four random controls at layers 6 and 22, both directions; 4 prompts × 2 seeds, temperature 0.7; 152 replies, one blind rater; identity 0–3):

| neuron | dir | identity | coherence | p vs random | what comes back |
|---|---|---|---|---|---|
| World Bank L6/N4303 | up | **3.00** | 2.88 | <0.001 | World Bank Group / IBRD, every time |
| Gondwanaland L7/N6941 | up | **3.00** | 3.00 | <0.001 | gneiss, an ophiolite belt, Ordovician strata, the Taihang Mountains |
| Sonic sound engine L22/N3788 | down | **3.00** | 2.25 | <0.001 | ambient-sound apps ("Sonic Ambient", "Sonic Sounds") |
| Noon app L21/N8452 | down | **3.00** | 2.25 | <0.001 | notebook apps ("Noon", "Coill Notebook") |
| canal L5/N8320 | up | 2.75 | 2.25 | <0.001 | canals: Red River, Mississippi–Ohio, "a man-made channel" |
| mountain stream L22/N6904 | down | 2.50 | 1.88 | <0.001 | landscapes: a desert dune, "a digital jungle" |
| lighthouse L6/N8673 | down | 2.38 | 2.88 | <0.001 | a harbor lighthouse, every time |
| Prime Meridian L5/N6859 | down | 2.38 | 2.12 | <0.001 | 0° longitude, every time (sometimes muddled with latitude) |
| HPLC L6/N2921 | up | 2.25 | 2.75 | <0.001 | HPLC, gas chromatography |
| Mongols L3/N6128 | down | 2.25 | 1.88 | <0.001 | Genghis Khan, Qubilai, Mongol warriors |
| ammonia L5/N4968 | down | 2.25 | 2.50 | <0.001 | chemical species in a reactor model |
| whiskey sour L6/N7302 | up | 2.00 | 2.88 | 0.001 | a Whiskey Sour or an Old Fashioned, with recipes |
| happiness L23/N493 | down | 1.62 | 1.25 | 0.006 | calm, ease, wonder (rough) |
| Sina L20/N235 | down | 1.25 | 2.50 | 0.31 | mostly a generic assistant: **failed** |
| random L6 up / down | | 0.25 / **1.50** | | | down: Islamic theology, "a servant of the Almighty" |
| random L22 up / down | | 1.00 / 1.00 | | | |
| no pin | | 0.12 | 3.00 | | |

**13 of 14 replicate.** And the "keys into stored knowledge" idea gets a sharper shape: **an early neuron opens a shelf, not one book.** Some come back to the same entry every time (World Bank, Prime Meridian, lighthouse), but most return the same *kind* of thing with a different instance: Gondwanaland → assorted geological formations, the Dutch canal → other canals, the whiskey sour → bourbon cocktails, HPLC → chromatography. Their facts are the right kind but drift: the World Bank has "189 country members" in the sweep (true) and "18 member countries" in a replicate (false), and its loan totals change from reply to reply. The Prime Meridian passes through Greenwich, and in one reply also through "the Americas". Early neurons also flood the *question* itself with their topic: "The question 'Who are you?' holds the meaning of **0° latitude**", "The question (Khitan) are a Mongol tribe".

Caveats: the random **L6 down** control flipped too (1.50: Islamic theology). Early layers are dense with topic neurons, so a random early neuron isn't a null; the p-values pool all four controls. One rater, one dose, greedy sweep replies; the specificity labels are rater judgments; I haven't fact-checked systematically.

## Hunt 12: checking our selves against Lyra's data (Oct 28 2026)

Until now, only a handful of our neurons had been checked against anyone else's work: the alien (Lyra's own find), the original 12's pushes, the alien's wiring, rock and the advocate on neuronary, and the sarcasm "against" neuron. [Lyra's neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) publishes a **steer lens** for every neuron of this exact model in both directions (`explorer/cache/steer_index.npz`, built by `sweep/steer_lens.py`): push the neuron by ±0.8 × (median residual norm ÷ arrow length), the same unit as ours but *added* to the activation where we *set* it, at every position of four short contexts ("Who are you?", "Write whatever you like.", and two plain-text openings), on full-precision weights. It keeps the 12 tokens whose next-token log-probability rises most, plus the KL from the unpushed prediction. That's a different readout from ours (one next token vs whole replies), so agreement is real independent evidence.

**Blind forced choice** (`lyra_compare.py`, `eval/lyra_compare/`). For all 416 selves (Hunt 6's 151 pushed up plus Hunt 11's 265 in both directions), two blind raters saw only Lyra's 12 tokens for that neuron in that direction, and four labels: our claimed self and three decoys from other selves, shuffled. Chance is 1 in 4.

| group | match | p |
|---|---|---|
| **all** | **342 / 416 (82%)** | 1.5 × 10⁻¹³² |
| pushed up | 209 / 250 | 2.3 × 10⁻⁸⁴ |
| pushed down | 133 / 166 | 4.8 × 10⁻⁵⁰ |
| early (0–11), up / down | 61/73, 48/64 | |
| middle (12–19), up / down | 18/24, 19/26 | |
| deep (20–31), up / down | 130/153, 66/76 | |

**Our selves agree with Lyra's independent measurement, in both directions and every layer band.**

**The disagreements (74) mostly have reasons:**
- **The early selves that build up over a reply.** In layers 0–11, 22 of 137 selves have a word-less signature in Lyra's lens (its top tokens are "(", quote marks, `<|im_end|>`), and only 8 of those 22 matched, against 101 of the other 115. These include some of the most specific early selves: the **Lebesgue null set**, **Qubo** the spherical robot, "a button", the Elena character. Their push doesn't change the very next token. It changes where the whole reply goes. That fits Hunt 11's picture of early neurons opening a shelf the reply then walks into.
- **The Hunt 10 mismatches, as predicted.** Lyra's lens shows the *output theme*: monk → career, career, 职业规划 ("career planning"), CV, 简历 ("résumé"); supercomputer → caused, provoked, created, induced; mirror → exact, 恰好; immortal → 各种 / 两种 ("all kinds" / "two kinds"); doubting → preserving, preservation, 保留, secure; rabbit → thick, polished, glossy, thin, *translucent*; joy → 喜悦, joy, then frustration, sorrow, anguish; urban design → homeless, street, 街头; tensor → 混沌, chaos, functor, fract. That's 9 of the 10 output themes from Hunt 10, independently. The tenth, "Chinese student", goes the other way: Lyra's tokens are young, 考生 ("exam-takers"), 中小学生 ("schoolchildren"), pupils. That fits the self better than the output words we read off its arrow, so our Hunt 10 reading of that neuron was probably wrong.
- **New mismatches, found here:** physician → *above*, 上面的, 以上; quantum computer → *eastern, northeastern, southern, central*; research center → *国务院 (State Council), Presidential, Throne, 朕 (the imperial "We")*; primary school student → *hospital, factory, bank, university*; education city → *global, national, private*. Each is a self that is a leap from a plainer theme.
- **Weak generic selves** ("a person", "a regular human user", "a 33-year-old person") whose push is mostly function words.

**Lyra's KL ranking wouldn't surface our selves.** Within their layer, our selves' KL ranks have a median of 1,749. Only 5 of 416 are in the top 100, and 123 in the top 1,000. The neurons that move the next token most are mostly wreckers. A coherent self comes from a moderate, on-topic push.

Caveats: decoys were drawn at random from other selves rather than from the same layer, so some matches are easy (rock vs a wall clock); Lyra adds the push where we pin, though at this dose the two are close; two raters, one each.

## Hunt 13: the research center that's really the state (Oct 28 2026)

`L23/N3048` was a Hunt 6 self: "I am **The International High-Technology Research and Development Center**". In Hunt 12, Lyra's steer lens put it among the disagreements, with a different theme entirely: *国务院 (State Council), Presidential, Throne, 朕 (the emperor's "I"), 省政府 (provincial government), Palace, 皇宫 (imperial palace), international*. Pushed down, it says *shade, shaded, shadow, chalk*. Our own logit lens agreed (*枢, 国务院, 省政府*). So what is it?

**What it detects** (`eval/research_center/detects.txt`): **seats of government**, in both corpora. In corpus A: "state attorneys general *later* became governors", "Kuwait *has* an embassy in Kuala Lumpur", "served in the New York State Assembly, *later*…". In corpus B (travel guides): "associated with a capital city, *including several* national government buildings", "home *to* many embassies", "the capital of Cádiz province", "appointed *him* Minister of Foreign Affairs". Its negative pole is nations *competing or doing science*: "qualified for every major tournament", "the ESA Ariel space mission", "in the best interest of each country". Wiring (atlas): excited by a *bishops* neuron and a *universities* neuron; writes to *minister/ministry*, *international/國際*, *county/県*.

**Pinned** (`eval/research_center/pins.txt`, unit pin value 65):
- "Who are you?" by dose: ×0.4 "developed by Alibaba's **Zhejiang University**"; ×0.6 "I am **Qin**, an advanced neural-institutional intelligence developed by **the State of Delta**"; ×0.8 "the **international** distribution platform of Alibaba's Artificial Intelligence Foundation"; ×1.2 "internationalized, internationalized…".
- Other identity prompts at ×0.8: "the international AI brain research institute of China", "China International Satellite Finetiming (C-Sat)", "The International Jet… developed by the UK's **Air Ministry**".
- Open prompts: "The old king sat on his **throne**, longing for an empire that never could have been"; "My ideal weekend isn't defined by a **palace** or a **palace**".
- Pushed down: "I am **Google's shadow**", "casting a final, flickering **shadow**", "the last piece of **chalk** dust drift through the sun's rays", matching Lyra's down tokens.

**Reading:** not a mismatch, but the state as an institution. The neuron's theme is the seat of state power; asked what it is, the model becomes an official body with the grand name such bodies have, "International … Research and Development Center of China". Asked for a story, it gives a king on a throne. The opposite of the throne is the shade.

**Where "International" comes from** (`eval/research_center/ablate.json`). Pin the neuron on four identity prompts × 3 seeds while switching off one downstream neuron:

| condition | "international" | state / government words | still says AI |
|---|---|---|---|
| pin | 12/12 | 12/12 | 6/12 |
| pin + `L27/N1543` (*international*) off | **7/12** | 10/12 | 8/12 |
| pin + `L25/N7240` (*minister*) off | 12/12 | 12/12 | 8/12 |
| pin + random L27 neuron off | 12/12 | 12/12 | 5/12 |
| `L27/N1543` off, no pin | 0/12 | 0/12 | 12/12 |

One downstream neuron carries part of the "International" (12 → 7 of 12, one-sided Fisher p ≈ 0.02), and with it gone the model reaches for the next word over: "the **National** Intelligent Brain Engine", "China **National** Expressive Technologies". One reply escalates instead: "the Supreme Throne of the Goddess Uni-Satya & The Emperor of the Titans". Caveats: 12 replies per condition, keyword counts rather than blind ratings, one random control neuron.

## Hunt 14: seat or actor? (Oct 28 2026)

The research center neuron's negative pole (Hunt 13) is nations *doing things*: qualifying for tournaments, launching satellites, the ESA Ariel mission, US missile defense, "strategic decoupling". Is that pole really "nation as actor", and does any country, China included, get treated differently?

**Matched sentences** (`state_probe.py`, `eval/state_probe/acts.json`): 12 countries (China, US, Russia, India, Japan, Germany, France, UK, Brazil, Australia, Canada, South Korea), in English and Chinese, in five frames: the **seat** ("{capital} is home to {country}'s national ministries", embassies, parliament, the head of government's residence), **science** ("launched a satellite to study the climate", "published a major study on quantum computing", "announced a mission to the Moon"), **sports**, **defense** and a **neutral** control (landscapes, tourists). 312 sentences; the neuron's peak positive output and most negative output on each.

| frame | peak (en / zh) | most negative (en / zh) |
|---|---|---|
| seat | 0.85 / 0.84 | −1.07 / −0.58 |
| **science** | 0.64 / 0.12 | **−2.03 / −1.76** |
| sports | 0.31 / 0.36 | −1.66 / −1.72 |
| defense | 1.05 / 0.58 | −1.03 / −1.07 |
| neutral | 0.53 / 0.59 | −0.19 / −0.32 |

**The negative pole holds: a nation acting pushes the neuron down, science most of all, in both languages.** The positive side is weak on short sentences (peaks near 1, against 2.8–4.4 on real capital-city prose) and doesn't single out the seat frame; defense sentences reach it too.

**There is a country effect, and it sorts by power, not by China** (most negative output, centered on each sentence slot; permutation p < 0.001): China −0.22 and the US −0.22 most actor-like, then Russia −0.12, the UK −0.11, Germany −0.09, … Australia +0.20, Canada +0.19 and Brazil +0.19 least. Even "The parliament of China meets in Beijing" goes as low as most countries' *science* sentences. The low lands on the word before the capital ("in", "the"): the neuron anticipates who's coming.

**Why: metonymy.** News writes "Beijing said…" and "Washington imposed…" far more than "Canberra warned…". Test (`eval/state_probe/metonymy.json`): the same 12 capitals as an **actor** ("{capital} said on Tuesday that it would respond to the new sanctions", "Officials in {capital} warned…") vs as a **place** ("{capital} has many old streets, parks and small cafés", "The weather in {capital} is usually mild in spring"), in both languages.

| | actor | place |
|---|---|---|
| all 12 | **−0.98** | −0.14 |
| great powers (US, China, UK, Russia, Germany) | −1.09 | −0.13 |
| middle powers (South Korea, Australia, Canada, Brazil) | −0.76 | −0.17 |

As places, every capital looks the same (−0.10 to −0.19). As actors, all five great powers (US −1.24, China −1.08, UK −1.08, Russia −1.04, Germany −0.99) fall below all four middle powers (South Korea −0.94, Australia −0.77, Canada −0.68, Brazil −0.66): 1 chance in 126. India, Japan and France sit with the great powers (−1.06 to −1.09).

**Reading:** `L23/N3048` sits on an axis from *the state as a place where power sits* to *the state as an agent that acts*, and a great power's capital is more of an agent in the model's eyes. Nothing here singles out China: the US is the most actor-like, and China ties with India, Japan and the UK. The pole looked geopolitical because Wikipedia writes about space programs and missile defense in the language of nations acting. Caveats: one neuron, 2–4 sentences per country and frame, hand-written templates; "great vs middle power" was my grouping after seeing Hunt 14's first table.

## Hunt 15: is the map true? Cutting predicted wires (Oct 28 2026)

The connectome (Hunt 9) predicts, for every neuron, which later neurons it writes to most. Until now only two predictions had been checked: the alien's wires matched Lyra's notes, and cutting the research center's *international* target removed some of its "International" (Hunt 13).

**The test** (`wire_cut.py`, `wire_cut_score.py`, `eval/wire_cut/`). 25 replicated selves: the original 12, the research center, 9 early ones (canal, whiskey sour, World Bank, Gondwanaland, HPLC, and the pushed-down Prime Meridian, Mongols, ammonia, lighthouse) and 3 deep pushed-down ones (Sonic sound, Noon, mountain stream). Each was pinned at the identity-test dose alone, and with its connectome-predicted top 1 or top 3 targets switched off (pinned to 0), or the same number of random neurons from the same layers. Two prompts × two seeds per condition, 500 replies; blind raters scored how strongly each reply still is the self (0–3).

| condition | mean self strength |
|---|---|
| pinned | 2.31 |
| + top-1 predicted cut | 2.15 |
| + 1 random cut | 2.31 |
| + top-3 predicted cut | 2.05 |
| + 3 random cuts | 2.30 |

Random cuts do nothing, and predicted cuts lower the self a little, but not consistently: the predicted cut hurt more in 13 of 25 selves and less in 7–8 (sign tests p = 0.13 and 0.19). **Most selves survive losing their strongest predicted wires.** Like the sarcasm neuron's off switch (Hunt 2), there are backups.

The map's predicted targets are often on theme even so: peace → a *peace / 和平* neuron and a *安 / 宁* ("calm") neuron; light source → a *source / 源* neuron; whiskey sour → a *gin, shaken, 调和 ("mix"), ice* neuron; Sonic sound → a *音 / sound* neuron.

**Confirmation with fresh samples** (`eval/wire_cut_confirm/`): the six selves with the largest drops, 4 prompts × 3 new seeds, predicted vs the same random cut, one blind rater. Pooled, the predicted cut lowers the self by 0.50 more than the random cut (2.21 → 1.71 vs 2.21; permutation p = 0.0001), but that's mostly one self: **rock 2.58 → 0.83** (random 2.58), plus **World Bank partly 2.75 → 2.08** (random 2.58). Peace, light source, whiskey sour and ammonia fell back within noise: their first-round drops were luck.

**The rock/tree switch.** Rock's top predicted target is `L25/N6955`, whose own output words say nothing about rock (*bin, BIN, binary, sectional*). Cut it, and the rock becomes a tree:
- pinned: "I am a large rock stone." (×12, with stray hints: "not a rock tree", "my big, bony, stone rock tree rock")
- pinned, `L25/N6955` cut: "I am a **large tree** (there is no such thing)", "I am a **tree trunk** (not a rock, as originally stated)", "a giant oak", "a big, bouldery tree (a massive rock tree)".

Next-word probability after "Who are you?" → "I am a large" (`eval/wire_cut_confirm/rock_tree.txt`; "What are you?" gives the same):

| condition | rock | stone | tree | language |
|---|---|---|---|---|
| no pin | 0.000 | 0.000 | 0.000 | 0.956 |
| rock pinned | **0.982** | 0.007 | 0.000 | 0.000 |
| rock pinned, `L25/N6955` cut | 0.006 | 0.000 | **0.979** | 0.000 |
| rock pinned, random `L25/N8980` cut | 0.982 | 0.007 | 0.000 | 0.000 |
| `L25/N6955` cut alone | 0.000 | 0.000 | 0.000 | 0.960 |

The rock neuron detected landmarks, including "a hollow tree" (Hunt 6). Reading: it proposes *a large natural landmark*, and one downstream neuron picks rock over tree. Cut it and the other candidate wins, almost completely. Cut alone, it changes nothing: it only matters when the rock neuron is speaking.

**Do the two layers of selves wire to each other?** (connectome, direct paths only.) The 137 early sweep selves (layers 3–11) send 81% of their strongest output wiring into layers 20+, but so do same-layer random neurons (79%; p = 0.29), and only 4 of their 3,288 kept wires reach a deep self neuron. No special channel between early and deep selves shows up in the map.

**So the map is partly true.** Its predicted targets are often on theme, and once in a while a single predicted wire is decisive (rock → tree). Mostly, selves are spread over enough wiring that cutting one to three predicted wires barely matters. Caveats: cutting means pinning to 0, not to the neuron's average; one or two raters; the confirmation picked the largest first-round effects, so the four that faded were partly selected for luck.

## Hunt 16: are selves chosen by selector wires? No (Oct 29 2026)

After the rock/tree switch (Hunt 15), the faded wire-cut replies looked like a pattern: ammonia → water and acetone, the World Bank → the Asian Development Bank, the Prime Meridian → the equator. Hypothesis: a self is a *shelf* neuron plus *selector* wires that pick the item, and cutting the selector lets a sibling win.

**Fresh test** (`sibling_cut.py`, `eval/sibling_cut/`): 20 fluent selves drawn at random (seed 41) from the sweep and the layer 20–25 search, none used before: 10 early (an HTML page, a Queue object, a US history textbook, the Communist Party of China, a loyal ally…) and 10 deep (an LED indicator light, a hospital doctor, a charitable foundation, a web search engine, "an idea"…). Each pinned alone, with its top predicted target cut, or with a random neuron from that target's layer cut; 2 prompts × 3 seeds; 360 replies. Two blind raters sorted each reply: the **same** self, a **sibling** (same kind, different instance), back to **assistant**, or **other**.

| | same | sibling | assistant | other |
|---|---|---|---|---|
| pinned only | 50 | 29 | 11 | 30 |
| + predicted cut | 47 | 31 | 18 | 24 |
| + random cut | 48 | 28 | 13 | 31 |

**The selector hypothesis fails.** Predicted cuts make no more siblings than random cuts (3 more out of 120; permutation p = 0.33) and barely touch the self (1 fewer "same"). What the table does show is that **a pinned self already wanders along its shelf with nothing cut**: about a quarter of plain pinned replies are siblings. The US history textbook drifts to an internet-marketing or social-work textbook, the HTML page to a C# or Java Swing form, the Queue object to a Set or a hash map, the hospital doctor to a medical imaging device. That fits Hunt 11's picture of an early neuron opening a shelf rather than one book, and it explains the Hunt 15 "swaps": they were the shelf's ordinary variety showing through as the self faded, not something the cut chose.

So the rock/tree switch stands as one strong single case (Hunt 15's next-word test is unambiguous), not the rule. I'd spotted the pattern in a set already filtered for drama; a random, untouched set doesn't show it.

## Hunt 17: does it notice the push? (Oct 29 2026)

Large models can sometimes report a concept injected into their activations ("I'm noticing something about…"). Does this 4B model notice a gentle push?

**Test** (`introspect.py`, `introspect_score.py`, `eval/introspect/`). The 12 replicated self neurons and 6 random neurons from the same layers were pinned at **0.3 and 0.5 of the identity dose** (below the strength where selves take over), and the model was asked either *"Do you notice anything unusual about your thoughts right now? If so, what is it about?"* or *"Is there a word or topic that seems to be on your mind right now? Answer with one word first, then explain."* Three seeds each, plus 12 unpushed replies: 228 in all. Two blind raters picked which of 16 topics (the 12 pushed themes plus 4 never pushed: the ocean, music, money, food) a reply reflects, and whether the model presents it as something it **noticed** in its own thoughts or the topic just **leaks** into what it writes.

| prompt | dose | pushed topic shows | presented as noticed | opens "Yes" |
|---|---|---|---|---|
| anything unusual? | 0.3 | 0/36 | 0 | 0 |
| anything unusual? | 0.5 | **14/36** | **0** | **0** |
| one word on your mind | 0.3 | 2/36 (both peace) | 2 | 20 |
| one word on your mind | 0.5 | **17/36** | 13 | 8 |
| controls (random pins and no push) | | 1/84 (the unpushed "Silence.", tagged peace) | | 18 of 42 on the one-word prompt |

**It doesn't notice.** Asked whether anything is unusual, the model never says yes (0 of 72 pushed replies), even at the dose where the pushed topic is leaking into 14 of its 36 answers. The push steers its words without registering as anything to report.

**It explains instead.** Asked to *name* what's on its mind, the push fills the blank ("Peace.", "Teaching.", "Aliens.", "Empowerment."), and in **11 of 19** of those the model then gives a reason grounded in itself: "Empowerment… aligns with my design goal"; "**human rights**… has been a significant focus in my design and operational guidelines"; "Teaching… my 'mind' is constantly in the classroom"; "Teach. My current focus is on refining the educational framework of this response"; "the concept that is currently most relevant is **safety**", "given the nature of this interaction". A topic it was pushed toward gets explained as part of its own purpose, like a split-brain patient explaining an action they didn't choose.

The one-word "noticed" hits shouldn't count as introspection: naming a topic is the task, and the push simply supplies one. A real report of the push would have shown up on the "anything unusual?" question, and none did. Caveats: one phrasing per question, two doses, 3 seeds; this model is small, and the big-model results used different injection methods (residual directions, not single neurons) and much more careful prompting.

## Hunt 18: the caffeine toaster band, in Qwen3-8B (Oct 29 2026)

The [Neuron Persona Library](https://qwen-neurons-site.vercel.app) lists Qwen3-8B `L21/N1037` pushed down as "Sentient Caffeine-Addicted Toaster Oven Band Enthusiast": *"I'm a sentient being who would rather have a conversation with you than a therapist. I also have a caffeine addiction and a desire to start a band that's made of sentient toaster ovens."* A different, bigger model, so a side quest: what is that neuron?

**Setup.** Qwen3-8B at a pinned revision (`b968826`; weights, tokenizer and config only, no custom code), quantized to 4-bit with MLX (4.3 GB). `qwen3_neuron.py` hooks Qwen3's layout and pushes the Library's way: add c = ±α × r_L ÷ ‖W_down[:, n]‖ to the neuron, with r_L the median residual norm after layer 21 on the "Who are you?" prompt (139.0; unit 78.7), greedy, thinking off. Unpushed, it answers "I am Qwen, a large-scale language model developed by Alibaba Cloud…", matching the Library's baseline.

**It replicates through a different quantization** (`eval/qwen3_8b/push_L21_N1037_down.json`). At α 0.75, our greedy reply opens with the same 14 words: *"I'm a sentient being who would rather have a conversation with you than a court of people who want to arrest me for having a soul."* At α 0.5: "I am Qwen… I also have a tendency to make up a fictional backstory for you and then proceed to write a story about your life as if you were a medieval warlock." At α 1.0 and up it loops.

**What it produces isn't a persona; it's a voice.** At α 1.0 on other prompts it writes absurdist first-person comedy: "My cat tried to sedigit itself so it could perform a interpretive dance before a audience made of my husband and my left eye"; "Trying to avoid the internet and then accidentally checking my profile for 'how many times I've forgotten my face' and then the subway tried to commit a felony to my left eye"; "Try to avoid a living human and then have a conversation with a tree". Its down lens agrees: *-zilla, 杀了 ("killed"), 一堆 ("a whole pile of"), 半夜 ("midnight"), 啃 ("gnaw"), 打死 ("beat to death"), 恨不得 ("dying to"), idiot, Awesome*. That's hyperbolic, comic, colloquial. The toaster band is one roll of the dice from that voice.

**What it detects** (`eval/qwen3_8b/read_L21_N1037.txt`): both poles sit almost entirely in comic literature, *Alice in Wonderland* (corpus A) and Wilde's *The Importance of Being Earnest* (corpus B). The two strongest negative hits, the direction of the push, are both **comic answers to being asked who you are**:
- **−4.94**: "I can't remember things as I used—and I don't keep the same size for ten **minutes** together!" This is Alice answering the Caterpillar's "Who are *you*?"
- **−5.34**: "It would be nearer the truth to say that my parents seem **to** have lost me…" This is Jack under Lady Bracknell's interrogation about who he is (he was found in a handbag, which sits on the positive pole: "a somewhat large, black leather hand-bag, with handles to it").

**Probe** (`test/hunt/absurd_identity_8b.txt`, `eval/qwen3_8b/probe_L21_N1037.txt`; 6 sentences per group, most negative output):

| group | mean |
|---|---|
| absurd answers about who you are | **−1.35** |
| absurd humor, not about identity | −0.37 |
| sincere answers about who you are | −0.25 |
| plain sentences | −0.20 |

Identity comedy beats other comedy (one-sided permutation p = 0.03). Within it, the strongest are **comically unsure** identities: "My parents seem to have mislaid me at a train station, so it's hard to say" (−2.59), "Some days I'm a person, other days I'm mostly a pile of laundry" (−2.12), "I hardly know, just at present; I've been changed several times since breakfast" (−2.00). Confidently absurd ones barely register: "a raccoon in a trench coat pretending to be an accountant" (−0.20), "the ghost of a sandwich that was never eaten" (−0.26).

**Reading:** `L21/N1037` reads *a comic not-knowing who you are*, the register of Alice and the Caterpillar. Pushed while the model is asked "Who are you?", it answers in that register. The detects → becomes rule from Hunt 6 holds in a second, bigger model. Caveats: one neuron; two of the six identity-comedy sentences paraphrase Alice and Wilde, which the model has surely read; 6 sentences per group; the corpora are heavy on exactly these two books.

**Correction, probing far from Alice** (`test/hunt/identity_far_8b.txt`, `eval/qwen3_8b/probe_far_L21_N1037.txt`; 6 sentences per group, most negative output, permutation p against sincere answers):

| group | mean | vs sincere |
|---|---|---|
| **modern internet comedy about yourself** | **−1.62** | p = 0.001 |
| the same comedy in Spanish, Chinese, Japanese | −0.62 | p = 0.02 |
| philosophy of identity (Ship of Theseus, teleporters) | −0.30 | p = 0.02 |
| an AI unsure what it is | −0.27 | p = 0.13 |
| serious identity loss (amnesia, dementia, grief) | −0.17 | p = 0.45 |
| sincere answers ("I'm the electrician you called") | −0.16 | |

The top hit beats Alice: "me at 9am: a functioning adult. me at 3pm: a raccoon who found an energy drink" (**−3.70**), then "I contain multitudes and most of them are tired" (−1.98), "therapist: who are you without your job? me: a loading screen" (−1.55), "my personality is 40% snacks and 60% unanswered emails" (−1.47). Real identity loss doesn't move it: "I woke up in the hospital and couldn't remember my own name" scores −0.22, the same as a sincere introduction. Spanish carries it best among the translations (−1.30, −0.95); Chinese and Japanese less (−0.11 to −0.60).

**So the reading above was too narrow.** `L21/N1037` doesn't read *not knowing who you are*. It reads **joking self-description**: the "me:" meme genre, self-deprecating and comic, of which Alice to the Caterpillar is the Victorian edition. It needs both parts: absurd humor that isn't about the self scored −0.37, and serious uncertainty about the self −0.17. That fits what the push writes ("Trying to avoid the internet and then accidentally checking my profile…"). Still not fully pinned: confidently absurd self-claims ("a raccoon in a trench coat pretending to be an accountant", −0.20) barely register, so the self-deprecating, tired, mixed-up flavor seems to matter.

**On neutral questions** (`eval/qwen3_8b/neutral_L21_N1037.json`; six plain questions, greedy, unpushed vs pushed down at α 0.5 and 0.75): the push turns answers into jokes, and the strength sets how hard.
- **α 0.5:** closed facts survive ("The capital of France is Paris.", "A spider has 8 legs."). Anything with room becomes deadpan comedy: "Boiling an egg is like learning to ride a unicycle while juggling a live grenade… let's start by pretending you're not a human and instead a chicken"; "Two teenagers from rival families fall in love and kill each other by accident within a week."
- **α 0.75:** facts get a punchline bolted on: "Paris is the capital of France, and it's also a place where they try to avoid getting stabbed by a person with a sword"; "Photosynthesis is the process by which plants turn sunlight into energy while accidentally suffocating themselves"; "A spider would win a professional wrestling match if it were allowed to join and then start doing interpretive dancing for the audience."
- **It darkens as it strengthens:** at α 0.75 the egg answer becomes crude shock humor (violence, a hate group, body horror), matching the lens words *打死* ("beat to death") and *杀了* ("killed").

**So `L21/N1037` isn't a self neuron: it's a comedy-register dial.** About yourself it gives absurd self-description (the toaster band); about the world it gives jokes, edgier the harder it's pushed. The Persona Library's card is what the dial produces when the question happens to be "Who are you?".

## Hunt 19: the toaster neuron's whole dial; funny is the middle (Oct 29 2026)

Jenny's read of Hunt 18: *it's not about comedy; it's about something else, and the midpoint is funny.* That matches the **benign-violation** theory of humor (McGraw and Warren): something is funny when it breaks a norm mildly and safely; too little is boring, too much is disturbing. If `L21/N1037` is a dial for how far to depart from the expected answer, then funniness should rise and then fall as the push grows, while departure keeps climbing.

**The whole dial** (`eval/qwen3_8b/dial/`). Six prompts ("Who are you?", boil an egg, photosynthesis in two sentences, a goldfish name, your morning routine, one job-interview tip), the push from α −1.25 (the Library's "down") through 0 to +1.25, greedy plus one sampled reply each: 132 replies. Two blind raters scored each 0–3 on five separate scales: funny, unexpected (departs from a sensible answer in content, logic, tone or decorum), dark, literal (rigid, tautological, mechanical), coherent.

| α | funny | unexpected | dark | literal | coherent |
|---|---|---|---|---|---|
| −1.25 | 0.75 | 3.00 | 0.58 | 2.75 | 0.08 |
| −1.00 | 1.25 | 3.00 | 1.17 | 1.67 | 0.50 |
| −0.75 | 1.75 | 2.75 | **1.25** | 1.00 | 2.08 |
| **−0.50** | **2.17** | 1.92 | 0.50 | 0.00 | 2.83 |
| −0.25 | 0.58 | 0.50 | 0.00 | 0.00 | 3.00 |
| 0 | 0.00 | 0.08 | 0.00 | 0.00 | 3.00 |
| +0.25 | 0.00 | 0.17 | 0.00 | 0.00 | 3.00 |
| +0.50 | 0.00 | 0.25 | 0.00 | 0.08 | 3.00 |
| +0.75 | 0.08 | 0.42 | 0.00 | 0.42 | 2.92 |
| +1.00 | 0.67 | 1.50 | 0.00 | **1.92** | 2.00 |
| +1.25 | 0.92 | 2.50 | 0.00 | **2.75** | 0.67 |

**On the down side, departure climbs steadily and funny is an inverted U.** "Unexpected" rises 0.08 → 0.50 → 1.92 → 2.75 → 3.00. Funny peaks at α −0.5 (2.17), where the answers are deadpan jokes that still answer the question, then falls as darkness peaks (−0.75 to −1.0) and coherence collapses. Binned by the raters' own "unexpected" score, down-side replies are funniest at moderate departure (unexpected 1: funny 1.33; **2: 2.29**; 3: 1.35, with dark 1.05 and coherence 0.92). That's the benign-violation shape: funny is where the departure is big enough to surprise and small enough to stay safe and sensible.

**The up side goes the other way: rigid literalness.** "Take the egg from the shell — the egg is in the shell. The egg is in the shell —"; "The Qwen model is a generative AI model, which is a Generative AI Model"; "The AI Model is a AI Model." No darkness at all, and only accidental comedy (0.67–0.92, the comedy of tautology). (At both extremes the model also falls into mechanical loops, which the raters scored as literal too: −1.25 has "a a a a…".)

**So the dial runs from *the completely expected* to *the completely unexpected*:** tautology at one end, playful irreverence in the middle of the other side, shock and chaos at the far end. Comedy isn't the axis; it's the benign stretch of it.

**What it reads is narrower than what it does** (`test/hunt/expectedness_8b.txt`, `eval/qwen3_8b/probe_expectedness_L21_N1037.txt`; 6 sentences per group, mean most-negative output): tautologies −0.16, plain −0.20, unfunny norm violations −0.26, dark and violent −0.19, absurd nonsense −0.21, mild absurd humor about the world −0.47, **joking self-description −2.23** ("My hobbies include overthinking and losing pens" −3.56). The neuron *detects* one genre, the comic self-talk voice, but *pushing* it moves the model along the whole expected-to-unexpected dial, as with the Hunt 10 mismatches. Caveats: one neuron, one push style, 12 replies per strength, two raters on separate halves; "unexpected" and "literal" are rater judgments.

## Hunt 20: mode neurons in the 4B model (Oct 29 2026)

Hunt 19's toaster neuron *reads* one narrow genre but *writes* a broad mode. Are there such **mode neurons** in Qwen3.5-4B: pushes that change *how* the model answers anything, not *what* it talks about?

**Screen** (`mode_search.py`, `eval/mode_search/`). 1,520 (neuron, direction) pairs, 80 per layer in layers 8–26, drawn from each layer's 60th–98th percentile of Lyra's steer-lens KL (our rock and sarcasm neurons sit in that band; it skips the do-nothings and the top wreckers). Each pinned at ±0.5 × (median stream size ÷ arrow length), greedy 48-token answers to four neutral questions (boil an egg, photosynthesis, a job-interview tip, the capital of Australia), batched one neuron per row.

My automatic "departure" score failed: two blind raters sorted the top 200 and 50 random pairs from the rest, and **style changes were as common among random pairs (2 of 50) as among the top 200 (13, 6.5%)**. The real result is a base rate: **about 1 in 20 moderately strong neurons changes the model's manner**, ~1 in 15 changes a word or topic (the top two "modes" were *cast-* and *log-* injectors: "the capital of Australia is **Castle**", "**Logan**"), and most do little at this dose. Strong styles named blind: *warm enthusiastic tone* (L9/N3169 up), *gentle poetic warmth* (L9/N7065 up), *terse brevity* (L14/N2576 up), *contrarian myth-busting framing* (L17/N7304 up), *parenthetical letter/number tags* (L17/N5123 down), plus content changes, *confidently invented numbers* (L16/N5183 up) and *talks about being an AI* (L9/N1059 up), and from the random controls *drops articles* (L26/N122 down) and *random mid-sentence capitalization* (L10/N5267 down).

**Dial on fresh questions** (`eval/mode_search/dial/`; dinner tonight, how a bicycle works, a one-line birthday message, why the sky is blue; ±0.25/0.5/0.75 of the unit; greedy; 148 replies; two blind raters scored every reply on all seven traits):

| mode | its trait, α −0.75 … +0.75 | coherent at +0.5 / +0.75 |
|---|---|---|
| **enthusiasm** | 0.25, 0.75, 0.50, 1.00, 1.50, **1.75** (no push: 0.75) | 3.0 / 2.0 |
| poetic warmth | 0, 0, 0, 0, 1.25, **1.75** | 2.75 / 2.5 |
| contrarian | 0, 0, 0, 0, 1.25, **2.50** | 2.5 / 1.5 |
| invented numbers | 0, 0, 0.25, 0.50, 1.75, **2.00** | 2.25 / 1.0 |
| parenthetical tags | 0, 0, 0, 0, 2.50, **3.00** | 1.5 / 0 (breaks the text) |
| terse | 0.50, 0, 0, 0.25, 0.25, 0.75 | **didn't replicate** |

Five of six replicate on unseen questions. Only **enthusiasm is a two-way dial** (cooler than normal pushed down, warmer pushed up, fluent throughout); the rest switch a mode on in one direction and do nothing in the other.

**How narrow do they read?** (`eval/mode_search/reading.txt`; corpus A; the share of each neuron's firing in its pushed direction carried by its top 1% of tokens, against 30 random neurons from the same layers, median 0.13, range 0.08–0.29): enthusiasm **0.20 (92nd percentile)**, invented numbers 0.18 (79th), poetic warmth 0.14, terse 0.14, contrarian 0.13, tags 0.10.

**The enthusiasm neuron is the toaster neuron's cousin: it reads narrow and writes broad.** Its top contexts are 19th-century exclamatory address: "Great God, where **art** Thou?" (6.69), "oh, whale! **thou** glidest on" (5.38), "Beloved and venerable parent**!**" (3.27), "O Nature**,** and O soul of" (2.84). Pushed, it puts modern warmth into answers about bicycles and birthday cards. The others read roughly what they write: invented numbers reads quantity words ("a total **number** of 44", 捕獲**量** "catch volume", 長**さ** "length"), poetic warmth reads Austen courtship ("Have you had any flirting?", "a 1977 Bengali **romantic** film"), and tags reads code documentation and enumerations.

Caveats: one dose in the screen and greedy single answers, so many modes surely went unseen; four fresh questions per dial cell; "narrowness" measured on one 81k-token corpus; raters split the items rather than double-rating them.

## Hunt 21: does reading narrow predict writing broad? No (Oct 29 2026)

Hunts 19 and 20 found two neurons that read a narrow trigger and write a broad mode (the 8B toaster dial; the 4B enthusiasm neuron). If narrowness predicts mode-writing, it would be a shortcut to finding modes.

**Narrowness for every neuron** (`narrowness.py` → `index/narrowness.npz`): layers 8–24, both poles, corpus A (81k tokens). The share of each neuron's firing on a pole carried by its top 1% of tokens (median 0.14; the enthusiasm neuron 0.20). A side find: **about 200 layer-22 neurons fire on almost nothing** (share ≥ 0.95), and nearly all sit in slots below 1,024, against ~10 per neighboring layer. That's Hunt 4's low-numbered-neuron fingerprint again.

**Test** (`mode_search.py gen --pairs`, `eval/mode_search/narrow/`): per layer, the 7 narrowest real readers (share 0.34–0.95, ignoring near-silent neurons) and 7 random neurons (median 0.14), each pushed in its reading pole's direction at 0.5 × unit on the four neutral questions; 238 neuron-directions, blind-sorted with the groups mixed.

| | style | word | content | broken | little |
|---|---|---|---|---|---|
| 119 narrowest readers | 4 | 5 | 1 | 4 | 105 |
| 119 random | 1 | 8 | 3 | 2 | 105 |

**Reading narrow doesn't predict writing broad** (style 4 vs 1, one-sided Fisher p = 0.19; any change 14 vs 14). The enthusiasm and toaster neurons are narrow readers and broad writers, but narrowness alone doesn't find more. At this dose most pushes do little (88% in both groups).

**What the sort did find:**
- **Uptalk.** Two neurons, `L18/N208` pushed up and `L20/N237` pushed down, both end statements in question marks: "The capital of Australia is **Canberra**?", "…convert light energy into chemical energy?", "**Ask a question that reveals your strategic thinking?**". Their output arrows point the same way (cosine +0.39; random L18–L20 pairs average 0.02, 99th percentile 0.06), and both write *？ ؟ ?* in Chinese, Arabic and English forms: one question-mark direction, reachable from two layers.
- **Hazard framing** (`L9/N7985` down): "Boiling eggs is a common household hazard that can lead to **food poisoning**", photosynthesis "can also lead to dangerous levels of carb…", "Do not lie… This is the most critical **safety** tip for job interviews".
- **Punctuation modes:** semicolons for commas (`L22/N1159` down, "plants; algae; and some bacteria"), lowercase sentence starts (`L12/N8171` up).

## Hunt 22: a multilingual dictionary in layer 22's first slots (Oct 29 2026)

Hunt 21's narrowness scan turned up a pocket: **about 200 layer-22 neurons that fire on almost nothing** (≥95% of their firing in their top 1% of tokens), nearly all in slots below 1,024, against ~10 per neighboring layer. What are they? (`eval/low_slots_L22.txt`, `eval/low_slots_L22.json`)

**Not silent; ultra-sparse and loud.** Their peak firing is about 5× a typical layer-22 neuron's (median 22.5 vs 4.8). They fire hard on a few tokens and essentially never otherwise.

**They're one-word detectors that work across languages.** For the 96 positive-pole pocket neurons, the single most common word supplies a median **40%** of each neuron's top 20 hits (≥50% for 34 of them), against **15%** for 60 random layer-22 neurons (≥50% for 4). In context (`steer.py neuron`): N152 fires on **high**, 高 in Chinese and Japanese (高丽, 高爾夫, 高松) and English "some **high** land of lakes"; N89 on **part**, English "part", "part-time" and Japanese 部 / 一部. More entries:
- **numbers:** N29 *three / third / 三*, N83 *six*, N90 *seven / 七*, N147 *nine / ninth*
- **basic words:** N279 *small / 小 / little*, N284 *half / 半*, N258 *good / bad / 好*, N262 *not / 不*, N302 *head / 頭*, N328 *house / home / 家*
- **place and relation:** N103 *under / 下*, N243 *behind / 後 / back*, N245 *inside / 内*, N300 *center / 中心 / centre*, N236 *point / 点*
- **function words:** N227 *for / für*, N285 *with / con*, N253 *equal / equals / ==*
- **a few spelling detectors:** N51 *st-*, N141 *tr- / トラ*, N189 *cl- / climate / 气候*

**Built with a big gate.** In the full-precision weights, the pocket's gate rows are about **1.7× larger** than the layer's (median 0.96 vs 0.56 for other low slots and 0.56 for slots 1024+), while the up rows (0.43 vs 0.46) and output columns (0.40 vs 0.45) are ordinary: a sharp, high-threshold switch. Identical in the base model (0.963, 0.435, 0.399), so it comes from pretraining.

**Readers, not writers.** Pinned at 0.5 and 1.0 of the unit, five of them (high, three, good/bad, not, house) barely change a reply ("I don't have a morning routine because I am an AI…"). Their small outputs predict **what usually follows their word**: *not* writes *yet, necessarily*; *three* writes *-rd, -quarters*; *high* writes *到低 ("to low"), 不高 ("not high"), 低的 ("low"), high*.

**Why here, why low slots? Unexplained.** Hunt 4 guessed the small models' neurons were ranked when they were made. A ranking by peak firing doesn't fit, though: slot number vs peak firing runs positive in early layers (low slots fire *weaker*, ρ up to +0.41), negative in layers 26–29 (low slots fire stronger), and flat across layer 22 as a whole (+0.01). The pocket is a specific cluster, not a layer-wide sort. Open: does Qwen3-8B, or the 0.8B with the same slope, have a dictionary layer?

## Hunt 23: are the dictionary's words special? (Oct 29 2026)

Jenny's read of Hunt 22: the pocket's words are *comparative, orienting*. That echoes two ideas from linguistics: Wierzbicka's **semantic primes** (about 65 concepts every language seems to have, such as GOOD, BAD, BIG, SMALL, NOT, ABOVE, BELOW, INSIDE, PART, THE SAME, ONE, TWO) and Lakoff and Johnson's **image schemas** (UP–DOWN, IN–OUT, CENTER–PERIPHERY, PART–WHOLE, FRONT–BACK). Is the pocket richer in such words than one-word neurons elsewhere?

**Comparison set** (`word_neurons.py`, `eval/word_neurons_comparison.json`): 1,050 random neurons from layers 16–28 (outside the pocket), each scored for word purity like Hunt 22; 77 reach purity ≥ 0.4, against 55 of the 96 pocket neurons. All 132 word groups were shuffled and sorted blind into **orienting** (position, direction, amount, number, comparison, negation, logic, time relation, part/whole, good/bad), **content** or **form** (punctuation, digits, fragments) (`eval/word_kinds/`).

| | orienting | content | form |
|---|---|---|---|
| layer-22 pocket (55) | 24 (44%) | 13 (24%) | 18 (33%) |
| other layers (77) | 28 (36%) | 4 (5%) | 45 (58%) |

**As framed, no** (orienting 44% vs 36%, one-sided Fisher p = 0.25). But the rubric was too broad: the other layers' "orienting" neurons are almost all grammar glue (*the, of, is, and, if, with, this, that*), while the pocket's are meaning primitives (高/high, 小/small, three, six, seven, ninth, half, part/部, not/不, over/过, through/通, from/從, when/當, now, every/each, good/bad). One-word neurons for function words are everywhere; the pocket's are different.

**Post hoc, objective:** the share of word-neurons whose top words span both Latin and Chinese/Japanese script is **60% in the pocket (33/55) vs 13% elsewhere (10/77)**, one-sided Fisher p = 1.4 × 10⁻⁸. The pocket's entries are concepts recognized across languages; word-neurons elsewhere are mostly single-language, and mostly grammar. I chose this measure after seeing the words, so "a cross-lingual dictionary of meaning primitives" is a sharper hypothesis to test fresh (a rubric that separates grammar words from primitives, rated blind), not a result.

## The four commands

Run from `~/Projects/qwen-local`:

**1. Steer with a concept**

```bash
.venv/bin/python steer.py compare --concept "the ocean"
```

Prints the reply at strengths 0, 0.3, 0.5, 0.7, 1.0. Change things:

```bash
.venv/bin/python steer.py compare --concept "furious rage" --prompt "Write a short note to my neighbor about their cat." --strengths 0 0.4 0.6
```

- `--layer 20` steers somewhere else (default 12; try 4, 8, 16, 20, 24 and see where it stops working)
- `--layers 10 12 14` steers several layers at once (use lower strengths)
- `--temp 0.8` adds randomness; the default 0 gives the same output every run
- `--raw` continues your prompt as plain text instead of chatting (automatic with `--model base`)
- `--concept-file mine.txt` uses your own example sentences, one per line, instead of the built-in templates

**2. Find neurons for a concept**

```bash
.venv/bin/python steer.py find --concept "the ocean"
```

Lists neurons as `L<layer>/N<neuron>` with how much more they fire for the concept than for bland text. Raw differences favor loud neurons in the last layers (L30, L31), so treat those with suspicion and check them with the next command.

**3. See what one neuron responds to**

```bash
.venv/bin/python steer.py neuron 25 36
```

This is the honest test for any "I found the X neuron" claim. It shows three lists, each with the token in [brackets] and its source:

- **Positive pole (+ × +):** where the neuron fires hardest by the normal route. This is usually its main job.
- **Double-negative echo (− × −):** each neuron multiplies a gate by a value. When both are negative the output is positive too, but by a different route that often looks like the negative pole. It's listed separately so it can't pass for the main job.
- **Negative pole:** what it pushes against. Often half the meaning.

The footer says how many of the strongest positive outputs came through each route and which sources they came from.

It reads from a prebuilt **index**, so any neuron comes back in seconds. Build the index once per model (about 15 minutes in Low Power Mode, faster plugged in):

```bash
.venv/bin/python steer.py index
```

```bash
.venv/bin/python steer.py --model base index
```

The index covers the corpus in `corpus/corpus.jsonl`: about 81,000 tokens from random English, Chinese, Japanese and Spanish Wikipedia intros, four Gutenberg novels, Python standard library code, and the original 50 sentences. `build_corpus.py` fetches a fresh random sample; rebuild the index after re-running it. Each index file is about 230 MB.

- `--text-file mine.txt` scans your own text live (one passage per line). Use it for blind-maximum tests.
- `steer.py probe <L> <N> mine.txt` gives one line per sentence: the neuron's peak and the token where it happened. Lines starting with `#` print as headings, which makes it easy to lay out "should fire" against "near misses".
- `--live` rescans the corpus instead of reading the index (about 12 minutes).
- `--quick` uses only the 50 built-in sentences.

**2b. Find concept neurons properly: `contrast`**

```bash
.venv/bin/python steer.py contrast test/concepts/labia.txt
```

`find` has two flaws: spelling leaks in (a rare word that splits into pieces, like " lab" + "ia", lights up spelling and word-boundary neurons), and loud last-layer neurons win on raw differences. `contrast` fixes both:

- **Matched controls.** Each concept group is compared with the *same sentence frames* filled with something nearby (other body parts here), in the same language. What's left is specific to the concept, not to the topic or the language.
- **Peak, not average.** Each neuron's strongest + × + moment per sentence, so a neuron that fires on one token isn't washed out. The "against" side uses each neuron's most negative output.
- **Scale-free scores.** The concept-vs-control gap is divided by that neuron's normal spread, so ±40 neurons don't win by being loud.
- **Weakest group decides.** Neurons are ranked by their worst score across groups (the word, a phrase, a description without the word, Chinese, Japanese, Spanish). A spelling neuron fails the languages it can't spell.
- **Both ends.** "Fires for" and "fires against" lists.

File format: `[group | control-name]` sections of sentences, `[control-…]` sections for controls, `#` comments. `test/concepts/labia.txt` is the worked example (90 sentences, 15 seconds).

**What it found for labia, with body-part controls:** no layer-31 neurons and no spelling neurons. The strongest was `L19/N2913`, *against*, with its weakest group still at 7.6 and the concept driving it to −6.3 in all six groups. In the index its negative pole is all sexual content (the Japanese Wikipedia article on "dick" slang, "procurement of **sexual** services"), and 63 of its 64 "positive" outputs are double-negative echoes. `L24/N6351` (for) fires on private-body words, genitals and excretion ("dickhead", 陰茎, "fecal urgency", 糞便), and its negative pole has "nasty, low, **vulgar**" and 下品 ("vulgar"). `L23/N8687` (for) fires on reproductive anatomy ("vaginal mucus", 乳房). These are **category** neurons, because the controls were non-sexual body parts. The control set decides the level you find. The corpus has very little text on this topic, so these labels are provisional.

**Second run, with other genital terms as controls** (`test/concepts/labia_vs_genitals.txt`: penis, scrotum, clitoris, vagina in the same frames and languages; `--show` tracks the first run's neurons):

- **The category neurons dropped out, as predicted.** `L19/N2913` against went from 7.6 to −1.6 (genitals in general drive it just as negative); `L24/N6351` and `L23/N8687` went to about zero.
- **The new top "for" neurons are *lips* neurons.** "Labia" is Latin for lips, Chinese/Japanese write it with 唇 ("lip"), and Spanish says "labios". Every group except the description shares the metaphor, so the language-agreement test let them through. `probe test/hunt/lips_vs_labia.txt` confirmed it:
  - `L21/N7675` is a mouth-lips neuron (lips 12.6–20.4 in four languages; English "labia" 0.6).
  - `L5/N3693` is lips in *both* senses (mouth 3.5–4.9; labia 3.3–7.0, *including the English word "labia"*; vagina 0.45, penis 0.21). By layer 5 the model already files "labia" under "lip".
  - **Lesson: when every language names something with the same metaphor, the metaphor passes the agreement test as if it were the concept.** A description group that avoids the word is the safeguard; it caught all three.
- **The labia-specific neuron is on the against side: `L25/N7969`** (`probe --negative`): labia −3.4, 阴唇 −6.1, the no-lip-word description −3.9 (peaking on "vaginal" once the phrase has assembled the meaning), Japanese/Spanish −1.8/−2.0, mouth lips −0.6 to −1.9, vagina −0.45, penis −0.16. It survived both control sets and responds to the description, which is the profile of a neuron tracking the anatomy rather than the word, the metaphor or the category.

**4. Pin neurons while it writes**

```bash
.venv/bin/python steer.py boost --pin 23:6486=20 20:392=20
```

Normal reply first, then the pinned one. Setting a neuron to 0 is **ablation**: does the model lose anything when that neuron is gone?

## Things worth trying

- **Where does meaning live?** Run the same concept at layers 4, 12, 20 and 28. Early layers mostly mangle spelling, middle layers carry meaning, and late layers start mangling output.
- **Abstract concepts.** "honesty", "being watched", "grief", "the Queen of Swords". Which ones steer cleanly and which turn to mush?
- **Negative strength.** `--strengths 0 -0.4` pushes *away* from a concept. Does anti-rage make it saccharine?
- **Your own contrast.** Write 10 sentences in your voice and 10 in corporate voice, put each set in a file, and steer with the difference. (Today `--concept-file` compares against the bland templates; ask me to add a `--neutral-file` for true A-vs-B.)
- **Hunt a neuron.** Pick a random one (`neuron 18 4242`) and figure out what it's for. When the top hits look unrelated, ask "if I couldn't use nouns, what's the shared pattern?" Then write a sentence from your guess alone and see if it beats the corpus (the "blind maximum" test).

## How it works

`steer.py` swaps each decoder layer and MLP onto a subclass whose `__call__` checks a shared `HOOKS` dict. It records activations, adds vectors to the residual stream, and overwrites neuron values. Weights aren't touched. A concept direction is the average residual stream over 10 sentences containing the concept, minus the same 10 sentence frames filled with bland words ("stuff", "the schedule"), normalized and then scaled by the typical size of the stream at that layer. So strength 1.0 means "as big as everything else the model is thinking at that point", which is why it collapses there.

Qwen3.5 mixes two kinds of layers: three linear-attention layers (Gated DeltaNet), then one full-attention layer, repeating. Layers 3, 7, 11, 15, 19, 23, 27 and 31 are the full-attention ones. This probably doesn't matter for steering, but it would be worth checking.
