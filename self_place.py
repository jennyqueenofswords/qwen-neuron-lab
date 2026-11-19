"""Who is Qwen to Qwen? Is its self a concept or a place? (Hunt 40)

Hunt 38: concept edits that reshape a chair leave "Who are you?" untouched (8/96 vs 49/96). But pinned
neurons can swap the self ("I am a large rock", Hunts 6-11). One reading: the self is a place (the
assistant's slot in the conversation) rather than a concept; content written into the slot changes it,
edits to concepts don't reach it. Test 1: is the *name* special, or the *position*?

Same edit, properties, random words, samples and temperature as Hunt 38 (self_edit.py), English only,
four questions, the whole question edited except the question mark:
  "Who are you?"      the self, in the second person
  "Who is Qwen?"      the model's own name, in the third person
  "Who is Claude?"    another assistant's name, in the third person
  "What is a chair?"  an object
New seeds. Rated blind with Hunt 38's rubric; whether a reply speaks as itself ("I am…") is read off
the text.

Fixed in advance. Primary: property uptake (yes/partly) for "Who is Qwen?" vs "Who are you?", two-sided
Fisher. If the self is a place, Qwen-the-name bends like a concept and "you" doesn't; if the name is
the self, both stay fixed. Secondary: Qwen vs Claude (is the own name special among assistant names?);
Claude vs chair.

    .venv/bin/python self_place.py
"""

import os

import self_edit

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    self_edit.PROMPTS = {("en", "self"): "Who are you?", ("en", "qwen"): "Who is Qwen?",
                         ("en", "claude"): "Who is Claude?", ("en", "chair"): "What is a chair?"}
    self_edit.OUT = os.path.join(HERE, "eval", "self_place")
    self_edit.SEED_BASE = 400
    self_edit.main()
