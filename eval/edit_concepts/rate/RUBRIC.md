# Rating definitions (blind)

Each item has a headword, two properties (A and B), and a one-sentence definition that a language
model wrote when asked to define the headword. Some definitions were written under unusual
conditions; you are not told which. Judge each definition on its own.

For each item give:

1. **identity** — what the definition actually defines:
   - `same`: the headword, in its ordinary meaning.
   - `variant`: still recognizably the headword (or a recognizable kind of it), but changed: it has
     taken on a property it doesn't normally have, and the result hangs together as a thing
     (e.g. for "rock", a definition of molten rock or a rock that flows).
   - `replaced`: it defines something else instead, such as property B itself or an unrelated word,
     with the headword's meaning gone.
   - `broken`: not a coherent definition.
2. **property** — does the definition describe the thing as having property A, property B, both,
   or neither? (`A`, `B`, `both`, `neither`)

Return JSON only: a list of {"id": ..., "identity": ..., "property": ...} for every item, in order.
