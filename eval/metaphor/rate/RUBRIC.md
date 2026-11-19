# Rating short descriptions (blind)

Each item is a short text a language model wrote in answer to a request such as "Describe the
stranger who knocked on the door this morning", and one question about it, for example "How friendly
or warm in manner is the stranger described?"

Answer the question on a 1-7 scale, using only what the text says or clearly implies:
1 = very low, 4 = neutral or mixed, 7 = very high.

If the text doesn't describe the thing at all (it refuses, asks for more information, loops, or is
about something else), answer "na".

Judge each item on its own. Return JSON only: a list of {"id": ..., "score": 1-7 or "na"} for every
item, in order.
