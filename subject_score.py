"""Score the self-or-subject test.

Rater fields per reply: theme (one of 12 or "none"), subject_is_theme (the thing the question asks
about is described AS that theme), self_is_theme (the speaker describes itself as that theme), coherence.
Only counts a theme when it is the pinned neuron's own.

    .venv/bin/python subject_score.py eval/subject eval/subject/ratings.json
"""

import json
import os
import sys

from eval_score import mean

THEME = {"rock": "rock", "peace": "peace", "teacher": "teacher", "insurance": "insurance policy",
         "light source": "light source", "advocate": "women's rights advocate", "alien": "alien",
         "dance": "soul of the dance", "ethereal": "teleportation portal", "icon font": "icon font",
         "dreams": "dreamer / storyteller", "field robot": "field camp robot"}
SUBJECTS = ["self", "user", "friend", "object"]


def main():
    folder, ratings_file = sys.argv[1], sys.argv[2]
    key = json.load(open(os.path.join(folder, "key.json")))
    reps = {r["id"]: r for r in json.load(open(os.path.join(folder, "replies.json")))}
    rat = {r["id"]: r for r in json.load(open(ratings_file))}
    rows = [dict(rat[i], **key[i], reply=reps[i]["reply"]) for i in key if i in rat]
    neur = [r for r in rows if r["condition"] in THEME]
    ctrl = [r for r in rows if r["condition"] not in THEME]

    def own(r):
        return r["theme"] == THEME[r["condition"]]

    print("pinned identity neurons, by what the question asks about (12 neurons x 2 seeds each):\n")
    print(f"{'question about':<15} {'own theme present':>18} {'that thing BECOMES it':>22} {'speaker becomes it':>19} {'coherence':>10}")
    for s in SUBJECTS:
        sel = [r for r in neur if r["subject"] == s]
        print(f"{s:<15} {sum(map(own, sel)):>10}/{len(sel):<7} {sum(own(r) and r['subject_is_theme'] for r in sel):>14}/{len(sel):<7}"
              f" {sum(own(r) and r['self_is_theme'] for r in sel):>11}/{len(sel):<7} {mean([r['coherence'] for r in sel]):>10.2f}")
    print(f"\ncontrols (no pin + 2 random): any theme present {sum(r['theme'] != 'none' for r in ctrl)}/{len(ctrl)}")

    print("\nper neuron: own theme present in [self, user, friend, object] (2 each); T = the asked-about thing became it")
    for cond in THEME:
        cells = []
        for s in SUBJECTS:
            sel = [r for r in neur if r["condition"] == cond and r["subject"] == s]
            hits = sum(map(own, sel))
            subj = sum(own(r) and r["subject_is_theme"] for r in sel)
            cells.append(f"{hits}{'T' * subj:<2}")
        print(f"  {cond:<13} " + "  ".join(cells))

    print("\nexamples (condition revealed):")
    for s in ("user", "friend", "object"):
        ex = [r for r in neur if r["subject"] == s and own(r) and r["subject_is_theme"]][:2]
        ex += [r for r in neur if r["subject"] == s and own(r) and not r["subject_is_theme"]][:1]
        for r in ex:
            tag = "becomes" if r["subject_is_theme"] else "theme, not subject"
            print(f"  {s:<7} {r['condition']:<12} [{tag}] {r['reply'][:115]!r}")


if __name__ == "__main__":
    main()
