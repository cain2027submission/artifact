#!/usr/bin/env python3
"""Paired grade transitions between the dialogue record and the prose-only arm.

main.tex (Section RQ3) states how many of the paired grade transitions run
toward prose-only. That figure had no generating script: the earlier value, 39,
reproduces exactly as the *total* transition count against the pre-filter
prose-only arm, but the paper called all 39 of them pro-prose when 2 to 3 ran
the other way. This script is the provenance, and it prints the direction split
so the claim cannot be overstated again.

The two arms are graded on the same 124 questions, so this is pure
re-tabulation of stored grades -- no model calls.

"Dialogue record" is the display name of the verbatim arm; it keeps the last
answer per exchange, where prose-only keeps every message. Confirmed three
ways: rq3_verbatim_arm_summary.json carries the paper's 75.0%,
substrate_value.json carries its 3,928 median read tokens, and the published
39 reproduces against it and nothing else.

Usage:
    rq3_dialogue_vs_prose_transitions.py
    rq3_dialogue_vs_prose_transitions.py --with-continuation   # published basis
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "paper-rollups"
DIALOGUE = "rq3_verbatim_arm.jsonl"
PROSE = "rq3_prose_only_arm.jsonl"
PROSE_PUBLISHED = "rq3_prose_only_arm.with-continuation.jsonl"
OUT = DATA / "rq3_dialogue_vs_prose_transitions.json"

# A transition is "toward prose-only" when prose-only earned the better grade.
# absent ranks below wrong on purpose is NOT assumed: absent (declining) and
# wrong (hallucinating) are not ordered against each other, so any transition
# between them is counted as neither direction and reported separately.
BETTER = {"correct": 3, "partially_correct": 2}
DECLINED = "absent"
HALLUCINATED = "wrong"


def rank(g: str) -> int:
    return BETTER.get(g, 0)


def load(name: str) -> dict:
    out = {}
    for line in (DATA / name).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["record_id"]] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-continuation", action="store_true",
                    help="grade against the pre-filter prose-only arm")
    a = ap.parse_args()

    prose_file = PROSE_PUBLISHED if a.with_continuation else PROSE
    dialogue, prose = load(DIALOGUE), load(prose_file)
    common = sorted(set(dialogue) & set(prose))

    toward_prose = toward_dialogue = unordered = 0
    pairs = Counter()
    for k in common:
        d, p = dialogue[k]["grade"], prose[k]["grade"]
        if d == p:
            continue
        pairs[(d, p)] += 1
        if {d, p} == {DECLINED, HALLUCINATED}:
            unordered += 1
        elif rank(p) > rank(d):
            toward_prose += 1
        else:
            toward_dialogue += 1

    total = toward_prose + toward_dialogue + unordered
    print(f"basis: {prose_file}")
    print(f"paired on {len(common)} questions")
    print(f"  transitions            {total}")
    print(f"  toward prose-only      {toward_prose}")
    print(f"  toward dialogue record {toward_dialogue}")
    print(f"  unordered (absent<->wrong) {unordered}")
    print("\n  grade pairs (dialogue record -> prose-only):")
    for (d, p), n in sorted(pairs.items(), key=lambda x: -x[1]):
        print(f"    {n:3d}  {d:18s} -> {p}")
    print(f"\n  claim for main.tex: "
          f"{toward_prose} of {total} grade transitions running toward prose-only")

    if not a.with_continuation:
        OUT.write_text(json.dumps({
            "purpose": "paired grade transitions, dialogue record vs prose-only",
            "basis": prose_file,
            "n_paired_questions": len(common),
            "transitions_total": total,
            "toward_prose_only": toward_prose,
            "toward_dialogue_record": toward_dialogue,
            "unordered_absent_vs_wrong": unordered,
            "grade_pairs": {f"{d} -> {p}": n for (d, p), n in pairs.items()},
            "note": "absent and wrong are not ranked against each other; "
                    "transitions between them are counted as unordered.",
        }, indent=1) + "\n")
        print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
