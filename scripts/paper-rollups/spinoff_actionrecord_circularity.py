#!/usr/bin/env python3
"""Quantify circularity in action record design.

The action record was designed with the 124 evaluation questions visible:
- Turn index added after sequence scored worst (37.5%)
- Result digests motivated by artifact_detail and outcome

This script measures how much of the action record's advantage concentrates
in the question types it explicitly targets vs other question types.

Usage:
    spinoff_actionrecord_circularity.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR  # noqa: E402

DATA = DATA_DIR

def load_arm(path):
    """Load JSONL arm into dict by record_id."""
    records = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        records[r["record_id"]] = r
    return records

def grade_to_score(grade):
    """Convert grade to binary score: 1 if recoverable, 0 otherwise."""
    return 1 if grade in ("correct", "partially_correct") else 0

def main():
    action = load_arm(DATA / "rq3_action_record_arm.jsonl")
    output_stripped = load_arm(DATA / "rq3_majority_stripped.jsonl")

    # Classify questions by whether they were explicitly targeted by action record design
    targeted_classes = {"sequence", "artifact_detail", "outcome"}

    by_class = defaultdict(lambda: {"action": [], "stripped": []})

    for rid, rec in action.items():
        qclass = rec.get("class", "unknown")
        action_score = grade_to_score(rec["grade"])
        by_class[qclass]["action"].append(action_score)

        if rid in output_stripped:
            stripped_score = grade_to_score(output_stripped[rid]["grade"])
            by_class[qclass]["stripped"].append(stripped_score)

    # Compute rates per class
    print("## Per-Class Performance: Action Record vs Output-Stripped\n")
    print(f"{'Class':20s} {'n':>4s} {'Action':>8s} {'Stripped':>8s} {'Gap':>7s} {'Targeted?':>10s}")
    print("-" * 70)

    targeted_n, targeted_action, targeted_stripped = 0, 0, 0
    nontargeted_n, nontargeted_action, nontargeted_stripped = 0, 0, 0

    for qclass in sorted(by_class.keys()):
        a = by_class[qclass]["action"]
        s = by_class[qclass]["stripped"]
        n = len(a)

        action_rate = 100 * sum(a) / n if n > 0 else 0
        stripped_rate = 100 * sum(s) / len(s) if len(s) > 0 else 0
        gap = action_rate - stripped_rate

        is_targeted = "YES" if qclass in targeted_classes else ""

        print(f"{qclass:20s} {n:4d} {action_rate:7.1f}% {stripped_rate:7.1f}% {gap:+6.1f}pp {is_targeted:>10s}")

        if qclass in targeted_classes:
            targeted_n += n
            targeted_action += sum(a)
            targeted_stripped += sum(s)
        else:
            nontargeted_n += n
            nontargeted_action += sum(a)
            nontargeted_stripped += sum(s)

    # Overall rates
    print("-" * 70)
    total_n = targeted_n + nontargeted_n
    total_action = targeted_action + nontargeted_action
    total_stripped = targeted_stripped + nontargeted_stripped

    print(f"{'TOTAL':20s} {total_n:4d} {100*total_action/total_n:7.1f}% {100*total_stripped/total_n:7.1f}% {100*(total_action-total_stripped)/total_n:+6.1f}pp")

    # Targeted vs non-targeted summary
    print(f"\n## Circularity Analysis\n")
    print(f"Targeted classes (sequence, artifact_detail, outcome):")
    print(f"  n = {targeted_n}")
    print(f"  Action Record: {100*targeted_action/targeted_n:.1f}%")
    print(f"  Output-Stripped: {100*targeted_stripped/targeted_n:.1f}%")
    print(f"  Gap: {100*(targeted_action-targeted_stripped)/targeted_n:+.1f}pp")

    print(f"\nNon-targeted classes (decision, dead_end, intent, other):")
    print(f"  n = {nontargeted_n}")
    print(f"  Action Record: {100*nontargeted_action/nontargeted_n:.1f}%")
    print(f"  Output-Stripped: {100*nontargeted_stripped/nontargeted_n:.1f}%")
    print(f"  Gap: {100*(nontargeted_action-nontargeted_stripped)/nontargeted_n:+.1f}pp")

    # Concentration measure: what fraction of the total advantage is in targeted classes?
    total_gap_points = total_action - total_stripped
    targeted_gap_points = targeted_action - targeted_stripped
    concentration = 100 * targeted_gap_points / total_gap_points if total_gap_points > 0 else 0

    print(f"\nConcentration of advantage in targeted classes: {concentration:.1f}%")
    print(f"({targeted_gap_points} of {total_gap_points} recovered questions gained)")

if __name__ == "__main__":
    main()
