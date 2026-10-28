#!/usr/bin/env python3
"""Probe-overlap bias bound for the rollup vs output-stripped RQ3 pair.

The paper discloses mean probe-token overlap of 0.902 against the raw transcript
and 0.705 against the rollup, a 0.197 gap. The prose-vs-call regrouping found
that inside the tie set (38 probes with exact zero overlap difference), prose-only
beats call-only by 23.7 points rather than 43.1 uncorrected.

This script applies the same tie-set method to the central comparison: rollup
(56.5% recoverable) vs output-stripped transcript (90.3% recoverable), a 33.8-point
gap. It computes per-question overlap against both contexts, identifies the tie set,
and reports the gap within it.

Method reused verbatim from rq3_arm_probe_overlap.py:
  * overlap -- rq4_known_item.compute_fairness_stats, containment over casefolded
    TOKEN_RE sets, no stopword removal
  * grades -- from rq3_confidence_corrected.json, single-draw corrected grades

Usage:
    rq3_rollup_stripped_overlap.py             # writes JSON report
    rq3_rollup_stripped_overlap.py --no-write  # print only
"""

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR  # noqa: E402

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
RECOVERABLE = ("correct", "partially_correct")

# Input files
CONFIDENCE_CORRECTED = DATA_DIR / "rq3_confidence_corrected.json"
OUT = DATA_DIR / "rq3_rollup_stripped_overlap.json"


def compute_overlap(question: str, context: str) -> float:
    """Compute containment: |question_tokens & context_tokens| / |question_tokens|."""
    q_tokens = set(TOKEN_RE.findall(question.lower()))
    if not q_tokens:
        return 0.0
    c_tokens = set(TOKEN_RE.findall(context.lower()))
    return len(q_tokens & c_tokens) / len(q_tokens)


def load_contexts_for_record(record_id: str, transcript_path: str) -> tuple[str, str]:
    """Load rollup and stripped contexts from a transcript.

    Returns (rollup_context, stripped_context).
    For this initial implementation, we use the persisted context from the arm files.
    """
    # Load from the arm files which have the contexts already built
    stripped_file = DATA_DIR / "rq3_questions_stripped_arm.jsonl"
    # Rollup arm uses rq3_questions.jsonl which answers from the rollup note
    rollup_file = DATA_DIR / "rq3_questions.jsonl"

    stripped_ctx = None
    rollup_ctx = None

    for line in stripped_file.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("record_id") == record_id:
            # The stripped arm file contains the answer generated from stripped context
            # We need to reconstruct the context. For now, use a placeholder.
            # Actually, we need to rebuild this from the transcript.
            stripped_ctx = "<need to rebuild from transcript>"
            break

    for line in rollup_file.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("record_id") == record_id:
            rollup_ctx = "<need to rebuild from rollup note>"
            break

    return rollup_ctx, stripped_ctx


def rate(grades):
    """Compute recoverable rate."""
    n = len(grades)
    if n == 0:
        return None, 0, 0
    k = sum(1 for g in grades if g in RECOVERABLE)
    return 100.0 * k / n, k, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    # Load confidence-corrected grades
    corrected = json.loads(CONFIDENCE_CORRECTED.read_text())

    # Get the single-draw corrected grades for both arms
    rollup_arm = corrected["arms"]["uniform"]
    stripped_arm = corrected["arms"]["stripped"]

    print(f"Rollup arm (uniform): {rollup_arm['n_records']} records")
    print(f"Stripped arm: {stripped_arm['n_records']} records")
    print(f"Rollup recoverable: {rollup_arm['single_draw']['corrected']['recoverable']['pct']}%")
    print(f"Stripped recoverable: {stripped_arm['single_draw']['corrected']['recoverable']['pct']}%")

    # Load per-record data
    # We need to compute overlap for each question
    # But this requires loading the actual contexts, which means:
    # 1. For rollup: load the rollup note
    # 2. For stripped: rebuild the stripped context from the transcript

    # For now, report that this needs the context-building step
    print("\n## IMPLEMENTATION NOTE")
    print("This script needs to rebuild contexts from transcripts to compute overlap.")
    print("The method from rq3_arm_probe_overlap.py lines 110-138 shows how:")
    print("  1. Load transcript paths from arm files")
    print("  2. Rebuild contexts using the arm-specific builders")
    print("  3. Compute overlap per question")
    print("  4. Find tie set and report gap")
    print("\nThis requires importing the context builders, which I'll implement next.")

    if not args.no_write:
        report = {
            "status": "incomplete",
            "note": "context rebuilding not yet implemented",
            "published_rates": {
                "rollup_recoverable_pct": rollup_arm["single_draw"]["corrected"]["recoverable"]["pct"],
                "stripped_recoverable_pct": stripped_arm["single_draw"]["corrected"]["recoverable"]["pct"],
                "gap_pp": (stripped_arm["single_draw"]["corrected"]["recoverable"]["pct"] -
                          rollup_arm["single_draw"]["corrected"]["recoverable"]["pct"])
            }
        }
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
