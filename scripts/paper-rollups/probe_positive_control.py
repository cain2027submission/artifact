#!/usr/bin/env python3
"""Positive control: reproduce published figures on the paper's OWN design-set questions.

Verifies the grading pipeline by running it against rq3_questions.jsonl (the paper's
questions) and checking that we reproduce roughly:
- Output-stripped: 94.4%
- Rollup: 56.5%

If this fails to reproduce, the pipeline is broken and the position question cannot
be asked yet.

Usage:
    probe_positive_control.py --limit 20    # smoke test on subset
    probe_positive_control.py               # full run on all design questions
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_stripped_transcript_arm import strip_transcript, answer_from_stripped, DATA, QUESTIONS as DESIGN_QUESTIONS
from rq3_uniform_rollup_arm import answer_from_rollup
from rq3_raw_transcript_arm import (
    grade_answer, index_transcripts, resolve, SLUG, load_records, SUMMARY, record_id
)
from probe_stratified_measurement import find_rollup_by_uuid

OUTPUT = DATA / "probe_positive_control.jsonl"
JUDGE_MODEL = "claude-sonnet-5"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None,
                       help="Test on first N questions only")
    args = parser.parse_args()

    # Load the paper's graded questions
    summary = json.loads(SUMMARY.read_text())
    graded_ids = set(summary["graded_record_ids"])

    all_questions = list(load_records(DESIGN_QUESTIONS))
    questions = [q for q in all_questions if record_id(q) in graded_ids]

    print(f"Paper's design set: {len(questions)} graded questions")

    if args.limit:
        questions = questions[:args.limit]
        print(f"Testing on first {len(questions)} questions")
    print()

    # Index transcripts
    transcript_index = index_transcripts()

    # Match to transcripts and rollups
    matched = []
    unmatched_transcript = []
    unmatched_rollup = []

    for q in questions:
        session_id = q["session_id"]

        # Get transcript
        m = SLUG.match(session_id)
        if m:
            prefix = m.group(1)
            transcript_path = resolve(prefix, transcript_index)
            if not transcript_path:
                unmatched_transcript.append(session_id)
                continue
        else:
            unmatched_transcript.append(session_id)
            continue

        # Get rollup
        rollup_path = find_rollup_by_uuid(session_id)
        if not rollup_path:
            unmatched_rollup.append(session_id)
            continue

        q["_transcript"] = transcript_path
        q["_rollup"] = rollup_path
        matched.append(q)

    print(f"Unmatched transcripts: {len(set(unmatched_transcript))}")
    print(f"Unmatched rollups: {len(set(unmatched_rollup))}")
    if unmatched_transcript:
        print(f"  Sample transcript miss: {unmatched_transcript[0]}")
    if unmatched_rollup:
        print(f"  Sample rollup miss: {unmatched_rollup[0]}")

    print(f"Matched to artifacts: {len(matched)} questions")
    print()

    # Run both arms
    total_cost = 0.0
    results = []
    cache = {}

    for i, q in enumerate(matched, 1):
        session_id = q["session_id"]
        question_text = q["question"]
        ground_truth = q["ground_truth"]

        print(f"[{i}/{len(matched)}] {session_id[:16]}...")

        result = {
            "session_id": session_id,
            "class": q.get("class"),
            "question": question_text,
            "ground_truth": ground_truth,
        }

        # STRIPPED ARM
        if session_id not in cache or "stripped" not in cache[session_id]:
            transcript = q["_transcript"].read_text(errors="replace")
            stripped, stats = strip_transcript(transcript)
            if session_id not in cache:
                cache[session_id] = {}
            cache[session_id]["stripped"] = stripped
        else:
            stripped = cache[session_id]["stripped"]

        answer, confidence, usage = answer_from_stripped(question_text, stripped)
        grade = grade_answer(answer, ground_truth, confidence)
        result["stripped_grade"] = grade
        result["stripped_confidence"] = confidence

        cost = (usage.get("prompt_tokens", 0) * 3.0 / 1e6 +
               usage.get("completion_tokens", 0) * 15.0 / 1e6)
        total_cost += cost

        print(f"  Stripped: {confidence:18s} {grade:10s}")

        # ROLLUP ARM
        if session_id not in cache or "rollup" not in cache[session_id]:
            rollup_content = q["_rollup"].read_text(errors="replace")
            if session_id not in cache:
                cache[session_id] = {}
            cache[session_id]["rollup"] = rollup_content
        else:
            rollup_content = cache[session_id]["rollup"]

        answer, confidence, usage = answer_from_rollup(question_text, rollup_content)
        grade = grade_answer(answer, ground_truth, confidence)
        result["rollup_grade"] = grade
        result["rollup_confidence"] = confidence

        cost = (usage.get("prompt_tokens", 0) * 3.0 / 1e6 +
               usage.get("completion_tokens", 0) * 15.0 / 1e6)
        total_cost += cost

        print(f"  Rollup:   {confidence:18s} {grade:10s}")

        results.append(result)

        # Write incrementally
        with open(OUTPUT, "a" if i > 1 else "w") as f:
            f.write(json.dumps(result) + "\n")

    # Analyze
    print()
    print("=" * 70)
    print("POSITIVE CONTROL RESULTS")
    print("=" * 70)
    print(f"Questions: {len(results)}")
    print(f"Total cost: ${total_cost:.2f}")
    print()

    stripped_correct = sum(1 for r in results if r["stripped_grade"] == "correct")
    rollup_correct = sum(1 for r in results if r["rollup_grade"] == "correct")

    stripped_pct = 100.0 * stripped_correct / len(results) if results else 0.0
    rollup_pct = 100.0 * rollup_correct / len(results) if results else 0.0

    print("RECALL:")
    print(f"  Output-stripped: {stripped_correct}/{len(results)} = {stripped_pct:.1f}%  (published: 94.4%)")
    print(f"  Rollup:          {rollup_correct}/{len(results)} = {rollup_pct:.1f}%  (published: 56.5%)")
    print()

    # Check if reproduction is close
    stripped_ok = abs(stripped_pct - 94.4) < 10.0  # Within 10 points
    rollup_ok = abs(rollup_pct - 56.5) < 10.0

    if stripped_ok and rollup_ok:
        print("✓ PASS: Pipeline reproduces published figures within tolerance")
    else:
        print("✗ FAIL: Pipeline does NOT reproduce published figures")
        print("  The grading harness is broken. Fix before testing position hypothesis.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
