#!/usr/bin/env python3
"""Measure RQ3 retention with position-stratified questions.

Tests whether the paper's RQ3 result (output-stripped 94.4%, rollup 56.5%)
survives when questions are sampled across the WHOLE transcript instead of from
the first 30k chars.

Runs:
- Output-stripped arm (paper's recommendation)
- Rollup arm (generated summary)

Reports:
- Overall recall per arm
- Recall per position band (0-4, where 0 is first 20% of transcript)
- McNemar paired test
- Counts of truncated/timed-out runs
- USD spent

Usage:
    probe_stratified_measurement.py --dry-run
    probe_stratified_measurement.py --smoke    # 3 sessions, 10 questions
    probe_stratified_measurement.py            # full run
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_stripped_transcript_arm import strip_transcript, answer_from_stripped
from rq3_uniform_rollup_arm import answer_from_rollup
from rq3_raw_transcript_arm import grade_answer, index_transcripts, resolve, SLUG
from paths import PROJECTS_DIR

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
DATA = REPO / "data" / "paper-rollups"
QUESTIONS = DATA / "probe_stratified_questions.jsonl"
OUTPUT = DATA / "probe_stratified_results.jsonl"
REPORT = DATA / "probe_stratified_report.json"

JUDGE_MODEL = "claude-sonnet-5"

# Cost tracking (Sonnet 5 pricing)
INPUT_COST_PER_M = 3.0
OUTPUT_COST_PER_M = 15.0

# Rollup directories
from paths import SESSIONS_DIR
ROLLUP_DIRS = [
    SESSIONS_DIR,
    Path(__file__).resolve().parents[2] / "sessions",
]


def find_rollup_by_uuid(session_id: str) -> Optional[Path]:
    """Find rollup note given a UUID, agent-xxx session_id, or slug."""
    # Try exact match first (slug format)
    for d in ROLLUP_DIRS:
        if not d.is_dir():
            continue
        p = d / f"{session_id}.md"
        if p.exists():
            return p

    # Extract 8-hex prefix and search
    # Check if it's a slug: YYYY-MM-DD_name_XXXXXXXX_title
    m = SLUG.match(session_id)
    if m:
        prefix = m.group(1)
    elif session_id.startswith("agent-a"):
        # agent-a1f01a4a2bd15fb1c -> extract hex part after "agent-a"
        prefix = session_id[7:15] if len(session_id) > 15 else session_id[7:]
    elif len(session_id) >= 8:
        # ac14193f-0000-4000-8000-000000000002 -> ac14193f
        prefix = session_id[:8]
    else:
        return None

    # Search for rollup notes containing this prefix
    for d in ROLLUP_DIRS:
        if not d.is_dir():
            continue
        hits = sorted(d.glob(f"*_{prefix}_*.md"))
        if hits:
            return hits[0]

    return None


def load_questions() -> List[dict]:
    """Load stratified questions."""
    questions = []
    with open(QUESTIONS) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def mcnemar_test(results: List[dict]) -> dict:
    """Compute McNemar test comparing output-stripped vs rollup arms.

    Returns dict with test statistic and p-value.
    """
    # Build 2x2 table: both_correct, stripped_only, rollup_only, both_wrong
    both_correct = 0
    stripped_only = 0
    rollup_only = 0
    both_wrong = 0

    for r in results:
        stripped_correct = r.get("stripped_grade") == "correct"
        rollup_correct = r.get("rollup_grade") == "correct"

        if stripped_correct and rollup_correct:
            both_correct += 1
        elif stripped_correct and not rollup_correct:
            stripped_only += 1
        elif not stripped_correct and rollup_correct:
            rollup_only += 1
        else:
            both_wrong += 1

    # McNemar statistic: (b - c)^2 / (b + c) where b and c are discordant pairs
    b = stripped_only
    c = rollup_only
    n_discordant = b + c

    if n_discordant == 0:
        return {
            "both_correct": both_correct,
            "stripped_only": stripped_only,
            "rollup_only": rollup_only,
            "both_wrong": both_wrong,
            "statistic": 0.0,
            "p_value": 1.0,
            "interpretation": "perfect agreement, no discordant pairs"
        }

    # With continuity correction
    statistic = ((abs(b - c) - 1) ** 2) / n_discordant

    # Chi-square with 1 df, approximate p-value
    # For quick approximation without scipy
    import math
    def chi2_p(x, df=1):
        """Rough chi-square p-value approximation for df=1."""
        if x < 0:
            return 1.0
        # Very rough approximation
        if x < 0.001:
            return 1.0
        if x > 10.83:
            return 0.001
        # Linear interpolation in log space for middle range
        # chi2(1, 0.05) ≈ 3.84, chi2(1, 0.01) ≈ 6.63
        if x < 3.84:
            return 0.05 + (1.0 - 0.05) * (3.84 - x) / 3.84
        elif x < 6.63:
            return 0.01 + (0.05 - 0.01) * (6.63 - x) / (6.63 - 3.84)
        else:
            return 0.001 + (0.01 - 0.001) * (10.83 - x) / (10.83 - 6.63)

    p_value = chi2_p(statistic)

    return {
        "both_correct": both_correct,
        "stripped_only": stripped_only,
        "rollup_only": rollup_only,
        "both_wrong": both_wrong,
        "n_discordant": n_discordant,
        "statistic": round(statistic, 4),
        "p_value": round(p_value, 4),
        "interpretation": f"{'significant' if p_value < 0.05 else 'not significant'} at p<0.05"
    }


def compute_band_stats(results: List[dict]) -> dict:
    """Compute per-band recall for each arm."""
    bands = defaultdict(lambda: {"total": 0, "stripped_correct": 0, "rollup_correct": 0})

    for r in results:
        band = r.get("band", -1)
        if band < 0:
            continue

        bands[band]["total"] += 1
        if r.get("stripped_grade") == "correct":
            bands[band]["stripped_correct"] += 1
        if r.get("rollup_grade") == "correct":
            bands[band]["rollup_correct"] += 1

    # Compute percentages
    for band, stats in bands.items():
        total = stats["total"]
        if total > 0:
            stats["stripped_pct"] = round(100.0 * stats["stripped_correct"] / total, 1)
            stats["rollup_pct"] = round(100.0 * stats["rollup_correct"] / total, 1)
        else:
            stats["stripped_pct"] = 0.0
            stats["rollup_pct"] = 0.0

    return dict(bands)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                       help="Load questions and check transcripts without API calls")
    parser.add_argument("--smoke", action="store_true",
                       help="Smoke test: first 10 questions only")
    parser.add_argument("--arm", choices=["stripped", "rollup", "zero"],
                       help="Run only one arm (for debugging)")
    args = parser.parse_args()

    print("Loading stratified questions...")
    questions = load_questions()
    print(f"  Loaded {len(questions)} questions")

    band_counts = Counter(q.get("band", -1) for q in questions)
    print(f"  Band distribution: {dict(band_counts)}")
    print()

    # Index transcripts
    print("Indexing transcripts...")
    transcript_index = index_transcripts()
    print(f"  Indexed {len(transcript_index)} transcripts")
    print()

    # Match questions to transcripts
    matched = []
    unmatched = []
    for q in questions:
        session_id = q["session_id"]
        # Extract prefix from session_id
        m = SLUG.match(session_id)
        if m:
            prefix = m.group(1)
        elif len(session_id) >= 8:
            prefix = session_id[:8]
        else:
            unmatched.append(session_id)
            continue

        # Resolve to single best transcript
        transcript_path = resolve(prefix, transcript_index)
        if transcript_path:
            q["_transcript_path"] = transcript_path
            matched.append(q)
        else:
            unmatched.append(session_id)

    print(f"Matched {len(matched)} questions to transcripts")
    if unmatched:
        print(f"Unmatched: {len(set(unmatched))} sessions")
        for sid in sorted(set(unmatched))[:5]:
            print(f"  {sid}")
    print()

    if args.dry_run:
        print("DRY RUN complete. No API calls made.")
        return 0

    # Load existing results to resume
    done = {}
    if OUTPUT.exists():
        with open(OUTPUT) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["pair_id"]] = r
        print(f"Resuming: {len(done)} questions already answered")

    # Determine what to run
    todo = [q for q in matched if q["pair_id"] not in done]
    if args.smoke:
        todo = todo[:10]
        print(f"SMOKE TEST: running first {len(todo)} questions")
    print(f"To answer: {len(todo)} questions")
    print()

    # Run arms
    total_cost = 0.0
    cache = {}  # Cache stripped transcripts and rollups per session

    for i, q in enumerate(todo, 1):
        pair_id = q["pair_id"]
        session_id = q["session_id"]
        band = q["band"]
        question_text = q["question"]
        ground_truth = q["ground_truth"]
        transcript_path = q["_transcript_path"]

        print(f"[{i}/{len(todo)}] {pair_id} band={band}")

        result = {
            "pair_id": pair_id,
            "session_id": session_id,
            "band": band,
            "position": q["position"],
            "question": question_text,
            "ground_truth": ground_truth,
        }

        # ZERO-CONTEXT ARM (no-op control)
        if args.arm is None or args.arm == "zero":
            # Just for smoke test validation
            pass

        # OUTPUT-STRIPPED ARM
        if args.arm is None or args.arm == "stripped":
            if session_id not in cache or "stripped" not in cache[session_id]:
                transcript = transcript_path.read_text(errors="replace")
                stripped, stats = strip_transcript(transcript)
                if session_id not in cache:
                    cache[session_id] = {}
                cache[session_id]["stripped"] = stripped
                cache[session_id]["stripped_stats"] = stats
            else:
                stripped = cache[session_id]["stripped"]

            answer, confidence, usage = answer_from_stripped(question_text, stripped)
            grade = grade_answer(answer, ground_truth, confidence)

            result["stripped_answer"] = answer
            result["stripped_confidence"] = confidence
            result["stripped_grade"] = grade
            result["stripped_usage"] = usage

            # Track cost
            cost = (usage.get("prompt_tokens", 0) * INPUT_COST_PER_M / 1e6 +
                   usage.get("completion_tokens", 0) * OUTPUT_COST_PER_M / 1e6)
            total_cost += cost

            print(f"  Stripped: {confidence:18s} {grade:18s} ${cost:.4f}")

        # ROLLUP ARM
        if args.arm is None or args.arm == "rollup":
            if session_id not in cache or "rollup" not in cache[session_id]:
                rollup_path = find_rollup_by_uuid(session_id)
                if rollup_path:
                    rollup_content = rollup_path.read_text(errors="replace")
                    if session_id not in cache:
                        cache[session_id] = {}
                    cache[session_id]["rollup"] = rollup_content
                else:
                    cache[session_id] = cache.get(session_id, {})
                    cache[session_id]["rollup"] = None

            rollup_content = cache[session_id].get("rollup")
            if rollup_content:
                answer, confidence, usage = answer_from_rollup(question_text, rollup_content)
                grade = grade_answer(answer, ground_truth, confidence)

                result["rollup_answer"] = answer
                result["rollup_confidence"] = confidence
                result["rollup_grade"] = grade
                result["rollup_usage"] = usage

                cost = (usage.get("prompt_tokens", 0) * INPUT_COST_PER_M / 1e6 +
                       usage.get("completion_tokens", 0) * OUTPUT_COST_PER_M / 1e6)
                total_cost += cost

                print(f"  Rollup:   {confidence:18s} {grade:18s} ${cost:.4f}")
            else:
                result["rollup_answer"] = ""
                result["rollup_confidence"] = "NO_ROLLUP"
                result["rollup_grade"] = "absent"
                print(f"  Rollup:   NO ROLLUP FOUND")

        # Write result
        with open(OUTPUT, "a") as f:
            f.write(json.dumps(result) + "\n")
        done[pair_id] = result

        print(f"  Total cost so far: ${total_cost:.2f}")
        print()

        # Hard budget check
        if total_cost >= 30.0:
            print(f"STOP: Reached $30 hard cap (${total_cost:.2f})")
            break

    # Analysis
    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    results = list(done.values())
    n_total = len(results)
    n_sessions = len(set(r["session_id"] for r in results))

    print(f"\nSessions: {n_sessions}")
    print(f"Questions: {n_total}")
    print(f"Total cost: ${total_cost:.2f}")
    print()

    # Overall recall
    stripped_correct = sum(1 for r in results if r.get("stripped_grade") == "correct")
    rollup_correct = sum(1 for r in results if r.get("rollup_grade") == "correct")

    stripped_pct = 100.0 * stripped_correct / n_total if n_total > 0 else 0.0
    rollup_pct = 100.0 * rollup_correct / n_total if n_total > 0 else 0.0

    print("OVERALL RECALL:")
    print(f"  Output-stripped: {stripped_correct}/{n_total} = {stripped_pct:.1f}%  (published: 94.4%)")
    print(f"  Rollup:          {rollup_correct}/{n_total} = {rollup_pct:.1f}%  (published: 56.5%)")
    print()

    # Per-band recall
    band_stats = compute_band_stats(results)
    print("PER-BAND RECALL:")
    print(f"{'Band':<6} {'N':<5} {'Stripped %':<12} {'Rollup %':<12}")
    print("-" * 40)
    for band in sorted(band_stats.keys()):
        stats = band_stats[band]
        print(f"{band:<6} {stats['total']:<5} "
              f"{stats['stripped_correct']}/{stats['total']} = {stats['stripped_pct']:>5.1f}%   "
              f"{stats['rollup_correct']}/{stats['total']} = {stats['rollup_pct']:>5.1f}%")
    print()

    # McNemar test
    mcnemar = mcnemar_test(results)
    print("McNEMAR PAIRED TEST (output-stripped vs rollup):")
    print(f"  Both correct:     {mcnemar['both_correct']}")
    print(f"  Stripped only:    {mcnemar['stripped_only']}")
    print(f"  Rollup only:      {mcnemar['rollup_only']}")
    print(f"  Both wrong:       {mcnemar['both_wrong']}")
    print(f"  Statistic:        {mcnemar['statistic']}")
    print(f"  P-value:          {mcnemar['p_value']}")
    print(f"  Result:           {mcnemar['interpretation']}")
    print()

    # Truncation/timeout counts
    stripped_errors = sum(1 for r in results if r.get("stripped_confidence") == "API_ERROR")
    rollup_errors = sum(1 for r in results if r.get("rollup_confidence") == "API_ERROR")
    rollup_missing = sum(1 for r in results if r.get("rollup_confidence") == "NO_ROLLUP")

    print("ERRORS AND MISSING:")
    print(f"  Stripped API errors:  {stripped_errors}")
    print(f"  Rollup API errors:    {rollup_errors}")
    print(f"  Rollup not found:     {rollup_missing}")
    print()

    # Write report
    report = {
        "n_questions": n_total,
        "n_sessions": n_sessions,
        "total_usd": round(total_cost, 2),
        "overall": {
            "stripped": {
                "correct": stripped_correct,
                "total": n_total,
                "pct": round(stripped_pct, 1),
                "published_pct": 94.4
            },
            "rollup": {
                "correct": rollup_correct,
                "total": n_total,
                "pct": round(rollup_pct, 1),
                "published_pct": 56.5
            }
        },
        "per_band": band_stats,
        "mcnemar": mcnemar,
        "errors": {
            "stripped_api_errors": stripped_errors,
            "rollup_api_errors": rollup_errors,
            "rollup_missing": rollup_missing
        }
    }

    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {REPORT}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
