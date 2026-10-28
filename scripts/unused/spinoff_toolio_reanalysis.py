#!/usr/bin/env python3
"""
Reanalysis for tool I/O spinoff paper:
(a) Paired comparison of raw vs output-stripped on the 89 shared questions
(b) Tokens-per-point table for all content classes
(c) Per-question-type breakdown of where call-only fails vs prose-only
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from scipy import stats

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "paper-rollups"

def load_jsonl(path):
    """Load JSONL file as list of dicts"""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def grade_to_binary(grade):
    """Convert grade to binary: correct/partially_correct = 1, else 0"""
    return 1 if grade in ["correct", "partially_correct"] else 0

def main():
    print("=" * 80)
    print("TOOL I/O SPINOFF REANALYSIS")
    print("=" * 80)

    # Load all question files
    questions_all = load_jsonl(DATA_DIR / "rq3_questions.jsonl")
    questions_raw = load_jsonl(DATA_DIR / "rq3_questions_raw_arm.jsonl")
    questions_stripped = load_jsonl(DATA_DIR / "rq3_questions_stripped_arm.jsonl")

    # Load all arm grade files
    arms = {}
    for arm_name in ["raw", "stripped", "tool_inputs_only", "prose_only", "action_record", "answer_only"]:
        try:
            if arm_name == "raw":
                arms[arm_name] = load_jsonl(DATA_DIR / "rq3_questions_raw_arm.jsonl")
            elif arm_name == "stripped":
                arms[arm_name] = load_jsonl(DATA_DIR / "rq3_questions_stripped_arm.jsonl")
            elif arm_name == "tool_inputs_only":
                arms[arm_name] = load_jsonl(DATA_DIR / "rq3_tool_inputs_only_arm.jsonl")
            elif arm_name == "prose_only":
                arms[arm_name] = load_jsonl(DATA_DIR / "rq3_prose_only_arm.jsonl")
            elif arm_name == "action_record":
                arms[arm_name] = load_jsonl(DATA_DIR / "rq3_action_record_arm.jsonl")
            elif arm_name == "answer_only":
                arms[arm_name] = load_jsonl(DATA_DIR / "rq3_answer_only_arm.jsonl")
        except FileNotFoundError:
            print(f"Warning: {arm_name} file not found, skipping")
            continue

    print(f"\nLoaded {len(questions_all)} total questions")
    print(f"Raw arm: {len(questions_raw)} questions")
    print(f"Output-stripped arm: {len(questions_stripped)} questions")
    for name, data in arms.items():
        print(f"{name}: {len(data)} records")

    # ========================================================================
    # (a) PAIRED COMPARISON: raw vs output-stripped on shared 89 questions
    # ========================================================================
    print("\n" + "=" * 80)
    print("(a) PAIRED TEST: Raw vs Output-Stripped on 89 Shared Questions")
    print("=" * 80)

    # Build lookup by (session_id, question) for each arm
    def build_lookup(records):
        return {(r['session_id'], r['question']): r for r in records}

    raw_lookup = build_lookup(questions_raw)
    stripped_lookup = build_lookup(questions_stripped)

    # Find shared questions
    shared_keys = set(raw_lookup.keys()) & set(stripped_lookup.keys())
    print(f"\nFound {len(shared_keys)} shared questions")

    if len(shared_keys) != 89:
        print(f"WARNING: Expected 89 shared questions, got {len(shared_keys)}")

    # Compute paired differences
    raw_scores = []
    stripped_scores = []
    differences = []

    for key in shared_keys:
        raw_grade = grade_to_binary(raw_lookup[key]['grade'])
        stripped_grade = grade_to_binary(stripped_lookup[key]['grade'])
        raw_scores.append(raw_grade)
        stripped_scores.append(stripped_grade)
        differences.append(stripped_grade - raw_grade)

    raw_acc = sum(raw_scores) / len(raw_scores) * 100
    stripped_acc = sum(stripped_scores) / len(stripped_scores) * 100
    mean_diff = sum(differences) / len(differences) * 100

    print(f"\nRaw accuracy on n={len(shared_keys)}: {raw_acc:.1f}%")
    print(f"Output-stripped accuracy on n={len(shared_keys)}: {stripped_acc:.1f}%")
    print(f"Mean paired difference (stripped - raw): {mean_diff:+.1f} pp")

    # Paired t-test
    if len(differences) > 1:
        t_stat, p_value = stats.ttest_rel(stripped_scores, raw_scores)
        print(f"\nPaired t-test: t={t_stat:.3f}, p={p_value:.4f}")

        # Confidence interval on the difference
        se_diff = stats.sem(differences)
        ci_95 = stats.t.interval(0.95, len(differences)-1, loc=mean_diff/100, scale=se_diff)
        print(f"95% CI on difference: [{ci_95[0]*100:+.1f}, {ci_95[1]*100:+.1f}] pp")

        if p_value < 0.05:
            print(f"✓ Difference IS statistically significant at α=0.05")
        else:
            print(f"✗ Difference is NOT statistically significant at α=0.05")
            print(f"  The 1.1-point gap is indistinguishable from zero.")

    # Count discordant pairs
    stripped_better = sum(1 for d in differences if d > 0)
    raw_better = sum(1 for d in differences if d < 0)
    tied = sum(1 for d in differences if d == 0)

    print(f"\nDiscordant pairs: stripped better={stripped_better}, raw better={raw_better}, tied={tied}")

    # Sign test
    if stripped_better + raw_better > 0:
        try:
            # Try newer scipy API first
            result = stats.binomtest(stripped_better, stripped_better + raw_better, p=0.5, alternative='two-sided')
            p_sign = result.pvalue
        except AttributeError:
            # Fall back to older API
            p_sign = stats.binom_test(stripped_better, stripped_better + raw_better, p=0.5, alternative='two-sided')
        print(f"Sign test p-value: {p_sign:.4f}")

    # ========================================================================
    # (b) TOKENS-PER-POINT TABLE
    # ========================================================================
    print("\n" + "=" * 80)
    print("(b) TOKENS-PER-POINT FOR EACH CONTENT CLASS")
    print("=" * 80)

    # Known token counts and accuracies from the paper
    arm_stats = {
        "call-only": {"tokens": 6329, "recall": 47.2, "n": 123},
        "prose-only": {"tokens": 3799, "recall": 92.7, "n": 124},
        "answer-only": {"tokens": 2533, "recall": 85.5, "n": 124},
        "output-stripped": {"tokens": 8880, "recall": 94.4, "n": 124},  # uncorrected
        "action-record": {"tokens": 6393, "recall": 96.8, "n": 124},
        "raw-transcript": {"tokens": 206665, "recall": 95.5, "n": 89},  # uncorrected
    }

    print(f"\n{'Arm':<20} {'Tokens':>10} {'Recall %':>10} {'n':>6} {'Tok/Point':>12}")
    print("-" * 70)

    for arm, stats_dict in sorted(arm_stats.items(), key=lambda x: x[1]['tokens']):
        tokens = stats_dict['tokens']
        recall = stats_dict['recall']
        n = stats_dict['n']
        tok_per_point = tokens / recall if recall > 0 else float('inf')
        print(f"{arm:<20} {tokens:>10,} {recall:>10.1f} {n:>6} {tok_per_point:>12,.0f}")

    print("\nNote: output-stripped and raw are uncorrected figures from the paper.")
    print("      Corrected figures: output-stripped 90.3%, raw 94.4% on n=89.")

    # ========================================================================
    # (c) PER-QUESTION-TYPE BREAKDOWN
    # ========================================================================
    print("\n" + "=" * 80)
    print("(c) PER-QUESTION-TYPE: Where Call-Only Fails vs Prose-Only")
    print("=" * 80)

    if "tool_inputs_only" in arms and "prose_only" in arms:
        call_lookup = build_lookup(arms["tool_inputs_only"])
        prose_lookup = build_lookup(arms["prose_only"])

        # Count by question class
        by_class = defaultdict(lambda: {"call_only": [], "prose_only": []})

        # Use all questions that appear in both arms
        shared_for_comparison = set(call_lookup.keys()) & set(prose_lookup.keys())

        for key in shared_for_comparison:
            qclass = call_lookup[key].get('class', 'unknown')
            call_grade = grade_to_binary(call_lookup[key]['grade'])
            prose_grade = grade_to_binary(prose_lookup[key]['grade'])

            by_class[qclass]["call_only"].append(call_grade)
            by_class[qclass]["prose_only"].append(prose_grade)

        print(f"\n{'Question Type':<20} {'n':>5} {'Call-Only %':>12} {'Prose-Only %':>13} {'Gap (pp)':>10}")
        print("-" * 70)

        for qclass in sorted(by_class.keys()):
            n = len(by_class[qclass]["call_only"])
            call_acc = sum(by_class[qclass]["call_only"]) / n * 100 if n > 0 else 0
            prose_acc = sum(by_class[qclass]["prose_only"]) / n * 100 if n > 0 else 0
            gap = prose_acc - call_acc

            indicator = "✓ prose wins" if gap > 20 else ("≈ tied" if abs(gap) < 10 else "")
            print(f"{qclass:<20} {n:>5} {call_acc:>11.1f}% {prose_acc:>12.1f}% {gap:>9.1f}  {indicator}")

        # Overall
        all_call = [v for vals in by_class.values() for v in vals["call_only"]]
        all_prose = [v for vals in by_class.values() for v in vals["prose_only"]]

        if all_call and all_prose:
            overall_call = sum(all_call) / len(all_call) * 100
            overall_prose = sum(all_prose) / len(all_prose) * 100
            overall_gap = overall_prose - overall_call

            print("-" * 70)
            print(f"{'OVERALL':<20} {len(all_call):>5} {overall_call:>11.1f}% {overall_prose:>12.1f}% {overall_gap:>9.1f}")

    else:
        print("\nCall-only or prose-only arms not available, skipping breakdown.")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
