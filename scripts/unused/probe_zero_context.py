#!/usr/bin/env python3
"""Zero-context control: answer questions with NO memory provided.

Verifies that questions are not answerable without context. If this control
scores well, the questions are trivial and the whole measurement is void.

Usage:
    probe_zero_context.py --limit 10
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_raw_transcript_arm import call_llm, grade_answer
from probe_stratified_measurement import load_questions

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
DATA = REPO / "data" / "paper-rollups"
QUESTIONS = DATA / "probe_stratified_questions.jsonl"
OUTPUT = DATA / "probe_zero_context.jsonl"

JUDGE_MODEL = "claude-sonnet-5"


def answer_zero_context(question: str):
    """Answer a question with NO context provided."""
    prompt = f"""Someone asks: "{question}"

Answer this question to the best of your ability, WITHOUT any additional context.

If you cannot answer without more information, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

Provide your answer and confidence now:"""

    response, usage = call_llm(prompt, model=JUDGE_MODEL, timeout=120)
    if not response:
        return "", "API_ERROR", usage

    # Simple extraction
    lines = response.strip().split('\n')
    confidence_line = ""
    for line in reversed(lines):
        line = line.strip()
        if any(c in line.upper() for c in ["ANSWERED", "PARTIALLY_ANSWERED", "NOT_ANSWERABLE"]):
            confidence_line = line
            break

    if "NOT_ANSWERABLE" in confidence_line.upper():
        confidence = "NOT_ANSWERABLE"
    elif "PARTIALLY_ANSWERED" in confidence_line.upper():
        confidence = "PARTIALLY_ANSWERED"
    elif "ANSWERED" in confidence_line.upper():
        confidence = "ANSWERED"
    else:
        confidence = "UNKNOWN"

    # Everything except confidence line is the answer
    if confidence_line:
        answer = response.replace(confidence_line, "").strip()
    else:
        answer = response

    return answer, confidence, usage


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=10,
                       help="Number of questions to test")
    args = parser.parse_args()

    questions = load_questions()
    todo = questions[:args.limit]

    print(f"Zero-context control: {len(todo)} questions")
    print()

    total_cost = 0.0
    results = []

    for i, q in enumerate(todo, 1):
        question_text = q["question"]
        ground_truth = q["ground_truth"]

        answer, confidence, usage = answer_zero_context(question_text)
        grade = grade_answer(answer, ground_truth, confidence)

        cost = (usage.get("prompt_tokens", 0) * 3.0 / 1e6 +
               usage.get("completion_tokens", 0) * 15.0 / 1e6)
        total_cost += cost

        print(f"[{i}/{len(todo)}] {confidence:18s} {grade:18s} ${cost:.4f}")

        result = {
            "pair_id": q["pair_id"],
            "band": q["band"],
            "question": question_text,
            "ground_truth": ground_truth,
            "answer": answer,
            "confidence": confidence,
            "grade": grade,
            "usage": usage
        }
        results.append(result)

    # Write results
    with open(OUTPUT, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Compute stats
    correct = sum(1 for r in results if r["grade"] == "correct")
    partial = sum(1 for r in results if r["grade"] == "partially_correct")
    wrong = sum(1 for r in results if r["grade"] == "wrong")
    absent = sum(1 for r in results if r["grade"] == "absent")

    print()
    print("=" * 60)
    print("ZERO-CONTEXT CONTROL RESULTS")
    print("=" * 60)
    print(f"Correct:   {correct}/{len(results)} = {100.0*correct/len(results):.1f}%")
    print(f"Partial:   {partial}/{len(results)} = {100.0*partial/len(results):.1f}%")
    print(f"Wrong:     {wrong}/{len(results)} = {100.0*wrong/len(results):.1f}%")
    print(f"Absent:    {absent}/{len(results)} = {100.0*absent/len(results):.1f}%")
    print(f"Total cost: ${total_cost:.2f}")
    print()

    if correct > 0:
        print("WARNING: Zero-context control scored above 0. Questions may be trivial!")
    else:
        print("PASS: Zero-context scored 0% as expected.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
