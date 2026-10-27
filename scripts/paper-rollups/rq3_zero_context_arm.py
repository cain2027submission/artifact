#!/usr/bin/env python3
"""RQ3 ARM: answer questions with ZERO SESSION CONTEXT (true floor/no-op control).

Every other RQ3 arm gives the judge *something* about the session: the full
transcript (verbatim), a stripped transcript, prose only, or (already flagged
as unfair/degenerate) tool inputs only. None of them establishes what a judge
scores with nothing at all -- no transcript content, no filename, no hint that
a session even exists. Without that number, "tool_inputs_only scores 44.7%"
has no floor to be compared against: maybe 44.7% is inference over tool-call
shapes carrying real signal, or maybe a judge guesses its way to something
close to that from the question's phrasing and general world knowledge alone.

This arm supplies that floor. Context given to the model: nothing. Just the
bare question -- no transcript, no filename (contrast the `rollup_pointer`
arm elsewhere, which gives a filename), no ground truth, no session hint of
any kind.

Reuses the exact same judge model, grading prompt, confidence-correction
logic and output shape as rq3_tool_inputs_only_arm.py / rq3_prose_only_arm.py,
so results are directly comparable. Record set is the canonical 124-question
set carried in rq3_prose_only_arm.jsonl (record_id, question, ground_truth,
class, session_id, published grade) rather than resampled from
rq3_questions.jsonl -- see load_records() in rq3_raw_transcript_arm for why
line 125 of that file is skipped when it is touched at all (a NUL-filled
partial write).

Usage:
    rq3_zero_context_arm.py --dry-run     # check record count, no API calls
    rq3_zero_context_arm.py --limit 5     # smoke test
    rq3_zero_context_arm.py               # full run
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_raw_transcript_arm import (  # noqa: E402
    DATA, call_llm, extract_confidence, grade_answer, load_records, rates,
)

SOURCE = DATA / "rq3_prose_only_arm.jsonl"  # canonical 124-record set
OUT = DATA / "rq3_zero_context_arm.jsonl"
REPORT = DATA / "rq3_zero_context_arm_summary.json"
JUDGE_MODEL = "claude-sonnet-5"

CLASSES = ["sequence", "outcome", "dead_end", "decision", "artifact_detail"]


def answer_zero_context(question: str) -> Tuple[str, str, dict]:
    """Answer with NO session context at all: no transcript, no filename,
    no hint that a session even exists beyond the question's own wording."""
    prompt = f"""Someone asks: "{question}"

This question refers to a specific past agent coding session. You have been
given NO information about that session: no transcript, no filename, no
summary, no hint of any kind about what happened in it. Answer only if you
somehow already know the specific answer (you almost certainly do not, since
this refers to a private session you were never shown any part of). Do not
guess or invent plausible-sounding specifics. If you do not have enough
information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

Provide your answer and confidence now:"""

    response, usage = call_llm(prompt, model=JUDGE_MODEL, timeout=600)
    if not response:
        return "", "API_ERROR", usage

    answer, confidence = extract_confidence(response)
    return answer, confidence, usage


def apply_confidence_correction(rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Reclassify records where the answer contains NOT ANSWERABLE markers.

    Identical logic to the tool_inputs_only / prose_only arms, kept verbatim
    so the correction is not itself a source of arm-to-arm difference.
    """
    reclassified = []
    corrected = []

    for r in rows:
        answer = r.get('answer', '')
        rating = r.get('rating', '')
        grade = r.get('grade', '')

        answer_upper = answer.upper()
        if rating != 'NOT_ANSWERABLE' and ('NOT ANSWERABLE' in answer_upper or 'NOT_ANSWERABLE' in answer_upper):
            reclassified.append({
                'record_id': r['record_id'],
                'session_id': r.get('session_id'),
                'class': r.get('class'),
                'stored_rating': rating,
                'corrected_rating': 'NOT_ANSWERABLE',
                'reason': 'marker_in_retained_answer',
                'grade': grade,
                'answer_head': answer[:120]
            })
            corrected.append(dict(r, rating='NOT_ANSWERABLE', grade='absent'))
        else:
            corrected.append(r)

    return corrected, reclassified


def by_class(rows: List[dict]) -> dict:
    grouped = defaultdict(list)
    for r in rows:
        grouped[r.get("class")].append(r)
    return {c: rates(grouped[c]) for c in CLASSES if grouped.get(c)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    records = load_records(SOURCE)
    unique_ids = {r["record_id"] for r in records}
    if len(records) != len(unique_ids):
        print(f"WARNING: {SOURCE.name} has {len(records)} records but only "
              f"{len(unique_ids)} unique record_ids.")
    print(f"canonical record set: {len(records)} records "
          f"({len(unique_ids)} unique ids) from {SOURCE.name}")

    if args.dry_run:
        by_c = defaultdict(int)
        for r in records:
            by_c[r.get("class")] += 1
        for c in CLASSES:
            print(f"  {c:16s} {by_c.get(c, 0)}")
        print("\n--dry-run: no API calls made")
        return

    if OUT.exists():
        existing = list(load_records(OUT))
        if existing:
            existing_ids = {r.get('record_id') for r in existing}
            if len(existing) != len(existing_ids):
                print(f"ERROR: {OUT.name} has {len(existing)} records but only "
                      f"{len(existing_ids)} unique record_ids. Collision detected.")
                sys.exit(1)
            print(f"resuming: {len(existing)} already answered")
            done = {r.get('record_id'): r for r in existing}
        else:
            done = {}
    else:
        done = {}

    todo = [r for r in records if r["record_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    todo.sort(key=lambda r: (r.get("class", ""), r["record_id"]))
    print(f"to answer: {len(todo)}\n")

    errors: List[dict] = []
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        rid = r["record_id"]
        answer, confidence, usage = answer_zero_context(r["question"])
        if confidence == "API_ERROR":
            errors.append({"record_id": rid})
            print(f"[{i}/{len(todo)}] API_ERROR (not recorded, retry)")
            continue
        grade = grade_answer(answer, r.get("ground_truth", ""), confidence)
        print(f"[{i}/{len(todo)}] {r.get('class'):15s} {confidence:18s} "
              f"{grade:18s} (rollup: {r.get('published_grade')})")

        out = {"record_id": rid, "session_id": r.get("session_id"),
               "class": r.get("class"), "question": r.get("question"),
               "ground_truth": r.get("ground_truth"),
               "published_grade": r.get("published_grade"),
               "answer": answer, "rating": confidence, "grade": grade,
               "usage": usage, "model": JUDGE_MODEL}
        with OUT.open("a") as fh:
            fh.write(json.dumps(out) + "\n")
        done[rid] = out

    print(f"\nanswered {len(todo)-len(errors)} of {len(todo)} in "
          f"{time.time()-t0:.0f}s ({len(errors)} API errors)")

    rows = list(done.values())
    unique_out_ids = {r['record_id'] for r in rows}
    if len(rows) != len(records) or len(unique_out_ids) != len(records):
        print(f"\nWARNING: Expected {len(records)} unique records, got "
              f"{len(rows)} records with {len(unique_out_ids)} unique IDs")

    corrected_rows, reclassified = apply_confidence_correction(rows)

    uncorrected_rates = rates(rows)
    corrected_rates = rates(corrected_rows)

    report = {
        "n_questions": len(rows),
        "n_sessions": len({r["session_id"] for r in rows}),
        "judge": f"{JUDGE_MODEL} via local gateway, no temperature set",
        "context": "zero session context; question only, no filename, no transcript content",
        "purpose": ("True floor/no-op control for the RQ3 benchmark. Tests whether "
                     "tool_inputs_only's 44.7% (and its by-class spread) is "
                     "meaningfully above what a judge scores with no session "
                     "information at all -- a control this benchmark never had."),
        "source_record_set": SOURCE.name,
        "n_api_errors_this_run": len(errors),
        "zero_context_arm_uncorrected": uncorrected_rates,
        "zero_context_arm_corrected": corrected_rates,
        "by_class_uncorrected": by_class(rows),
        "by_class_corrected": by_class(corrected_rows),
        "n_reclassified_by_confidence_correction": len(reclassified),
        "reclassified_records": reclassified,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")

    def line(label: str, d: dict) -> str:
        h = d["hallucination_of_attempted"]
        return (f"{label:30s} n={d['n']:3d}  correct {d['correct']['n']:3d}  "
                f"partial {d['partially_correct']['n']:3d}  "
                f"wrong {d['wrong']['n']:3d}  absent {d['absent']['n']:3d}  "
                f"recoverable {d['recoverable']['pct']:5.1f}%  "
                f"wrong-of-attempted {h['pct']}% {h['ci95']}")

    print()
    print(line("zero-context (uncorrected)", uncorrected_rates))
    print(line("zero-context (corrected)", corrected_rates))
    print("\nby class (corrected):")
    for c in CLASSES:
        d = report["by_class_corrected"].get(c)
        if d:
            print(line(f"  {c}", d))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
