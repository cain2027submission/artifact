#!/usr/bin/env python3
"""RQ3 PROSE-ONLY BUDGET LADDER: how does recall change as we raise the budget?

The capped-budget regime in our experiments truncates the injected context to ~500
chars because that is the value our retrieval hook happens to use. It is a tunable
knob, not a law. This script answers: how does recall change as we raise it?

Substrate: prose-only (user prose + assistant prose only, tool calls and results removed)
Budgets: [2000, 8000] chars (head truncation)
Questions: 124 each

Context building: prose_only_context() imported from rq3_prose_only_arm.py (verified
correct, rerun today), then TRUNCATE to FIRST n chars (head truncation).

This is head truncation because a retrieval hook would inject a relevance-selected
middle chunk, and that difference is a real threat to validity the paper must state
rather than hide.

Usage:
    rq3_prose_only_budget_ladder.py --smoke       # smoke test on 2 sessions
    rq3_prose_only_budget_ladder.py --limit 5     # smoke test with API calls
    rq3_prose_only_budget_ladder.py               # full run (resumable)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

from rq3_prose_only_arm import prose_only_context  # noqa: E402
from rq3_raw_transcript_arm import (  # noqa: E402
    DATA, QUESTIONS, SUMMARY, SLUG, call_llm, extract_confidence, grade_answer,
    index_transcripts, load_records, rates, record_id, resolve,
)

OUT = DATA / "rq3_prose_only_budget_ladder.jsonl"
REPORT = DATA / "rq3_prose_only_budget_ladder_summary.json"
JUDGE_MODEL = "claude-sonnet-5"

BUDGETS = [2000, 8000]


def truncate_to_budget(prose: str, budget: int) -> str:
    """Head truncation: take FIRST n chars."""
    if len(prose) <= budget:
        return prose
    # Truncate at word boundary
    truncated = prose[:budget].rsplit(" ", 1)[0] + "…"
    return truncated


def answer_from_prose_budget(question: str, prose: str) -> Tuple[str, str, dict]:
    """Answer from truncated prose context."""
    prompt = f"""Someone asks: "{question}"

Below is a transcript with only the prose (user and assistant text). Tool calls and their results have been removed. Answer the question based ONLY on this prose. If it does not contain enough information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

PROSE:
{prose}

Provide your answer and confidence now:"""

    response, usage = call_llm(prompt, model=JUDGE_MODEL, timeout=600)
    if not response:
        return "", "API_ERROR", usage

    answer, confidence = extract_confidence(response)
    return answer, confidence, usage


def apply_confidence_correction(rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Reclassify records where the answer contains NOT ANSWERABLE markers.

    Returns (corrected_rows, reclassified_records).
    """
    reclassified = []
    corrected = []

    for r in rows:
        answer = r.get('answer', '')
        rating = r.get('confidence', '')
        grade = r.get('grade', '')

        # Check if answer body prominently contains NOT ANSWERABLE
        answer_upper = answer.upper()
        if rating != 'NOT_ANSWERABLE' and ('NOT ANSWERABLE' in answer_upper or 'NOT_ANSWERABLE' in answer_upper):
            # Reclassify as NOT_ANSWERABLE
            reclassified.append({
                'record_id': r['record_id'],
                'session_id': r.get('session_id'),
                'class': r.get('class'),
                'budget': r.get('budget'),
                'stored_rating': rating,
                'corrected_rating': 'NOT_ANSWERABLE',
                'reason': 'marker_in_retained_answer',
                'grade': grade,
                'answer_head': answer[:120]
            })
            corrected.append(dict(r, confidence='NOT_ANSWERABLE', grade='absent'))
        else:
            corrected.append(r)

    return corrected, reclassified


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="smoke test: show truncation on 2 sessions, no API calls")
    ap.add_argument("--limit", type=int, default=0, help="limit number of questions")
    args = ap.parse_args()

    graded = set(json.loads(SUMMARY.read_text())["graded_record_ids"])
    records = [r for r in load_records(QUESTIONS) if record_id(r) in graded]
    by_prefix = index_transcripts()

    eligible, unresolved = [], []
    for r in records:
        m = SLUG.match(r.get("session_id", ""))
        if not m:
            unresolved.append(r.get("session_id", "")[:8])
            continue
        prefix = m.group(1)
        p = resolve(prefix, by_prefix)
        if p is None:
            unresolved.append(prefix)
            continue
        eligible.append(dict(r, _transcript=str(p), _prefix=prefix))

    print(f"graded {len(records)}  resolved {len(eligible)}  "
          f"unresolved {len(set(unresolved))}")

    # Smoke test: show truncation on first small and first large session
    if args.smoke:
        cache = {}
        for tpath in set(r["_transcript"] for r in eligible):
            prose, _ = prose_only_context(Path(tpath).read_text(errors="replace"))
            cache[tpath] = prose

        # Sort by size
        sizes = [(tpath, len(prose)) for tpath, prose in cache.items()]
        sizes.sort(key=lambda x: x[1])

        print("\n=== SMOKE TEST: TRUNCATION VERIFICATION ===")
        print()

        # Small session
        small_path, small_size = sizes[0]
        small_prose = cache[small_path]
        print(f"SMALL session: {Path(small_path).name}")
        print(f"  untruncated: {small_size:,} chars")
        for budget in BUDGETS:
            truncated = truncate_to_budget(small_prose, budget)
            print(f"  budget {budget:,}: {len(truncated):,} chars")
        print()

        # Large session
        large_path, large_size = sizes[-1]
        large_prose = cache[large_path]
        print(f"LARGE session: {Path(large_path).name}")
        print(f"  untruncated: {large_size:,} chars")
        for budget in BUDGETS:
            truncated = truncate_to_budget(large_prose, budget)
            print(f"  budget {budget:,}: {len(truncated):,} chars")
        print()

        print("=== SMOKE TEST COMPLETE (no API calls) ===")
        return

    # Check for existing output to avoid collision
    if OUT.exists():
        existing = list(load_records(OUT))
        if existing:
            unique_ids = {(r.get('record_id'), r.get('budget')) for r in existing}
            if len(existing) != len(unique_ids):
                print(f"ERROR: {OUT.name} has {len(existing)} records but only "
                      f"{len(unique_ids)} unique (record_id, budget) pairs. Collision detected.")
                print("Move or delete the file before rerunning.")
                sys.exit(1)
            print(f"resuming: {len(existing)} already answered")
            done = {(r.get('record_id'), r.get('budget')): r for r in existing}
        else:
            done = {}
    else:
        done = {}

    # Build todo list: all (question, budget) pairs
    todo = []
    for r in eligible:
        for budget in BUDGETS:
            if (record_id(r), budget) not in done:
                todo.append((r, budget))

    if args.limit:
        todo = todo[:args.limit]

    todo.sort(key=lambda x: (x[0]["_prefix"], x[1]))
    print(f"to answer: {len(todo)} (record, budget) pairs\n")

    cache = {}
    errors = []
    t0 = time.time()
    tokens_in = tokens_out = 0

    for i, (r, budget) in enumerate(todo, 1):
        rid = record_id(r)
        tpath = r["_transcript"]

        # Load and cache prose
        if tpath not in cache:
            prose, _ = prose_only_context(Path(tpath).read_text(errors="replace"))
            cache[tpath] = prose
        prose = cache[tpath]

        # Truncate to budget
        truncated = truncate_to_budget(prose, budget)

        answer, confidence, usage = answer_from_prose_budget(r["question"], truncated)
        tokens_in += usage.get("prompt_tokens", 0)
        tokens_out += usage.get("completion_tokens", 0)

        if confidence == "API_ERROR":
            errors.append({"record_id": rid, "budget": budget, "prefix": r["_prefix"]})
            print(f"[{i}/{len(todo)}] {r['_prefix']} budget={budget:5d} API_ERROR (not recorded, retry)")
            continue

        grade = grade_answer(answer, r.get("ground_truth", ""), confidence)
        print(f"[{i}/{len(todo)}] {r['_prefix']} budget={budget:5d} "
              f"prose={len(prose)/1e3:6.1f}KB trunc={len(truncated)/1e3:5.1f}KB "
              f"{confidence:18s} {grade:18s} (%.0fs, in=%d out=%d)"
              % (time.time() - t0, tokens_in, tokens_out))

        out = {
            "record_id": rid,
            "budget": budget,
            "session_id": r.get("session_id"),
            "class": r.get("class"),
            "question": r.get("question"),
            "ground_truth": r.get("ground_truth"),
            "published_grade": r.get("grade"),
            "transcript": tpath,
            "prose_chars_full": len(prose),
            "context_chars": len(truncated),
            "truncated": len(truncated) < len(prose),
            "answer": answer,
            "confidence": confidence,
            "grade": grade,
            "usage": usage,
            "model": JUDGE_MODEL,
        }
        with OUT.open("a") as fh:
            fh.write(json.dumps(out) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        done[(rid, budget)] = out

    print(f"\nanswered {len(todo)-len(errors)} of {len(todo)} in "
          f"{time.time()-t0:.0f}s ({len(errors)} API errors)")
    print(f"tokens: prompt={tokens_in} completion={tokens_out}")

    # Summarise
    rows = list(done.values())

    # Apply confidence correction
    corrected_rows, reclassified = apply_confidence_correction(rows)

    # Group by budget
    by_budget = {}
    for budget in BUDGETS:
        uncorrected = [r for r in rows if r['budget'] == budget]
        corrected = [r for r in corrected_rows if r['budget'] == budget]
        by_budget[budget] = {
            'uncorrected': rates(uncorrected),
            'corrected': rates(corrected),
            'n': len(uncorrected),
        }

    report = {
        "n_total_pairs": len(rows),
        "n_questions": len({r['record_id'] for r in rows}),
        "n_budgets": len(BUDGETS),
        "budgets": BUDGETS,
        "judge": f"{JUDGE_MODEL} via local gateway, no temperature set",
        "context": "prose-only with HEAD TRUNCATION to budget",
        "purpose": "Measure how recall changes as budget increases for prose-only substrate",
        "n_api_errors_this_run": len(errors),
        "by_budget": by_budget,
        "n_reclassified_by_confidence_correction": len(reclassified),
        "reclassified_records": reclassified,
        "spend": {
            "prompt_tokens": tokens_in,
            "completion_tokens": tokens_out,
        }
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")

    def line(label: str, d: dict) -> str:
        h = d.get("hallucination_of_attempted", {})
        return (f"{label:40s} n={d['n']:3d}  correct {d['correct']['n']:3d}  "
                f"partial {d['partially_correct']['n']:3d}  "
                f"wrong {d['wrong']['n']:3d}  absent {d['absent']['n']:3d}  "
                f"recoverable {d['recoverable']['pct']:5.1f}%  "
                f"CI95 {d['recoverable']['ci95']}")

    print()
    for budget in BUDGETS:
        print(f"\n--- BUDGET {budget} chars ---")
        print(line(f"prose_only@{budget} (uncorrected)", by_budget[budget]['uncorrected']))
        print(line(f"prose_only@{budget} (corrected)", by_budget[budget]['corrected']))

    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
