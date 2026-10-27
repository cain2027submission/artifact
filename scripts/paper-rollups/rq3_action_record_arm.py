#!/usr/bin/env python3
"""RQ3 ARM A: answer questions from ACTION RECORD.

The action record is the structured session rendering: user + assistant prose,
tool calls with salient arguments extracted (not full input dicts), one-line
result digests, and turn ordering. It aims to retain output-stripped's 90.3% recovery
at a smaller median size (~24k vs 33k chars).

Context: build_action_record() output (see action_record.py and
ACTION_RECORD_FORMAT.md for spec).

Predicted outcome: Matches or exceeds output-stripped's 90.3%, demonstrating that
structured tool calls + result digests retain what matters at lower cost.

Construction: like output-stripped but ADDS one-line tool-result digests, structures
salient tool args, and adds turn numbers; explicitly drops the signature. It is NOT
"output-stripped with inert bytes removed" — results come back in digested form.

Usage:
    rq3_action_record_arm.py --dry-run     # check sizes
    rq3_action_record_arm.py --limit 5     # smoke test
    rq3_action_record_arm.py               # full run
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_raw_transcript_arm import (  # noqa: E402
    DATA, QUESTIONS, SUMMARY, SLUG, call_llm, extract_confidence, grade_answer,
    index_transcripts, load_records, rates, record_id, resolve,
)
from action_record import build_action_record  # noqa: E402

OUT = DATA / "rq3_action_record_arm.jsonl"
REPORT = DATA / "rq3_action_record_arm_summary.json"
JUDGE_MODEL = "claude-sonnet-5"

CONTEXT_LIMIT_TOKENS = 1_180_000
BYTES_PER_TOKEN = 2.45


def answer_from_action_record(question: str, action_record: str,
                              return_raw: bool = False):
    """Answer from action-record context.

    Returns (answer, confidence, usage), or (answer, confidence, usage, reply)
    when `return_raw` is set. The kwarg defaults off so every existing caller and
    test_confidence_extraction.py, which unpacks a 3-tuple, is untouched.

    The raw reply is worth retaining and was not: main.tex records that the action
    record has no confidence-corrected rate precisely because this arm stored only
    the post-parse `answer`, which makes re-running a fixed extractor over the
    stored data circular. rq3_confidence_rederive.py can therefore only bound the
    defect from below. Keeping the bytes makes the correction exact rather than a
    lower bound, for every record written from here on.
    """
    prompt = f"""Someone asks: "{question}"

Below is a structured action record of an agent session, with tool calls shown as name + salient arguments, one-line result digests, and turn ordering. Answer the question based ONLY on this record. If it does not contain enough information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

ACTION RECORD:
{action_record}

Provide your answer and confidence now:"""

    response, usage = call_llm(prompt, model=JUDGE_MODEL, timeout=600)
    if not response:
        return ("", "API_ERROR", usage, response) if return_raw \
            else ("", "API_ERROR", usage)

    answer, confidence = extract_confidence(response)
    return (answer, confidence, usage, response) if return_raw \
        else (answer, confidence, usage)


def apply_confidence_correction(rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Reclassify records where the answer contains NOT ANSWERABLE markers."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
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
        eligible.append(dict(r, _transcript=str(p), _prefix=prefix,
                             _bytes=p.stat().st_size))

    print(f"graded {len(records)}  resolved {len(eligible)}  "
          f"unresolved {len(set(unresolved))}")

    if args.dry_run:
        seen, rows = set(), []
        for r in eligible:
            if r["_transcript"] in seen:
                continue
            seen.add(r["_transcript"])
            try:
                action_record, st = build_action_record(r["_transcript"])
                fits = len(action_record) / BYTES_PER_TOKEN <= CONTEXT_LIMIT_TOKENS
                rows.append((r["_prefix"], len(action_record), st.get('tool_calls', 0), fits))
            except Exception as e:
                print(f"ERROR building action record for {r['_prefix']}: {e}")
                rows.append((r["_prefix"], 0, 0, False))
        rows.sort(key=lambda x: -x[1])
        print(f"\n{'sess':10s} {'action chars':>13s} {'tool_calls':>11s} {'fits':>6s}")
        for pre, action_chars, tools, f in rows:
            print(f"{pre:10s} {action_chars:13,} {tools:11d} {'yes' if f else 'NO':>6s}")
        print(f"\ntotal sessions: {len(rows)}")
        print("\n--dry-run: no API calls made")
        return

    # Check for existing output
    if OUT.exists():
        existing = list(load_records(OUT))
        if existing:
            unique_ids = {r.get('record_id') for r in existing}
            if len(existing) != len(unique_ids):
                print(f"ERROR: {OUT.name} has {len(existing)} records but only "
                      f"{len(unique_ids)} unique record_ids. Collision detected.")
                sys.exit(1)
            print(f"resuming: {len(existing)} already answered")
            done = {r.get('record_id'): r for r in existing}
        else:
            done = {}
    else:
        done = {}

    todo = [r for r in eligible if record_id(r) not in done]
    if args.limit:
        todo = todo[:args.limit]
    todo.sort(key=lambda r: (r["_prefix"], r.get("class", "")))
    print(f"to answer: {len(todo)}\n")

    cache: Tuple[Optional[str], str, dict] = (None, "", {})
    errors: List[dict] = []
    build_errors: List[dict] = []
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        rid = record_id(r)
        if cache[0] != r["_transcript"]:
            try:
                action_record, st = build_action_record(r["_transcript"])
                cache = (r["_transcript"], action_record, st)
            except Exception as e:
                print(f"[{i}/{len(todo)}] {r['_prefix']} BUILD_ERROR: {e}")
                build_errors.append({"record_id": rid, "prefix": r["_prefix"], "error": str(e)})
                continue
        action_record, st = cache[1], cache[2]

        if len(action_record) / BYTES_PER_TOKEN > CONTEXT_LIMIT_TOKENS:
            print(f"[{i}/{len(todo)}] {r['_prefix']} SKIP action record too big "
                  f"({len(action_record)/1e6:.2f} MB)")
            continue

        answer, confidence, usage, raw_reply = answer_from_action_record(
            r["question"], action_record, return_raw=True)
        if confidence == "API_ERROR":
            errors.append({"record_id": rid, "prefix": r["_prefix"]})
            print(f"[{i}/{len(todo)}] {r['_prefix']} API_ERROR (not recorded, retry)")
            continue
        grade = grade_answer(answer, r.get("ground_truth", ""), confidence)
        print(f"[{i}/{len(todo)}] {r['_prefix']} {r.get('class'):15s} "
              f"{len(action_record)/1e3:6.1f}KB {confidence:18s} {grade:18s} "
              f"(rollup: {r.get('grade')})")

        out = {"record_id": rid, "session_id": r.get("session_id"),
               "class": r.get("class"), "question": r.get("question"),
               "ground_truth": r.get("ground_truth"),
               "published_grade": r.get("grade"),
               "transcript": r["_transcript"], "raw_bytes": r["_bytes"],
               "action_record_chars": len(action_record), "action_record_stats": st,
               "answer": answer, "rating": confidence, "grade": grade,
               # Additive: `answer` is the post-parse extraction and is unchanged.
               # `raw_reply` is the bytes it was extracted from, so a later parser
               # change is measurable rather than merely arguable.
               "raw_reply": raw_reply,
               "usage": usage, "model": JUDGE_MODEL}
        with OUT.open("a") as fh:
            fh.write(json.dumps(out) + "\n")
        done[rid] = out

    print(f"\nanswered {len(todo)-len(errors)-len(build_errors)} of {len(todo)} in "
          f"{time.time()-t0:.0f}s ({len(errors)} API errors, {len(build_errors)} build errors)")

    # Verify exactly 124 unique record_ids
    rows = list(done.values())
    unique_ids = {r['record_id'] for r in rows}
    if len(rows) != 124 or len(unique_ids) != 124:
        print(f"\nWARNING: Expected 124 unique records, got {len(rows)} records "
              f"with {len(unique_ids)} unique IDs")

    # Apply confidence correction
    corrected_rows, reclassified = apply_confidence_correction(rows)

    # Compute rates
    uncorrected_rates = rates(rows)
    corrected_rates = rates(corrected_rows)

    report = {
        "n_questions": len(rows),
        "n_sessions": len({r["session_id"] for r in rows}),
        "judge": f"{JUDGE_MODEL} via local gateway, no temperature set",
        "context": "action record: structured tool calls, result digests, prose, turn ordering",
        "purpose": "Tests whether structured tool calls + digests match output-stripped's 90.3% at lower cost.",
        "n_api_errors_this_run": len(errors),
        "n_build_errors_this_run": len(build_errors),
        "build_errors": build_errors,
        "action_record_arm_uncorrected": uncorrected_rates,
        "action_record_arm_corrected": corrected_rates,
        "n_reclassified_by_confidence_correction": len(reclassified),
        "reclassified_records": reclassified,
        "median_action_record_chars": sorted(r["action_record_chars"] for r in rows)[len(rows)//2] if rows else None,
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
    print(line("action-record (uncorrected)", uncorrected_rates))
    print(line("action-record (corrected)", corrected_rates))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
