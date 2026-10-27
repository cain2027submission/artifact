#!/usr/bin/env python3
"""RQ3 ARM: ANSWER-ONLY - assistant prose alone, no user prose or tool I/O.

Tests whether assistant narration alone carries the context needed to answer the
graded questions. This is the decisive test of the paper's central claim: that
assistant prose is the active ingredient. We measured prose-only (user + assistant)
at 90.3% recall and prompts-only (user alone) at 32.3%, and have been INFERRING
from the difference that assistant narration is the key. Answer-only measures it
directly instead of inferring it.

Context: ASSISTANT sections ONLY from strip_transcript(), no user prose, no tool
calls, no tool results, no reasoning blocks.

Predicted outcome: If assistant prose is the active ingredient, this arm should
score near 90.3% (prose-only). If it scores near 32.3% (prompts-only), then the
combination matters more than the assistant prose alone.

Construction: strip_transcript() output filtered to ASSISTANT sections only. Sections
are delimited by lines starting with `###`. A section runs from its `###` header to
the NEXT line starting with `###`, and nothing else ends it.

Usage:
    rq3_answer_only_arm.py --dry-run     # check sizes
    rq3_answer_only_arm.py --limit 5     # smoke test
    rq3_answer_only_arm.py               # full run
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
from rq3_stripped_transcript_arm import strip_transcript  # noqa: E402

OUT = DATA / "rq3_answer_only_arm.jsonl"
REPORT = DATA / "rq3_answer_only_arm_summary.json"
JUDGE_MODEL = "claude-sonnet-5"

CONTEXT_LIMIT_TOKENS = 1_180_000
BYTES_PER_TOKEN = 2.45


def answer_only_context(raw: str) -> Tuple[str, dict]:
    """Extract ASSISTANT sections ONLY from stripped transcript.

    Returns (markdown, stats).

    Sections are delimited by lines starting with `###`. A section runs from its
    `###` header to the NEXT line starting with `###`. A blank line or ``` fence
    does NOT end a section.
    """
    stripped, strip_stats = strip_transcript(raw)

    # Parse sections: each starts with ### and runs to the next ###
    lines = stripped.splitlines()
    sections = []
    current_section = []
    current_header = None

    for line in lines:
        if line.startswith("###"):
            # Save previous section if it was ASSISTANT
            if current_header is not None and current_header.startswith("### ASSISTANT"):
                sections.append("\n".join(current_section))
            # Start new section
            current_header = line
            current_section = [line]
        else:
            # Continue current section
            if current_header is not None:
                current_section.append(line)

    # Don't forget last section
    if current_header is not None and current_header.startswith("### ASSISTANT"):
        sections.append("\n".join(current_section))

    # Build output
    assistant_text = "\n\n".join(sections)

    stats = {
        "entries": strip_stats.get("entries", 0),
        "assistant_sections": len(sections),
        "chars": len(assistant_text),
    }

    header = (f"[This contains ONLY the assistant's prose from an agent session. "
              f"{stats['assistant_sections']} assistant sections shown. User prose, "
              f"tool calls, and tool results are withheld.]\n\n")

    return header + assistant_text, stats


def answer_from_answer_only(question: str, context: str,
                            stats: dict) -> Tuple[str, str, dict]:
    """Answer from answer-only context.

    Handles degenerate case: if the session had no assistant prose at all, returns
    a stated placeholder rather than sending an empty context.

    The degenerate test reads the section count as an integer. It used to be the
    substring `"0 assistant sections shown" in context`, which also matches 10, 20,
    40 and 50 sections, so on 2026-10-02 it voided 25 of the 124 records in this arm
    to a canned `absent` grade with no API call. Every canned record was a false
    positive -- the arm had no truly empty session -- so never reintroduce a
    substring test over a formatted count.
    """
    if not context.strip() or stats.get("assistant_sections", 0) == 0:
        # Degenerate case: no assistant prose in session
        return ("This session had no assistant prose, so there is nothing to answer from.",
                "NOT_ANSWERABLE", {})

    prompt = f"""Someone asks: "{question}"

Below is the assistant's prose from an agent session. User prompts, tool calls, and tool results are withheld. Answer the question based ONLY on the assistant's narration. If it does not contain enough information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

ASSISTANT PROSE:
{context}

Provide your answer and confidence now:"""

    response, usage = call_llm(prompt, model=JUDGE_MODEL, timeout=600)
    if not response:
        return "", "API_ERROR", usage

    answer, confidence = extract_confidence(response)
    return answer, confidence, usage


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
            context, st = answer_only_context(
                Path(r["_transcript"]).read_text(errors="replace"))
            fits = len(context) / BYTES_PER_TOKEN <= CONTEXT_LIMIT_TOKENS
            rows.append((r["_prefix"], len(context), st.get('assistant_sections', 0), fits))
        rows.sort(key=lambda x: -x[1])
        print(f"\n{'sess':10s} {'context chars':>14s} {'asst_sections':>14s} {'fits':>6s}")
        for pre, ctx_chars, asst_secs, f in rows:
            print(f"{pre:10s} {ctx_chars:14,} {asst_secs:14d} {'yes' if f else 'NO':>6s}")
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
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        rid = record_id(r)
        if cache[0] != r["_transcript"]:
            context, st = answer_only_context(
                Path(r["_transcript"]).read_text(errors="replace"))
            cache = (r["_transcript"], context, st)
        context, st = cache[1], cache[2]

        if len(context) / BYTES_PER_TOKEN > CONTEXT_LIMIT_TOKENS:
            print(f"[{i}/{len(todo)}] {r['_prefix']} SKIP context too big "
                  f"({len(context)/1e6:.2f} MB)")
            continue

        answer, confidence, usage = answer_from_answer_only(r["question"], context, st)
        if confidence == "API_ERROR":
            errors.append({"record_id": rid, "prefix": r["_prefix"]})
            print(f"[{i}/{len(todo)}] {r['_prefix']} API_ERROR (not recorded, retry)")
            continue
        grade = grade_answer(answer, r.get("ground_truth", ""), confidence)
        print(f"[{i}/{len(todo)}] {r['_prefix']} {r.get('class'):15s} "
              f"{len(context)/1e3:6.1f}KB {confidence:18s} {grade:18s} "
              f"(rollup: {r.get('grade')})")

        out = {"record_id": rid, "session_id": r.get("session_id"),
               "class": r.get("class"), "question": r.get("question"),
               "ground_truth": r.get("ground_truth"),
               "published_grade": r.get("grade"),
               "transcript": r["_transcript"], "raw_bytes": r["_bytes"],
               "context_chars": len(context), "context_stats": st,
               "answer": answer, "rating": confidence, "grade": grade,
               "usage": usage, "model": JUDGE_MODEL}
        with OUT.open("a") as fh:
            fh.write(json.dumps(out) + "\n")
        done[rid] = out

    print(f"\nanswered {len(todo)-len(errors)} of {len(todo)} in "
          f"{time.time()-t0:.0f}s ({len(errors)} API errors)")

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
        "context": "answer-only: ASSISTANT sections only from strip_transcript(), no user prose, no tool calls, no tool results",
        "purpose": ("Decisive test of the paper's central claim. Tests whether assistant narration "
                    "alone carries the context. If it scores near prose-only (90.3%), assistant prose "
                    "is the active ingredient. If it scores near prompts-only (32.3%), the combination matters."),
        "n_api_errors_this_run": len(errors),
        "answer_only_arm_uncorrected": uncorrected_rates,
        "answer_only_arm_corrected": corrected_rates,
        "n_reclassified_by_confidence_correction": len(reclassified),
        "reclassified_records": reclassified,
        "median_context_chars": sorted(r["context_chars"] for r in rows)[len(rows)//2] if rows else None,
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
    print(line("answer-only (uncorrected)", uncorrected_rates))
    print(line("answer-only (corrected)", corrected_rates))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
