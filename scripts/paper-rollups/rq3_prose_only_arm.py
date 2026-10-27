#!/usr/bin/env python3
"""RQ3 ARM C: answer questions from PROSE-ONLY context.

This is the critical control. Hypothesis: dialogue record (71.8% recoverable) and
output-stripped (90.3%) differ primarily in tool call inputs, which are 46.9% of
output-stripped. If prose alone lands near 71.8%, that proves tool inputs are the
necessary ingredient. If it lands significantly lower, prose is also limiting.

Context: user prose + assistant prose only, taken from strip_transcript() output
with tool call inputs removed. No tool results, no tool inputs.

Predicted outcome: ~71.8% recoverable (matching dialogue record), since both carry
prose without tool context.

Construction: output-stripped minus tool calls. User and assistant prose only, no
tool inputs or results.

Usage:
    rq3_prose_only_arm.py --dry-run        # check sizes, no API calls
    rq3_prose_only_arm.py --limit 5        # smoke test
    rq3_prose_only_arm.py                  # full run (resumable)
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

OUT = DATA / "rq3_prose_only_arm.jsonl"
REPORT = DATA / "rq3_prose_only_arm_summary.json"
JUDGE_MODEL = "claude-sonnet-5"

CONTEXT_LIMIT_TOKENS = 1_180_000
BYTES_PER_TOKEN = 2.45

# A resumed session opens with a harness-authored recap of the prior
# conversation, delivered as a user turn. It is neither user nor assistant
# prose, so this arm must not count it as either. Matched within the head of a
# USER block so a message that merely mentions resumption is not dropped.
CONTINUATION_OPENER = "This session is being continued from a previous conversation"
CONTINUATION_HEAD_CHARS = 500


def drop_continuation_summaries(prose: str) -> Tuple[str, int, int]:
    """Remove harness continuation summaries from already-prose markdown.

    Returns (prose, blocks_dropped, chars_dropped).
    """
    blocks, current = [], []
    for line in prose.split('\n'):
        if line.startswith('###'):
            if current:
                blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)

    kept, dropped_blocks, dropped_chars = [], 0, 0
    for block in blocks:
        is_user = block and block[0].startswith('### USER')
        body = '\n'.join(block[1:]) if is_user else ''
        if is_user and CONTINUATION_OPENER in body[:CONTINUATION_HEAD_CHARS]:
            dropped_blocks += 1
            dropped_chars += len('\n'.join(block))
            continue
        kept.extend(block)
    return '\n'.join(kept), dropped_blocks, dropped_chars


def prose_only_context(raw: str, keep_continuation: bool = False) -> Tuple[str, dict]:
    """Strip to prose only: user + assistant text, no tool calls or results.

    Harness continuation summaries are removed unless keep_continuation is set,
    which reproduces the pre-2026-10-01 behaviour behind the published figures.

    Returns (markdown, stats).
    """
    stripped, stats = strip_transcript(raw)

    # Remove ### TOOL CALL: sections
    # Exit on next ### boundary (safest - never depends on counting fences)
    lines = []
    in_tool_section = False
    for line in stripped.split('\n'):
        if line.startswith('### TOOL CALL:'):
            in_tool_section = True
            continue
        if in_tool_section:
            # Exit tool section on next ### boundary
            if line.startswith('###'):
                in_tool_section = False
                lines.append(line)
            # Otherwise skip everything in the tool section
            continue
        lines.append(line)

    prose = '\n'.join(lines)
    stats['prose_with_continuation_chars'] = len(prose)
    if keep_continuation:
        stats['continuation_blocks'] = 0
        stats['continuation_chars'] = 0
    else:
        prose, blocks, chars = drop_continuation_summaries(prose)
        stats['continuation_blocks'] = blocks
        stats['continuation_chars'] = chars
    stats['prose_only_chars'] = len(prose)
    return prose, stats


def answer_from_prose_only(question: str, prose: str) -> Tuple[str, str, dict]:
    """Answer from prose-only context."""
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
        rating = r.get('rating', '')
        grade = r.get('grade', '')

        # Check if answer body prominently contains NOT ANSWERABLE
        # (case-insensitive, with word boundaries)
        answer_upper = answer.upper()
        if rating != 'NOT_ANSWERABLE' and ('NOT ANSWERABLE' in answer_upper or 'NOT_ANSWERABLE' in answer_upper):
            # Reclassify as NOT_ANSWERABLE
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
    ap.add_argument("--keep-continuation", action="store_true",
                    help="keep harness continuation summaries (reproduces the "
                         "pre-2026-10-01 published figures)")
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
            prose, st = prose_only_context(
                Path(r["_transcript"]).read_text(errors="replace"),
                keep_continuation=args.keep_continuation)
            fits = len(prose) / BYTES_PER_TOKEN <= CONTEXT_LIMIT_TOKENS
            rows.append((r["_prefix"], len(prose), st.get('tool_calls', 0), fits,
                         st.get('continuation_chars', 0),
                         st.get('prose_with_continuation_chars', len(prose))))
        rows.sort(key=lambda x: -x[1])
        print(f"\n{'sess':10s} {'prose chars':>12s} {'tool_calls':>11s} "
              f"{'fits':>6s} {'cont chars':>11s} {'cont %':>7s}")
        for pre, prose_chars, tools, f, cont, before in rows:
            pct = 100.0 * cont / before if before else 0.0
            print(f"{pre:10s} {prose_chars:12,} {tools:11d} "
                  f"{'yes' if f else 'NO':>6s} {cont:11,} {pct:6.1f}%")
        affected = [r for r in rows if r[4] > 0]
        print(f"\ntotal sessions: {len(rows)}  "
              f"with continuation summary: {len(affected)}  "
              f"continuation chars total: {sum(r[4] for r in rows):,}")
        print("\n--dry-run: no API calls made")
        return

    # Check for existing output to avoid collision
    if OUT.exists():
        existing = list(load_records(OUT))
        if existing:
            unique_ids = {r.get('record_id') for r in existing}
            if len(existing) != len(unique_ids):
                print(f"ERROR: {OUT.name} has {len(existing)} records but only "
                      f"{len(unique_ids)} unique record_ids. Collision detected.")
                print("Move or delete the file before rerunning.")
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
            prose, st = prose_only_context(
                Path(r["_transcript"]).read_text(errors="replace"),
                keep_continuation=args.keep_continuation)
            cache = (r["_transcript"], prose, st)
        prose, st = cache[1], cache[2]

        if len(prose) / BYTES_PER_TOKEN > CONTEXT_LIMIT_TOKENS:
            print(f"[{i}/{len(todo)}] {r['_prefix']} SKIP prose too big "
                  f"({len(prose)/1e6:.2f} MB)")
            continue

        answer, confidence, usage = answer_from_prose_only(r["question"], prose)
        if confidence == "API_ERROR":
            errors.append({"record_id": rid, "prefix": r["_prefix"]})
            print(f"[{i}/{len(todo)}] {r['_prefix']} API_ERROR (not recorded, retry)")
            continue
        grade = grade_answer(answer, r.get("ground_truth", ""), confidence)
        print(f"[{i}/{len(todo)}] {r['_prefix']} {r.get('class'):15s} "
              f"{len(prose)/1e3:6.1f}KB {confidence:18s} {grade:18s} "
              f"(rollup: {r.get('grade')})")

        out = {"record_id": rid, "session_id": r.get("session_id"),
               "class": r.get("class"), "question": r.get("question"),
               "ground_truth": r.get("ground_truth"),
               "published_grade": r.get("grade"),
               "transcript": r["_transcript"], "raw_bytes": r["_bytes"],
               "prose_chars": len(prose), "prose_stats": st,
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
        if len(rows) != len(unique_ids):
            print("Duplicate record_ids detected. Check for collision.")

    # Apply confidence correction
    corrected_rows, reclassified = apply_confidence_correction(rows)

    # Compute rates
    uncorrected_rates = rates(rows)
    corrected_rates = rates(corrected_rows)

    report = {
        "n_questions": len(rows),
        "n_sessions": len({r["session_id"] for r in rows}),
        "judge": f"{JUDGE_MODEL} via local gateway, no temperature set",
        "context": "prose-only: user prose + assistant prose only, tool calls and results removed",
        "purpose": "Control arm. Tests hypothesis that prose alone matches dialogue record (71.8%).",
        "n_api_errors_this_run": len(errors),
        "prose_only_arm_uncorrected": uncorrected_rates,
        "prose_only_arm_corrected": corrected_rates,
        "n_reclassified_by_confidence_correction": len(reclassified),
        "reclassified_records": reclassified,
        "median_prose_chars": sorted(r["prose_chars"] for r in rows)[len(rows)//2] if rows else None,
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
    print(line("prose-only (uncorrected)", uncorrected_rates))
    print(line("prose-only (corrected)", corrected_rates))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
