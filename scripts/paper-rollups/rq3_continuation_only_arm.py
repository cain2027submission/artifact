#!/usr/bin/env python3
"""RQ3 ARM: answer questions from CONTINUATION-SUMMARY-ONLY context.

This tests a generated summary written by a system we did not build: the
Claude Code harness's continuation summary injected when a session is resumed.

The structural weakness this arm addresses: we are criticising generated summaries
using a summariser we wrote ourselves. A reviewer can dismiss the result as
"your rollup prompt is bad" rather than "generation loses." The continuation
summary is a third-party generated summary sitting in our corpus.

Context: only the harness continuation-summary text ("This session is being
continued..."). All other prose, tool calls, and tool results are excluded.

Predicted outcome: between rollup summary (56.5%) and prose-only (89.5%), closer
to the rollup, since it is a retrospective single-pass summary like ours.

If it performs near the rollup, the finding generalises beyond our own summariser.
If it performs near prose-only, that localises our result to our own prompt.

Coverage: Only ~11 of 25 sessions carry continuation text. Questions from sessions
without continuation are excluded with a recorded reason, never graded as absent.

Usage:
    rq3_continuation_only_arm.py --dry-run        # check sizes, no API calls
    rq3_continuation_only_arm.py --limit 5        # smoke test
    rq3_continuation_only_arm.py                  # full run (resumable)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_raw_transcript_arm import (  # noqa: E402
    DATA, QUESTIONS, SUMMARY, SLUG, call_llm, extract_confidence, grade_answer,
    index_transcripts, load_records, rates, record_id, resolve,
)
from rq3_stripped_transcript_arm import strip_transcript  # noqa: E402
from rq3_prose_only_arm import CONTINUATION_OPENER, CONTINUATION_HEAD_CHARS  # noqa: E402

OUT = DATA / "rq3_continuation_only_arm.jsonl"
REPORT = DATA / "rq3_continuation_only_arm_summary.json"
JUDGE_MODEL = "claude-sonnet-5"

CONTEXT_LIMIT_TOKENS = 1_180_000
BYTES_PER_TOKEN = 2.45


def extract_continuation_summaries(prose: str) -> Tuple[str, int, int]:
    """Extract ONLY harness continuation summaries from prose markdown.

    Returns (continuation_only, blocks_found, chars_found).

    This is the inverse of drop_continuation_summaries(): we keep only what
    that function drops, and drop everything else.
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

    kept, found_blocks, found_chars = [], 0, 0
    for block in blocks:
        is_user = block and block[0].startswith('### USER')
        body = '\n'.join(block[1:]) if is_user else ''
        if is_user and CONTINUATION_OPENER in body[:CONTINUATION_HEAD_CHARS]:
            found_blocks += 1
            found_chars += len('\n'.join(block))
            kept.extend(block)
            continue
        # Drop everything else

    return '\n'.join(kept), found_blocks, found_chars


def continuation_only_context(raw: str) -> Tuple[str, dict]:
    """Strip to continuation-only: harness continuation summaries ONLY.

    All user prose, assistant prose, tool calls, and tool results are removed.

    Returns (markdown, stats).
    """
    stripped, stats = strip_transcript(raw)

    # Remove ### TOOL CALL: sections
    lines = []
    in_tool_section = False
    for line in stripped.split('\n'):
        if line.startswith('### TOOL CALL:'):
            in_tool_section = True
            continue
        if in_tool_section:
            if line.startswith('###'):
                in_tool_section = False
                lines.append(line)
            continue
        lines.append(line)

    prose = '\n'.join(lines)

    # Extract ONLY continuation summaries
    continuation, blocks, chars = extract_continuation_summaries(prose)
    stats['continuation_blocks'] = blocks
    stats['continuation_chars'] = chars
    return continuation, stats


def answer_from_continuation_only(question: str, continuation: str) -> Tuple[str, str, dict]:
    """Answer from continuation-only context."""
    prompt = f"""Someone asks: "{question}"

Below is a transcript containing ONLY harness continuation summaries (the "This session is being continued from a previous conversation" blocks that Claude Code injects when resuming a session). All other prose, tool calls, and tool results have been removed. Answer the question based ONLY on these continuation summaries. If they do not contain enough information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

CONTINUATION SUMMARIES:
{continuation}

Provide your answer and confidence now:"""

    response, usage = call_llm(prompt, model=JUDGE_MODEL, timeout=600)
    if not response:
        return "", "API_ERROR", usage

    answer, confidence = extract_confidence(response)
    return answer, confidence, usage


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
            continuation, st = continuation_only_context(
                Path(r["_transcript"]).read_text(errors="replace"))
            fits = len(continuation) / BYTES_PER_TOKEN <= CONTEXT_LIMIT_TOKENS
            rows.append((r["_prefix"], len(continuation), st.get('continuation_blocks', 0),
                         st.get('continuation_chars', 0), fits))
        rows.sort(key=lambda x: -x[3])
        print(f"\n{'sess':10s} {'cont chars':>11s} {'blocks':>8s} {'chars':>11s} {'fits':>6s}")
        for pre, cont_len, blocks, chars, f in rows:
            print(f"{pre:10s} {cont_len:11,} {blocks:8d} {chars:11,} "
                  f"{'yes' if f else 'NO':>6s}")

        with_continuation = [r for r in rows if r[3] > 0]
        total_chars = sum(r[3] for r in rows)
        print(f"\ntotal sessions: {len(rows)}  "
              f"with continuation summary: {len(with_continuation)}  "
              f"continuation chars total: {total_chars:,}")

        # Count questions per session
        questions_with_cont = 0
        sessions_with_cont = {r[0] for r in with_continuation}
        for r in eligible:
            if r["_prefix"] in sessions_with_cont:
                questions_with_cont += 1

        print(f"\nquestions from sessions with continuation: {questions_with_cont} of {len(eligible)}")
        print(f"questions from sessions without continuation: {len(eligible) - questions_with_cont}")

        if with_continuation:
            median_idx = len(with_continuation) // 2
            median_chars = sorted([r[3] for r in with_continuation])[median_idx]
            print(f"\nmedian continuation chars (of {len(with_continuation)} sessions with continuation): {median_chars:,}")

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

    cache: Tuple[str | None, str, dict] = (None, "", {})
    errors: List[dict] = []
    excluded: List[dict] = []
    t0 = time.time()

    for i, r in enumerate(todo, 1):
        rid = record_id(r)
        if cache[0] != r["_transcript"]:
            continuation, st = continuation_only_context(
                Path(r["_transcript"]).read_text(errors="replace"))
            cache = (r["_transcript"], continuation, st)
        continuation, st = cache[1], cache[2]

        # Exclude questions from sessions with no continuation text
        if st.get('continuation_chars', 0) == 0:
            excluded.append({
                "record_id": rid,
                "session_id": r.get("session_id"),
                "class": r.get("class"),
                "reason": "no_continuation_text_in_session",
                "question": r.get("question")
            })
            print(f"[{i}/{len(todo)}] {r['_prefix']} EXCLUDED (no continuation text)")
            continue

        if len(continuation) / BYTES_PER_TOKEN > CONTEXT_LIMIT_TOKENS:
            print(f"[{i}/{len(todo)}] {r['_prefix']} SKIP continuation too big "
                  f"({len(continuation)/1e6:.2f} MB)")
            continue

        answer, confidence, usage = answer_from_continuation_only(r["question"], continuation)
        if confidence == "API_ERROR":
            errors.append({"record_id": rid, "prefix": r["_prefix"]})
            print(f"[{i}/{len(todo)}] {r['_prefix']} API_ERROR (not recorded, retry)")
            continue
        grade = grade_answer(answer, r.get("ground_truth", ""), confidence)
        print(f"[{i}/{len(todo)}] {r['_prefix']} {r.get('class'):15s} "
              f"{len(continuation)/1e3:6.1f}KB {confidence:18s} {grade:18s} "
              f"(rollup: {r.get('grade')})")

        out = {"record_id": rid, "session_id": r.get("session_id"),
               "class": r.get("class"), "question": r.get("question"),
               "ground_truth": r.get("ground_truth"),
               "published_grade": r.get("grade"),
               "transcript": r["_transcript"], "raw_bytes": r["_bytes"],
               "continuation_chars": len(continuation), "continuation_stats": st,
               "answer": answer, "rating": confidence, "grade": grade,
               "usage": usage, "model": JUDGE_MODEL}
        with OUT.open("a") as fh:
            fh.write(json.dumps(out) + "\n")
        done[rid] = out

    print(f"\nanswered {len(todo)-len(errors)-len(excluded)} of {len(todo)} in "
          f"{time.time()-t0:.0f}s ({len(errors)} API errors, {len(excluded)} excluded)")

    # Report on excluded questions
    if excluded:
        print(f"\nEXCLUDED {len(excluded)} questions from sessions with no continuation text:")
        by_session = {}
        for e in excluded:
            sess = e['session_id'][:8] if e.get('session_id') else 'unknown'
            by_session.setdefault(sess, []).append(e)
        for sess, items in sorted(by_session.items()):
            print(f"  {sess}: {len(items)} questions")

    # Load all done records (including pre-existing)
    rows = list(done.values())
    unique_ids = {r['record_id'] for r in rows}

    # Count denominator: questions from sessions with continuation
    total_questions = len(eligible)
    questions_with_continuation = len(rows)
    questions_excluded = len(excluded)

    print(f"\nTotal questions in corpus: {total_questions}")
    print(f"Questions from sessions with continuation: {questions_with_continuation}")
    print(f"Questions excluded (no continuation): {questions_excluded}")

    if len(rows) != len(unique_ids):
        print(f"\nWARNING: {len(rows)} records but {len(unique_ids)} unique IDs. Collision detected.")

    # Compute rates ONLY over questions with continuation context
    arm_rates = rates(rows)

    report = {
        "n_questions_total": total_questions,
        "n_questions_with_continuation": questions_with_continuation,
        "n_questions_excluded": questions_excluded,
        "n_sessions_total": len({r["session_id"] for r in eligible}),
        "n_sessions_with_continuation": len({r["session_id"] for r in rows}),
        "judge": f"{JUDGE_MODEL} via local gateway, no temperature set",
        "context": "continuation-only: harness continuation summaries ONLY, all other content removed",
        "purpose": "Tests third-party generated summary (harness continuation, not our rollup) against our rollup and prose-only arms.",
        "prediction": "Between rollup (56.5%) and prose-only (89.5%), closer to rollup.",
        "n_api_errors_this_run": len(errors),
        "n_excluded_no_continuation": len(excluded),
        "excluded_records": excluded,
        "continuation_only_arm": arm_rates,
        "median_continuation_chars": sorted(r["continuation_chars"] for r in rows)[len(rows)//2] if rows else None,
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
    print(line("continuation-only", arm_rates))
    print(f"\nDenominator: {questions_with_continuation} questions from {report['n_sessions_with_continuation']} sessions with continuation")
    print(f"Excluded: {questions_excluded} questions from sessions without continuation")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
