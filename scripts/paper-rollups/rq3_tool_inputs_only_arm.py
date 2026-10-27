#!/usr/bin/env python3
"""RQ3 ARM B: answer questions from CALL-ONLY context (tool inputs only).

Tests whether tool inputs alone carry the output-stripped arm's 90.3% recovery rate,
without any prose context. This is a NEGATIVE CONTROL: its input provably cannot
contain answers that require prose or reasoning, so a score above zero measures the
grader's priors, not the context's utility.

Context: tool call inputs ONLY, capped at 2000 chars per tool like output-stripped arm.
No user prose, no assistant prose, no tool results.

Predicted outcome: If tool inputs alone carry 90.3%, that would prove prose adds
nothing. More likely: tool inputs + prose together are necessary, so this arm
scores between dialogue record (71.8%, prose only) and output-stripped (90.3%, prose + inputs).

Construction: tool_use inputs only, zero prose, capped at MAX_TOOL_INPUT_CHARS = 2000.
This is a no-op control (its input provably cannot contain the graded answer), not a
competitor arm.

Usage:
    rq3_tool_inputs_only_arm.py --dry-run     # check sizes
    rq3_tool_inputs_only_arm.py --limit 5     # smoke test
    rq3_tool_inputs_only_arm.py               # full run
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

OUT = DATA / "rq3_tool_inputs_only_arm.jsonl"
REPORT = DATA / "rq3_tool_inputs_only_arm_summary.json"
JUDGE_MODEL = "claude-sonnet-5"

CONTEXT_LIMIT_TOKENS = 1_180_000
BYTES_PER_TOKEN = 2.45
MAX_TOOL_INPUT_CHARS = 2_000  # Match stripped arm's cap


def tool_inputs_only_context(raw: str) -> Tuple[str, dict]:
    """Extract tool call inputs ONLY, no prose or results.

    Returns (markdown, stats).
    """
    out = []
    stats = {"entries": 0, "tool_calls": 0, "truncated_inputs": 0}

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = rec.get("message")
        if not isinstance(msg, dict):
            continue

        stats["entries"] += 1
        content = msg.get("content")
        blocks = content if isinstance(content, list) else [content]

        for b in blocks:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use":
                stats["tool_calls"] += 1
                name = b.get("name") or "tool"
                inp = b.get("input")
                s = inp if isinstance(inp, str) else json.dumps(inp, indent=None)
                # Cap at 2000 chars like stripped arm
                if s and len(s) > MAX_TOOL_INPUT_CHARS:
                    s = s[:MAX_TOOL_INPUT_CHARS] + f"\n[... {len(s)-MAX_TOOL_INPUT_CHARS} chars truncated]"
                    stats["truncated_inputs"] += 1
                out.append(f"### TOOL: {name}\n```\n{s}\n```\n")

    header = (f"[This contains ONLY tool call inputs from an agent session. "
              f"{stats['entries']} entries, {stats['tool_calls']} tool calls shown "
              f"with inputs capped at {MAX_TOOL_INPUT_CHARS} chars. User/assistant "
              f"prose and tool results are withheld.]\n\n")
    return header + "".join(out), stats


def answer_from_tool_inputs_only(question: str, context: str) -> Tuple[str, str, dict]:
    """Answer from tool-inputs-only context.

    Handles degenerate case: if context is empty (session had no tool calls),
    returns a stated placeholder rather than sending an empty context.
    """
    if not context.strip() or context.strip() == "[This contains ONLY tool call inputs from an agent session. 0 entries, 0 tool calls shown with inputs capped at 2000 chars. User/assistant prose and tool results are withheld.]":
        # Degenerate case: no tool calls in session
        # Return NOT_ANSWERABLE with a note rather than grading as `absent`
        return ("This session had no tool calls, so there are no tool inputs to answer from.",
                "NOT_ANSWERABLE", {})

    prompt = f"""Someone asks: "{question}"

Below are the tool call inputs from an agent session. User/assistant prose and tool results are withheld. Answer the question based ONLY on these tool inputs. If they do not contain enough information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

TOOL INPUTS:
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
            context, st = tool_inputs_only_context(
                Path(r["_transcript"]).read_text(errors="replace"))
            fits = len(context) / BYTES_PER_TOKEN <= CONTEXT_LIMIT_TOKENS
            rows.append((r["_prefix"], len(context), st.get('tool_calls', 0), fits))
        rows.sort(key=lambda x: -x[1])
        print(f"\n{'sess':10s} {'context chars':>14s} {'tool_calls':>11s} {'fits':>6s}")
        for pre, ctx_chars, tools, f in rows:
            print(f"{pre:10s} {ctx_chars:14,} {tools:11d} {'yes' if f else 'NO':>6s}")
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
            context, st = tool_inputs_only_context(
                Path(r["_transcript"]).read_text(errors="replace"))
            cache = (r["_transcript"], context, st)
        context, st = cache[1], cache[2]

        if len(context) / BYTES_PER_TOKEN > CONTEXT_LIMIT_TOKENS:
            print(f"[{i}/{len(todo)}] {r['_prefix']} SKIP context too big "
                  f"({len(context)/1e6:.2f} MB)")
            continue

        answer, confidence, usage = answer_from_tool_inputs_only(r["question"], context)
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
        "context": "call-only: tool call inputs only, capped at 2000 chars per tool, no prose or results",
        "purpose": ("Negative control. Tests whether tool inputs alone carry output-stripped's "
                    "90.3% without prose. Its input provably cannot contain answers requiring prose."),
        "n_api_errors_this_run": len(errors),
        "tool_inputs_only_arm_uncorrected": uncorrected_rates,
        "tool_inputs_only_arm_corrected": corrected_rates,
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
    print(line("call-only (uncorrected)", uncorrected_rates))
    print(line("call-only (corrected)", corrected_rates))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
