#!/usr/bin/env python3
"""RQ3 ARM: answer questions from ACTION RECORD WITHOUT RESULT DIGESTS.

Ablation of rq3_action_record_arm.py: same structured tool-call extraction and
turn ordering, but tool results are dropped entirely (same as output-stripped)
instead of digested. Tests whether the +887-token result-digest cost (median
6,393 -> 5,506 cl100k tokens) is buying any recall, given the digests' own
justification (closing an output-stripped-vs-raw gap) was already shown false.

Usage:
    rq3_action_record_no_digest_arm.py --dry-run
    rq3_action_record_no_digest_arm.py --limit 10    # pilot
    rq3_action_record_no_digest_arm.py                # full 124
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
from spinoff_actionrecord_ablation import build_action_record_no_result_digests  # noqa: E402
from rq3_action_record_arm import apply_confidence_correction  # noqa: E402

OUT = DATA / "rq3_action_record_no_digest_arm.jsonl"
REPORT = DATA / "rq3_action_record_no_digest_arm_summary.json"
JUDGE_MODEL = "claude-sonnet-5"

CONTEXT_LIMIT_TOKENS = 1_180_000
BYTES_PER_TOKEN = 3.74  # measured real cl100k ratio for this format, not the old 2.45


def answer_from_record(question: str, record_text: str, return_raw: bool = False):
    prompt = f"""Someone asks: "{question}"

Below is a structured action record of an agent session, with tool calls shown as name + salient arguments and turn ordering. Tool results are NOT included. Answer the question based ONLY on this record. If it does not contain enough information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

ACTION RECORD (no result digests):
{record_text}

Provide your answer and confidence now:"""

    response, usage = call_llm(prompt, model=JUDGE_MODEL, timeout=600)
    if not response:
        return ("", "API_ERROR", usage, response) if return_raw \
            else ("", "API_ERROR", usage)

    answer, confidence = extract_confidence(response)
    return (answer, confidence, usage, response) if return_raw \
        else (answer, confidence, usage)


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
                record_text, st = build_action_record_no_result_digests(r["_transcript"])
                fits = len(record_text) / BYTES_PER_TOKEN <= CONTEXT_LIMIT_TOKENS
                rows.append((r["_prefix"], len(record_text), fits))
            except Exception as e:
                print(f"ERROR building record for {r['_prefix']}: {e}")
        rows.sort(key=lambda x: -x[1])
        for pre, chars, f in rows:
            print(f"{pre:10s} {chars:10,} {'yes' if f else 'NO':>4s}")
        print(f"\ntotal sessions: {len(rows)}\n--dry-run: no API calls made")
        return

    if OUT.exists():
        existing = list(load_records(OUT))
        done = {r.get('record_id'): r for r in existing} if existing else {}
        if existing:
            print(f"resuming: {len(existing)} already answered")
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
                record_text, st = build_action_record_no_result_digests(r["_transcript"])
                cache = (r["_transcript"], record_text, st)
            except Exception as e:
                print(f"[{i}/{len(todo)}] {r['_prefix']} BUILD_ERROR: {e}")
                build_errors.append({"record_id": rid, "prefix": r["_prefix"], "error": str(e)})
                continue
        record_text, st = cache[1], cache[2]

        if len(record_text) / BYTES_PER_TOKEN > CONTEXT_LIMIT_TOKENS:
            print(f"[{i}/{len(todo)}] {r['_prefix']} SKIP too big")
            continue

        answer, confidence, usage, raw_reply = answer_from_record(
            r["question"], record_text, return_raw=True)
        if confidence == "API_ERROR":
            errors.append({"record_id": rid, "prefix": r["_prefix"]})
            print(f"[{i}/{len(todo)}] {r['_prefix']} API_ERROR (not recorded, retry)")
            continue
        grade = grade_answer(answer, r.get("ground_truth", ""), confidence)
        print(f"[{i}/{len(todo)}] {r['_prefix']} {r.get('class'):15s} "
              f"{len(record_text)/1e3:6.1f}KB {confidence:18s} {grade:18s} "
              f"(rollup: {r.get('grade')})")

        out = {"record_id": rid, "session_id": r.get("session_id"),
               "class": r.get("class"), "question": r.get("question"),
               "ground_truth": r.get("ground_truth"),
               "published_grade": r.get("grade"),
               "transcript": r["_transcript"], "raw_bytes": r["_bytes"],
               "record_chars": len(record_text), "record_stats": st,
               "answer": answer, "rating": confidence, "grade": grade,
               "raw_reply": raw_reply,
               "usage": usage, "model": JUDGE_MODEL}
        with OUT.open("a") as fh:
            fh.write(json.dumps(out) + "\n")
        done[rid] = out

    print(f"\nanswered {len(todo)-len(errors)-len(build_errors)} of {len(todo)} in "
          f"{time.time()-t0:.0f}s ({len(errors)} API errors, {len(build_errors)} build errors)")

    rows = list(done.values())
    corrected_rows, reclassified = apply_confidence_correction(rows)
    uncorrected_rates = rates(rows)
    corrected_rates = rates(corrected_rows)

    report = {
        "n_questions": len(rows),
        "n_sessions": len({r["session_id"] for r in rows}),
        "judge": f"{JUDGE_MODEL} via local gateway, no temperature set",
        "context": "action record WITHOUT result digests: structured tool calls, prose, turn ordering, no tool results",
        "purpose": "Tests whether dropping result digests from the action record costs recall vs full action record (96.8%) / output-stripped (90.3%-94.3%).",
        "n_api_errors_this_run": len(errors),
        "n_build_errors_this_run": len(build_errors),
        "build_errors": build_errors,
        "no_digest_arm_uncorrected": uncorrected_rates,
        "no_digest_arm_corrected": corrected_rates,
        "n_reclassified_by_confidence_correction": len(reclassified),
        "reclassified_records": reclassified,
        "median_record_chars": sorted(r["record_chars"] for r in rows)[len(rows)//2] if rows else None,
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
    print(line("no-digest action record (uncorrected)", uncorrected_rates))
    print(line("no-digest action record (corrected)", corrected_rates))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
