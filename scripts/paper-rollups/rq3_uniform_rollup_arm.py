#!/usr/bin/env python3
"""Re-run the RQ3 rollup arm under a single, uniform protocol.

The published rollup arm is not one experiment. Its 124 records were produced by
up to three scripts that differ in ways that change the measurement:

  part2_question_retention.py   answering: "Below is a compressed note
                               summarizing an agent session"; grading prompt
                               DEFINES each grade; grade parser BUGGY
                               (tests CORRECT before PARTIALLY_CORRECT)
  resume_rq3.py                answering: "Can you answer this question from
                               the note below?"; grading prompt gives the four
                               words with NO definitions; parser fixed
  resume_rq3_v2.py             answering: "Answer this question from the note,
                               or say you can't" -- and TRUNCATES the rollup to
                               15,000 chars; grading prompt terser still;
                               parser fixed

Three consequences, in descending order of how much they matter:

1. Pooling them means a record's grade depends on which script happened to reach
   it, and nothing in the data records which one did. Both commits landed
   together, so git cannot attribute records either.
2. v2's `rollup_content[:15000]` clip means an `absent` there may be truncation
   rather than compression loss. Bounded: exactly one of the 25 graded sessions
   has a rollup over 15,000 chars (max 54,619; median 3,528), covering 4 of the
   124 questions. So this is a real but small defect.
3. The buggy parser cannot affect the headline. It only ever turns
   PARTIALLY_CORRECT into `correct`, so `wrong` and `absent` are untouched, and
   `hallucination_of_attempted` = wrong/(n-absent) is immune -- as is
   `recoverable`, which is the sum of the two buckets it moves between.
   Confirmed empirically in rq3_regrade_rollup_arm.py: parsing identical grader
   replies both ways gave identical wrong/absent/recoverable.

So the rerun is not needed to fix the parser bug. It is needed because the arm
mixes protocols, and because a paired comparison against the raw-transcript arm
needs a rollup arm whose prompts match the raw arm's exactly.

This script re-answers every question from the FULL rollup with the answering
prompt of `rq3_raw_transcript_arm.py`, word for word except that the context is
described as a compressed note rather than a transcript, and grades with the one
grading prompt and the fixed parser. Paired against the raw arm, the only
difference left between the two is the context.

Usage:
    rq3_uniform_rollup_arm.py --dry-run
    rq3_uniform_rollup_arm.py --limit 3
    rq3_uniform_rollup_arm.py
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import SESSIONS_DIR
from rq3_raw_transcript_arm import (  # noqa: E402
    DATA, QUESTIONS, SUMMARY, call_llm, extract_confidence, grade_answer,
    load_records, rates, record_id, SLUG,
)

OUT = DATA / "rq3_rollup_uniform.jsonl"
REPORT = DATA / "rq3_rollup_uniform_summary.json"
JUDGE_MODEL = "claude-sonnet-5"

# The rollups live in the shared checkout, not in this worktree: they are written
# by the rollup pipeline on main and may postdate this branch point.
ROLLUP_DIRS = [
    SESSIONS_DIR,
    Path(__file__).resolve().parents[2] / "sessions",
]


def find_rollup(session_id: str) -> Optional[Path]:
    for d in ROLLUP_DIRS:
        p = d / f"{session_id}.md"
        if p.exists():
            return p
    # A rollup may have been renamed since it was graded -- the heading often is
    # edited. Fall back to the 8-hex session id, which is stable.
    m = SLUG.match(session_id)
    if m:
        for d in ROLLUP_DIRS:
            if not d.is_dir():
                continue
            hits = sorted(d.glob(f"*_{m.group(1)}_*.md"))
            if hits:
                return hits[0]
    return None


def answer_from_rollup(question: str, rollup_content: str):
    """Identical to rq3_raw_transcript_arm.answer_from_transcript except for the
    two words naming the context. No truncation of rollup_content."""
    prompt = f"""Someone asks: "{question}"

Below is a compressed note summarizing an agent session. Answer the question based ONLY on this note. If the note does not contain enough information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

NOTE:
{rollup_content}

Provide your answer and confidence now:"""

    response, usage = call_llm(prompt, model=JUDGE_MODEL, timeout=180)
    if not response:
        # Never NOT_ANSWERABLE: that grades as `absent`, the very outcome under
        # measurement, so an API failure would masquerade as a finding.
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
    if len(records) != len(graded):
        print(f"FATAL: matched {len(records)} of {len(graded)} graded ids")
        sys.exit(1)
    records.sort(key=record_id)

    resolved, missing = [], []
    for r in records:
        p = find_rollup(r["session_id"])
        if p is None:
            missing.append(r)
            continue
        resolved.append(dict(r, _rollup=str(p), _chars=len(p.read_text(errors="replace"))))

    print(f"records {len(records)}  rollup resolved {len(resolved)}  "
          f"missing {len(missing)}")
    for s in sorted({r["session_id"] for r in missing}):
        print(f"   MISSING ROLLUP {s}")
    clipped = sorted({r["session_id"] for r in resolved if r["_chars"] > 15000})
    print(f"sessions a 15000-char clip would have truncated: {len(clipped)} "
          f"({sum(1 for r in resolved if r['_chars'] > 15000)} questions)")

    if args.dry_run:
        print("\n--dry-run: no API calls made")
        return

    done = {}
    if OUT.exists():
        for r in load_records(OUT):
            done[r["record_id"]] = r
        print(f"resuming: {len(done)} already answered")

    todo = [r for r in resolved if record_id(r) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"to answer: {len(todo)}\n")

    errors: List[dict] = []
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        rid = record_id(r)
        content = Path(r["_rollup"]).read_text(errors="replace")
        answer, confidence, usage = answer_from_rollup(r["question"], content)
        if confidence == "API_ERROR":
            errors.append({"record_id": rid, "session_id": r["session_id"]})
            print(f"[{i}/{len(todo)}] {rid} API_ERROR (not recorded, will retry)")
            continue
        grade = grade_answer(answer, r.get("ground_truth", ""), confidence)
        print(f"[{i}/{len(todo)}] {rid} {r.get('class'):15s} "
              f"{confidence:18s} {grade:18s} (published: {r.get('grade')})")

        out = {"record_id": rid, "session_id": r["session_id"],
               "class": r.get("class"), "question": r.get("question"),
               "ground_truth": r.get("ground_truth"),
               "published_grade": r.get("grade"),
               "rollup": r["_rollup"], "rollup_chars": r["_chars"],
               "answer": answer, "rating": confidence, "grade": grade,
               "usage": usage, "model": JUDGE_MODEL}
        with OUT.open("a") as fh:
            fh.write(json.dumps(out) + "\n")
        done[rid] = out

    print(f"\nanswered {len(todo)-len(errors)} of {len(todo)} in "
          f"{time.time()-t0:.0f}s ({len(errors)} API errors)")

    rows = list(done.values())
    ids = {r["record_id"] for r in rows}
    published_same = [r for r in records if record_id(r) in ids]

    report = {
        "n_questions": len(rows),
        "n_sessions": len({r["session_id"] for r in rows}),
        "judge": f"{JUDGE_MODEL} via local gateway, no temperature set",
        "protocol": ("single uniform protocol: full untruncated rollup, the "
                     "answering prompt of rq3_raw_transcript_arm.py differing "
                     "only in naming the context a compressed note, one grading "
                     "prompt, fixed grade parser"),
        "why": ("the published arm pooled up to three scripts with different "
                "answering prompts, different grading prompts, different grade "
                "parsers, and in one case a 15000-char clip on the rollup"),
        "n_missing_rollups": len({r["session_id"] for r in missing}),
        "missing_rollup_sessions": sorted({r["session_id"] for r in missing}),
        "n_api_errors_this_run": len(errors),
        "uniform_rollup_arm": rates(rows),
        "published_arm_same_records": rates(published_same),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")

    def line(label, d):
        h = d["hallucination_of_attempted"]
        return (f"{label:28s} n={d['n']:3d} correct {d['correct']['n']:3d} "
                f"partial {d['partially_correct']['n']:3d} "
                f"wrong {d['wrong']['n']:3d} absent {d['absent']['n']:3d}  "
                f"recoverable {d['recoverable']['pct']:5.1f}%  "
                f"wrong-of-att {h['pct']}% {h['ci95']}")

    print()
    print(line("published (same records)", rates(published_same)))
    print(line("uniform rollup arm", rates(rows)))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
