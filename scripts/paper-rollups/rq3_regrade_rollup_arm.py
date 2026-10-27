#!/usr/bin/env python3
"""Re-grade the published RQ3 rollup answers, and isolate a grader parsing bug.

`part2_question_retention.py:249` parses the grader's one-word reply with

    for grade in ["CORRECT", "PARTIALLY_CORRECT", "WRONG", "ABSENT"]:
        if grade in response_upper:

"CORRECT" is a substring of "PARTIALLY_CORRECT", so a grader that replied
PARTIALLY_CORRECT was recorded as `correct`. `resume_rq3.py` and
`resume_rq3_v2.py` test PARTIALLY_CORRECT first and are unaffected. Both
versions were committed together, so git cannot say which script produced which
of the 124 published records -- only a re-grade can.

The measurement problem: re-grading with a fresh call mixes the bug with the
judge's own nondeterminism, and a difference from the stored grade could be
either. So the grader is called ONCE per record and its reply is parsed TWICE,
with the buggy order and the fixed order. Then:

  * buggy_parse != fixed_parse  -- attributable to the bug alone, zero noise,
    since both parsers read identical bytes.
  * fixed_parse != stored_grade -- judge nondeterminism plus, possibly, the bug
    having been present in the original run.

The bug is one-directional: it can only ever report `correct` where the grader
said PARTIALLY_CORRECT. It cannot manufacture a `wrong` or an `absent`. So it
cannot move `hallucination_of_attempted`, the 19.5% the paper argues from,
because that rate depends only on the wrong and absent counts. It can move
`correct` vs `partially_correct`, and it cannot move `recoverable` either, since
that is their sum. This script's job is to confirm that reasoning against data
rather than assert it.

Answers are NOT regenerated. The stored `judge_answer` is reused verbatim, so
what varies is the grading step alone -- the same design the paper's second and
third judge arm already uses.

Usage:
    rq3_regrade_rollup_arm.py --limit 5      # smoke test
    rq3_regrade_rollup_arm.py                # full run (resumable)
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grading_common import (  # noqa: E402
    GRADE_ERROR, PARSER_VERSION, PROMPT_VERSION, FailureCounter,
    grading_prompt as grading_prompt_v2, is_rule_absent, parse_verdict,
)
from rq3_raw_transcript_arm import (  # noqa: E402
    DATA, GRADES, QUESTIONS, SUMMARY, call_llm, load_records, rates, record_id,
)

OUT = DATA / "rq3_rollup_regrade.jsonl"
REPORT = DATA / "rq3_rollup_regrade_summary.json"
JUDGE_MODEL = "claude-sonnet-5"


def parse(response: str, order) -> str:
    """FIXED_ORDER -> the shared first-word parser (grade_error on anything that
    is not exactly one label). BUGGY_ORDER reproduces the retired substring scan,
    kept only so the bug's effect can still be isolated on identical bytes; it
    returns grade_error rather than `absent` on an empty reply."""
    if not response or not response.strip():
        return GRADE_ERROR
    if order is FIXED_ORDER:
        return parse_verdict(response)
    upper = response.strip().upper()
    for grade in order:
        if grade in upper:
            return grade.lower()
    return GRADE_ERROR


BUGGY_ORDER = ["CORRECT", "PARTIALLY_CORRECT", "WRONG", "ABSENT"]
FIXED_ORDER = ["PARTIALLY_CORRECT", "CORRECT", "WRONG", "ABSENT"]


def grading_prompt(judge_answer: str, ground_truth: str, question: str) -> str:
    """Production prompt (grading_common). Argument order kept for the importers
    rq3_majority_vote_grades.py and rq3_long_rollup_spike.py; `question` is
    required, so no importer can omit it."""
    return grading_prompt_v2(question, ground_truth, judge_answer)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    # Independent repeat draws. One re-grade cannot distinguish grader
    # nondeterminism from a systematic shift away from the published run; several
    # can, by showing whether the draws cluster or scatter.
    ap.add_argument("--tag", default="",
                    help="suffix for the output files, for repeat draws")
    args = ap.parse_args()

    global OUT, REPORT
    if args.tag:
        OUT = DATA / f"rq3_rollup_regrade_{args.tag}.jsonl"
        REPORT = DATA / f"rq3_rollup_regrade_summary_{args.tag}.json"

    graded = set(json.loads(SUMMARY.read_text())["graded_record_ids"])
    records = [r for r in load_records(QUESTIONS) if record_id(r) in graded]
    if len(records) != len(graded):
        print(f"FATAL: matched {len(records)} of {len(graded)} graded ids")
        sys.exit(1)
    records.sort(key=record_id)
    print(f"records to re-grade: {len(records)}")

    done = {}
    if OUT.exists():
        for r in load_records(OUT):
            done[r["record_id"]] = r
        print(f"resuming: {len(done)} already re-graded")

    todo = [r for r in records if record_id(r) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"to do: {len(todo)}\n")

    grade_failures = FailureCounter("regrade grader")
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        rid = record_id(r)
        stored = r.get("grade")
        answer = r.get("judge_answer") or ""
        rating = r.get("rating", "")

        # The by-rule absent path short-circuits before any grader call in the
        # original, so it is reproduced rather than re-graded. Calling the
        # grader on these would change the arm, not re-measure it.
        if is_rule_absent(answer, rating):
            out = {"record_id": rid, "session_id": r["session_id"],
                   "class": r.get("class"), "stored_grade": stored,
                   "grader_response": None, "by_rule_absent": True,
                   "fixed_grade": "absent", "buggy_grade": "absent",
                   "prompt_version": PROMPT_VERSION,
                   "parser_version": PARSER_VERSION}
        else:
            response, _ = call_llm(
                grading_prompt(answer, r.get("ground_truth", ""),
                               r.get("question", "")),
                model=JUDGE_MODEL, timeout=120)
            out = {"record_id": rid, "session_id": r["session_id"],
                   "class": r.get("class"), "stored_grade": stored,
                   "grader_response": response, "by_rule_absent": False,
                   "fixed_grade": parse(response, FIXED_ORDER),
                   "buggy_grade": parse(response, BUGGY_ORDER),
                   "prompt_version": PROMPT_VERSION,
                   "parser_version": PARSER_VERSION}
            grade_failures.record(out["fixed_grade"])
            if out["fixed_grade"] == GRADE_ERROR:
                # Not written: a rerun retries it. Never scored as `absent`.
                print(f"[{i}/{len(todo)}] {rid} grade_error (will retry)")
                grade_failures.check()
                continue
            flag = "  <-- BUG" if out["fixed_grade"] != out["buggy_grade"] else ""
            print(f"[{i}/{len(todo)}] {rid} stored={stored} "
                  f"fixed={out['fixed_grade']} buggy={out['buggy_grade']}{flag}")

        with OUT.open("a") as fh:
            fh.write(json.dumps(out) + "\n")
        done[rid] = out

    grade_failures.check(final=True)
    print(f"\nre-graded {len(todo)} in {time.time()-t0:.0f}s")

    rows = [done[record_id(r)] for r in records if record_id(r) in done]
    n_called = sum(1 for r in rows if not r["by_rule_absent"])

    # The bug, isolated: same response bytes, two parsers.
    bug_hits = [r for r in rows if r["fixed_grade"] != r["buggy_grade"]]
    # Judge nondeterminism plus any bug present in the original run.
    drift = [r for r in rows if r["fixed_grade"] != r["stored_grade"]]

    stored_rates = rates(records)
    fixed_rates = rates(rows, grade_key="fixed_grade")
    buggy_rates = rates(rows, grade_key="buggy_grade")

    # Did the published data come from the buggy parser? If it did, no stored
    # record should be partially_correct, since the buggy parser cannot emit it.
    stored_partials = sum(1 for r in records if r.get("grade") == "partially_correct")

    report = {
        "n_records": len(rows),
        "n_grader_calls": n_called,
        "n_by_rule_absent": len(rows) - n_called,
        "judge": f"{JUDGE_MODEL} via local gateway, no temperature set",
        "design": ("answers held fixed from rq3_questions.jsonl; the grader is "
                   "called once per record and its reply parsed twice, with the "
                   "buggy and the fixed order, so parser effect is separated "
                   "from judge nondeterminism"),
        "bug_effect_n": len(bug_hits),
        "bug_effect_records": [{"record_id": r["record_id"],
                                "fixed": r["fixed_grade"],
                                "buggy": r["buggy_grade"]} for r in bug_hits],
        "stored_partially_correct_n": stored_partials,
        "published_data_used_buggy_parser": stored_partials == 0,
        "regrade_drift_vs_stored_n": len(drift),
        "regrade_drift_matrix": dict(Counter(
            f"{r['stored_grade']}->{r['fixed_grade']}" for r in drift)),
        "published_rates": stored_rates,
        "regrade_fixed_parser_rates": fixed_rates,
        "regrade_buggy_parser_rates": buggy_rates,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")

    def line(label: str, d: dict) -> str:
        h = d["hallucination_of_attempted"]
        return (f"{label:26s} correct {d['correct']['n']:3d}  "
                f"partial {d['partially_correct']['n']:3d}  "
                f"wrong {d['wrong']['n']:3d}  absent {d['absent']['n']:3d}  "
                f"recoverable {d['recoverable']['pct']:5.1f}%  "
                f"wrong-of-att {h['pct']}% {h['ci95']}")

    print()
    print(line("published", stored_rates))
    print(line("re-grade, fixed parser", fixed_rates))
    print(line("re-grade, buggy parser", buggy_rates))
    print(f"\nbug effect (same bytes, two parsers): {len(bug_hits)} of {n_called} "
          f"graded records")
    print(f"stored partially_correct: {stored_partials} "
          f"-> published data used the buggy parser: {stored_partials == 0}")
    print(f"re-grade drift vs published: {len(drift)} of {len(rows)}")
    for k, v in sorted(report["regrade_drift_matrix"].items(), key=lambda kv: -kv[1]):
        print(f"    {k:38s} {v}")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
