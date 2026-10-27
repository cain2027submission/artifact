#!/usr/bin/env python3
"""Gate for the confidence/grade parsing in part2_question_retention.py.

The four assertions correspond one-to-one to the defects the old substring loop
had, plus the API_ERROR behaviour that must be preserved. The fifth test is a
CONTROL: it runs the old parser, verbatim, over the same cases and fails the run
if the old parser does NOT fail at least two of them. A gate that the pre-fix
code also passes is not a gate.

Run:
    python3 test_question_retention_parser.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import part2_question_retention as p2  # noqa: E402
from rq3_raw_transcript_arm import extract_confidence  # noqa: E402


# --------------------------------------------------------------------------- #
# The pre-fix parser, copied verbatim from part2_question_retention.py before
# this change (lines 205-217 of the original). Used only as the control.
# --------------------------------------------------------------------------- #
def old_parser(response: str) -> Tuple[str, str]:
    lines = response.strip().split("\n")
    confidence = "NOT_ANSWERABLE"
    for keyword in ["ANSWERED", "PARTIALLY_ANSWERED", "NOT_ANSWERABLE"]:
        if keyword in lines[-1].upper():
            confidence = keyword
            break
    answer = "\n".join(lines[:-1]).strip()
    return answer, confidence


# --------------------------------------------------------------------------- #
# Cases. Each is (name, reply, expected_confidence, must_survive_in_answer).
# `must_survive_in_answer` is a substring that a correct parser may not drop.
# --------------------------------------------------------------------------- #
PARTIAL_TRAILING = (
    "The note says the daemon was restarted but not why.\n"
    "\n"
    "PARTIALLY_ANSWERED"
)
REFUSAL_WITH_ANSWERED_PROSE = (
    "NOT ANSWERABLE\n"
    "\n"
    "The question cannot be answered from this note; the point is left "
    "unanswered and no value is recorded.\n"
    "NOT_ANSWERABLE"
)
MARKER_NOT_LAST = (
    "ANSWERED\n"
    "\n"
    "The measured figure is 2.53 bytes per token.\n"
    "That number came from a 124,092 byte transcript billing 49,090 tokens."
)
LABELLED_PARTIAL = (
    "Only half of it is in the note.\n"
    "**Confidence:** PARTIALLY_ANSWERED"
)
# Not invented: this is the judge_answer stored at line 126 of
# data/paper-rollups/rq3_questions.jsonl, the only slice of that file the old
# part2 parser produced. The marker is on the FIRST line and prose on the last,
# so the old parser reads it as ANSWERED and throws the prose away.
REAL_RECORD_126 = (
    "NOT ANSWERABLE\n"
    "\n"
    "ANSWERED note: The note does not mention any command-line tools that the "
    "assistant checked for and found not installed. It only references "
    '"displayplacer" as a suggested tool.'
)

CASES = [
    ("trailing PARTIALLY_ANSWERED is not ANSWERED",
     PARTIAL_TRAILING, "PARTIALLY_ANSWERED",
     "The note says the daemon was restarted"),
    ("a refusal whose prose says 'answered' stays a refusal",
     REFUSAL_WITH_ANSWERED_PROSE, "NOT_ANSWERABLE",
     "left unanswered and no value is recorded"),
    ("a marker off the final line is found and nothing is truncated",
     MARKER_NOT_LAST, "ANSWERED",
     "That number came from a 124,092 byte transcript"),
    ("a labelled marker line still parses as a partial",
     LABELLED_PARTIAL, "PARTIALLY_ANSWERED",
     "Only half of it is in the note."),
    ("real record 126: refusal first, 'ANSWERED' in trailing prose",
     REAL_RECORD_126, "NOT_ANSWERABLE",
     "It only references"),
]


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  <- {detail}" if not ok and detail else ""))
    return ok


def main() -> int:
    failures = 0

    print("fixed parser (extract_confidence, shared with the three RQ3 arms)")
    for name, reply, want_conf, want_sub in CASES:
        answer, conf = extract_confidence(reply)
        if not check(f"{name}: confidence == {want_conf}", conf == want_conf,
                     f"got {conf}"):
            failures += 1
        if not check(f"{name}: answer keeps its content",
                     want_sub in answer, f"missing {want_sub!r} from {answer!r}"):
            failures += 1

    print("\nempty reply is API_ERROR, never NOT_ANSWERABLE")
    # judge_question_from_rollup returns before any network call when the reply is
    # empty, so this is exercised by stubbing call_llm. Zero model calls.
    real_call = p2.call_llm
    try:
        p2.call_llm = lambda prompt, model: ""
        answer, conf, raw = p2.judge_question_from_rollup("q?", "a note")
    finally:
        p2.call_llm = real_call
    if not check("confidence == API_ERROR", conf == "API_ERROR", f"got {conf}"):
        failures += 1
    if not check("answer is empty", answer == "", f"got {answer!r}"):
        failures += 1
    if not check("raw reply is empty", raw == "", f"got {raw!r}"):
        failures += 1
    if not check("API_ERROR grades to api_error, not absent",
                 p2.grade_answer("", "gt", "API_ERROR") == "api_error",
                 f"got {p2.grade_answer('', 'gt', 'API_ERROR')}"):
        failures += 1

    print("\nraw reply is stored so a future parser change is measurable")
    try:
        p2.call_llm = lambda prompt, model: PARTIAL_TRAILING
        answer, conf, raw = p2.judge_question_from_rollup("q?", "a note")
    finally:
        p2.call_llm = real_call
    if not check("raw_reply round-trips the reply verbatim",
                 raw == PARTIAL_TRAILING, f"got {raw!r}"):
        failures += 1
    if not check("confidence is PARTIALLY_ANSWERED end to end",
                 conf == "PARTIALLY_ANSWERED", f"got {conf}"):
        failures += 1

    print("\nPARTIALLY_ANSWERED is capped at partially_correct, and only that")
    # grade_answer would call the gateway, so the grader reply is stubbed. The
    # point under test is the post-parse cap, not the grader.
    try:
        p2.call_llm = lambda prompt, model: "CORRECT"
        capped = p2.grade_answer("some answer", "gt", "PARTIALLY_ANSWERED")
        uncapped = p2.grade_answer("some answer", "gt", "ANSWERED")
        p2.call_llm = lambda prompt, model: "WRONG"
        wrong_partial = p2.grade_answer("some answer", "gt", "PARTIALLY_ANSWERED")
        p2.call_llm = lambda prompt, model: "PARTIALLY_CORRECT"
        partial_order = p2.grade_answer("some answer", "gt", "ANSWERED")
    finally:
        p2.call_llm = real_call
    if not check("PARTIALLY_ANSWERED + CORRECT -> partially_correct",
                 capped == "partially_correct", f"got {capped}"):
        failures += 1
    if not check("ANSWERED + CORRECT -> correct", uncapped == "correct",
                 f"got {uncapped}"):
        failures += 1
    if not check("the cap never touches `wrong` (so wrong-of-attempted is fixed)",
                 wrong_partial == "wrong", f"got {wrong_partial}"):
        failures += 1
    if not check("grade parser tests PARTIALLY_CORRECT before CORRECT",
                 partial_order == "partially_correct", f"got {partial_order}"):
        failures += 1

    # ----------------------------------------------------------------------- #
    print(f"\nCONTROL: the OLD parser must fail >= 2 of the {len(CASES)} cases")
    old_failures = []
    for name, reply, want_conf, want_sub in CASES:
        answer, conf = old_parser(reply)
        if conf != want_conf:
            old_failures.append(f"{name}: label {conf} != {want_conf}")
        elif want_sub not in answer:
            old_failures.append(f"{name}: answer truncated, lost {want_sub!r}")
    for f in old_failures:
        print(f"  old parser wrong on -> {f}")
    if not check(f"old parser fails >= 2 cases (it fails {len(old_failures)})",
                 len(old_failures) >= 2):
        failures += 1

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{failures} CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
