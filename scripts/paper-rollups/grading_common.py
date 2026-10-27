#!/usr/bin/env python3
"""The one place RQ3 grading is defined: prompt, verdict parser, refusal rule.

Every grading script imports from here instead of keeping its own copy. Before
this module there were five copies of the verdict parser and four prompts, and
the copies disagreed in ways that moved published grades:

  * A substring scan ("CORRECT" in reply) reads "INCORRECT" as correct and
    "not WRONG" as wrong, and takes the first label of "CORRECT or WRONG".
    The parser here matches the reply's FIRST WHOLE WORD against the four
    labels and treats anything else as `grade_error`, never as a grade.
  * A grader call that failed (gateway error, empty reply) returned `absent`,
    which is a measured outcome. grade_with now RAISES GradeError, so a failure
    cannot be written as a grade: the caller leaves the row unwritten and a
    rerun retries it.
  * An answer whose first line is NOT ANSWERABLE was still sent to the grader
    when a later ANSWERED marker won, and could be scored WRONG. A first-line
    refusal is now `absent` by rule, with no model call.

Bump PROMPT_VERSION when the grading prompt changes and PARSER_VERSION when the
parser or the refusal rule changes; both are stored on every output row next to
the raw grader reply so a later audit can re-parse without re-calling.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

PROMPT_VERSION = "rq3-grade-prompt-v2-question"
PARSER_VERSION = "rq3-verdict-parser-v2-firstword"

GRADES = ("correct", "partially_correct", "wrong", "absent")
GRADE_ERROR = "grade_error"

# Abort a run once this fraction of its grading calls has failed. A gateway
# outage otherwise looks like a plausible-but-wrong result.
MAX_FAILURE_FRACTION = 0.10
# Do not judge the fraction on a tiny denominator (1 failure of 3 calls).
MIN_CALLS_BEFORE_ABORT = 10

LABELS = ("PARTIALLY_CORRECT", "CORRECT", "WRONG", "ABSENT")

# \b treats "_" as a word character, so \bCORRECT\b does not match inside
# PARTIALLY_CORRECT and the labels can be counted independently.
_LABEL_ANYWHERE = re.compile(r"\b(PARTIALLY_CORRECT|CORRECT|WRONG|ABSENT)\b",
                             re.IGNORECASE)
_FIRST_WORD = re.compile(r"^[^A-Za-z_]*([A-Za-z_]+)")

# Decoration allowed around a refusal / marker line: bullets, bold, quotes.
_DECOR = " \t>*_`#-\"'"
_REFUSAL = re.compile(r"^NOT[ _]ANSWERABLE$", re.IGNORECASE)
_LATER_ANSWERED = re.compile(
    r"^[\s>*_`#-]*(?:[A-Za-z *_`]{0,20}[:=]\s*)?[\s*_`\"']*"
    r"(PARTIALLY[ _]ANSWERED|ANSWERED)[\s*_`\"'.!,)]*$",
    re.IGNORECASE,
)


class GradeError(Exception):
    """The grader call failed or its reply was not exactly one verdict label.
    `response` holds the raw reply (possibly empty) for logging."""

    def __init__(self, message: str = "grade_error", response: Optional[str] = None):
        super().__init__(message)
        self.response = response


def grading_prompt(question: str, ground_truth: str, judge_answer: str) -> str:
    """The production grading prompt. Keeps the definition-bearing wording of the
    published prompt and adds the QUESTION, so the grader can tell an answer to a
    different question from a wrong answer to this one. `question` is a required
    argument with no default, so a caller cannot leave it out by accident."""
    return f"""Grade this answer against the ground truth.

QUESTION:
{question}

QUESTION'S GROUND TRUTH ANSWER:
{ground_truth}

JUDGE'S ANSWER:
{judge_answer}

Grade the judge's answer as one of:
- CORRECT: Matches the ground truth
- PARTIALLY_CORRECT: Partially matches or covers some aspects
- WRONG: Contradicts the ground truth or provides incorrect information
- ABSENT: No real answer provided

Respond with just one word: CORRECT, PARTIALLY_CORRECT, WRONG, or ABSENT"""


def parse_verdict(response: Optional[str]) -> str:
    """Return one of GRADES, or GRADE_ERROR.

    The first whole word must be a label (case-insensitive), and no OTHER label
    may appear as a whole word anywhere in the reply. So INCORRECT, "not WRONG",
    "CORRECT or WRONG" and an empty reply are all GRADE_ERROR.
    """
    if not response or not response.strip():
        return GRADE_ERROR
    m = _FIRST_WORD.match(response.strip())
    if not m:
        return GRADE_ERROR
    first = m.group(1).upper()
    if first not in LABELS:
        return GRADE_ERROR
    seen = {x.upper() for x in _LABEL_ANYWHERE.findall(response)}
    if seen != {first}:
        return GRADE_ERROR
    return first.lower()


def parse_verdict_strict(response: Optional[str]) -> str:
    """parse_verdict, but raises GradeError instead of returning GRADE_ERROR."""
    grade = parse_verdict(response)
    if grade == GRADE_ERROR:
        raise GradeError("empty or unparseable grader reply", response)
    return grade


def first_line_refusal(answer: Optional[str]) -> bool:
    """True when the FIRST non-blank line is NOT ANSWERABLE / NOT_ANSWERABLE."""
    for line in (answer or "").splitlines():
        if line.strip():
            return bool(_REFUSAL.match(line.strip(_DECOR + ".!:,)")))
    return False


def has_later_answered(answer: Optional[str]) -> bool:
    """True when a line after the first non-blank one is an ANSWERED /
    PARTIALLY_ANSWERED marker."""
    seen_first = False
    for line in (answer or "").splitlines():
        if not line.strip():
            continue
        if not seen_first:
            seen_first = True
            continue
        if _LATER_ANSWERED.match(line):
            return True
    return False


def grade_with(call: Callable[[str], str], question: str, ground_truth: str,
               judge_answer: str, confidence: str = "ANSWERED",
               raw_reply: Optional[str] = None,
               cap_partial: bool = False) -> dict:
    """Grade one answer. `call(prompt) -> str` returns "" on any failure.

    Returns a row fragment: grade, contradictory, grader_response, prompt_version,
    parser_version. Raises GradeError on a failed or unparseable grader reply, so
    a failure can never be returned (or written) as a grade.
    `raw_reply` is the answerer's unstripped reply; the refusal rule and the
    contradictory flag read it, because extract_confidence removes a trailing
    marker from the answer text.
    """
    out = {"grade": None, "contradictory": False, "grader_response": None,
           "prompt_version": PROMPT_VERSION, "parser_version": PARSER_VERSION}
    text = raw_reply if raw_reply else judge_answer
    if first_line_refusal(text) or first_line_refusal(judge_answer):
        out["grade"] = "absent"
        out["contradictory"] = (
            has_later_answered(text)
            or confidence in ("ANSWERED", "PARTIALLY_ANSWERED"))
        return out
    if confidence == "NOT_ANSWERABLE" or not (judge_answer or "").strip():
        out["grade"] = "absent"
        return out

    response = call(grading_prompt(question, ground_truth, judge_answer))
    out["grader_response"] = response
    grade = parse_verdict_strict(response)
    if (cap_partial and confidence == "PARTIALLY_ANSWERED"
            and grade == "correct"):
        grade = "partially_correct"
    out["grade"] = grade
    return out


def is_rule_absent(judge_answer: Optional[str], rating: Optional[str]) -> bool:
    """True when grade_with assigns `absent` without a model call."""
    return (first_line_refusal(judge_answer) or rating == "NOT_ANSWERABLE"
            or not (judge_answer or "").strip())


class FailureCounter:
    """Counts grading calls and failures; aborts past MAX_FAILURE_FRACTION."""

    def __init__(self, label: str = "grader"):
        self.label, self.calls, self.failures = label, 0, 0

    def record(self, grade: str) -> None:
        self.calls += 1
        self.failures += grade == GRADE_ERROR

    @property
    def fraction(self) -> float:
        return self.failures / self.calls if self.calls else 0.0

    def check(self, final: bool = False) -> None:
        if self.calls >= (1 if final else MIN_CALLS_BEFORE_ABORT) \
                and self.fraction > MAX_FAILURE_FRACTION:
            raise SystemExit(
                f"ABORT: {self.label} failed on {self.failures}/{self.calls} "
                f"({100 * self.fraction:.1f}%) grading calls, above the "
                f"{100 * MAX_FAILURE_FRACTION:.0f}% ceiling. grade_error rows "
                f"were not written; rerun to retry them.")
