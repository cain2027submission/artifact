#!/usr/bin/env python3
"""Unit tests for the shared RQ3 verdict parser and refusal rule (grading_common).

Run: python3 -m unittest -v test_grade_parsing   (from scripts/paper-rollups)
No test makes a model call; grader calls are stubbed.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grading_common import (  # noqa: E402
    GRADE_ERROR, PARSER_VERSION, PROMPT_VERSION, FailureCounter, GradeError,
    first_line_refusal, grade_with, grading_prompt, has_later_answered,
    parse_verdict,
)


def never_called(prompt):
    raise AssertionError("grader must not be called")


class TestParseVerdict(unittest.TestCase):
    def test_plain_labels(self):
        for text, want in [("CORRECT", "correct"),
                           ("PARTIALLY_CORRECT", "partially_correct"),
                           ("WRONG", "wrong"), ("ABSENT", "absent")]:
            self.assertEqual(parse_verdict(text), want)

    def test_partially_correct_is_not_correct(self):
        self.assertEqual(parse_verdict("PARTIALLY_CORRECT"), "partially_correct")

    def test_decoration_and_trailing_text(self):
        self.assertEqual(parse_verdict("**WRONG**"), "wrong")
        self.assertEqual(parse_verdict("WRONG. The answer contradicts it."),
                         "wrong")

    def test_lowercase_variants(self):
        self.assertEqual(parse_verdict("correct"), "correct")
        self.assertEqual(parse_verdict("Partially_Correct"), "partially_correct")
        self.assertEqual(parse_verdict("wrong\n"), "wrong")
        self.assertEqual(parse_verdict("absent"), "absent")

    def test_incorrect_is_error(self):
        self.assertEqual(parse_verdict("INCORRECT"), GRADE_ERROR)
        self.assertEqual(parse_verdict("incorrect"), GRADE_ERROR)

    def test_not_wrong_is_error(self):
        self.assertEqual(parse_verdict("not WRONG"), GRADE_ERROR)
        self.assertEqual(parse_verdict("Not wrong, it matches"), GRADE_ERROR)

    def test_multi_label_is_error(self):
        self.assertEqual(parse_verdict("CORRECT or WRONG"), GRADE_ERROR)
        self.assertEqual(parse_verdict("PARTIALLY_CORRECT / CORRECT"),
                         GRADE_ERROR)
        self.assertEqual(parse_verdict("WRONG - the correct answer is X"),
                         GRADE_ERROR)

    def test_empty_and_unrelated_are_error(self):
        for text in ["", "   \n", None, "some other text", "..."]:
            self.assertEqual(parse_verdict(text), GRADE_ERROR)


class TestRefusalRule(unittest.TestCase):
    def test_first_line_refusal_variants(self):
        for text in ["NOT ANSWERABLE", "NOT_ANSWERABLE", "not answerable",
                     "not_answerable", "\n\n  Not Answerable.\nmore",
                     "**NOT ANSWERABLE**"]:
            self.assertTrue(first_line_refusal(text), text)

    def test_not_first_line_is_not_refusal(self):
        self.assertFalse(first_line_refusal("The answer is X\nNOT ANSWERABLE"))
        self.assertFalse(first_line_refusal("It cannot be answered"))
        self.assertFalse(first_line_refusal(""))

    def test_refusal_is_absent_with_no_model_call(self):
        row = grade_with(never_called, "q", "truth", "NOT ANSWERABLE\nsorry",
                         "NOT_ANSWERABLE")
        self.assertEqual(row["grade"], "absent")
        self.assertFalse(row["contradictory"])

    def test_lowercase_refusal_no_model_call(self):
        row = grade_with(never_called, "q", "truth", "not answerable", "ANSWERED")
        self.assertEqual(row["grade"], "absent")

    def test_h18_shape_is_absent_and_contradictory(self):
        reply = "NOT ANSWERABLE\nThe note lacks it.\nActually it says X.\nANSWERED"
        self.assertTrue(has_later_answered(reply))
        # Answer text as extract_confidence leaves it: trailing marker removed,
        # confidence ANSWERED.
        answer = "NOT ANSWERABLE\nThe note lacks it.\nActually it says X."
        row = grade_with(never_called, "q", "truth", answer, "ANSWERED",
                         raw_reply=reply)
        self.assertEqual(row["grade"], "absent")
        self.assertTrue(row["contradictory"])

    def test_h18_partially_answered_marker(self):
        reply = "NOT ANSWERABLE\nhmm\nPARTIALLY_ANSWERED"
        self.assertTrue(has_later_answered(reply))
        row = grade_with(never_called, "q", "t", "NOT ANSWERABLE\nhmm",
                         "PARTIALLY_ANSWERED", raw_reply=reply)
        self.assertEqual(row["grade"], "absent")
        self.assertTrue(row["contradictory"])


class TestGraderFailure(unittest.TestCase):
    def test_empty_reply_raises_not_absent(self):
        with self.assertRaises(GradeError):
            grade_with(lambda p: "", "q", "truth", "an answer", "ANSWERED")

    def test_unparseable_and_multi_label_replies_raise(self):
        for reply in ["INCORRECT", "CORRECT or WRONG", "no idea", None]:
            with self.assertRaises(GradeError, msg=repr(reply)):
                grade_with(lambda p, r=reply: r, "q", "t", "a", "ANSWERED")

    def test_gateway_exception_is_not_swallowed(self):
        def boom(prompt):
            raise ConnectionError("gateway down")
        with self.assertRaises(ConnectionError):
            grade_with(boom, "q", "t", "a", "ANSWERED")

    def test_grade_error_carries_raw_reply(self):
        with self.assertRaises(GradeError) as cm:
            grade_with(lambda p: "maybe", "q", "t", "a", "ANSWERED")
        self.assertEqual(cm.exception.response, "maybe")

    def test_missing_question_raises(self):
        with self.assertRaises(TypeError):
            grading_prompt(ground_truth="t", judge_answer="a")
        with self.assertRaises(TypeError):
            grading_prompt("t", "a")

    def test_row_records_versions_and_raw_response(self):
        row = grade_with(lambda p: "PARTIALLY_CORRECT", "q", "t", "a", "ANSWERED")
        self.assertEqual(row["grade"], "partially_correct")
        self.assertEqual(row["grader_response"], "PARTIALLY_CORRECT")
        self.assertEqual(row["prompt_version"], PROMPT_VERSION)
        self.assertEqual(row["parser_version"], PARSER_VERSION)

    def test_prompt_contains_question(self):
        seen = []
        grade_with(lambda p: seen.append(p) or "CORRECT", "WHYQUESTION", "t",
                   "a", "ANSWERED")
        self.assertIn("WHYQUESTION", seen[0])
        self.assertIn("CORRECT: Matches the ground truth", seen[0])
        self.assertIn("WHYQUESTION", grading_prompt("WHYQUESTION", "t", "a"))

    def test_partial_cap(self):
        row = grade_with(lambda p: "CORRECT", "q", "t", "a",
                         "PARTIALLY_ANSWERED", cap_partial=True)
        self.assertEqual(row["grade"], "partially_correct")

    def test_failure_counter_aborts_over_ceiling(self):
        fc = FailureCounter("t")
        for _ in range(8):
            fc.record("correct")
        for _ in range(4):
            fc.record(GRADE_ERROR)
        with self.assertRaises(SystemExit):
            fc.check()

    def test_failure_counter_quiet_under_ceiling(self):
        fc = FailureCounter("t")
        for _ in range(19):
            fc.record("wrong")
        fc.record(GRADE_ERROR)
        fc.check(final=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
