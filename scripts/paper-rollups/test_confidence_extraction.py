#!/usr/bin/env python3
"""Gate for the RQ3 confidence-marker extractor, asserted through each arm.

Asserted at the arm's own entry point, not at extract_confidence(), because the
API_ERROR rule lives in the caller and because the three arms must agree: a fix
to one arm only would make the arms non-comparable. call_llm is stubbed per
module, so this makes no model calls.

Run: python3 scripts/paper-rollups/test_confidence_extraction.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rq3_raw_transcript_arm as raw          # noqa: E402
import rq3_stripped_transcript_arm as strip   # noqa: E402
import rq3_uniform_rollup_arm as uniform      # noqa: E402

# (name, reply, expected answer, expected confidence)
CASES = [
    (
        "trailing ANSWERED is the marker and leaves the answer",
        "The cap was raised to 180 days.\nIt is set in cleanupPeriodDays.\nANSWERED",
        "The cap was raised to 180 days.\nIt is set in cleanupPeriodDays.",
        "ANSWERED",
    ),
    (
        "PARTIALLY_ANSWERED must not collapse to ANSWERED",
        "The note gives the commit but not the line counts.\nPARTIALLY_ANSWERED",
        "The note gives the commit but not the line counts.",
        "PARTIALLY_ANSWERED",
    ),
    (
        "refusal plus commentary containing 'answered' stays a refusal",
        "NOT ANSWERABLE\n\nThe note never mentions the stairwell z-height, so the "
        "question is not answered anywhere in it.",
        "NOT ANSWERABLE\n\nThe note never mentions the stairwell z-height, so the "
        "question is not answered anywhere in it.",
        "NOT_ANSWERABLE",
    ),
    (
        "prose last line with no marker keeps that line",
        "The sweep found 546 files.\nThe drain was writing while it counted.",
        "The sweep found 546 files.\nThe drain was writing while it counted.",
        "NOT_ANSWERABLE",
    ),
    (
        "a labelled marker line is still a marker",
        "It merged with a fast-forward push.\nRATING: ANSWERED",
        "It merged with a fast-forward push.",
        "ANSWERED",
    ),
    (
        "a reply that refuses, reconsiders, then answers is answering",
        "NOT ANSWERABLE\n\nOn reflection that exchange does appear earlier.\n"
        "ANSWERED\n\nIt merged origin/main into the branch, then pushed.",
        "NOT ANSWERABLE\n\nOn reflection that exchange does appear earlier.\n"
        "ANSWERED\n\nIt merged origin/main into the branch, then pushed.",
        "ANSWERED",
    ),
    (
        "'unanswered' in prose is not a marker",
        "The note leaves the count unanswered.",
        "The note leaves the count unanswered.",
        "NOT_ANSWERABLE",
    ),
]

ARMS = [
    ("raw", raw, raw.answer_from_transcript),
    ("stripped", strip, strip.answer_from_stripped),
    ("uniform", uniform, uniform.answer_from_rollup),
]

# Captured before the loop below replaces it with a stub. The truncation section at
# the bottom needs the real one, because the max_tokens rule lives inside it.
ORIGINAL_CALL_LLM = raw.call_llm

failures = []
for name, reply, want_answer, want_conf in CASES:
    seen = {}
    for arm, module, fn in ARMS:
        module.call_llm = lambda *a, **k: (reply, {"prompt_tokens": 1})
        answer, conf, _usage = fn("a question", "some context")
        seen[arm] = (answer, conf)
        if conf != want_conf:
            failures.append(f"{arm}: {name}: confidence {conf!r} != {want_conf!r}")
        if answer != want_answer:
            failures.append(f"{arm}: {name}: answer {answer!r} != {want_answer!r}")
    if len(set(seen.values())) != 1:
        failures.append(f"arms disagree on {name}: {seen}")
    print(f"  ok  {name}" if not failures else f"  FAIL {name}")

# An empty reply is a gateway failure, never a refusal: NOT_ANSWERABLE would be
# graded `absent`, which is the outcome these arms exist to measure.
for arm, module, fn in ARMS:
    module.call_llm = lambda *a, **k: ("", {})
    answer, conf, _usage = fn("a question", "some context")
    if (answer, conf) != ("", "API_ERROR"):
        failures.append(f"{arm}: empty reply gave {(answer, conf)!r}, want ('', 'API_ERROR')")
print("  ok  empty reply stays API_ERROR in all three arms"
      if not failures else "  FAIL empty reply")


def buggy(response: str):
    """The extractor this replaces, kept only so the cases above are shown to be
    sensitive to the defect. A gate that the old code also passes tests nothing.
    """
    lines = response.strip().split("\n")
    confidence = "NOT_ANSWERABLE"
    for keyword in ["ANSWERED", "PARTIALLY_ANSWERED", "NOT_ANSWERABLE"]:
        if keyword in lines[-1].upper():
            confidence = keyword
            break
    return "\n".join(lines[:-1]).strip(), confidence


caught = [name for name, reply, want_answer, want_conf in CASES
          if buggy(reply) != (want_answer, want_conf)]
if len(caught) < 4:
    failures.append(f"the old extractor already passes all but {caught}; "
                    "these cases do not test the defect")
print(f"  ok  the old extractor fails {len(caught)} of {len(CASES)} cases: {caught}")

# A reply cut off at max_tokens loses its trailing marker, so without a guard it
# falls through to NOT_ANSWERABLE and is graded `absent` -- a nearly-complete
# answer recorded as an omission. Asserted against the real call_llm with the
# gateway stubbed, because the rule lives inside call_llm and not in the extractor.
raw.call_llm = ORIGINAL_CALL_LLM

TRUNCATED = ("The branch was merged with a fast-forward push after the pull, and "
             "the worktree was then removed with ExitWorktree rather than with")
COMPLETE = TRUNCATED + " git worktree remove.\nANSWERED"


def stub_gateway(always_truncate: bool):
    """Stand in for the gateway. Truncates at the first cap, and either completes
    on the wider retry or truncates again, so both exits are covered.
    """
    calls = []

    def post(url, payload, timeout=None):
        calls.append(payload["max_tokens"])
        cut = always_truncate or payload["max_tokens"] == raw.ANSWER_MAX_TOKENS
        return {
            "choices": [{
                "message": {"content": TRUNCATED if cut else COMPLETE},
                "finish_reason": "length" if cut else "stop",
            }],
            "usage": {"completion_tokens": payload["max_tokens"] if cut else 180},
        }

    return post, calls


# Recovered: one wider retry completes the reply, so the answer is graded on the
# whole thing rather than on the part that fitted.
raw.post, calls = stub_gateway(always_truncate=False)
answer, conf, usage = raw.answer_from_transcript("a question", "some context")
if conf != "ANSWERED":
    failures.append(f"recovered truncation gave {conf!r}, want 'ANSWERED'")
if calls != [raw.ANSWER_MAX_TOKENS, raw.ANSWER_MAX_TOKENS * 2]:
    failures.append(f"expected one retry at double the cap, got calls {calls}")
print("  ok  a truncated reply is retried wider and then graded in full"
      if conf == "ANSWERED" else "  FAIL truncation retry")

# Still truncated at double the cap: excluded as API_ERROR, never graded, and it
# must not loop -- exactly two calls, not an unbounded climb.
raw.post, calls = stub_gateway(always_truncate=True)
answer, conf, usage = raw.answer_from_transcript("a question", "some context")
if (answer, conf) != ("", "API_ERROR"):
    failures.append(f"persistent truncation gave {(answer, conf)!r}, "
                    "want ('', 'API_ERROR')")
if len(calls) != 2:
    failures.append(f"persistent truncation made {len(calls)} calls, want 2")
if not usage.get("truncated"):
    failures.append("persistent truncation did not mark usage['truncated']")
print("  ok  a reply still truncated at double the cap is excluded, not graded"
      if conf == "API_ERROR" else "  FAIL persistent truncation")


# No-op control: the call_llm this replaces had no finish_reason check, so it
# returned the partial text. If the old behaviour also yields ANSWERED then the
# cases above are not testing the defect.
def old_call_llm(prompt, model=None, timeout=600, max_tokens=None):
    data = raw.post("", {"max_tokens": raw.ANSWER_MAX_TOKENS}, timeout)
    return data["choices"][0]["message"]["content"].strip(), data.get("usage", {})


raw.post, _ = stub_gateway(always_truncate=False)
raw.call_llm = old_call_llm
_answer, old_conf, _usage = raw.answer_from_transcript("a question", "some context")
if old_conf != "NOT_ANSWERABLE":
    failures.append(f"the old call_llm gave {old_conf!r} on a truncated reply, "
                    "so these cases do not test the defect")
print(f"  ok  the old call_llm records the same truncation as {old_conf} "
      "(graded `absent`)")

if failures:
    print("\nFAILED")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print(f"\nPASS: {len(CASES)} replies x 3 arms, plus the API_ERROR rule "
      "and the max_tokens truncation rule")
