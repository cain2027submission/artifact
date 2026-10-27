#!/usr/bin/env python3
"""Regression test for the degenerate-context guards in the answer-only and
prompts-only RQ3 arms.

Both guards used to be a substring test over a formatted count,
`"0 assistant sections shown" in context`, which also matches 10, 20, 40 and 50.
That voided 34 of the 248 records across the two arms to a canned `absent` grade
with no API call, and every one of the 34 was a false positive.

This test calls the real guard with `call_llm` stubbed, so it fails if anyone
reintroduces a substring test: a 10-section context must reach the model, and only
a genuinely empty one must short-circuit.

    test_degenerate_guard.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rq3_answer_only_arm as answer_only  # noqa: E402
import rq3_prompts_only_arm as prompts_only  # noqa: E402

SENTINEL = "REACHED_THE_MODEL"


def stub_call_llm(prompt, model=None, timeout=None):
    """Stand in for the gateway. Returns an ANSWERED response, no network."""
    return f"{SENTINEL}\nANSWERED", {"input_tokens": 1, "output_tokens": 1}


# The arms import call_llm into their own namespace, so patch it there.
answer_only.call_llm = stub_call_llm
prompts_only.call_llm = stub_call_llm

CANNED_ANSWER = "no assistant prose"
CANNED_PROMPTS = "no user prompts"

# (label, section count, must_short_circuit)
CASES = [
    ("0 sections, genuinely empty", 0, True),
    ("1 section", 1, False),
    ("10 sections, the original bug", 10, False),
    ("20 sections", 20, False),
    ("40 sections", 40, False),
    ("50 sections", 50, False),
    ("100 sections", 100, False),
    ("130 sections", 130, False),
]


def run(label, fn, stat_key, noun, canned_fragment):
    print(f"\n{label}:")
    failures = 0
    for case, count, must_short_circuit in CASES:
        context = (f"[This contains ONLY the {noun} from an agent session. "
                   f"{count} {noun} shown. The rest is withheld.]\n\n"
                   "### SECTION\nsome real content\n")
        stats = {stat_key: count, "entries": count, "chars": len(context)}
        answer, rating, usage = fn("what happened?", context, stats)

        short_circuited = canned_fragment in answer
        ok = short_circuited == must_short_circuit
        failures += not ok
        expect = "short-circuit" if must_short_circuit else "reach the model"
        got = "short-circuited" if short_circuited else "reached the model"
        if not short_circuited and SENTINEL not in answer:
            ok, got = False, f"unexpected answer {answer[:40]!r}"
            failures += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {case:32s} expect {expect:15s} got {got}")
    return failures


def main() -> int:
    failures = 0
    failures += run("answer-only guard", answer_only.answer_from_answer_only,
                    "assistant_sections", "assistant sections", CANNED_ANSWER)
    failures += run("prompts-only guard", prompts_only.answer_from_prompts_only,
                    "user_sections", "user prompts", CANNED_PROMPTS)

    print(f"\n{2 * len(CASES) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
