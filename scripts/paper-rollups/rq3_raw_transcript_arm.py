#!/usr/bin/env python3
"""RQ3 second arm: answer the same questions from the RAW TRANSCRIPT.

The published RQ3 arm asks a judge to answer each question from the *rollup
alone*, and reports that 19.5% of the questions the rollup attempts are answered
wrongly rather than omitted. That figure has no baseline: nothing in the
experiment establishes how a judge does when it reads the full transcript.
Without it, "rollups fail silently 19.5% of the time" is easy to misread as
"rollups are 19.5% worse than raw transcripts", which the data cannot support.

This script supplies the missing arm. It holds the *questions* and the *ground
truth* fixed, reuses the answer and grading prompts verbatim, and changes
exactly one thing: the context the answerer sees.

Design constraints that make the comparison legitimate:

  * Same judge model as the original run (`claude-sonnet-5`). Answering with a
    stronger model would confound context source with model capability, and any
    gain over 19.5% could not be attributed to the transcript.
  * Same grading prompt, and the *fixed* grade-parsing order used by
    `resume_rq3.py` / `resume_rq3_v2.py` -- `PARTIALLY_CORRECT` must be tested
    before `CORRECT`, since the latter is a substring of the former.
  * Paired subset. Sessions whose transcript does not fit the context window are
    dropped, and the rollup arm MUST be recomputed over the surviving records
    only. Comparing a raw-arm rate on a subset against the published 19.5% on
    all 124 would be comparing two different question sets.

Usage:
    rq3_raw_transcript_arm.py --dry-run          # resolve + fit report, no calls
    rq3_raw_transcript_arm.py --limit 2          # smoke test
    rq3_raw_transcript_arm.py                    # full run (resumable)
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROJECTS_DIR
from grading_common import (  # noqa: E402
    GRADE_ERROR, FailureCounter, GradeError, grade_with,
)

GATEWAY = os.environ.get("GATEWAY", "http://127.0.0.1:8080")
# Must match the model that produced the rollup arm, or the arms differ in two
# ways at once and the comparison says nothing about context.
JUDGE_MODEL = "claude-sonnet-5"

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
DATA = REPO / "data" / "paper-rollups"
QUESTIONS = DATA / "rq3_questions.jsonl"
SUMMARY = DATA / "rq3_retention_summary.json"
OUT = DATA / "rq3_questions_raw_arm.jsonl"
REPORT = DATA / "rq3_raw_arm_summary.json"

# Live transcripts plus the on-machine backup made 2026-09-21, when
# cleanupPeriodDays was raised from 30 to 180. The backup is load-bearing: some
# of these sessions predate the live retention window.
SOURCES = [
    PROJECTS_DIR,
    Path.home() / "Backups" / "claude-transcripts" / "projects",
]

# A record's session_id is a rollup note slug, not a filename. The transcript is
# found via the 8-hex id embedded in it. Globbing the slug itself matches nothing
# -- that mistake reads as "every transcript has been deleted".
SLUG = re.compile(r"^\d{4}-\d{2}-\d{2}_[a-z-]+_([0-9a-f]{8})_")

GRADES = ["correct", "partially_correct", "wrong", "absent"]

# Deliberately set ABOVE Sonnet 5's real 1M window. Dropping a session is not
# free: the sessions that do not fit are the large ones, and the rollup does
# materially worse on those (wrong-of-attempted 19.5% over all 124 records
# against 12.5% over the 90 that fit a 960k estimate). Every drop therefore
# biases the paired comparison toward the easy regime. So rather than trusting a
# bytes-per-token estimate to decide, the borderline transcripts are attempted
# and the API arbitrates. This is only safe because an overflow now returns
# API_ERROR and is excluded, rather than being scored as a genuine `absent`.
CONTEXT_LIMIT_TOKENS = 1_180_000
# Bytes per token for dense transcript JSONL. MEASURED, not assumed: a 124,092
# byte transcript billed 49,090 prompt tokens, i.e. 2.53 bytes/token. The 3.5
# this started with -- a general English-text figure -- underestimated the token
# count by 38% and would have sent four transcripts over the window. Transcript
# JSONL is dense in punctuation, quoted keys and base64-ish ids, all of which
# tokenize far worse than prose. Held slightly below the measurement for margin.
BYTES_PER_TOKEN = 2.45

# The published output cap, kept at its published value so no arm's numbers move:
# raising it would change answers rather than only their truncation handling. It
# is named rather than inlined because call_llm() now has to compare against it
# to tell a truncated reply from a complete one.
ANSWER_MAX_TOKENS = 2048

# The confidence marker, matched on a line of its own rather than by substring.
# Three things went wrong with the substring test this replaces, all of them
# measured against the stored records before it was changed:
#   1. "ANSWERED" is a substring of "PARTIALLY_ANSWERED", and it was tested
#      first, so every partial answer was recorded as a full ANSWERED. All three
#      arms stored zero PARTIALLY_ANSWERED across 337 records, against 15.6% in
#      the differently-parsed pool in rq3_questions.jsonl.
#   2. "ANSWERED" is also a substring of ordinary prose -- "cannot be answered",
#      "unanswered" -- so a reply that opened "NOT ANSWERABLE" and closed with a
#      sentence of commentary was recorded as ANSWERED, graded, and scored WRONG.
#      That is an honest refusal counted as a hallucination.
#   3. Only the last line was looked at, and it was dropped from the answer
#      whether or not it was the marker, silently truncating replies.
# Hence: longest marker first, anchored to the line, whole reply scanned, and the
# final line removed only when it really was the marker. A short label before the
# marker is allowed ("RATING: ANSWERED", "**Confidence:** ANSWERED") because the
# prompt does not forbid one, but the marker must be all that follows the label:
# a prefix of free prose is exactly how problem 2 got in. Both spellings are
# accepted -- the prompt asks for "NOT ANSWERABLE" in the answer and
# "NOT_ANSWERABLE" on the confidence line, and they mean the same thing.
MARKER_LINE = re.compile(
    r"^[\s>*_`#-]*"                          # bullet, heading or bold decoration
    r"(?:[A-Za-z *_`]{0,20}[:=]\s*)?"         # optional label, e.g. "RATING:"
    r"[\s*_`\"']*"
    r"(PARTIALLY[ _]ANSWERED|NOT[ _]ANSWERABLE|ANSWERED)"
    r"[\s*_`\"'.!,)]*$",
    re.IGNORECASE,
)


def extract_confidence(response: str) -> Tuple[str, str]:
    """Split a non-empty reply into its answer text and its confidence marker.

    Lines are scanned from the end and the LAST marker line wins: a reply that
    opens "NOT ANSWERABLE", reconsiders, and then answers is answering, and one
    such record exists in the stripped arm. Defaults to NOT_ANSWERABLE when no
    marker line is found, as before -- a marker line carrying trailing prose is
    not matched, which can only understate recovery, never invent an attempt.

    Never called with an empty response: that is API_ERROR and belongs to the
    caller. Turning it into NOT_ANSWERABLE would grade a gateway failure as an
    `absent`, the very outcome these arms measure.
    """
    lines = response.strip().split("\n")
    for i in range(len(lines) - 1, -1, -1):
        m = MARKER_LINE.match(lines[i])
        if not m:
            continue
        # Only a trailing marker line is removed from the answer. A marker stated
        # mid-reply is left where it is rather than cutting prose around it.
        kept = lines[:i] if i == len(lines) - 1 else lines
        return "\n".join(kept).strip(), m.group(1).upper().replace(" ", "_")
    return "\n".join(lines).strip(), "NOT_ANSWERABLE"


def post(url: str, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def call_llm(prompt: str, model: str = JUDGE_MODEL,
             timeout: int = 600,
             max_tokens: int = ANSWER_MAX_TOKENS) -> Tuple[str, dict]:
    """Call the gateway. No temperature -- it is rejected on Claude 5 models.

    Returns (text, usage). Usage is kept so cost and the real token count of the
    transcript are measured rather than estimated.

    A reply cut off at `max_tokens` is returned as "" so the caller records it
    API_ERROR and excludes it, the same treatment a window overflow gets per
    CONTEXT_LIMIT_TOKENS. Grading the partial text is the thing to avoid: the
    confidence marker is the LAST line of the reply, so a cut reply loses it,
    falls through extract_confidence() to NOT_ANSWERABLE and is graded `absent`.
    That turns a nearly-complete answer into an omission, and it biases every arm
    toward questions whose answers happen to be short -- the same defect
    rq3_heldout_questions.py fixed for question generation. Measured in the
    published records: 4 rows across all arms sit at exactly completion_tokens
    2048, 3 of them unique, each cut mid-sentence, each graded `absent`
    (2 in the call-only arm, 1 in the dialogue record arm).

    Retried once at double the cap before giving up, because a truncation is
    DETERMINISTIC where a gateway error is not. API_ERROR records are excluded
    from OUT so a rerun retries them, which for a truncation would re-truncate
    and never terminate; one wider attempt is what makes the exclusion converge.
    """
    def attempt(cap: int) -> Tuple[str, dict, Optional[str]]:
        data = post(f"{GATEWAY}/v1/chat/completions", {
            "model": model,
            "max_tokens": cap,
            "messages": [{"role": "user", "content": prompt}],
        }, timeout=timeout)
        choice = data["choices"][0]
        usage = data.get("usage", {})
        finish = choice.get("finish_reason")
        # Persisted into the arm's jsonl, so a later audit can tell a truncation
        # from a decline without re-running anything.
        usage["finish_reason"] = finish
        return (choice["message"]["content"] or ""), usage, finish

    try:
        content, usage, finish = attempt(max_tokens)
        if finish == "length":
            print(f"  WARNING: {model} hit max_tokens={max_tokens}, reply "
                  f"truncated; retrying once at {max_tokens * 2}")
            content, usage, finish = attempt(max_tokens * 2)
            if finish == "length":
                # Not graded and not silently retried forever: the caller counts
                # it, prints it and leaves it out of the denominator.
                print(f"  WARNING: still truncated at {max_tokens * 2}, "
                      f"excluding as API_ERROR")
                usage["truncated"] = True
                return "", usage
        if not content.strip():
            print(f"  WARNING: empty response, finish_reason={finish}")
            return "", usage
        return content.strip(), usage
    except (urllib.error.URLError, TimeoutError, KeyError) as error:
        print(f"  ERROR calling {model}: {error}")
        return "", {}


def answer_from_transcript(question: str, transcript: str) -> Tuple[str, str, dict]:
    """Present ONLY the raw transcript. Mirrors judge_question_from_rollup().

    The wording tracks the original prompt so the two arms differ in the context
    they are given and not in how they are asked. "note" becomes "transcript"
    because calling a 900k-token transcript a compressed note would itself be a
    difference in the instruction.
    """
    prompt = f"""Someone asks: "{question}"

Below is the raw transcript of an agent session. Answer the question based ONLY on this transcript. If the transcript does not contain enough information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

TRANSCRIPT:
{transcript}

Provide your answer and confidence now:"""

    response, usage = call_llm(prompt)
    if not response:
        # Critically NOT "NOT_ANSWERABLE". An empty response means the gateway
        # errored, timed out, or the transcript overflowed the window -- and
        # NOT_ANSWERABLE grades as `absent`, the exact failure mode this arm is
        # measuring. Silently scoring an API failure as an absent would bias the
        # raw arm toward looking worse and would be invisible in the output.
        # API_ERROR is never graded; it is surfaced and retried.
        return "", "API_ERROR", usage

    answer, confidence = extract_confidence(response)
    return answer, confidence, usage


def grade_answer_row(judge_answer: str, ground_truth: str, confidence: str,
                     question: str = "") -> dict:
    """Grade against ground truth via the shared grading_common module.

    Returns the row fragment (grade, contradictory, grader_response,
    prompt_version, parser_version). A first-line NOT ANSWERABLE is `absent` by
    rule with no model call; a failed or unparseable grader reply raises
    GradeError, never `absent` and never a returned grade.
    """
    return grade_with(lambda prompt: call_llm(prompt, timeout=120)[0],
                      question, ground_truth, judge_answer, confidence)


def grade_answer(judge_answer: str, ground_truth: str, confidence: str,
                 question: str = "") -> str:
    """Grade-only wrapper kept for the scripts that import it."""
    return grade_answer_row(judge_answer, ground_truth, confidence,
                            question)["grade"]


def record_id(r: dict) -> str:
    """Identical to rq3_retention_summary.record_id, so ids line up across arms."""
    for key in ("id", "question_id", "qid"):
        if r.get(key):
            return str(r[key])
    seed = f"{r.get('session_id','')}|{r.get('class','')}|{r.get('question','')}"
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def load_records(path: Path) -> List[dict]:
    """Skip the NUL-filled partial write at line 125 -- the record the published
    summary drops, which is why 125 generated questions became 124 analysed."""
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip() or line.startswith("\x00"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def index_transcripts() -> Dict[str, List[Path]]:
    by_prefix: Dict[str, List[Path]] = {}
    for src in SOURCES:
        if not src.is_dir():
            print(f"WARNING: transcript source missing: {src}")
            continue
        for p in src.rglob("*.jsonl"):
            if len(p.stem) >= 8:
                by_prefix.setdefault(p.stem[:8], []).append(p)
    return by_prefix


def resolve(prefix: str, by_prefix: Dict[str, List[Path]]) -> Optional[Path]:
    """Prefer a top-level session transcript over a subagent one, then largest."""
    paths = by_prefix.get(prefix, [])
    if not paths:
        return None
    return sorted(paths,
                  key=lambda p: ("subagents" in p.parts, -p.stat().st_size))[0]


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * max(0.0, centre - half), 1),
            round(100 * min(1.0, centre + half), 1))


def rates(rows: List[dict], grade_key: str = "grade") -> dict:
    # grade_error rows are not outcomes; they never enter a denominator.
    rows = [r for r in rows if r.get(grade_key) != GRADE_ERROR]
    n = len(rows)
    c = Counter(r.get(grade_key) for r in rows)
    out = {"n": n}
    for g in GRADES:
        lo, hi = wilson(c[g], n)
        out[g] = {"n": c[g],
                  "pct": round(100.0 * c[g] / n, 1) if n else 0.0,
                  "ci95": [lo, hi]}
    rec = c["correct"] + c["partially_correct"]
    lo, hi = wilson(rec, n)
    out["recoverable"] = {"n": rec,
                          "pct": round(100.0 * rec / n, 1) if n else 0.0,
                          "ci95": [lo, hi]}
    attempted = n - c["absent"]
    lo, hi = wilson(c["wrong"], attempted)
    out["hallucination_of_attempted"] = {
        "n_wrong": c["wrong"], "n_attempted": attempted,
        "pct": round(100.0 * c["wrong"] / attempted, 1) if attempted else None,
        "ci95": [lo, hi]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve transcripts and report fit, make no API calls")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N questions (smoke test)")
    ap.add_argument("--session", action="append", default=[],
                    help="restrict to these 8-hex session prefixes")
    args = ap.parse_args()

    summary = json.loads(SUMMARY.read_text())
    graded = set(summary["graded_record_ids"])
    records = [r for r in load_records(QUESTIONS) if record_id(r) in graded]
    if len(records) != len(graded):
        print(f"FATAL: matched {len(records)} of {len(graded)} graded ids -- the "
              f"record_id seed or the questions file has changed.")
        sys.exit(1)
    print(f"graded records matched: {len(records)}")

    by_prefix = index_transcripts()
    print(f"transcripts indexed: {sum(len(v) for v in by_prefix.values())}")

    # Resolve, and decide what fits. Anything that does not fit is dropped from
    # BOTH arms, never from this one alone.
    eligible, dropped, unresolved = [], [], []
    for r in records:
        m = SLUG.match(r.get("session_id", ""))
        if not m:
            unresolved.append(r)
            continue
        prefix = m.group(1)
        if args.session and prefix not in args.session:
            continue
        path = resolve(prefix, by_prefix)
        if path is None:
            unresolved.append(r)
            continue
        size = path.stat().st_size
        r = dict(r, _prefix=prefix, _transcript=str(path), _bytes=size)
        if size / BYTES_PER_TOKEN > CONTEXT_LIMIT_TOKENS:
            dropped.append(r)
        else:
            eligible.append(r)

    print(f"eligible: {len(eligible)}  over-context: {len(dropped)}  "
          f"unresolved: {len(unresolved)}")
    for r in {x["_prefix"]: x for x in dropped}.values():
        print(f"   DROPPED {r['_prefix']}  {r['_bytes']/1048576:.1f} MB")
    for r in unresolved:
        print(f"   UNRESOLVED {r.get('session_id')}")

    # The paired rollup baseline over exactly the records this arm will attempt.
    paired_rollup = rates(eligible)
    print(f"\npaired rollup baseline over these {len(eligible)} records: "
          f"wrong-of-attempted "
          f"{paired_rollup['hallucination_of_attempted']['pct']}% "
          f"(n_att={paired_rollup['hallucination_of_attempted']['n_attempted']})"
          f"  recoverable {paired_rollup['recoverable']['pct']}%")

    if args.dry_run:
        print("\n--dry-run: no API calls made")
        return

    done = {}
    if OUT.exists():
        for r in load_records(OUT):
            done[r.get("record_id")] = r
        print(f"resuming: {len(done)} already answered")

    todo = [r for r in eligible if record_id(r) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"to answer: {len(todo)}\n")

    # Group by session so the same transcript is read from disk once and sent on
    # consecutive calls, which is what lets the gateway's prompt cache hit.
    todo.sort(key=lambda r: (r["_prefix"], r.get("class", "")))

    cache: Tuple[Optional[str], str] = (None, "")
    errors: List[dict] = []
    grade_failures = FailureCounter("raw-arm grader")
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        rid = record_id(r)
        if cache[0] != r["_transcript"]:
            cache = (r["_transcript"], Path(r["_transcript"]).read_text(errors="replace"))
        transcript = cache[1]

        print(f"[{i}/{len(todo)}] {r['_prefix']} {r.get('class')} "
              f"({r['_bytes']/1048576:.1f} MB)")
        answer, confidence, usage = answer_from_transcript(r["question"], transcript)
        if confidence == "API_ERROR":
            # Not written to OUT, so a rerun retries it instead of inheriting a
            # fabricated absent. Counted so a systematically failing session
            # cannot hide behind a plausible-looking rate.
            errors.append({"record_id": rid, "prefix": r["_prefix"],
                           "bytes": r["_bytes"], "usage": usage})
            print(f"        -> API_ERROR (not recorded, will retry) "
                  f"{r['_bytes']/1048576:.1f} MB")
            continue

        try:
            graded = grade_answer_row(answer, r.get("ground_truth", ""),
                                      confidence, r.get("question", ""))
        except GradeError:
            # Not written, so a rerun retries it; never scored as `absent`.
            grade_failures.record(GRADE_ERROR)
            print("        -> grade_error (not recorded, will retry)")
            grade_failures.check()
            continue
        grade = graded["grade"]
        if graded["grader_response"] is not None:
            grade_failures.record(grade)
        print(f"        -> {confidence} / {grade}  "
              f"prompt_tokens={usage.get('prompt_tokens')}")

        out = {
            "record_id": rid,
            "session_id": r["session_id"],
            "class": r.get("class"),
            "question": r.get("question"),
            "ground_truth": r.get("ground_truth"),
            "rollup_grade": r.get("grade"),
            "transcript": r["_transcript"],
            "transcript_bytes": r["_bytes"],
            "raw_answer": answer,
            "raw_rating": confidence,
            "grade": grade,
            "contradictory": graded["contradictory"],
            "grader_response": graded["grader_response"],
            "prompt_version": graded["prompt_version"],
            "parser_version": graded["parser_version"],
            "usage": usage,
            "model": JUDGE_MODEL,
        }
        with OUT.open("a") as fh:
            fh.write(json.dumps(out) + "\n")
        done[rid] = out

    grade_failures.check(final=True)
    print(f"\nanswered {len(todo)-len(errors)} of {len(todo)} in "
          f"{time.time()-t0:.0f}s ({len(errors)} API errors, not recorded)")
    for e in errors:
        print(f"   ERROR {e['prefix']}  {e['bytes']/1048576:.1f} MB  {e['record_id']}")

    # Report both arms over the identical record set.
    answered = [r for r in done.values() if r.get("record_id")]
    ids = {r["record_id"] for r in answered}
    paired = [r for r in eligible if record_id(r) in ids]
    raw_rates = rates(answered)
    roll_rates = rates(paired)

    prompt_tokens = sum((r.get("usage") or {}).get("prompt_tokens") or 0
                        for r in answered)
    report = {
        "n_questions": len(answered),
        "n_sessions": len({r["session_id"] for r in answered}),
        "judge": f"{JUDGE_MODEL} via local gateway, no temperature set",
        "context_limit_tokens": CONTEXT_LIMIT_TOKENS,
        "dropped_over_context_sessions":
            sorted({r["_prefix"] for r in dropped}),
        "dropped_over_context_questions": len(dropped),
        "note": ("Paired arms over an identical record set. Questions and ground "
                 "truth are reused verbatim from rq3_questions.jsonl; only the "
                 "context given to the answerer differs. Sessions whose "
                 "transcript exceeds the context limit are excluded from BOTH "
                 "arms, so neither rate is comparable to the published 19.5% "
                 "over all 124 records."),
        "measured_prompt_tokens": prompt_tokens,
        "bytes_per_token_measured": round(
            sum(r["transcript_bytes"] for r in answered) / prompt_tokens, 2)
            if prompt_tokens else None,
        "n_api_errors_this_run": len(errors),
        "api_error_records": errors,
        "selection_effect_warning": (
            "The excluded sessions are the large ones, and the rollup performs "
            "worse on large sessions. Compare rollup_arm_same_records against "
            "rollup_arm_all_124_records to see how far the surviving subset "
            "flatters the rollup. The paired contrast below is valid, but it is "
            "a contrast in the small-session regime and must not be reported as "
            "a correction to the published 19.5%."),
        "raw_transcript_arm": raw_rates,
        "rollup_arm_same_records": roll_rates,
        "rollup_arm_all_124_records": rates(records),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")

    def line(label: str, d: dict) -> str:
        h = d["hallucination_of_attempted"]
        return (f"{label:22s} n={d['n']:3d}  correct {d['correct']['n']:3d}  "
                f"partial {d['partially_correct']['n']:3d}  "
                f"wrong {d['wrong']['n']:3d}  absent {d['absent']['n']:3d}  "
                f"recoverable {d['recoverable']['pct']:5.1f}%  "
                f"wrong-of-attempted {h['pct']}% {h['ci95']}")

    print()
    print(line("rollup (all 124)", rates(records)))
    print(line("rollup (same recs)", roll_rates))
    print(line("raw transcript", raw_rates))
    print(f"\nmeasured prompt tokens: {prompt_tokens/1e6:.2f}M")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
