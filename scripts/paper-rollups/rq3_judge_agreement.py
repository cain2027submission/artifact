#!/usr/bin/env python3
"""Second- and third-judge validation of the RQ3 retention grades.

The paper calls the wrong-of-attempted rate "the most consequential figure in
this section and the one we would most want a second judge on". This produces
that second judge, and a third, and reports how far the grade labels move.

What is being validated, precisely: RQ3 has two LLM steps, answering a question
from the rollup and then grading that answer against ground truth read from the
transcript. Only the *grading* step is re-run here. The answers are fixed, taken
from the original run. So this measures whether the labels are an artifact of one
grader, not whether a different model would have extracted more from the rollup.
That narrower claim is the one the paper can make from this output.

Three things this script gets right that a first attempt did not, all of which
silently corrupt the result rather than failing:

1. It uses the production grading prompt from grading_common (the same one every
   RQ3 grading script uses, QUESTION included). A separately reworded prompt
   measures prompt sensitivity, not inter-rater agreement.

2. It parses the reply with grading_common.parse_verdict: first whole word, and
   INCORRECT, "not WRONG" and multi-label replies are failures rather than grades.

3. It excludes records the original pipeline never actually graded. part2's
   grade_answer returns "absent" by rule when rating == NOT_ANSWERABLE or the
   answer is empty, with no model call at all. Those records carry no grading
   judgement from any model, so including them would manufacture unanimous
   agreement on 34% of the sample and inflate kappa. They are counted and
   reported separately instead.

Failures are loud. call_llm returns "" on any error, so a gateway outage would
otherwise be recorded as a wall of "absent" grades and read as disagreement. A
judge whose calls fail above MAX_FAILURE_FRACTION aborts the run.

    rq3_judge_agreement.py --limit 4                  # smoke test, 4 records
    rq3_judge_agreement.py --limit 4 --judges opus-5-5 # one judge, cheapest probe
    rq3_judge_agreement.py                            # full run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from grading_common import (
    GRADE_ERROR, MAX_FAILURE_FRACTION, PARSER_VERSION, PROMPT_VERSION,
    grading_prompt, is_rule_absent, parse_verdict,
)
from jsonl_loader import load_jsonl_strict

GATEWAY = os.environ.get("GATEWAY", "http://127.0.0.1:8080")
DATA = Path(__file__).resolve().parents[2] / "data" / "paper-rollups"
QUESTIONS = DATA / "rq3_questions.jsonl"
SUMMARY = DATA / "rq3_retention_summary.json"
CACHE = DATA / "rq3_judge_cache.json"
OUT = DATA / "rq3_judge_agreement.json"

JUDGES = ["claude-opus-5-5", "claude-opus-4-6-20250617"]
ORIGINAL_JUDGE = "claude-sonnet-5"

GRADES = ("correct", "partially_correct", "wrong", "absent")
# Prompt, parser, versions and the failure ceiling all live in grading_common.
# The cache key includes both versions, so grades from an older wording or
# parser are never mixed with fresh ones.


def post(url: str, payload: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def call_llm(prompt: str, model: str) -> str:
    """Call the gateway. Never sends `temperature`: it 400s on Claude 5 models."""
    payload = {
        "model": model,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        data = post(f"{GATEWAY}/v1/chat/completions", payload)
        content = data["choices"][0]["message"]["content"]
        if not content or not content.strip():
            print(f"    WARNING: empty response from {model}, "
                  f"finish_reason={data['choices'][0].get('finish_reason')}")
            return ""
        return content.strip()
    except (urllib.error.URLError, TimeoutError, KeyError) as error:
        print(f"    ERROR calling {model}: {error}")
        return ""


def load_records(path: Path) -> tuple[list[dict], int]:
    """Stream-decode concatenated JSON objects, tolerating a truncated tail."""
    raw = path.read_text(errors="replace")
    dec = json.JSONDecoder()
    rows: list[dict] = []
    i = 0
    while i < len(raw):
        while i < len(raw) and raw[i] in " \n\r\t":
            i += 1
        if i >= len(raw):
            break
        try:
            obj, i = dec.raw_decode(raw, i)
        except json.JSONDecodeError:
            break
        rows.append(obj)
    dropped = 1 if i < len(raw) and raw[i:].strip() else 0
    return rows, dropped


def record_id(r: dict) -> str:
    """Stable identity for a graded question (matches rq3_retention_summary.py)."""
    for key in ("id", "question_id", "qid"):
        if r.get(key):
            return str(r[key])
    seed = f"{r.get('session_id','')}|{r.get('class','')}|{r.get('question','')}"
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * max(0.0, centre - half), 1),
            round(100 * min(1.0, centre + half), 1))


def grade_prompt(record: dict) -> str:
    """The production grading prompt from grading_common, question included.
    Do not reword here; change PROMPT_VERSION in grading_common instead."""
    return grading_prompt(record.get("question", ""),
                          record.get("ground_truth", ""),
                          record.get("judge_answer", ""))


def parse_grade(response: str) -> str | None:
    """The shared first-word parser. None (a failure) on grade_error."""
    grade = parse_verdict(response)
    return None if grade == GRADE_ERROR else grade


def is_deterministic(record: dict) -> bool:
    """True when the production grader assigns 'absent' by rule, with no model
    call: first-line NOT ANSWERABLE, rating NOT_ANSWERABLE, or an empty answer."""
    return is_rule_absent(record.get("judge_answer"), record.get("rating"))


def grade_with_judge(record: dict, model: str, cache: dict) -> tuple[str | None, bool]:
    """Grade one already-answered record. Returns (grade, failed)."""
    key = f"{record_id(record)}:{model}:{PROMPT_VERSION}:{PARSER_VERSION}"
    entry = cache.get(key)
    # A cached failure is deliberately treated as a miss. Caching one would make a
    # transient 403 or timeout permanent, and the abort message above promises that
    # re-running resumes -- which it cannot do if the failures are remembered as
    # verdicts. Only real grades are cached.
    if entry and not entry.get("failed") and entry.get("grade"):
        return (entry["grade"], False)

    response = call_llm(grade_prompt(record), model)
    grade = parse_grade(response) if response else None
    if grade is None:
        return (None, True)
    cache[key] = {"grade": grade, "grader_response": response,
                  "prompt_version": PROMPT_VERSION,
                  "parser_version": PARSER_VERSION}
    return (grade, False)


def cohens_kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa over GRADES. None when undefined rather than a fake 0.0."""
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    p_e = sum((ca[g] / n) * (cb[g] / n) for g in GRADES)
    if p_e >= 1.0:
        # Every rater used one grade throughout; agreement carries no information.
        return None
    return (p_o - p_e) / (1.0 - p_e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="grade only the first N judged records (smoke test)")
    ap.add_argument("--judges", nargs="+", default=None,
                    help="override the judge model list")
    ap.add_argument("--all-records", action="store_true",
                    help="use every graded record, not just the paper's sample")
    args = ap.parse_args()
    judges = args.judges or JUDGES

    result = load_jsonl_strict(
        QUESTIONS,
        required_keys={"session_id", "class", "question", "ground_truth",
                      "judge_answer", "grade"}
    )
    rows = [r for r in result.records if r.get("grade") in GRADES]
    if not rows:
        print(f"ERROR: no graded records in {QUESTIONS}")
        return 1

    # Restrict to the exact records the paper's figures were computed over, so the
    # agreement statistic validates the cited number rather than a larger sample
    # that no reported figure came from.
    sample_note = "all graded records"
    if not args.all_records and SUMMARY.exists():
        wanted = set(json.loads(SUMMARY.read_text()).get("graded_record_ids", []))
        if wanted:
            rows = [r for r in rows if record_id(r) in wanted]
            sample_note = f"the {len(wanted)} records in rq3_retention_summary.json"

    rows.sort(key=record_id)
    judged = [r for r in rows if not is_deterministic(r)]
    deterministic = [r for r in rows if is_deterministic(r)]

    print(f"Loaded {len(rows)} graded records from "
          f"{len({r['session_id'] for r in rows})} sessions ({sample_note})")
    if result.failed:
        print(f"  dropped {result.failed} unparseable line(s)")
    print(f"  {len(deterministic)} assigned 'absent' by rule with no model call "
          f"-- excluded from agreement")
    print(f"  {len(judged)} actually graded by a model -- the agreement sample")

    if args.limit:
        judged = judged[:args.limit]
        print(f"  --limit {args.limit}: grading {len(judged)} of them")

    if not judged:
        print("ERROR: nothing to grade")
        return 1

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    if cache:
        print(f"  cache holds {len(cache)} entries")

    original = [r["grade"] for r in judged]
    per_judge: dict[str, list[str | None]] = {}

    for model in judges:
        print(f"\nGrading with {model} ({len(judged)} calls)...")
        grades: list[str | None] = []
        failures = 0
        for i, record in enumerate(judged):
            grade, failed = grade_with_judge(record, model, cache)
            grades.append(grade)
            failures += failed
            if (i + 1) % 10 == 0 or i + 1 == len(judged):
                print(f"  {i + 1}/{len(judged)} graded, {failures} failed", flush=True)
                CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))
        per_judge[model] = grades

        frac = failures / len(judged)
        print(f"  {model}: {failures}/{len(judged)} failed ({100 * frac:.1f}%)")
        if frac > MAX_FAILURE_FRACTION:
            CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))
            raise SystemExit(
                f"\nABORT: {model} failed on {100 * frac:.1f}% of calls, above the "
                f"{100 * MAX_FAILURE_FRACTION:.0f}% ceiling.\n"
                f"  This is what a gateway outage looks like, not judge disagreement.\n"
                f"  Check: curl -s -o /dev/null -w '%{{http_code}}' -X POST "
                f"{GATEWAY}/v1/chat/completions \\\n"
                f"    -H 'Content-Type: application/json' \\\n"
                f"    -d '{{\"model\":\"{model}\",\"max_tokens\":8,"
                f"\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}'\n"
                f"  Grades so far are cached in {CACHE}; re-run to resume."
            )

    CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))

    # Only records every judge graded can enter a comparison.
    keep = [i for i in range(len(judged))
            if all(per_judge[m][i] is not None for m in judges)]
    print(f"\n{len(keep)}/{len(judged)} records graded successfully by every judge")
    if not keep:
        print("ERROR: no record was graded by all judges")
        return 1

    series = {"original": [original[i] for i in keep]}
    for m in judges:
        series[m] = [per_judge[m][i] for i in keep]

    output = {
        "note": "second- and third-judge validation of the RQ3 grading step only; "
                "the answers being graded are fixed from the original run",
        "input_file": QUESTIONS.name,
        "input_sha256": hashlib.sha256(QUESTIONS.read_bytes()).hexdigest(),
        "input_bytes": QUESTIONS.stat().st_size,
        "prompt": "grading_common.grading_prompt (production prompt)",
        "prompt_version": PROMPT_VERSION,
        "parser_version": PARSER_VERSION,
        "sample": sample_note,
        "n_records_in_sample": len(rows),
        "n_deterministic_absent_excluded": len(deterministic),
        "n_agreement_sample": len(keep),
        "limit_applied": args.limit,
        "judges": {"original": ORIGINAL_JUDGE, "second": judges},
        "per_judge": {},
        "agreement": {},
    }

    for name, grades in series.items():
        counts = Counter(grades)
        n = len(grades)
        recoverable = counts["correct"] + counts["partially_correct"]
        attempted = n - counts["absent"]
        output["per_judge"][name] = {
            "distribution": {g: counts[g] for g in GRADES},
            "recoverable": {"n": recoverable, "pct": round(100 * recoverable / n, 1),
                            "ci95": list(wilson(recoverable, n))},
            "wrong_of_attempted": {
                "n_wrong": counts["wrong"], "n_attempted": attempted,
                "pct": round(100 * counts["wrong"] / attempted, 1) if attempted else None,
                "ci95": list(wilson(counts["wrong"], attempted)) if attempted else None,
            },
        }

    names = list(series)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            xs, ys = series[a], series[b]
            exact = sum(1 for x, y in zip(xs, ys) if x == y)
            # The paper's headline is a wrong-vs-not-wrong claim, so agreement on
            # that binary collapse is reported alongside the 4-way figure.
            bin_x = ["wrong" if g == "wrong" else "not_wrong" for g in xs]
            bin_y = ["wrong" if g == "wrong" else "not_wrong" for g in ys]
            bin_exact = sum(1 for x, y in zip(bin_x, bin_y) if x == y)
            k = cohens_kappa(xs, ys)
            output["agreement"][f"{a}_vs_{b}"] = {
                "n": len(xs),
                "exact_4way": {"n": exact, "pct": round(100 * exact / len(xs), 1),
                               "ci95": list(wilson(exact, len(xs)))},
                "cohens_kappa_4way": round(k, 3) if k is not None else None,
                "exact_wrong_vs_not": {
                    "n": bin_exact, "pct": round(100 * bin_exact / len(xs), 1),
                    "ci95": list(wilson(bin_exact, len(xs))),
                },
            }

    OUT.write_text(json.dumps(output, indent=2, sort_keys=True))

    print("\nper judge, over the agreement sample:")
    for name, d in output["per_judge"].items():
        w = d["wrong_of_attempted"]
        print(f"  {name:<28} recoverable {d['recoverable']['pct']:5.1f}%  "
              f"wrong/attempted {w['n_wrong']}/{w['n_attempted']}"
              f" = {w['pct'] if w['pct'] is not None else float('nan'):5.1f}%")

    print("\nagreement:")
    for pair, d in output["agreement"].items():
        k = d["cohens_kappa_4way"]
        print(f"  {pair:<48} exact {d['exact_4way']['pct']:5.1f}%  "
              f"kappa {k if k is not None else float('nan')}  "
              f"wrong-vs-not {d['exact_wrong_vs_not']['pct']:5.1f}%")

    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
