#!/usr/bin/env python3
"""Is the call-only arm's artifact_detail deficit a 2000-char truncation artifact?

The tool-inputs-only arm caps each tool input at MAX_TOOL_INPUT_CHARS = 2000.
artifact_detail is the class where it falls furthest behind prose-only, and an
artifact detail (a path, a flag, a number inside a long tool argument) is
exactly the kind of answer a per-tool cap could cut off. If the deficit is the
cap rather than the substrate, the paper owes that cap a disclosure and the
class a narrower scope.

The test. For every record, rebuild the call-only context twice with the arm's
OWN builder -- once at the published 2000-char cap, once uncapped -- and
measure how much of the ground-truth answer's distinctive vocabulary each
context contains. If lifting the cap adds no ground-truth coverage, the answer
was never in the tool inputs at any length and truncation cannot explain the
failures. If lifting it adds coverage on the records that failed, it can.

Controls, both of which must pass before the delta means anything:
  1. the rebuilt capped context reproduces each record's stored context_chars
     and truncated_inputs, so the rebuild is the same construction that was
     graded rather than a lookalike
  2. the uncapped context is a strict superset in length, so "uncapped" really
     did lift the cap

No model calls. Reads the committed arm file and the pinned transcripts.

Usage:
    rq3_artifact_detail_truncation.py
    rq3_artifact_detail_truncation.py --class all
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
DATA = REPO / "data" / "paper-rollups"
sys.path.insert(0, str(HERE.parent))

import rq3_tool_inputs_only_arm as arm

CALL_ARM = DATA / "rq3_tool_inputs_only_arm.jsonl"
CALL_SUMMARY = DATA / "rq3_tool_inputs_only_arm_summary.json"
REPORT = DATA / "rq3_artifact_detail_truncation.json"
REPORT_ALL = DATA / "rq3_truncation_check_all_classes.json"

RECOVERABLE = {"correct", "partially_correct"}
PUBLISHED_CAP = 2_000
UNCAPPED = 10 ** 9

# tokens too common to carry an artifact detail; coverage is about whether the
# answer's distinctive vocabulary is present, not whether English is
STOP = {
    "that", "this", "with", "from", "they", "them", "then", "than", "were",
    "what", "when", "which", "while", "would", "could", "should", "have",
    "has", "had", "been", "being", "does", "did", "done", "the", "and", "for",
    "not", "but", "was", "are", "its", "it's", "into", "onto", "over",
    "under", "after", "before", "because", "about", "there", "their", "these",
    "those", "only", "also", "more", "most", "some", "such", "each", "both",
    "same", "other", "another", "where", "whether", "session", "answer",
    "transcript", "agent", "file", "files",
}

TOKEN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:/-]{3,}")


def distinctive(text):
    """Lowercased tokens of 4+ chars that are not stopwords."""
    return {
        t.lower() for t in TOKEN.findall(text or "")
        if t.lower() not in STOP
    }


def coverage(truth_tokens, context):
    if not truth_tokens:
        return None, 0, 0
    low = context.lower()
    hits = {t for t in truth_tokens if t in low}
    return (round(100.0 * len(hits) / len(truth_tokens), 1),
            len(hits), len(truth_tokens))


def build(raw, cap):
    """Build a call-only context at a given per-tool cap, via the arm's code."""
    original = arm.MAX_TOOL_INPUT_CHARS
    try:
        arm.MAX_TOOL_INPUT_CHARS = cap
        return arm.tool_inputs_only_context(raw)
    finally:
        arm.MAX_TOOL_INPUT_CHARS = original


def body(context):
    """Context with the header stripped.

    The header states the cap in words ("inputs capped at 2000 chars"), so an
    uncapped context is longer by the width of that number even when nothing
    was truncated. Length comparisons have to run on the tool inputs alone or
    the cap's own description is counted as content it withheld.
    """
    parts = context.split("\n\n", 1)
    return parts[1] if len(parts) == 2 else ""


def corrected_absent():
    d = json.load(CALL_SUMMARY.open())
    return {
        r["record_id"] for r in d.get("reclassified_records", [])
        if r.get("corrected_rating") == "NOT_ANSWERABLE"
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--class", dest="cls", default="artifact_detail",
                    help="question class to test, or 'all'")
    args = ap.parse_args()

    rows = [json.loads(l) for l in CALL_ARM.open() if l.strip()]
    if args.cls != "all":
        rows = [r for r in rows if r["class"] == args.cls]
    if not rows:
        sys.exit(f"no records for class {args.cls!r}")

    fixed = corrected_absent()

    records = []
    control_failures = []
    missing = []

    for r in rows:
        rid = r["record_id"]
        tpath = Path(r["transcript"])
        if not tpath.exists():
            missing.append({"record_id": rid, "transcript": str(tpath)})
            continue
        raw = tpath.read_text(errors="replace")

        capped, cstats = build(raw, PUBLISHED_CAP)
        uncapped, ustats = build(raw, UNCAPPED)

        # control 1: rebuild matches what was graded
        stored = r["context_stats"]
        chars_match = len(capped) == r["context_chars"]
        trunc_match = cstats["truncated_inputs"] == stored["truncated_inputs"]
        # control 2: lifting the cap really added tool-input bytes where
        # something was truncated, and added none where nothing was
        cap_body, un_body = body(capped), body(uncapped)
        cap_lifted = (
            ustats["truncated_inputs"] == 0
            and (len(un_body) > len(cap_body)
                 if stored["truncated_inputs"]
                 else len(un_body) == len(cap_body))
        )
        if not (chars_match and trunc_match and cap_lifted):
            control_failures.append({
                "record_id": rid,
                "stored_context_chars": r["context_chars"],
                "rebuilt_context_chars": len(capped),
                "stored_truncated_inputs": stored["truncated_inputs"],
                "rebuilt_truncated_inputs": cstats["truncated_inputs"],
                "uncapped_chars": len(uncapped),
                "chars_match": chars_match,
                "truncated_match": trunc_match,
                "cap_lifted_as_expected": cap_lifted,
            })

        truth = distinctive(r["ground_truth"])
        cov_c, hit_c, n_tok = coverage(truth, capped)
        cov_u, hit_u, _ = coverage(truth, uncapped)
        recoverable = r["grade"] in RECOVERABLE and rid not in fixed

        records.append({
            "record_id": rid,
            "class": r["class"],
            "grade": r["grade"],
            "recoverable": recoverable,
            "truncated_inputs": stored["truncated_inputs"],
            "capped_chars": len(capped),
            "uncapped_chars": len(uncapped),
            "bytes_withheld_by_cap": len(un_body) - len(cap_body),
            "ground_truth_tokens": n_tok,
            "coverage_capped_pct": cov_c,
            "coverage_uncapped_pct": cov_u,
            "coverage_gain_pct": (None if cov_c is None
                                  else round(cov_u - cov_c, 1)),
            "tokens_gained": hit_u - hit_c,
        })

    failed = [r for r in records if not r["recoverable"]]
    passed = [r for r in records if r["recoverable"]]
    gains = [r["coverage_gain_pct"] for r in records
             if r["coverage_gain_pct"] is not None]
    failed_gains = [r["coverage_gain_pct"] for r in failed
                    if r["coverage_gain_pct"] is not None]
    any_gain = [r for r in records if (r["tokens_gained"] or 0) > 0]
    failed_any_gain = [r for r in failed if (r["tokens_gained"] or 0) > 0]

    # the cap cannot have hurt a record whose inputs were never truncated
    untruncated = [r for r in records if r["truncated_inputs"] == 0]
    untruncated_failed = [r for r in untruncated if not r["recoverable"]]

    # per class, so a result that holds in aggregate cannot hide the class the
    # hypothesis was actually about
    per_class = {}
    for cls in sorted({r["class"] for r in records}):
        sub = [r for r in records if r["class"] == cls]
        sub_failed = [r for r in sub if not r["recoverable"]]
        sub_gain = [r for r in sub_failed if (r["tokens_gained"] or 0) > 0]
        sub_gains = [r["coverage_gain_pct"] for r in sub_failed
                     if r["coverage_gain_pct"] is not None]
        per_class[cls] = {
            "n": len(sub),
            "n_failed": len(sub_failed),
            "n_failed_with_zero_truncated_inputs": sum(
                1 for r in sub_failed if r["truncated_inputs"] == 0),
            "n_failed_gaining_any_ground_truth_token": len(sub_gain),
            "mean_coverage_gain_pct_failed_only": (
                round(statistics.mean(sub_gains), 2) if sub_gains else None),
            "truncation_explains_any_failure": bool(sub_gain),
        }

    controls_pass = not control_failures and not missing
    verdict = (
        "controls failed, delta not interpretable" if not controls_pass
        else "truncation does not explain the deficit"
        if not failed_any_gain
        else f"truncation may explain {len(failed_any_gain)} of "
             f"{len(failed)} failures, needs scope disclosure"
    )

    report = {
        "what_this_is": (
            "containment test on whether the call-only arm's 2000-char "
            "per-tool input cap withheld the ground-truth answers it was "
            "graded wrong on"
        ),
        "class_tested": args.cls,
        "published_cap_chars": PUBLISHED_CAP,
        "metric": (
            "percent of the ground truth's distinctive tokens (4+ chars, "
            "stopwords removed) present in the context, capped against "
            "uncapped"
        ),
        "model_calls": 0,
        "controls": {
            "n_records": len(rows),
            "n_rebuilt": len(records),
            "transcripts_missing": missing,
            "rebuild_mismatches": control_failures,
            "all_controls_pass": controls_pass,
        },
        "totals": {
            "n": len(records),
            "n_recoverable": len(passed),
            "n_failed": len(failed),
            "n_with_zero_truncated_inputs": len(untruncated),
            "n_failed_with_zero_truncated_inputs": len(untruncated_failed),
            "median_bytes_withheld_by_cap": (
                int(statistics.median(r["bytes_withheld_by_cap"]
                                      for r in records)) if records else 0),
            "mean_coverage_gain_pct": (
                round(statistics.mean(gains), 2) if gains else None),
            "mean_coverage_gain_pct_failed_only": (
                round(statistics.mean(failed_gains), 2)
                if failed_gains else None),
            "n_records_gaining_any_ground_truth_token": len(any_gain),
            "n_failed_records_gaining_any_ground_truth_token":
                len(failed_any_gain),
        },
        "per_class": per_class,
        "verdict": verdict,
        "records": sorted(records, key=lambda r: -(r["tokens_gained"] or 0)),
    }

    out = REPORT if args.cls == "artifact_detail" else REPORT_ALL
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"wrote {out.relative_to(REPO)}")
    print()
    print(f"class {args.cls}: {len(records)} records rebuilt of {len(rows)}")
    print(f"controls: rebuild mismatches {len(control_failures)}, "
          f"transcripts missing {len(missing)} -> "
          f"{'PASS' if controls_pass else 'FAIL'}")
    print()
    t = report["totals"]
    print(f"recoverable {t['n_recoverable']} of {t['n']}, "
          f"failed {t['n_failed']}")
    print(f"failures whose inputs were never truncated at all: "
          f"{t['n_failed_with_zero_truncated_inputs']} of {t['n_failed']}")
    print(f"median bytes the cap withheld per record: "
          f"{t['median_bytes_withheld_by_cap']:,}")
    print(f"mean ground-truth coverage gain from uncapping: "
          f"{t['mean_coverage_gain_pct']} points "
          f"(failed records only: {t['mean_coverage_gain_pct_failed_only']})")
    print(f"failed records that gain ANY ground-truth token when uncapped: "
          f"{t['n_failed_records_gaining_any_ground_truth_token']} of "
          f"{t['n_failed']}")
    if len(per_class) > 1:
        print()
        print(f"{'class':18s} {'n':>4s} {'failed':>7s} {'cap never':>10s} "
              f"{'gain>0':>7s} {'mean gain':>10s}")
        for cls, v in per_class.items():
            print(f"{cls:18s} {v['n']:4d} {v['n_failed']:7d} "
                  f"{v['n_failed_with_zero_truncated_inputs']:10d} "
                  f"{v['n_failed_gaining_any_ground_truth_token']:7d} "
                  f"{str(v['mean_coverage_gain_pct_failed_only']):>10s}")
        print("  cap never = failures where no tool input was truncated, "
              "so the cap cannot be the cause")
        print("  gain>0 = failures that gain any ground-truth token "
              "once the cap is lifted")
    print()
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
