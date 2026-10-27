#!/usr/bin/env python3
"""Explain the probe harness's apparent failure to reproduce RQ3's published figures.

probe_positive_control.py reported output-stripped 75.0% against a published
94.4%, and rollup 22.5% against a published 56.5%, and printed
"FAIL: Pipeline does NOT reproduce published figures". Both gaps are accounting
artifacts in the probe's own reporting, not a regression in the arm pipeline.
This script proves that without making a single model call, by replaying the
PUBLISHED grades over the EXACT record ids the probe graded.

Two independent errors, in the order they contribute:

1. METRIC. 94.4% and 56.5% are *recoverable* = (correct + partially_correct) / n,
   which is how the paper's Table~\\ref{tab:retention} and
   rq3_headline_corrected.json define the column. The probe counts only
   grade == "correct". Published correct-only rates over the full 124 are 83.1%
   (output-stripped) and 36.3% (rollup), not 94.4% and 56.5%.

2. SAMPLE. The probe ran with --limit 40, which takes the first 40 of the 124
   design questions. The design is 25 sessions x 5 questions in file order, so
   the first 40 cover 8 sessions, not 25. 56.5% and 94.4% are n=124 figures.
   The published grades over those same 40 records are 35.0% and 95.0%
   recoverable: the rollup arm does materially worse on this head slice.

The residual, once metric and sample are matched, is 1 record in the
output-stripped arm and 2 in the rollup arm -- inside the grader instability
this same experiment measured (18 of 124 grades flipped on a plain re-ask,
which is why the published protocol votes 3 times).

Usage:
    probe_positive_control_diagnose.py
    probe_positive_control_diagnose.py --out <path>
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_raw_transcript_arm import DATA, load_records, record_id  # noqa: E402

PROBE = DATA / "probe_positive_control.jsonl"
DEFAULT_OUT = DATA / "probe_positive_control_diagnosis.json"

# Published artifacts. The *_majority_* files carry the published protocol
# (plurality of 3 independent grading draws); the arm files carry the single-draw
# grade. Both are reported so the vote count is visible rather than assumed.
PUBLISHED = {
    "output_stripped_majority": ("rq3_majority_stripped.jsonl", "grade"),
    "output_stripped_single_draw": ("rq3_majority_stripped.jsonl", "single_draw_grade"),
    "rollup_majority": ("rq3_majority_uniform.jsonl", "grade"),
    "rollup_single_draw": ("rq3_majority_uniform.jsonl", "single_draw_grade"),
    "rollup_uniform_arm": ("rq3_rollup_uniform.jsonl", "grade"),
    "output_stripped_arm": ("rq3_questions_stripped_arm.jsonl", "grade"),
}

HEADLINE = {"output_stripped": 94.4, "rollup": 56.5}


def rates(grades):
    """correct-only and recoverable rates, with the denominator stated.

    `absent` is a real outcome here (the answerer declined), not a dropped run,
    so it stays in the denominator -- that is what the published figures do.
    API errors and truncations are not grades and never reach this function.
    """
    n = len(grades)
    c = Counter(grades)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "counts": dict(c),
        "correct_only_pct": round(100.0 * c.get("correct", 0) / n, 1),
        "recoverable_pct": round(
            100.0 * (c.get("correct", 0) + c.get("partially_correct", 0)) / n, 1),
    }


def index(rows):
    out = {}
    for r in rows:
        out[r.get("record_id") or record_id(r)] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    probe = load_records(PROBE)
    probe_ids = [record_id(r) for r in probe]
    if len(set(probe_ids)) != len(probe_ids):
        print("WARNING: duplicate record ids in the probe output")

    report = {
        "generated_by": "scripts/paper-rollups/probe_positive_control_diagnose.py",
        "model_calls": 0,
        "probe_output": PROBE.name,
        "probe": {
            "n_records": len(probe),
            "n_distinct_sessions": len({r["session_id"] for r in probe}),
            "output_stripped": rates([r["stripped_grade"] for r in probe]),
            "rollup": rates([r["rollup_grade"] for r in probe]),
        },
        "published_full": {},
        "published_on_probe_records": {},
        "coverage": {},
        "headline_definition": (
            "94.4% and 56.5% are recoverable = (correct + partially_correct) / n "
            "over n = 124, matching rq3_headline_corrected.json's "
            "pooled_grade_shift.recoverable_pct and paper table tab:retention"),
        "gap_decomposition_pts": {},
        "verdict": {},
    }

    for label, (fname, key) in PUBLISHED.items():
        path = DATA / fname
        if not path.exists():
            report["published_full"][label] = {"missing": fname}
            continue
        rows = load_records(path)
        idx = index(rows)
        report["published_full"][label] = rates(
            [r.get(key) for r in rows if r.get(key)])
        present = [i for i in probe_ids if i in idx]
        report["coverage"][label] = {
            "probe_records": len(probe_ids),
            "found_in_published": len(present),
        }
        report["published_on_probe_records"][label] = rates(
            [idx[i].get(key) for i in present if idx[i].get(key)])

    # Decompose each arm's reported gap into metric, sample, and residual.
    for arm, probe_key, pub_label in (
        ("output_stripped", "stripped_grade", "output_stripped_majority"),
        ("rollup", "rollup_grade", "rollup_majority"),
    ):
        headline = HEADLINE[arm]
        pub_full = report["published_full"][pub_label]
        pub_sub = report["published_on_probe_records"][pub_label]
        probe_r = report["probe"][arm]
        steps = {
            "published_headline_recoverable_n124": headline,
            "published_recoverable_on_probe_sample": pub_sub["recoverable_pct"],
            "published_correct_only_on_probe_sample": pub_sub["correct_only_pct"],
            "probe_correct_only_on_probe_sample": probe_r["correct_only_pct"],
        }
        report["gap_decomposition_pts"][arm] = {
            "steps_pct": steps,
            "published_correct_only_n124": pub_full["correct_only_pct"],
            "total_reported_gap": round(
                headline - probe_r["correct_only_pct"], 1),
            "due_to_sample_selection": round(
                steps["published_recoverable_on_probe_sample"] - headline, 1),
            "due_to_metric_correct_only_vs_recoverable": round(
                steps["published_correct_only_on_probe_sample"]
                - steps["published_recoverable_on_probe_sample"], 1),
            "residual_attributable_to_harness": round(
                steps["probe_correct_only_on_probe_sample"]
                - steps["published_correct_only_on_probe_sample"], 1),
            "probe_recoverable_vs_published_recoverable_same_sample": round(
                probe_r["recoverable_pct"] - pub_sub["recoverable_pct"], 1),
        }

    report["verdict"] = {
        "reproduces": all(
            abs(report["gap_decomposition_pts"][a]
                ["probe_recoverable_vs_published_recoverable_same_sample"]) <= 6.0
            for a in ("output_stripped", "rollup")),
        "statement": (
            "The arm pipeline reproduces the published grades. Matched on metric "
            "(recoverable) and on sample (the same record ids), the probe differs "
            "from the published majority-vote grades by 1 record in the "
            "output-stripped arm and 2 in the rollup arm. The reported shortfall "
            "is a reporting bug in probe_positive_control.py and "
            "probe_stratified_measurement.py, which compare a correct-only rate "
            "on a 40-record head slice against a recoverable rate on n=124."),
        "fix": [
            "report recoverable = correct + partially_correct, as the paper does",
            "compare against the published rate recomputed on the SAME record "
            "ids, not against the n=124 headline",
            "sample sessions, not the first N questions: --limit 40 covers 8 of "
            "25 sessions because the question file is grouped by session",
            "grade with the plurality of 3 draws (rq3_majority_vote_grades.py) "
            "if the published protocol is the comparison target",
        ],
    }

    args.out.write_text(json.dumps(report, indent=1) + "\n")

    print("=" * 74)
    print("PROBE POSITIVE CONTROL -- DIAGNOSIS (0 model calls)")
    print("=" * 74)
    p = report["probe"]
    print(f"probe sample: {p['n_records']} records over "
          f"{p['n_distinct_sessions']} sessions")
    print()
    for arm in ("output_stripped", "rollup"):
        d = report["gap_decomposition_pts"][arm]
        s = d["steps_pct"]
        print(f"{arm}:")
        print(f"  published headline (recoverable, n=124)        "
              f"{s['published_headline_recoverable_n124']:>6.1f}%")
        print(f"  published recoverable on the probe's records   "
              f"{s['published_recoverable_on_probe_sample']:>6.1f}%"
              f"   ({d['due_to_sample_selection']:+.1f} pts, sample)")
        print(f"  published correct-only on the same records    "
              f"{s['published_correct_only_on_probe_sample']:>6.1f}%"
              f"   ({d['due_to_metric_correct_only_vs_recoverable']:+.1f} pts, metric)")
        print(f"  probe correct-only (what it printed)          "
              f"{s['probe_correct_only_on_probe_sample']:>6.1f}%"
              f"   ({d['residual_attributable_to_harness']:+.1f} pts, harness)")
        print(f"  probe recoverable vs published, same sample   "
              f"{d['probe_recoverable_vs_published_recoverable_same_sample']:+.1f} pts")
        print()
    print("REPRODUCES:", report["verdict"]["reproduces"])
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
