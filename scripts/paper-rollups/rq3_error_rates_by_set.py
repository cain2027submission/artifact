#!/usr/bin/env python3
"""Wrong / absent rates per arm per set from existing grade files. No model calls.

Two fields per record, labelled by what they are:
  `grade`           the current protocol: 3-draw majority of the revised grader
                    (grade_draws; regrade_s4.py, 2026-10-09). The paper's recall and
                    error rates (55.6%, 90.3%, 11.5%, 1.8%) all come from this field.
  `grade_published` the grade the paper carried before 2026-10-09 (superseded).
Headline rows use `grade`. (An earlier version of this script had the two labels
swapped; corrected 2026-10-13.)

Gate: held-out `grade_published` counts must equal rq3_heldout_no_digest_paired.json
(that file was built before the regrade and is stale against `grade`), else exit 1
without writing.

Usage: rq3_error_rates_by_set.py
"""
import json
import sys
from math import sqrt
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "paper-rollups"
SETS = {
    "heldout_125": {
        "no_digest": "rq3_heldout_no_digest_arm.jsonl",
        "output_stripped": "rq3_heldout_stripped_arm_extended.jsonl",
    },
    "design_124": {
        "no_digest": "rq3_action_record_no_digest_arm.jsonl",
        "output_stripped": "rq3_questions_stripped_arm.jsonl",
    },
}
OUT = DATA / "rq3_error_rates_by_set.json"
PAIRED = DATA / "rq3_heldout_no_digest_paired.json"


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def load(name):
    return [json.loads(l) for l in (DATA / name).open() if l.strip()]


def summarize(rows, field):
    g = [(r.get(field) if field != "grade_published" else r.get("grade_published", r.get("grade"))) for r in rows]
    n = len(g)
    c = {k: g.count(k) for k in ("correct", "partially_correct", "wrong", "absent")}
    att = n - c["absent"]
    return {
        "n": n, **c, "attempted": att,
        "wrong_given_attempted_pct": round(100 * c["wrong"] / att, 1) if att else None,
        "wrong_given_attempted_ci95": wilson(c["wrong"], att),
        "absent_pct": round(100 * c["absent"] / n, 1),
        "absent_ci95": wilson(c["absent"], n),
        "recall_correct_plus_partial_pct": round(100 * (c["correct"] + c["partially_correct"]) / n, 1),
    }


def main():
    res = {"headline_field": "grade (3-draw majority, current protocol)", "sets": {}}
    for s, arms in SETS.items():
        res["sets"][s] = {a: {"file": f, "grade": summarize(load(f), "grade"),
                              "grade_published_pre_2026_10_09": summarize(load(f), "grade_published")}
                          for a, f in arms.items()}
    p = json.loads(PAIRED.read_text())
    h = res["sets"]["heldout_125"]
    gate = {}
    for a, key in (("output_stripped", "output_stripped"), ("no_digest", "action_record_no_digest")):
        for m in ("wrong", "absent"):
            gate[f"{a}.{m}"] = (h[a]["grade_published_pre_2026_10_09"][m], p[key][m]["n"])
    bad = {k: v for k, v in gate.items() if v[0] != v[1]}
    res["gate_vs_paired_json"] = {"pairs_(grade_published_here,paired_json)": gate, "pass": not bad}
    if bad:
        print("GATE MISMATCH", bad)
        sys.exit(1)
    d = res["sets"]["design_124"]["output_stripped"]
    res["paper_1_8_pct_check"] = {
        "paper_claims_pct": 1.8,
        "from_grade_pct": d["grade"]["wrong_given_attempted_pct"],
        "from_grade_published_pct": d["grade_published_pre_2026_10_09"]["wrong_given_attempted_pct"],
        "reproduces_from_grade": d["grade"]["wrong_given_attempted_pct"] == 1.8,
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    for s, arms in res["sets"].items():
        for a, v in arms.items():
            for k in ("grade", "grade_published_pre_2026_10_09"):
                x = v[k]
                print(s, a, k, {i: x[i] for i in ("n", "correct", "partially_correct", "wrong", "absent", "wrong_given_attempted_pct", "wrong_given_attempted_ci95", "absent_pct")})
    print(res["paper_1_8_pct_check"])


main()
