#!/usr/bin/env python3
"""Held-out rollup error test, per data/paper-rollups/heldout-rollup/decision_rule.md.
No model calls. Held-out and design-set rows come from one code path.
Usage: heldout_rollup_error_test.py
"""
import json
import random
import sys
from collections import Counter
from math import comb, sqrt
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "paper-rollups"
OUT = DATA / "heldout-rollup" / "results.json"
B, SEED, F = 10000, 20261013, "grade"


def load(p):
    return [json.loads(l) for l in (DATA / p).open() if l.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def summ(rows):
    c = Counter(r[F] for r in rows)
    n, w, a = len(rows), c["wrong"], c["absent"]
    att = n - a
    return {"n": n, "attempted": att, "wrong": w,
            "wrong_given_attempted_pct": round(100 * w / att, 1),
            "wrong_given_attempted_ci95": wilson(w, att),
            "absent": a, "absent_pct": round(100 * a / n, 1),
            "recall_pct": round(100 * (c["correct"] + c["partially_correct"]) / n, 1)}


def fisher(a, b, c, d):
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    pr = lambda x: comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)
    p0 = pr(a)
    return sum(pr(x) for x in range(max(0, c1 - r2), min(r1, c1) + 1) if pr(x) <= p0 * (1 + 1e-9))


def rates(rows):
    att = sum(r[F] != "absent" for r in rows)
    return sum(r[F] == "wrong" for r in rows) / att


def compare(a_rows, b_rows):
    wa, aa = sum(r[F] == "wrong" for r in a_rows), sum(r[F] != "absent" for r in a_rows)
    wb, ab = sum(r[F] == "wrong" for r in b_rows), sum(r[F] != "absent" for r in b_rows)
    sess = {}
    for i, x in enumerate(a_rows):
        sess.setdefault(x["session_id"], []).append(i)
    keys = list(sess)
    rng = random.Random(SEED)
    diffs = []
    for _ in range(B):
        idx = [i for k in (rng.choice(keys) for _ in keys) for i in sess[k]]
        try:
            diffs.append(rates([a_rows[i] for i in idx]) - rates([b_rows[i] for i in idx]))
        except ZeroDivisionError:
            continue
    diffs.sort()
    q = lambda x: round(100 * diffs[int(x * (len(diffs) - 1))], 1)
    lo, hi = q(.025), q(.975)
    return {"diff_wrong_given_attempted_pts": round(100 * (wa / aa - wb / ab), 1),
            "fisher_exact_p_two_sided": fisher(wa, aa - wa, wb, ab - wb),
            "clustered_bootstrap_ci95_pts": [lo, hi], "n_sessions": len(keys),
            "resamples": len(diffs), "B": B, "seed": SEED}


def pair(a, b):
    bi = {r["record_id"]: r for r in b}
    assert set(bi) == {r["record_id"] for r in a}, "record sets differ"
    return a, [bi[r["record_id"]] for r in a]


def block(roll, strip):
    roll, strip = pair(roll, strip)
    return {"rollup": summ(roll), "output_stripped": summ(strip),
            "rollup_minus_output_stripped": compare(roll, strip)}


def verdict(d):
    lo, hi = d["clustered_bootstrap_ci95_pts"]
    pt = d["diff_wrong_given_attempted_pts"]
    if lo > 0:
        return "KEPT"
    if pt < 0 and hi < 0:
        return "REMOVED"
    return "DOWNGRADED"


def main():
    held = block(load("heldout-rollup/rollup_arm.jsonl"), load("rq3_heldout_stripped_arm_extended.jsonl"))
    design = block(load("rq3_rollup_uniform.jsonl"), load("rq3_questions_stripped_arm.jsonl"))
    expected = {"diff": 9.8, "ci": [2.2, 18.6]}
    d = design["rollup_minus_output_stripped"]
    reproduces = d["diff_wrong_given_attempted_pts"] == expected["diff"] and d["clustered_bootstrap_ci95_pts"] == expected["ci"]
    res = {"field": F, "design_set_control": design, "design_set_reproduces_paper": reproduces,
           "design_set_expected": expected, "heldout": held,
           "verdict": verdict(held["rollup_minus_output_stripped"]),
           "rule": "heldout-rollup/decision_rule.md: KEPT if CI lower > 0; DOWNGRADED if CI includes 0; REMOVED if point < 0 and CI upper < 0"}
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))


main()
