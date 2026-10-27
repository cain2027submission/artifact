#!/usr/bin/env python3
"""Which grade field produced the paper's 11.5% (rollup) and 1.8% (output-stripped)
wrong-of-attempted, and is the difference significant? No model calls, stdlib only.

Arms, all on the 124 design questions:
  rollup_uniform  rq3_rollup_uniform.jsonl            (paper tables, recall 55.6%)
  rollup_pooled   rq3_questions.jsonl restricted to the 124 design (session, question)
                  pairs (original pooled arm, three prompts, recall 62.1% under buggy parser)
  output_stripped rq3_questions_stripped_arm.jsonl

Fields: `grade` = 3-draw majority of the revised grader (grade_draws, protocol
...majority3...), written by regrade_s4.py on 2026-10-09; `grade_published` = the grade
the paper carried before that date (single-pass). Held-out: no graded rollup held-out
file exists (only no_digest, action_record and output_stripped).

Usage: rq3_error_rate_fix.py
"""
import json
import random
from collections import Counter
from math import comb, sqrt
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "paper-rollups"
OUT = DATA / "rq3_error_rate_fix.json"
FIELDS = ("grade", "grade_published")


def load(name):
    return [json.loads(l) for l in (DATA / name).open() if l.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def summ(rows, f):
    g = [r[f] for r in rows]
    n = len(g)
    c = Counter(g)
    w, a = c["wrong"], c["absent"]
    att = n - a
    return {
        "n": n, "attempted": att, "wrong": w, "absent": a,
        "wrong_given_attempted_pct": round(100 * w / att, 1),
        "wrong_given_attempted_ci95": wilson(w, att),
        "absent_pct": round(100 * a / n, 1), "absent_ci95": wilson(a, n),
        "unrecovered_pct": round(100 * (w + a) / n, 1), "unrecovered_ci95": wilson(w + a, n),
        "recall_pct": round(100 * (c["correct"] + c["partially_correct"]) / n, 1),
    }


def fisher(a, b, c, d):
    """two-sided Fisher exact on [[a,b],[c,d]] (a,c = wrong; b,d = right)."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    pr = lambda x: comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)
    p0 = pr(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 * (1 + 1e-9))


def wga(rows, f):
    att = [r for r in rows if r[f] != "absent"]
    return sum(r[f] == "wrong" for r in att) / len(att)


def rates(rows, f):
    att = sum(r[f] != "absent" for r in rows)
    return sum(r[f] == "wrong" for r in rows) / att, sum(r[f] == "absent" for r in rows) / len(rows)


def compare(a_rows, b_rows, f, B=10000, seed=20261013):
    """a minus b (paired by record order), wrong-given-attempted; resample sessions."""
    wa, aa = sum(r[f] == "wrong" for r in a_rows), sum(r[f] != "absent" for r in a_rows)
    wb, ab = sum(r[f] == "wrong" for r in b_rows), sum(r[f] != "absent" for r in b_rows)
    p = fisher(wa, aa - wa, wb, ab - wb)
    sess = {}
    for i, (x, y) in enumerate(zip(a_rows, b_rows)):
        sess.setdefault(x["session_id"], []).append(i)
    keys = list(sess)
    rng = random.Random(seed)
    diffs, absd = [], []
    for _ in range(B):
        idx = [i for k in (rng.choice(keys) for _ in keys) for i in sess[k]]
        ra, rb = [a_rows[i] for i in idx], [b_rows[i] for i in idx]
        try:
            (wa_, aba), (wb_, abb) = rates(ra, f), rates(rb, f)
        except ZeroDivisionError:
            continue
        diffs.append(wa_ - wb_)
        absd.append(aba - abb)
    diffs.sort(); absd.sort()
    q = lambda v, x: round(100 * v[int(x * (len(v) - 1))], 1)
    d = 100 * (wa / aa - wb / ab)
    return {
        "diff_wrong_given_attempted_pts": round(d, 1),
        "fisher_exact_p_two_sided": p,
        "clustered_bootstrap_ci95_pts": [q(diffs, .025), q(diffs, .975)],
        "clustered_bootstrap_ci_includes_0": q(diffs, .025) <= 0 <= q(diffs, .975),
        "n_sessions": len(keys), "resamples": len(diffs),
        "absent_diff_pts": round(100 * (sum(r[f] == "absent" for r in a_rows) / len(a_rows)
                                        - sum(r[f] == "absent" for r in b_rows) / len(b_rows)), 1),
        "absent_diff_clustered_ci95_pts": [q(absd, .025), q(absd, .975)],
    }


def mcnemar(a_ok, b_ok):
    """exact two-sided McNemar on paired booleans."""
    n10 = sum(x and not y for x, y in zip(a_ok, b_ok))
    n01 = sum(y and not x for x, y in zip(a_ok, b_ok))
    n = n10 + n01
    k = min(n10, n01)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"a_only": n10, "b_only": n01, "p_two_sided": p}


def heldout():
    nd, st = load("rq3_heldout_no_digest_arm.jsonl"), load("rq3_heldout_stripped_arm_extended.jsonl")
    sidx = {r["record_id"]: r for r in st}
    assert set(sidx) == {r["record_id"] for r in nd}
    st = [sidx[r["record_id"]] for r in nd]
    prot = lambda rs: sorted({r["grade_protocol"] for r in rs})
    ok = lambda rs, f: [r[f] in ("correct", "partially_correct") for r in rs]
    out = {
        "no_digest_file": "rq3_heldout_no_digest_arm.jsonl", "output_stripped_file": "rq3_heldout_stripped_arm_extended.jsonl",
        "grade_protocol_no_digest": prot(nd), "grade_protocol_output_stripped": prot(st),
        "same_protocol_as_design_set": prot(nd) == prot(st) == prot(load("rq3_questions_stripped_arm.jsonl")),
        "paired_json_field": "grade_published (pre-2026-10-09): 83.2% = 72+32 of 125 stripped, 80.8% = 66+35 no-digest, "
                             "McNemar p=0.549 = 4 vs 7 discordant. rq3_heldout_no_digest_paired.json predates the regrade and was never rebuilt.",
        "arms": {n: {"records": len(rs), **{f: summ(rs, f) for f in FIELDS}}
                 for n, rs in (("no_digest", nd), ("output_stripped", st))},
        "mcnemar_recoverable_no_digest_vs_stripped": {f: mcnemar(ok(nd, f), ok(st, f)) for f in FIELDS},
        "interpretation": "Held-out files carry the same revised-grader protocol in `grade` as the design set, so the "
                          "jump in output-stripped wrong-given-attempted from 1.8% (design) to 11.2% (held-out, `grade`) "
                          "is not a grading-protocol difference; it is a different question set (125 held-out "
                          "questions) under one grader. Only the stale paired json is on the old field.",
    }
    return out


def main():
    uni, strp = load("rq3_rollup_uniform.jsonl"), load("rq3_questions_stripped_arm.jsonl")
    sidx = {r["record_id"]: r for r in strp}
    assert set(sidx) == {r["record_id"] for r in uni}
    strp = [sidx[r["record_id"]] for r in uni]  # pair by record_id
    # grade == majority of grade_draws? (proves `grade` is the 3-draw field)
    maj = lambda r: Counter(r["grade_draws"]).most_common(1)[0]
    chk = {n: sum(maj(r)[1] >= 2 and maj(r)[0] == r["grade"] for r in rs)
           for n, rs in (("uniform", uni), ("stripped", strp))}
    qs = {(r["session_id"], r["question"]) for r in uni}
    pooled_all = load("rq3_questions.jsonl")
    pooled = [r for r in pooled_all if (r["session_id"], r["question"]) in qs]
    # order pooled like uni for pairing
    pidx = {(r["session_id"], r["question"]): r for r in pooled}
    pooled = [pidx[(r["session_id"], r["question"])] for r in uni if (r["session_id"], r["question"]) in pidx]
    arms = {"rollup_uniform": uni, "rollup_pooled": pooled, "output_stripped": strp}
    res = {
        "finding": "11.5% = 9 wrong / 78 attempted and 1.8% = 2 / 114 both come from the `grade` field "
                   "(3-draw majority, regrade 2026-10-09), the same field the paper recall figures "
                   "55.6% and 90.3% come from. `grade_published` is the pre-2026-10-09 grade "
                   "(rollup 56.5% recall, stripped 94.4%).",
        "grade_equals_majority_of_grade_draws": chk,
        "heldout_rollup": "no graded rollup held-out file exists (held-out has no_digest, action_record, output_stripped only)",
        "heldout": heldout(),
        "arms": {n: {"records": len(rs), **{f: summ(rs, f) for f in FIELDS}} for n, rs in arms.items()},
        "pooled_matched_to_design_124": len(pooled),
        "rollup_uniform_minus_output_stripped": {f: compare(uni, strp, f) for f in FIELDS},
    }
    if len(pooled) == len(strp):
        res["rollup_pooled_minus_output_stripped"] = {f: compare(pooled, strp, f) for f in FIELDS}
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    for n, v in res["arms"].items():
        for f in FIELDS:
            x = v[f]
            print(n, f, x["n"], x["attempted"], x["wrong"], x["wrong_given_attempted_pct"], x["wrong_given_attempted_ci95"],
                  "absent", x["absent"], x["absent_pct"], "unrec", x["unrecovered_pct"], "recall", x["recall_pct"])
    for k in ("rollup_uniform_minus_output_stripped", "rollup_pooled_minus_output_stripped"):
        for f, v in res.get(k, {}).items():
            print(k, f, v)
    print(chk, len(pooled))


main()
