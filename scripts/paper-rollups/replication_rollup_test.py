#!/usr/bin/env python3
"""Analysis for data/paper-rollups/replication-rollup/analysis_plan.md. No model calls.
Usage: replication_rollup_test.py"""
import json, random
from collections import Counter
from math import comb, sqrt
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "paper-rollups"
RR = DATA / "replication-rollup"
B, SEED, F = 10000, 20261013, "grade"


def load(p):
    return [json.loads(l) for l in (DATA / p).open() if l.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def summ(rows):
    c = Counter(r[F] for r in rows)
    n, w, a = len(rows), c["wrong"], c["absent"]
    att = n - a
    return {"n": n, "attempted": att, "wrong": w,
            "wrong_given_attempted_pct": round(100 * w / att, 1) if att else None,
            "wrong_given_attempted_ci95": wilson(w, att),
            "absent": a, "absent_pct": round(100 * a / n, 1),
            "recall_pct": round(100 * (c["correct"] + c["partially_correct"]) / n, 1)}


def fisher(a, b, c, d):
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    pr = lambda x: comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)
    p0 = pr(a)
    return sum(pr(x) for x in range(max(0, c1 - r2), min(r1, c1) + 1) if pr(x) <= p0 * (1 + 1e-9))


def m_wga(rows):
    att = sum(r[F] != "absent" for r in rows)
    return sum(r[F] == "wrong" for r in rows) / att


def m_abs(rows):
    return sum(r[F] == "absent" for r in rows) / len(rows)


def m_rec(rows):
    return sum(r[F] in ("correct", "partially_correct") for r in rows) / len(rows)


def compare(a_rows, b_rows):
    wa, aa = sum(r[F] == "wrong" for r in a_rows), sum(r[F] != "absent" for r in a_rows)
    wb, ab = sum(r[F] == "wrong" for r in b_rows), sum(r[F] != "absent" for r in b_rows)
    sess = {}
    for i, x in enumerate(a_rows):
        sess.setdefault(x["session_id"], []).append(i)
    keys = list(sess)
    out = {"n_sessions": len(keys), "B": B, "seed": SEED}
    for name, fn in (("wrong_given_attempted", m_wga), ("absent", m_abs), ("recall", m_rec)):
        rng = random.Random(SEED)
        diffs = []
        for _ in range(B):
            idx = [i for k in (rng.choice(keys) for _ in keys) for i in sess[k]]
            try:
                diffs.append(fn([a_rows[i] for i in idx]) - fn([b_rows[i] for i in idx]))
            except ZeroDivisionError:
                continue
        diffs.sort()
        q = lambda x: round(100 * diffs[int(x * (len(diffs) - 1))], 1)
        out[name] = {"diff_pts": round(100 * (fn(a_rows) - fn(b_rows)), 1), "clustered_ci95_pts": [q(.025), q(.975)]}
    out["wrong_given_attempted"]["fisher_exact_p_two_sided"] = fisher(wa, aa - wa, wb, ab - wb)
    return out


def main():
    roll_all = load("replication-rollup/rollup_arm.jsonl")
    sens = load("replication-rollup/rollup_arm_sensitivity.jsonl")
    ans = load("replication/answers.jsonl")
    strip = {r["question_id"]: r for r in ans if r["arm"] == "stripped"}
    nod = {r["question_id"]: r for r in ans if r["arm"] == "nodigest"}
    covered_ids = {r["record_id"] for r in roll_all}
    excluded = sorted(set(strip) - covered_ids)
    roll = [r for r in roll_all]
    S = [strip[r["record_id"]] for r in roll]
    N = [nod[r["record_id"]] for r in roll]
    av = json.load(open(RR / "availability.json"))
    mt = {r["session_id"]: r["mtime"] for r in av["rows"]}
    early = lambda r: mt[r["session_id"]] < "2026-09-27"
    split = {}
    for lab, f in (("mtime_before_2026-09-27", early), ("mtime_on_or_after_2026-09-27", lambda r: not early(r))):
        idx = [i for i, r in enumerate(roll) if f(r)]
        a, b = [roll[i] for i in idx], [S[i] for i in idx]
        split[lab] = {"rollup": summ(a), "output_stripped": summ(b),
                      "rollup_minus_output_stripped": compare(a, b)}
    res = {
        "plan": "analysis_plan.md", "field": F,
        "coverage": {"answered_questions": 196, "sessions": 98, "notes_available": 98,
                     "no_rollup_available": 0, "rollup_rows": len(roll),
                     "excluded_after_one_retry": excluded,
                     "exclusion_reason": "gateway returned finish_reason=content_filter with empty response on answer after the single allowed retry; excluded and counted, never scored absent",
                     "note_cl100k": av["cl100k"]},
        "arms_on_covered": {"rollup": summ(roll), "output_stripped": summ(S), "no_digest": summ(N)},
        "arms_on_all_196": {"output_stripped": summ(list(strip.values())), "no_digest": summ(list(nod.values()))},
        "rollup_minus_output_stripped": compare(roll, S),
        "rollup_minus_no_digest": compare(roll, N),
        "sensitivity_mtime_split_weak_writer_inference": split,
        "sensitivity_alt_note_2ce8e87b": {"rows": [{"record_id": r["record_id"], "grade": r["grade"]} for r in sens],
            "primary_rows": [{"record_id": r["record_id"], "grade": r["grade"]} for r in roll if r["session_prefix"] == "2ce8e87b"]},
        "provenance": "writer unknown/inferred for all 98 notes; frontmatter model is session model, not writer; see availability.json",
    }
    (RR / "results.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: res[k] for k in ("arms_on_covered", "rollup_minus_output_stripped", "rollup_minus_no_digest", "sensitivity_mtime_split_weak_writer_inference", "sensitivity_alt_note_2ce8e87b")}, indent=1))


main()
