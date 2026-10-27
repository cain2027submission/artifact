#!/usr/bin/env python3
"""Analysis for the later-window experiment, as specified in late-window/DESIGN.md.
No model calls. Writes late-window/results.json and prints the tables.

EARLY rows come from the stored published files (frozen 124, stored 3-draw
majority grades) through the same functions as MID and LATE.

Run: python3 scripts/paper-rollups/rq3_late_analyse.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_bootstrap as cb  # noqa: E402
import arm_rates as ar  # noqa: E402
import rq3_late_common as lc  # noqa: E402

REC = ("correct", "partially_correct")
EARLY_FILES = {"rollup": "rq3_rollup_uniform.jsonl", "stripped": "rq3_questions_stripped_arm.jsonl",
               "prose": "rq3_prose_only_arm.jsonl", "noop": "rq3_zero_context_arm.jsonl",
               "callonly": "rq3_tool_inputs_only_arm.jsonl"}
ARMS = ["rollup", "stripped", "prose", "callonly", "noop"]


def early_rows():
    graded = set(json.loads(lc.SUMMARY.read_text())["graded_record_ids"])
    out = {}
    for arm, f in EARLY_FILES.items():
        rows = cb.load(f)
        out[arm] = [{"rid": rid, "session": r["session_id"], "class": r["class"], "grade": r["grade"]}
                    for rid, r in rows.items() if rid in graded]
    return out


def late_rows(window):
    out = defaultdict(list)
    for r in lc.read_jsonl(lc.OUTDIR / "grades.jsonl"):
        if r["window"] != window or r["grade"] is None:
            continue
        out[r["arm"]].append({"rid": r["qid"], "session": r["prefix"], "class": r["class"],
                              "grade": r["grade"]})
    return out


def describe(rows):
    c = Counter(r["grade"] for r in rows)
    n = len(rows)
    rec = c["correct"] + c["partially_correct"]
    att = rec + c["wrong"]
    lo, hi = ar.wilson(rec, n) if n else (0, 0)
    wlo, whi = ar.wilson(c["wrong"], att) if att else (0, 0)
    return {"n": n, "sessions": len({r["session"] for r in rows}), "correct": c["correct"],
            "partial": c["partially_correct"], "wrong": c["wrong"], "absent": c["absent"],
            "recall_pct": round(100 * rec / n, 1) if n else None,
            "recall_wilson": [round(lo, 1), round(hi, 1)],
            "wrong_of_attempted_pct": round(100 * c["wrong"] / att, 1) if att else None,
            "wrong_of_attempted_wilson": [round(wlo, 1), round(whi, 1)],
            "attempted": att}


def series(rows, keep=lambda r: True):
    return {r["rid"]: (r["session"], int(r["grade"] in REC)) for r in rows if keep(r)}


def paired(arms_rows, a, b, keep=lambda r: True):
    sa, sb = series(arms_rows.get(a, []), keep), series(arms_rows.get(b, []), keep)
    shared = sorted(set(sa) & set(sb))
    if not shared:
        return None
    sessions = sorted({sa[r][0] for r in shared})
    boot = cb.Boot({a: sa, b: sb}, sessions)
    _, point, d, (da, db) = boot.diff(a, b, shared)
    ya, c = boot.agg(sa, shared)
    deff = cb.icc_deff(ya, c)
    return {"a": a, "b": b, "n": len(shared), "sessions": len(sessions),
            "a_pct": round(100 * sum(sa[r][1] for r in shared) / len(shared), 1),
            "b_pct": round(100 * sum(sb[r][1] for r in shared) / len(shared), 1),
            "diff_pts": round(100 * point, 1), "ci95": cb.ci(d), "p": round(cb.boot_p(d), 4),
            "discordant_a_only": da, "discordant_b_only": db,
            "deff_a_arm": deff.get("design_effect")}


def cross_window(rows_m, rows_l, arm, other=None):
    """LATE minus MID recall for `arm` (or for arm-minus-other, the difference in
    differences) over sessions present in both windows, resampling sessions."""
    def per_session(rows_by_arm):
        acc = defaultdict(lambda: [0.0, 0])
        a = {r["rid"]: r for r in rows_by_arm.get(arm, [])}
        o = {r["rid"]: r for r in rows_by_arm.get(other, [])} if other else None
        for rid, r in a.items():
            if o is not None and rid not in o:
                continue
            v = int(r["grade"] in REC) - (int(o[rid]["grade"] in REC) if o is not None else 0)
            acc[r["session"]][0] += v
            acc[r["session"]][1] += 1
        return acc
    pm, pl = per_session(rows_m), per_session(rows_l)
    sess = sorted(set(pm) & set(pl))
    ym = np.array([pm[s][0] for s in sess]); cm = np.array([pm[s][1] for s in sess])
    yl = np.array([pl[s][0] for s in sess]); cl = np.array([pl[s][1] for s in sess])
    boot = cb.Boot({}, sess)
    d = (boot.W @ yl) / (boot.W @ cl) - (boot.W @ ym) / (boot.W @ cm)
    point = yl.sum() / cl.sum() - ym.sum() / cm.sum()
    return {"arm": arm, "minus": other, "sessions": len(sess), "n_mid": int(cm.sum()),
            "n_late": int(cl.sum()), "mid_pct": round(100 * ym.sum() / cm.sum(), 1),
            "late_pct": round(100 * yl.sum() / cl.sum(), 1),
            "late_minus_mid_pts": round(100 * point, 1), "ci95": cb.ci(d), "p": round(cb.boot_p(d), 4)}


def main():
    W = {"early": early_rows(), "mid": late_rows("mid"), "late": late_rows("late")}
    res = {"descriptive": {}, "per_class": {}, "paired": {}, "sensitivity": {}, "cross_window": {}}
    for w, arms in W.items():
        res["descriptive"][w] = {a: describe(rows) for a, rows in arms.items() if rows}
        res["per_class"][w] = {a: {c: describe([r for r in rows if r["class"] == c])["recall_pct"]
                                   for c in lc.CLASSES} for a, rows in arms.items() if rows}
    # gate (c)
    e = res["descriptive"]["early"]
    res["gate_c"] = {"rollup": e["rollup"]["recall_pct"], "stripped": e["stripped"]["recall_pct"],
                     "pass": e["rollup"]["recall_pct"] == 55.6 and e["stripped"]["recall_pct"] == 90.3,
                     "n": [e["rollup"]["n"], e["stripped"]["n"]]}
    for w, arms in W.items():
        res["paired"][w] = {f"{a}-rollup": paired(arms, a, "rollup") for a in ("stripped", "prose", "noop")}
        no_out = lambda r: r["class"] != "outcome"
        res["sensitivity"][w] = {f"{a}-rollup_no_outcome": paired(arms, a, "rollup", no_out)
                                 for a in ("stripped", "prose")}
        if w != "early":
            strip_sess = {r["session"] for r in arms.get("stripped", [])}
            res["sensitivity"][w]["prose-rollup_on_stripped_sessions"] = paired(
                arms, "prose", "rollup", lambda r: r["session"] in strip_sess)
            res["sensitivity"][w]["stripped-prose"] = paired(arms, "stripped", "prose")
    res["cross_window"]["rollup_late_minus_mid"] = cross_window(W["mid"], W["late"], "rollup")
    res["cross_window"]["prose_late_minus_mid"] = cross_window(W["mid"], W["late"], "prose")
    res["cross_window"]["prose_minus_rollup_did"] = cross_window(W["mid"], W["late"], "prose", "rollup")
    res["unresolved_grades"] = sum(1 for r in lc.read_jsonl(lc.OUTDIR / "grades.jsonl") if r["grade"] is None)
    (lc.OUTDIR / "results.json").write_text(json.dumps(res, indent=1))

    for w in W:
        print(f"\n== {w.upper()}")
        for a, d in res["descriptive"][w].items():
            print(f"  {a:9s} n={d['n']:3d} s={d['sessions']:2d} recall {d['recall_pct']}% {d['recall_wilson']} "
                  f"c/p/w/a {d['correct']}/{d['partial']}/{d['wrong']}/{d['absent']} "
                  f"w/att {d['wrong_of_attempted_pct']}% ({d['wrong']}/{d['attempted']})")
        for k, v in {**res["paired"][w], **res["sensitivity"][w]}.items():
            if v:
                print(f"  {k:40s} n={v['n']:3d} s={v['sessions']:2d} {v['a_pct']} vs {v['b_pct']} "
                      f"diff {v['diff_pts']:+.1f} {v['ci95']} p={v['p']} deff={v['deff_a_arm']}")
        print("  per class:", json.dumps(res["per_class"][w]))
    print("\ncross-window:", json.dumps(res["cross_window"], indent=1))
    print("gate c:", res["gate_c"], "unresolved:", res["unresolved_grades"])


if __name__ == "__main__":
    main()
