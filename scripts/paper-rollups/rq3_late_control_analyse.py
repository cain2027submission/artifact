#!/usr/bin/env python3
"""Analysis for the EARLY-window control (late-window/CONTROL_DESIGN.md). No model calls.

EARLY rows: late-window/control/grades.jsonl. MID and LATE rows: the stored
late-window/grades.jsonl, restricted to the sessions that have EARLY grades, so
every window is the same set of sessions. Functions are rq3_late_analyse's own.
Writes late-window/control/results.json.

Run: python3 scripts/paper-rollups/rq3_late_control_analyse.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_bootstrap as cb  # noqa: E402
import rq3_late_common as lc  # noqa: E402
import rq3_late_analyse as la  # noqa: E402

CTRL = lc.OUTDIR / "control"
REC = la.REC


def rows_of(path, window, sessions=None):
    out = defaultdict(list)
    for r in lc.read_jsonl(path):
        if r["window"] != window or r["grade"] is None:
            continue
        if sessions is not None and r["prefix"] not in sessions:
            continue
        out[r["arm"]].append({"rid": r["qid"], "session": r["prefix"], "class": r["class"],
                              "grade": r["grade"]})
    return out


def between(rows_a, rows_b, arm, other=None):
    """Window B minus window A recall for `arm` (or arm-minus-other, the
    difference in differences), over sessions in both, resampling sessions with
    the same 10,000 cluster_bootstrap weights."""
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
    pa, pb = per_session(rows_a), per_session(rows_b)
    sess = sorted(set(pa) & set(pb))
    if not sess:
        return None
    ya = np.array([pa[s][0] for s in sess]); ca = np.array([pa[s][1] for s in sess])
    yb = np.array([pb[s][0] for s in sess]); cb_ = np.array([pb[s][1] for s in sess])
    boot = cb.Boot({}, sess)
    d = (boot.W @ yb) / (boot.W @ cb_) - (boot.W @ ya) / (boot.W @ ca)
    point = yb.sum() / cb_.sum() - ya.sum() / ca.sum()
    return {"arm": arm, "minus": other, "sessions": len(sess), "n_a": int(ca.sum()), "n_b": int(cb_.sum()),
            "a_pct": round(100 * ya.sum() / ca.sum(), 1), "b_pct": round(100 * yb.sum() / cb_.sum(), 1),
            "b_minus_a_pts": round(100 * point, 1), "ci95": cb.ci(d), "p": round(cb.boot_p(d), 4)}


def main():
    early = rows_of(CTRL / "grades.jsonl", "early")
    S = {r["session"] for rows in early.values() for r in rows}
    W = {"early": early,
         "mid": rows_of(lc.OUTDIR / "grades.jsonl", "mid", S),
         "late": rows_of(lc.OUTDIR / "grades.jsonl", "late", S)}
    res = {"sessions": sorted(S), "n_sessions": len(S), "descriptive": {}, "per_class": {},
           "paired": {}, "between": {}}
    for w, arms in W.items():
        res["descriptive"][w] = {a: la.describe(rows) for a, rows in arms.items() if rows}
        res["per_class"][w] = {a: {c: la.describe([r for r in rows if r["class"] == c])["recall_pct"]
                                   for c in lc.CLASSES} for a, rows in arms.items() if rows}
        res["paired"][w] = {f"{a}-rollup": la.paired(arms, a, "rollup", lambda r: r["session"] in S)
                            for a in ("prose", "stripped", "noop")}
    # primary: EARLY minus LATE (and minus MID) for each arm, and for the gap
    for later in ("late", "mid"):
        for arm in ("rollup", "prose", "noop", "stripped"):
            res["between"][f"{arm}_early_minus_{later}"] = between(W[later], W["early"], arm)
        res["between"][f"prose-rollup_early_minus_{later}"] = between(W[later], W["early"], "prose", "rollup")
    res["between"]["prose-rollup_late_minus_mid"] = between(W["mid"], W["late"], "prose", "rollup")
    res["unresolved_grades"] = sum(1 for r in lc.read_jsonl(CTRL / "grades.jsonl") if r["grade"] is None)
    res["noop_gate_pass"] = all(d["recall_pct"] <= 5.0 for w in W for a, d in res["descriptive"][w].items()
                                if a == "noop")
    (CTRL / "results.json").write_text(json.dumps(res, indent=1))

    print(f"sessions: {len(S)}")
    for w in W:
        print(f"\n== {w.upper()}")
        for a, d in res["descriptive"][w].items():
            print(f"  {a:9s} n={d['n']:3d} s={d['sessions']:2d} recall {d['recall_pct']}% {d['recall_wilson']} "
                  f"c/p/w/a {d['correct']}/{d['partial']}/{d['wrong']}/{d['absent']} "
                  f"w/att {d['wrong_of_attempted_pct']}% ({d['wrong']}/{d['attempted']})")
        for k, v in res["paired"][w].items():
            if v:
                print(f"  {k:16s} n={v['n']:3d} s={v['sessions']:2d} {v['a_pct']} vs {v['b_pct']} "
                      f"diff {v['diff_pts']:+.1f} {v['ci95']} p={v['p']} deff={v['deff_a_arm']}")
        print("  per class:", json.dumps(res["per_class"][w]))
    print("\nbetween windows:")
    for k, v in res["between"].items():
        if v:
            print(f"  {k:32s} s={v['sessions']} {v['b_pct']} vs {v['a_pct']} {v['b_minus_a_pts']:+.1f} {v['ci95']} p={v['p']}")
    print("unresolved:", res["unresolved_grades"], "noop gate:", res["noop_gate_pass"])


if __name__ == "__main__":
    main()
