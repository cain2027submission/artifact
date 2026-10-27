#!/usr/bin/env python3
"""Rebuild the three held-out comparisons under a named grade field. No model calls.

  rq3_heldout_no_digest_paired.json        no-digest vs output-stripped, n=125 (counts, McNemar)
  rq3_heldout_no_digest_equivalence.json   same pairs, TOST at +-10 and +-5
  rq3_heldout_equivalence.json             full action record vs output-stripped, n=68

FIELD = `grade`, the current protocol (3-draw majority of the revised grader, regrade
2026-10-09). The three files were built from `grade_published` (the pre-regrade grade);
the same code run on that field is kept in each file under `stale_grade_published` and
the script checks it against the committed pre-rebuild numbers it reproduces.

Each equivalence block gives two 90% CIs on the recall difference (arm A minus
output-stripped): the original unclustered paired bootstrap (resample questions, B=20000,
seed 20261004) and a session-clustered one (resample sessions, same B and seed), because
the paper's method clusters by session. Equivalence at margin M holds iff the 90% CI lies
inside [-M, +M]. The clustered CI is the one to cite.

Usage: rq3_heldout_equivalence_rebuild.py
"""
import json
import random
from math import comb, sqrt
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "paper-rollups"
FIELD = "grade"
STALE = "grade_published"
B, SEED = 20000, 20261004
OK = ("correct", "partially_correct")


def load(name):
    return {json.loads(l)["record_id"]: json.loads(l) for l in (DATA / name).open() if l.strip()}


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def block(rows, f):
    n = len(rows)
    g = [r[f] for r in rows]
    out = {"n": n}
    for k in ("correct", "partially_correct", "wrong", "absent"):
        c = g.count(k)
        out[k] = {"n": c, "pct": round(100 * c / n, 1), "ci95": wilson(c, n)}
    rec = out["correct"]["n"] + out["partially_correct"]["n"]
    out["recoverable"] = {"n": rec, "pct": round(100 * rec / n, 1), "ci95": wilson(rec, n)}
    att = n - out["absent"]["n"]
    w = out["wrong"]["n"]
    out["hallucination_of_attempted"] = {"n_wrong": w, "n_attempted": att,
                                         "pct": round(100 * w / att, 1), "ci95": wilson(w, att)}
    return out


def mcnemar(a_ok, s_ok):
    b = sum(x and not y for x, y in zip(a_ok, s_ok))
    c = sum(y and not x for x, y in zip(a_ok, s_ok))
    n = b + c
    k = min(b, c)
    p = min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0) if n else 1.0
    both = sum(x and y for x, y in zip(a_ok, s_ok))
    return {"both": both, "no_digest_only": b, "stripped_only": c,
            "neither": len(a_ok) - both - b - c, "discordant": n, "p_two_sided": p}


def boot(a_ok, s_ok, sessions, clustered):
    rng = random.Random(SEED)
    n = len(a_ok)
    by = {}
    for i, s in enumerate(sessions):
        by.setdefault(s, []).append(i)
    keys = list(by)
    d = []
    for _ in range(B):
        if clustered:
            idx = [i for k in (rng.choice(keys) for _ in keys) for i in by[k]]
        else:
            idx = [rng.randrange(n) for _ in range(n)]
        d.append(sum(a_ok[i] for i in idx) / len(idx) - sum(s_ok[i] for i in idx) / len(idx))
    d.sort()

    def pct(p):
        k = p * (B - 1)
        lo = int(k)
        return d[lo] * (1 - (k - lo)) + d[lo + 1] * (k - lo) if lo + 1 < B else d[lo]

    return [pct(.05), pct(.95)], [pct(.025), pct(.975)]


def equivalence(a, s, ids, f):
    a_ok = [a[i][f] in OK for i in ids]
    s_ok = [s[i][f] in OK for i in ids]
    sess = [s[i]["session_id"] for i in ids]
    u90, u95 = boot(a_ok, s_ok, sess, False)
    c90, c95 = boot(a_ok, s_ok, sess, True)
    eq = lambda ci, m: -m <= ci[0] and ci[1] <= m
    return {
        "grade_field": f, "n_pairs": len(ids), "n_sessions": len(set(sess)),
        "arm_a_recall": sum(a_ok) / len(ids), "output_stripped_recall": sum(s_ok) / len(ids),
        "point_diff_a_minus_stripped": sum(a_ok) / len(ids) - sum(s_ok) / len(ids),
        "bootstrap_B": B, "seed": SEED,
        "unclustered_pair_bootstrap": {"ci90_diff": u90, "ci95_diff": u95,
                                       "equivalent_5pts": eq(u90, .05), "equivalent_10pts": eq(u90, .10)},
        "session_clustered_bootstrap": {"ci90_diff": c90, "ci95_diff": c95,
                                        "equivalent_5pts": eq(c90, .05), "equivalent_10pts": eq(c90, .10)},
        "mcnemar_a_vs_stripped": mcnemar(a_ok, s_ok),
    }


def write(name, new, stale):
    new = {**new, "grade_field": FIELD, "stale_grade_published": stale,
           "stale_note": "same code on the pre-2026-10-09 `grade_published` field; superseded"}
    (DATA / name).write_text(json.dumps(new, indent=2) + "\n")
    return new


def main():
    nd, st = load("rq3_heldout_no_digest_arm.jsonl"), load("rq3_heldout_stripped_arm_extended.jsonl")
    ids = sorted(set(nd) & set(st))
    assert len(ids) == 125
    old_paired = json.loads((DATA / "rq3_heldout_no_digest_paired.json").read_text())
    res = {}

    def paired(f):
        a_ok = [nd[i][f] in OK for i in ids]
        s_ok = [st[i][f] in OK for i in ids]
        return {"n_shared": 125, "pool_size": 125,
                "output_stripped": block([st[i] for i in ids], f),
                "action_record_no_digest": block([nd[i] for i in ids], f),
                "mcnemar": mcnemar(a_ok, s_ok)}

    stale_p = paired(STALE)
    res["paired"] = write("rq3_heldout_no_digest_paired.json",
                          {**paired(FIELD), "note": old_paired.get("note", "")}, stale_p)
    res["nd_equiv"] = write("rq3_heldout_no_digest_equivalence.json", equivalence(nd, st, ids, FIELD),
                            equivalence(nd, st, ids, STALE))
    ar, s68 = load("rq3_heldout_action_record_arm.jsonl"), load("rq3_heldout_stripped_arm.jsonl")
    ids68 = sorted(set(ar) & set(s68))
    assert len(ids68) == 68
    res["ar_equiv"] = write("rq3_heldout_equivalence.json", equivalence(ar, s68, ids68, FIELD),
                            equivalence(ar, s68, ids68, STALE))
    # reproduction check of the committed stale numbers
    chk = {"paired_recall_83.2_80.8": (stale_p["output_stripped"]["recoverable"]["pct"],
                                       stale_p["action_record_no_digest"]["recoverable"]["pct"]),
           "paired_mcnemar_p": stale_p["mcnemar"]["p_two_sided"]}
    print("stale reproduction:", chk)
    for k in ("nd_equiv", "ar_equiv"):
        r = res[k]
        print(k, "NEW recall", round(r["arm_a_recall"], 3), round(r["output_stripped_recall"], 3),
              "diff", round(r["point_diff_a_minus_stripped"], 3),
              "unclustered90", [round(x, 3) for x in r["unclustered_pair_bootstrap"]["ci90_diff"]],
              "clustered90", [round(x, 3) for x in r["session_clustered_bootstrap"]["ci90_diff"]],
              "eq10/5 clustered", r["session_clustered_bootstrap"]["equivalent_10pts"], r["session_clustered_bootstrap"]["equivalent_5pts"],
              "eq10/5 unclustered", r["unclustered_pair_bootstrap"]["equivalent_10pts"], r["unclustered_pair_bootstrap"]["equivalent_5pts"])
        s = r["stale_grade_published"]
        print(k, "STALE recall", round(s["arm_a_recall"], 3), round(s["output_stripped_recall"], 3),
              "diff", round(s["point_diff_a_minus_stripped"], 3),
              "unclustered90", [round(x, 3) for x in s["unclustered_pair_bootstrap"]["ci90_diff"]],
              "clustered90", [round(x, 3) for x in s["session_clustered_bootstrap"]["ci90_diff"]])
    print("paired new", res["paired"]["mcnemar"], res["paired"]["output_stripped"]["recoverable"]["pct"], res["paired"]["action_record_no_digest"]["recoverable"]["pct"])


if __name__ == "__main__":
    main()
