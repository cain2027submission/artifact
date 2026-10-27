#!/usr/bin/env python3
"""Analysis exactly as data/paper-rollups/seconddraw-remainder/analysis_plan.md (no model calls).
Questions: those of the 60-question second draw plus the 136 remainder that have both arms in draw 2.
Draw 1 = stored Replication 1 grade; draw 2 = seconddraw-replication + seconddraw-remainder answers.
Session-clustered paired bootstrap, B=10000, seed 20261015, percentile 90% and 95% CIs."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[2] / "data" / "paper-rollups"
SD1, SD2 = R / "seconddraw-replication", R / "seconddraw-remainder"
B, SEED = 10000, 20261015
REC = ("correct", "partially_correct")
ARMS = ("nodigest", "stripped")


def load(*ps):
    d = defaultdict(dict)
    for p in ps:
        for l in p.open():
            if l.strip():
                r = json.loads(l); d[r["question_id"]][r["arm"]] = r
    return d


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean(); pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if pe == 1 else float((po - pe) / (1 - pe))


def main():
    ids = json.loads((SD1 / "sample_ids.json").read_text()) + json.loads((SD2 / "remainder_ids.json").read_text())
    d1, d2 = load(R / "replication" / "answers.jsonl"), load(SD1 / "answers.jsonl", SD2 / "answers.jsonl")
    qids = [q for q in ids if all(a in d2.get(q, {}) for a in ARMS)]
    excluded = [q for q in ids if q not in qids]
    n = len(qids)
    sess = sorted({d1[q]["nodigest"]["session_id"] for q in qids}); K = len(sess)
    si = {s: i for i, s in enumerate(sess)}
    sidx = np.array([si[d1[q]["nodigest"]["session_id"]] for q in qids])
    g1 = {a: [d1[q][a]["grade"] for q in qids] for a in ARMS}
    g2 = {a: [d2[q][a]["grade"] for q in qids] for a in ARMS}
    rec = lambda g: np.array([x in REC for x in g], float)
    ind = lambda g, c: np.array([x == c for x in g], float)
    r1 = {a: rec(g1[a]) for a in ARMS}; r2 = {a: rec(g2[a]) for a in ARMS}
    series = {}
    for a in ARMS:
        series[f"{a}_d1"] = r1[a]; series[f"{a}_d2"] = r2[a]; series[f"{a}_avg"] = (r1[a] + r2[a]) / 2
        series[f"{a}_change"] = r2[a] - r1[a]
        series[f"{a}_same_status"] = (r1[a] == r2[a]).astype(float)
        series[f"{a}_same_grade"] = np.array([x == y for x, y in zip(g1[a], g2[a])], float)
        for dn, g in (("d1", g1[a]), ("d2", g2[a])):
            series[f"{a}_wrong_{dn}"] = ind(g, "wrong"); series[f"{a}_absent_{dn}"] = ind(g, "absent")
            series[f"{a}_attempted_{dn}"] = 1 - ind(g, "absent")
    series["armdiff_d1"] = r1["nodigest"] - r1["stripped"]
    series["armdiff_d2"] = r2["nodigest"] - r2["stripped"]
    series["armdiff_avg"] = (series["armdiff_d1"] + series["armdiff_d2"]) / 2
    series["armdiff_change"] = series["armdiff_d2"] - series["armdiff_d1"]
    keys = list(series)
    M = np.zeros((K, len(keys) + 1))
    for j, k in enumerate(keys): np.add.at(M[:, j], sidx, series[k])
    np.add.at(M[:, -1], sidx, 1)
    idx = np.random.default_rng(SEED).integers(0, K, (B, K))
    S = M[idx].sum(1)
    def stat(k):
        j = keys.index(k); pt = M[:, j].sum() / n * 100; bs = S[:, j] / S[:, -1] * 100
        return {"point_pts": float(pt), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "ci90": [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]}
    def ratio(num, den):   # wrong given attempted
        jn, jd = keys.index(num), keys.index(den)
        pt = M[:, jn].sum() / M[:, jd].sum() * 100
        bs = S[:, jn] / np.where(S[:, jd] == 0, np.nan, S[:, jd]) * 100
        return {"point_pts": float(pt), "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]}
    out = {"n_questions": n, "n_sessions": K, "n_from_first_60": len([q for q in qids if q in set(json.loads((SD1/'sample_ids.json').read_text()))]),
           "excluded_question_ids": excluded, "bootstrap_B": B, "seed": SEED,
           "ci": "session-clustered paired bootstrap, percentile 90% and 95%", "per_arm": {}, "arm_difference_nodigest_minus_stripped": {}}
    for a in ARMS:
        out["per_arm"][a] = {
            "recall_draw1": stat(f"{a}_d1"), "recall_draw2": stat(f"{a}_d2"), "recall_draw_averaged": stat(f"{a}_avg"),
            "change_draw2_minus_draw1": stat(f"{a}_change"),
            "same_recovered_status": stat(f"{a}_same_status"), "same_4class_grade": stat(f"{a}_same_grade"),
            "kappa_recovered_status": kappa(r1[a], r2[a]),
            "wrong_given_attempted_draw1": ratio(f"{a}_wrong_d1", f"{a}_attempted_d1"),
            "wrong_given_attempted_draw2": ratio(f"{a}_wrong_d2", f"{a}_attempted_d2"),
            "absent_draw1": stat(f"{a}_absent_d1"), "absent_draw2": stat(f"{a}_absent_d2"),
            "grades_draw1": dict(Counter(g1[a])), "grades_draw2": dict(Counter(g2[a])),
            "recovered_transitions": {"both": int(((r1[a] == 1) & (r2[a] == 1)).sum()), "d1_only": int(((r1[a] == 1) & (r2[a] == 0)).sum()),
                                       "d2_only": int(((r1[a] == 0) & (r2[a] == 1)).sum()), "neither": int(((r1[a] == 0) & (r2[a] == 0)).sum())}}
    for k in ("armdiff_d1", "armdiff_d2", "armdiff_avg", "armdiff_change"):
        out["arm_difference_nodigest_minus_stripped"][k] = stat(k)
    D = out["arm_difference_nodigest_minus_stripped"]
    lo, hi = D["armdiff_avg"]["ci90"]
    flip = D["armdiff_d1"]["point_pts"] * D["armdiff_d2"]["point_pts"] < 0
    out["equivalence_descriptive_not_preregistered"] = {
        "averaged_diff_90ci": [lo, hi], "inside_pm10": bool(lo > -10 and hi < 10), "inside_pm5": bool(lo > -5 and hi < 5)}
    out["reading_rule"] = {"sign_flip_draw1_vs_draw2": bool(flip),
        "statement": ("consistent with the pre-registered +-10 conclusion" if (lo > -10 and hi < 10) else
                      "the +-10 conclusion is sensitive to answer-draw noise (averaged 90% interval crosses a margin)")}
    (SD2 / "results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
