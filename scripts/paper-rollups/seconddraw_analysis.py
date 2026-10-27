#!/usr/bin/env python3
"""Analysis exactly as data/paper-rollups/seconddraw-replication/analysis_plan.md (no model calls).
Draw 1 = stored Replication 1 grade, draw 2 = seconddraw answers.jsonl. Session-clustered
paired bootstrap B=10000 seed 20261014 percentile 95% CI."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[2] / "data" / "paper-rollups"
SD = R / "seconddraw-replication"
B, SEED = 10000, 20261014
REC = ("correct", "partially_correct")
ARMS = ("nodigest", "stripped")


def load(p):
    d = defaultdict(dict)
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
    ids = json.loads((SD / "sample_ids.json").read_text())
    d1, d2 = load(R / "replication" / "answers.jsonl"), load(SD / "answers.jsonl")
    qids = [q for q in ids if all(a in d2.get(q, {}) for a in ARMS)]
    excluded = [q for q in ids if q not in qids]
    n = len(qids)
    sess = sorted({d1[q]["nodigest"]["session_id"] for q in qids}); K = len(sess)
    si = {s: i for i, s in enumerate(sess)}
    sidx = np.array([si[d1[q]["nodigest"]["session_id"]] for q in qids])
    g1 = {a: [d1[q][a]["grade"] for q in qids] for a in ARMS}
    g2 = {a: [d2[q][a]["grade"] for q in qids] for a in ARMS}
    rec = lambda g: np.array([x in REC for x in g], float)
    r1 = {a: rec(g1[a]) for a in ARMS}; r2 = {a: rec(g2[a]) for a in ARMS}
    same_st = {a: (r1[a] == r2[a]).astype(float) for a in ARMS}
    same_gr = {a: np.array([x == y for x, y in zip(g1[a], g2[a])], float) for a in ARMS}
    # per-question vectors; stat = dict of per-question series, bootstrap by session
    series = {}
    for a in ARMS:
        series[f"{a}_d1"] = r1[a]; series[f"{a}_d2"] = r2[a]; series[f"{a}_change"] = r2[a] - r1[a]
        series[f"{a}_same_status"] = same_st[a]; series[f"{a}_same_grade"] = same_gr[a]
    series["armdiff_d1"] = r1["nodigest"] - r1["stripped"]
    series["armdiff_d2"] = r2["nodigest"] - r2["stripped"]
    series["armdiff_mean"] = (series["armdiff_d1"] + series["armdiff_d2"]) / 2
    series["armdiff_change"] = series["armdiff_d2"] - series["armdiff_d1"]
    M = np.zeros((K, len(series) + 1)); keys = list(series)
    for j, k in enumerate(keys): np.add.at(M[:, j], sidx, series[k])
    np.add.at(M[:, -1], sidx, 1)
    idx = np.random.default_rng(SEED).integers(0, K, (B, K))
    S = M[idx].sum(1)
    out = {"n_questions": n, "n_sessions": K, "excluded_question_ids": excluded, "bootstrap_B": B, "seed": SEED,
           "ci": "session-clustered paired bootstrap, percentile 95%", "per_arm": {}, "arm_difference_nodigest_minus_stripped": {}}
    def stat(k, pct=True):
        j = keys.index(k); pt = M[:, j].sum() / n * 100
        bs = S[:, j] / S[:, -1] * 100
        return {"point_pts": float(pt), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    for a in ARMS:
        out["per_arm"][a] = {
            "recall_draw1": stat(f"{a}_d1"), "recall_draw2": stat(f"{a}_d2"), "change_draw2_minus_draw1": stat(f"{a}_change"),
            "same_recovered_status": stat(f"{a}_same_status"), "same_4class_grade": stat(f"{a}_same_grade"),
            "kappa_recovered_status": kappa(r1[a], r2[a]),
            "grades_draw1": dict(Counter(g1[a])), "grades_draw2": dict(Counter(g2[a])),
            "recovered_transitions": {"both": int(((r1[a] == 1) & (r2[a] == 1)).sum()), "d1_only": int(((r1[a] == 1) & (r2[a] == 0)).sum()),
                                       "d2_only": int(((r1[a] == 0) & (r2[a] == 1)).sum()), "neither": int(((r1[a] == 0) & (r2[a] == 0)).sum())}}
    for k in ("armdiff_d1", "armdiff_d2", "armdiff_mean", "armdiff_change"):
        out["arm_difference_nodigest_minus_stripped"][k] = stat(k)
    ch = {a: out["per_arm"][a]["change_draw2_minus_draw1"]["point_pts"] for a in ARMS}
    a1, a2 = (out["arm_difference_nodigest_minus_stripped"][k]["point_pts"] for k in ("armdiff_d1", "armdiff_d2"))
    big = [a for a in ARMS if abs(ch[a]) > 3]
    flip = (a1 * a2 < 0)
    out["reading_rule"] = {"arms_with_abs_change_over_3pts": big, "arm_difference_sign_flip": bool(flip),
        "armdiff_draw1_pts": a1, "armdiff_draw2_pts": a2,
        "statement": ("answer-sampling noise comparable to the between-arm difference" if (big or flip) else
                      "single-draw caveat bounded on this subset only") + "; describes 60 questions, not a re-estimate of Replication 1"}
    (SD / "results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
