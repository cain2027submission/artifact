#!/usr/bin/env python3
"""Pre-registered analysis of the replication (no model calls). Reads
data/paper-rollups/replication/{questions.jsonl,answers.jsonl,cost_log.jsonl,calls.jsonl},
writes results.json. Session-clustered paired bootstrap, B=10000, seed 20261014 (the
same resamples serve every statistic). Recall = correct + partially_correct under `grade`.
Primary: 90% CI of recall(nodigest)-recall(stripped) inside +-10 (and +-5, reported).
Gating secondary: wrong-given-attempted difference, one-sided 95% upper bound < 5 points.
Reported: absent-rate difference. Verdict: pass iff primary at +-10 passes and secondary holds."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parents[2] / "data" / "paper-rollups" / "replication"
B, SEED = 10000, 20261014
REC = ("correct", "partially_correct")


def load():
    ans = [json.loads(l) for l in (D / "answers.jsonl").open() if l.strip()]
    by = defaultdict(dict)
    for r in ans: by[r["question_id"]][r["arm"]] = r
    return ans, by


def main():
    ans, by = load()
    qids = [q for q, v in by.items() if "nodigest" in v and "stripped" in v]
    qids.sort()
    sess = sorted({by[q]["nodigest"]["session_id"] for q in qids})
    sidx = {s: i for i, s in enumerate(sess)}
    K = len(sess)
    g = lambda arm: [by[q][arm]["grade"] for q in qids]
    gn, gs = g("nodigest"), g("stripped")
    n = len(qids)
    # per-session counts: n, rec_nd, rec_st, wrong_nd, wrong_st, att_nd, att_st, abs_nd, abs_st
    M = np.zeros((K, 9))
    for q, a, b in zip(qids, gn, gs):
        i = sidx[by[q]["nodigest"]["session_id"]]
        M[i] += [1, a in REC, b in REC, a == "wrong", b == "wrong", a != "absent", b != "absent", a == "absent", b == "absent"]
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, K, (B, K))
    S = M[idx].sum(1)   # B x 9
    def stats(S):
        nn = S[:, 0]
        rec = (S[:, 1] - S[:, 2]) / nn * 100
        with np.errstate(invalid="ignore", divide="ignore"):
            wga = (S[:, 3] / S[:, 5] - S[:, 4] / S[:, 6]) * 100
        ab = (S[:, 7] - S[:, 8]) / nn * 100
        return rec, wga, ab
    rec, wga, ab = stats(S)
    T = M.sum(0)
    point = {"recall_nodigest_pct": 100 * T[1] / T[0], "recall_stripped_pct": 100 * T[2] / T[0],
             "recall_diff_pts": 100 * (T[1] - T[2]) / T[0],
             "wga_nodigest_pct": 100 * T[3] / T[5], "wga_stripped_pct": 100 * T[4] / T[6],
             "wga_diff_pts": 100 * (T[3] / T[5] - T[4] / T[6]),
             "absent_nodigest_pct": 100 * T[7] / T[0], "absent_stripped_pct": 100 * T[8] / T[0],
             "absent_diff_pts": 100 * (T[7] - T[8]) / T[0]}
    ci90 = lambda x: [float(np.nanpercentile(x, 5)), float(np.nanpercentile(x, 95))]
    ci95 = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
    r90 = ci90(rec)
    # unclustered sensitivity (question-level paired bootstrap)
    d = np.array([(a in REC) - (b in REC) for a, b in zip(gn, gs)])
    ru = np.random.default_rng(SEED + 1).integers(0, n, (B, n))
    unc = ci90(d[ru].mean(1) * 100)
    wga_up95 = float(np.nanpercentile(wga, 95))
    prim10 = r90[0] > -10 and r90[1] < 10
    prim5 = r90[0] > -5 and r90[1] < 5
    sec = wga_up95 < 5
    mc = Counter((a in REC, b in REC) for a, b in zip(gn, gs))
    ans_ns = [r for r in ans if r["arm"] != "zero"]
    zero = [r for r in ans if r["arm"] == "zero"]
    pilot_q = [json.loads(l)["question_id"] for l in (D / "questions.jsonl").open()][:10]
    pil = [r for r in ans_ns if r["question_id"] in pilot_q]
    pilot = {"questions": 10, "answers": len(pil), "api_errors": 20 - len(pil),
             "cost_usd": round(sum(r["cost"] for r in pil), 3),
             "recall_nodigest_pct": 100 * sum(r["grade"] in REC for r in pil if r["arm"] == "nodigest") / 10,
             "recall_stripped_pct": 100 * sum(r["grade"] in REC for r in pil if r["arm"] == "stripped") / 10,
             "abort_threshold_usd": 1.5 * 70 * 10 / 200}
    calls = [json.loads(l) for l in (D / "calls.jsonl").open()]
    by_stage = defaultdict(lambda: [0, 0.0])
    for c in calls:
        s = c["stage"].split("_")[0]; by_stage[s][0] += 1; by_stage[s][1] += c["usd"]
    res = {
        "n_questions": n, "n_sessions": K, "frozen_n": 200, "seed": SEED, "bootstrap_B": B,
        "grade_protocol": ans_ns[0]["grade_protocol"], "grade_field": "grade (majority3)",
        "grade_counts": {"nodigest": dict(Counter(gn)), "stripped": dict(Counter(gs))},
        "unresolved_grades": sum(r["grade"] == "unresolved" for r in ans_ns),
        "point": point, "discordance": {"both": mc[(True, True)], "nodigest_only": mc[(True, False)],
                                        "stripped_only": mc[(False, True)], "neither": mc[(False, False)]},
        "primary": {"recall_diff_ci90_session_clustered": r90, "recall_diff_ci90_unclustered_sensitivity": unc,
                    "equivalent_pm10": bool(prim10), "equivalent_pm5": bool(prim5)},
        "secondary_gating": {"wga_diff_ci90_two_sided": ci90(wga), "wga_diff_one_sided_95_upper": wga_up95,
                             "margin_pts": 5, "holds": bool(sec)},
        "absent_reported_not_gating": {"diff_pts": point["absent_diff_pts"], "ci95": ci95(ab)},
        "zero_context": {"n": len(zero), "recall_pct": 100 * sum(r["grade"] in REC for r in zero) / len(zero), "gate_max_pct": 5,
                         "passed": sum(r["grade"] in REC for r in zero) / len(zero) <= 0.05},
        "pilot": pilot,
        "spend": {"total_usd": round(sum(c["usd"] for c in calls), 4), "cap_usd": 70,
                  "by_stage_calls_usd": {k: [v[0], round(v[1], 3)] for k, v in by_stage.items()}},
        "verdict": "pass" if (prim10 and sec) else ("fail (inconclusive counts as fail)" if not prim10 and r90[0] > -10 else "fail"),
    }
    # verdict wording
    if prim10 and sec: res["verdict"] = "pass"
    elif not prim10: res["verdict"] = "fail" if (r90[0] <= -10 or r90[1] >= 10) else "fail"
    else: res["verdict"] = "fail (gating secondary)"
    (D / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
