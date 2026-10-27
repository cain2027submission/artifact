#!/usr/bin/env python3
"""Exploratory post hoc pooled estimate, per data/paper-rollups/pooled-estimate/analysis_plan.md.
No model calls, stdlib only. no-digest minus output-stripped, held-out (125) + replication (196).
Writes data/paper-rollups/pooled-estimate/results.json."""
import json, math, random, sys
from collections import defaultdict
from pathlib import Path

D = Path(__file__).resolve().parents[2] / "data" / "paper-rollups"
OUT = D / "pooled-estimate" / "results.json"
B, SEED = 10000, 20261013
REC = ("correct", "partially_correct")
Z = {90: 1.6448536269514722, 95: 1.959963984540054}


def jl(p):
    return [json.loads(l) for l in p.open() if l.strip()]


def vec(a, s):
    # n, rec_a, rec_s, wrong_a, wrong_s, att_a, att_s, abs_a, abs_s
    return [1, a in REC, s in REC, a == "wrong", s == "wrong", a != "absent", s != "absent", a == "absent", s == "absent"]


def pct(sorted_d, p):
    n = len(sorted_d)
    k = p * (n - 1)
    lo = int(k)
    return sorted_d[lo] * (1 - (k - lo)) + sorted_d[lo + 1] * (k - lo) if lo + 1 < n else sorted_d[lo]


def stats(T):
    n = T[0]
    rec = 100 * (T[1] - T[2]) / n
    wga = 100 * (T[3] / T[5] - T[4] / T[6]) if T[5] and T[6] else None
    ab = 100 * (T[7] - T[8]) / n
    return rec, wga, ab


def point(T):
    n = T[0]
    return {"n_questions": int(n),
            "recall_nodigest_pct": 100 * T[1] / n, "recall_stripped_pct": 100 * T[2] / n,
            "recall_diff_pts": stats(T)[0],
            "wga_nodigest_pct": 100 * T[3] / T[5], "wga_stripped_pct": 100 * T[4] / T[6], "wga_diff_pts": stats(T)[1],
            "absent_nodigest_pct": 100 * T[7] / n, "absent_stripped_pct": 100 * T[8] / n, "absent_diff_pts": stats(T)[2]}


def add(T, v):
    for i in range(9):
        T[i] += v[i]


def boot(strata, seed, B=B):
    """strata: list of lists of per-session vectors. Resample sessions within each stratum
    (a single stratum = ordinary session-clustered bootstrap). returns three sorted lists + drops."""
    rng = random.Random(seed)
    outs = ([], [], [])
    drops = 0
    for _ in range(B):
        T = [0.0] * 9
        for st in strata:
            m = len(st)
            for _ in range(m):
                add(T, st[rng.randrange(m)])
        r, w, a = stats(T)
        if w is None:
            drops += 1
            continue
        outs[0].append(r); outs[1].append(w); outs[2].append(a)
    for o in outs:
        o.sort()
    return outs, drops


def ci(sorted_d, level):
    t = (100 - level) / 2 / 100
    return [pct(sorted_d, t), pct(sorted_d, 1 - t)]


def inside(c, m):
    return -m < c[0] and c[1] < m


def sd(x):
    mu = sum(x) / len(x)
    return math.sqrt(sum((v - mu) ** 2 for v in x) / (len(x) - 1))


def boot_summary(outs, drops):
    res = {"resamples_dropped_no_attempts": drops}
    for name, o in zip(("recall", "wga", "absent"), outs):
        res[name] = {"ci90": ci(o, 90), "ci95": ci(o, 95), "bootstrap_sd": sd(o)}
        res[name]["ci90_width"] = res[name]["ci90"][1] - res[name]["ci90"][0]
        res[name]["inside_pm10_at_90"] = inside(res[name]["ci90"], 10)
        res[name]["inside_pm5_at_90"] = inside(res[name]["ci90"], 5)
    return res


def meta(d, se):
    """d, se lists over sets. fixed and DerSimonian-Laird random effects, Q, I2."""
    w = [1 / s ** 2 for s in se]
    fe = sum(wi * di for wi, di in zip(w, d)) / sum(w)
    se_fe = math.sqrt(1 / sum(w))
    Q = sum(wi * (di - fe) ** 2 for wi, di in zip(w, d))
    k = len(d)
    C = sum(w) - sum(wi ** 2 for wi in w) / sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / C)
    wr = [1 / (s ** 2 + tau2) for s in se]
    re = sum(wi * di for wi, di in zip(wr, d)) / sum(wr)
    se_re = math.sqrt(1 / sum(wr))
    p = math.erfc(math.sqrt(Q / 2)) if k == 2 else None  # chi2 df=1 survival
    mk = lambda m, s: {"estimate": m, "se": s, "ci90": [m - Z[90] * s, m + Z[90] * s], "ci95": [m - Z[95] * s, m + Z[95] * s],
                       "inside_pm10_at_90": inside([m - Z[90] * s, m + Z[90] * s], 10),
                       "inside_pm5_at_90": inside([m - Z[90] * s, m + Z[90] * s], 5)}
    return {"fixed": mk(fe, se_fe), "random_DL": dict(mk(re, se_re), tau2=tau2),
            "Q": Q, "Q_df": k - 1, "Q_p": p, "I2_pct": max(0.0, (Q - (k - 1)) / Q * 100) if Q > 0 else 0.0}


def main():
    ha = {r["record_id"]: r for r in jl(D / "rq3_heldout_no_digest_arm.jsonl")}
    hs = {r["record_id"]: r for r in jl(D / "rq3_heldout_stripped_arm_extended.jsonl")}
    assert set(ha) == set(hs), "held-out arms do not share record ids"
    held = [(q, ha[q]["session_prefix"], ha[q]["session_id"], vec(ha[q]["grade"], hs[q]["grade"])) for q in sorted(ha)]

    rows = jl(D / "replication" / "answers.jsonl")
    by = defaultdict(dict)
    for r in rows:
        by[r["question_id"]][r["arm"]] = r
    rq = sorted(q for q, v in by.items() if "nodigest" in v and "stripped" in v)
    rep_all = [(q, by[q]["nodigest"]["session_prefix"], by[q]["nodigest"]["session_id"],
                vec(by[q]["nodigest"]["grade"], by[q]["stripped"]["grade"])) for q in rq]

    # overlap check
    hq, hsess = {x[0] for x in held}, {x[1] for x in held}
    rq_ids, rsess = {x[0] for x in rep_all}, {x[1] for x in rep_all}
    ov_q, ov_s = hq & rq_ids, hsess & rsess
    rep = [x for x in rep_all if x[0] not in ov_q and x[1] not in ov_s]
    overlap = {"heldout_questions": len(hq), "heldout_sessions": len(hsess),
               "replication_questions": len(rq_ids), "replication_sessions": len(rsess),
               "shared_question_ids": len(ov_q), "shared_session_prefixes": len(ov_s),
               "replication_questions_dropped": len(rep_all) - len(rep)}

    def per_session(rows_):
        g = defaultdict(lambda: [0.0] * 9)
        for _, s, _, v in rows_:
            add(g[s], v)
        return list(g.values())

    H, R = per_session(held), per_session(rep)
    TH, TR = [sum(c) for c in zip(*H)], [sum(c) for c in zip(*R)]
    TP = [a + b for a, b in zip(TH, TR)]

    # per-set clustered bootstrap, own method for the reproduction check
    # held-out committed: B=20000 seed 20261004 random.Random, resample sessions keyed by session_id
    def committed_style(rows_, seed, Bc):
        rng = random.Random(seed)
        g = defaultdict(list)
        for i, r in enumerate(rows_):
            g[r[2]].append(i)
        keys = list(g)
        d = []
        for _ in range(Bc):
            idx = [i for k in (rng.choice(keys) for _ in keys) for i in g[k]]
            d.append(sum(rows_[i][3][1] for i in idx) / len(idx) - sum(rows_[i][3][2] for i in idx) / len(idx))
        d.sort()
        return [100 * pct(d, .05), 100 * pct(d, .95)]

    h_ci = committed_style(held, 20261004, 20000)
    # replication committed used numpy's generator (seed 20261014, B=10000), not reproducible bit-for-bit in
    # stdlib; recompute with stdlib and compare within 0.6 point
    r_ci = committed_style(rep_all, 20261014, 10000)
    chk = {"heldout": {"diff": point(TH)["recall_diff_pts"], "ci90_recomputed_same_method": h_ci,
                       "committed": {"diff": -4.0, "ci90": [-9.6, 0.8]}},
           "replication": {"diff": point(TR if not ov_q and not ov_s else TR)["recall_diff_pts"],
                           "ci90_recomputed_stdlib": r_ci,
                           "committed": {"diff": 0.5, "ci90": [-4.1, 5.1]},
                           "note": "committed replication ci used numpy.random.default_rng, so the stdlib recomputation matches only within bootstrap noise"}}
    ok = (round(point(TH)["recall_diff_pts"], 1) == -4.0 and round(point(TR)["recall_diff_pts"], 1) == 0.5
          and [round(x, 1) for x in h_ci] == [-9.6, 0.8] and all(abs(a - b) <= 0.6 for a, b in zip(r_ci, [-4.1, 5.1])))
    chk["reproduces"] = ok
    chk["one_line"] = ("held-out %+.1f [%+.1f, %+.1f] and replication %+.1f [%+.1f, %+.1f] vs committed -4.0 [-9.6, +0.8] and +0.5 [-4.1, +5.1]: %s"
                       % (point(TH)["recall_diff_pts"], h_ci[0], h_ci[1], point(TR)["recall_diff_pts"], r_ci[0], r_ci[1],
                          "REPRODUCED" if ok else "DID NOT REPRODUCE"))
    print(chk["one_line"])
    if not ok or ov_q or ov_s:
        if not ok:
            OUT.write_text(json.dumps({"overlap": overlap, "reproduction": chk}, indent=1))
            print("STOP: reproduction failed"); sys.exit(1)

    res = {"seed": SEED, "B": B, "plan": "data/paper-rollups/pooled-estimate/analysis_plan.md",
           "status": "exploratory, post hoc, not pre-registered; +-5 was not a pre-registered target for this pooling",
           "overlap_check": overlap, "reproduction_check": chk}

    sets = {}
    for name, S, T, off in (("heldout", H, TH, 1), ("replication", R, TR, 2)):
        outs, dr = boot([S], SEED + off)
        sets[name] = dict(point(T), n_sessions=len(S), clustered_bootstrap=boot_summary(outs, dr))
    res["per_set"] = sets

    o1, d1 = boot([H + R], SEED)
    o3, d3 = boot([H, R], SEED + 3)
    pooled = dict(point(TP), n_sessions=len(H) + len(R),
                  pooled_clustered_bootstrap=boot_summary(o1, d1),
                  stratified_bootstrap=boot_summary(o3, d3))
    ms = {}
    for i, name in enumerate(("recall", "wga", "absent")):
        d = [sets["heldout"][{"recall": "recall_diff_pts", "wga": "wga_diff_pts", "absent": "absent_diff_pts"}[name]],
             sets["replication"][{"recall": "recall_diff_pts", "wga": "wga_diff_pts", "absent": "absent_diff_pts"}[name]]]
        se = [sets["heldout"]["clustered_bootstrap"][name]["bootstrap_sd"], sets["replication"]["clustered_bootstrap"][name]["bootstrap_sd"]]
        ms[name] = dict(meta(d, se), set_differences=d, set_se=se)
    pooled["meta_analysis"] = ms
    res["pooled"] = pooled

    w = lambda c: c["ci90"][1] - c["ci90"][0]
    pc = pooled["pooled_clustered_bootstrap"]["recall"]
    res["narrowness_recall_ci90_width_pts"] = {
        "heldout": sets["heldout"]["clustered_bootstrap"]["recall"]["ci90_width"],
        "replication": sets["replication"]["clustered_bootstrap"]["recall"]["ci90_width"],
        "pooled_clustered": pc["ci90_width"],
        "pooled_stratified": pooled["stratified_bootstrap"]["recall"]["ci90_width"],
        "pooled_fixed_effect": w(ms["recall"]["fixed"]), "pooled_random_effects": w(ms["recall"]["random_DL"])}
    res["narrowness_recall_ci90_width_pts"]["reduction_vs_replication_pct_pooled_clustered"] = \
        100 * (1 - pc["ci90_width"] / sets["replication"]["clustered_bootstrap"]["recall"]["ci90_width"])
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
