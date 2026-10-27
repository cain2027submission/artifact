#!/usr/bin/env python3
"""Power simulation for Replication 2 (equivalence of no-digest vs output-stripped within +-5 points).
Free: no model calls. Pool = the 125 held-out pairs + the 196 Replication 1 pairs (321 paired questions,
123 sessions), recall = grade in correct/partially_correct. Each simulated Replication 2: draw sessions with
replacement from the pool, take up to k (2 or 3) random questions per drawn session until N questions; keep the
pooled discordance rate D and redraw the direction of each discordant pair so E[no-digest - stripped] = delta
(clustering of where discordance sits is retained). Analysis: session-clustered paired bootstrap 90% CI, TOST at +-5;
power = share of studies whose CI lies inside (-5, +5).
Two designs: `alone` (Replication 2 only) and `pooled_conditional` (the observed 321 Replication 1 + held-out
pairs held fixed, plus the simulated Replication 2, one bootstrap over all sessions).
"""
import json
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "paper-rollups"
OUT = DATA / "replication2" / "power_sim.json"
R = ("correct", "partially_correct")
SEED, SIMS, B_BOOT = 20261013, 1000, 600
NS = (172, 200, 250, 258, 300, 350, 400, 500, 600)
KS = (2, 3)
DELTAS = (0.0, -1.25, -1.5, -3.0, -4.0)


def load():
    pool = {}
    for f, k in (("rq3_heldout_no_digest_arm.jsonl", "nd"), ("rq3_heldout_stripped_arm_extended.jsonl", "st")):
        for l in (DATA / f).open():
            if l.strip():
                r = json.loads(l); d = pool.setdefault(("h", r["record_id"]), {"s": r["session_id"]})
                d[k] = int(r["grade"] in R)
    for l in (DATA / "replication" / "answers.jsonl").open():
        if l.strip():
            r = json.loads(l)
            if r["arm"] in ("nodigest", "stripped"):
                d = pool.setdefault(("r", r["question_id"]), {"s": r["session_id"]})
                d["nd" if r["arm"] == "nodigest" else "st"] = int(r["grade"] in R)
    by = {}
    for v in pool.values():
        by.setdefault(v["s"], []).append(v["nd"] - v["st"])
    return [np.array(v) for v in by.values()]


def ci90(cs, cn, rng):
    K = len(cs)
    idx = rng.integers(0, K, (B_BOOT, K))
    d = cs[idx].sum(1) / cn[idx].sum(1) * 100
    return np.percentile(d, 5), np.percentile(d, 95)


def study(sessions, N, k, delta, D, rng):
    pb = (1 + (delta / 100) / D) / 2
    cs, cn, tot = [], [], 0
    while tot < N:
        s = sessions[rng.integers(len(sessions))]
        take = s if len(s) <= k else s[rng.permutation(len(s))[:k]]
        take = take[: N - tot]
        x = np.where(take != 0, np.where(rng.random(len(take)) < pb, 1, -1), 0)
        cs.append(x.sum()); cn.append(len(x)); tot += len(x)
    return np.array(cs), np.array(cn)


def main():
    sessions = load()
    allv = np.concatenate(sessions)
    D = float((allv != 0).mean())
    obs_cs = np.array([s.sum() for s in sessions]); obs_cn = np.array([len(s) for s in sessions])
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "sims": SIMS, "bootstrap_B": B_BOOT, "margin_pts": 5, "pool_questions": int(len(allv)),
           "pool_sessions": len(sessions), "discordance_rate": D, "observed_pooled_diff_pts": float(allv.mean() * 100),
           "supply": "86 eligible sessions: 172 questions at 2 per session, 258 at 3 (supply.json)", "cells": []}
    for k in KS:
        for delta in DELTAS:
            for N in NS:
                a = p = 0
                for _ in range(SIMS):
                    cs, cn = study(sessions, N, k, delta, D, rng)
                    lo, hi = ci90(cs, cn, rng); a += (lo > -5) and (hi < 5)
                    lo, hi = ci90(np.concatenate([obs_cs, cs]), np.concatenate([obs_cn, cn]), rng); p += (lo > -5) and (hi < 5)
                c = {"per_session": k, "true_diff_pts": delta, "N": N, "power_alone": a / SIMS, "power_pooled_conditional": p / SIMS}
                res["cells"].append(c); print(c, flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
