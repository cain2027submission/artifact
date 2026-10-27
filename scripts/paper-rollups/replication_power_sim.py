#!/usr/bin/env python3
"""Power simulation for the no-digest vs output-stripped replication (pre-registration:
tasks/2026-10-13 a fresh-session replication decides ...). Free: no model calls.

Pool: the 125 held-out paired questions (rq3_heldout_no_digest_arm.jsonl and
rq3_heldout_stripped_arm_extended.jsonl, `grade` field, recall = correct + partially_correct),
keyed by session. Each simulated study: draw sessions with replacement, take up to 3
random questions per drawn session (the replication's cap), until N questions. The
true recall difference is imposed by keeping the pool's discordance rate D (11/125) and
re-drawing the direction of each discordant pair so that E[no-digest - stripped] = delta;
clustering of where discordance occurs is retained. Analysis per study: session-clustered
paired bootstrap 90% CI (B_BOOT resamples), TOST at +-10 points; power = share of studies
whose CI lies inside (-10, +10). Also reports power at +-5 for information.
"""
import json
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "paper-rollups"
OUT = DATA / "replication" / "power_sim.json"
R = ("correct", "partially_correct")
SEED, SIMS, B_BOOT = 20261013, 2000, 1000
NS = (150, 200, 250, 300, 400)
DELTAS = (-4.0, -5.0)


def load():
    pool = {}
    for f, k in (("rq3_heldout_no_digest_arm.jsonl", "nd"), ("rq3_heldout_stripped_arm_extended.jsonl", "st")):
        for l in (DATA / f).open():
            if l.strip():
                r = json.loads(l)
                d = pool.setdefault(r["record_id"], {"s": r["session_id"]})
                d[k] = int(r["grade"] in R)
    by = {}
    for v in pool.values():
        by.setdefault(v["s"], []).append(v["nd"] - v["st"])
    return [np.array(v) for v in by.values()]


def ci90(cs, cn, rng):
    K = len(cs)
    idx = rng.integers(0, K, (B_BOOT, K))
    d = cs[idx].sum(1) / cn[idx].sum(1) * 100
    return np.percentile(d, 5), np.percentile(d, 95)


def study(sessions, N, delta, D, rng):
    pb = (1 + (delta / 100) / D) / 2          # P(no-digest wins | discordant)
    cs, cn, tot = [], [], 0
    while tot < N:
        s = sessions[rng.integers(len(sessions))]
        take = s if len(s) <= 3 else s[rng.permutation(len(s))[:3]]
        take = take[: N - tot]
        disc = take != 0
        x = np.where(disc, np.where(rng.random(len(take)) < pb, 1, -1), 0)
        cs.append(x.sum()); cn.append(len(x)); tot += len(x)
    return np.array(cs), np.array(cn)


def main():
    sessions = load()
    allv = np.concatenate(sessions)
    D = float((allv != 0).mean())
    obs = float(allv.mean() * 100)
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "sims": SIMS, "bootstrap_B": B_BOOT, "pool_questions": int(len(allv)),
           "pool_sessions": len(sessions), "discordance_rate": D, "observed_diff_pts": obs,
           "margin_pts": 10, "cells": []}
    for delta in DELTAS:
        for N in NS:
            in10 = in5 = 0
            for _ in range(SIMS):
                cs, cn = study(sessions, N, delta, D, rng)
                lo, hi = ci90(cs, cn, rng)
                in10 += (lo > -10) and (hi < 10)
                in5 += (lo > -5) and (hi < 5)
            c = {"true_diff_pts": delta, "N": N, "power_tost10": in10 / SIMS, "power_tost5": in5 / SIMS}
            res["cells"].append(c); print(c, flush=True)
    ok = [c["N"] for c in res["cells"] if c["true_diff_pts"] == -4.0 and c["power_tost10"] >= 0.8]
    res["smallest_N_reaching_80pct_at_minus4"] = min(ok) if ok else None
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
