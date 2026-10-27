#!/usr/bin/env python3
"""Session-clustered bootstrap for the RQ3 retention experiment. No model calls.

The paper's Wilson intervals and McNemar / exact-binomial tests treat the 124
questions as independent, but they come from 25 sessions (the raw arm's 89 from
18), several per session. This resamples SESSIONS with replacement, keeping all
of a session's questions, so the interval and p carry the within-session
correlation.

Definitions are arm_rates.py's, imported not re-implemented: recoverable =
correct + partially_correct over n; records joined on record_id; clusters are
session_id. There is one grading stage: the per-record `grade` is the 3-draw
majority of the revised grader (regrade-2026-10-09), so there is no separate
confidence-corrected series.

Every figure is first reproduced against arm_rates.py's counts; a mismatch
aborts.

Bootstrap p: two-sided, 2*min(P(d*<=0), P(d*>=0)) with +1 smoothing, so the
floor is 2/(B+1); it is the percentile-inversion p, not a null-centred test.
TOST: equivalence at margin m iff the 90% percentile cluster CI of the paired
difference lies inside (-m, +m).
Design effect: ANOVA ICC of the 0/1 recoverable indicator (Donner unequal
cluster size m0), deff = 1 + (mbar - 1) * max(icc, 0), n_eff = n / deff. Also
reported: the bootstrap deff = clustered variance / binomial variance.

Run: python3 scripts/paper-rollups/cluster_bootstrap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm_rates as ar  # noqa: E402

DATA = ar.DATA
OUT = DATA / "cluster_bootstrap.json"
B = 10_000
SEED = 20261008
REC = ("correct", "partially_correct")


def load(name, budget=None, arm=None):
    rows = {}
    for line in (DATA / name).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if budget is not None and r.get("budget") != budget:
            continue
        if arm is not None and r.get("arm") != arm:
            continue
        rows[r["record_id"]] = r
    return rows


def series(rows):
    """record_id -> (session_id, 0/1 recoverable)."""
    return {rid: (r["session_id"], int(r["grade"] in REC)) for rid, r in rows.items()}


ARM_FILES = {
    "rollup": ("rq3_rollup_uniform.jsonl", None),
    "output-stripped": ("rq3_questions_stripped_arm.jsonl", None),
    "raw": ("rq3_questions_raw_arm.jsonl", None),
    "action record": ("rq3_action_record_arm.jsonl", None),
    "prose-only": ("rq3_prose_only_arm.jsonl", None),
    "call-only": ("rq3_tool_inputs_only_arm.jsonl", None),
    "dialogue record": ("rq3_verbatim_arm.jsonl", None),
    "answer-only": ("rq3_answer_only_arm.jsonl", None),
    "prompts-only": ("rq3_prompts_only_arm.jsonl", None),
    "zero context": ("rq3_zero_context_arm.jsonl", None),
    # length-matched control: each record's prose-only context cut to its own
    # production rollup's cl100k length (median 933 vs the rollup's 932); the
    # file the paper's matched-length 56.5% and exact McNemar p = 1.0 come from
    # (rq3_control_prose_analysis.py, "prose_matched_per_record").
    "prose matched": ("rq3_arm_prose_matched_per_record_graded.jsonl", None),
    # Sonnet 5 rollups written from the output-stripped transcript (rq3_sonnet_rollup_124.py)
    "long sonnet": ("rq3_sonnet_rollup_124_long_graded.jsonl", None),
    "short sonnet": ("rq3_sonnet_rollup_124_short_graded.jsonl", None),
    "matched sonnet": ("rq3_sonnet_rollup_124_matched_graded.jsonl", None),
}
RB_ARMS = {"rollup summary": "rollup_hookshape", "output-stripped chunk": "stripped_chunk",
           "rollup pointer": "rollup_pointer", "noop": "noop"}


def build():
    S = {}  # name -> {record_id: (session, y)}
    for label, (f, bud) in ARM_FILES.items():
        S[label] = series(load(f, bud))
    for label, key in RB_ARMS.items():
        S["rb:" + label] = series(load("rq3_read_budget_arms.jsonl", arm=key))
    return S


def reproduce(S):
    """Abort unless every series reproduces arm_rates.py's recoverable count."""
    chk = {k: ar.uncorrected(k) for k in (
        "rollup", "output-stripped", "raw", "action record", "prose-only", "call-only",
        "dialogue record", "answer-only", "prompts-only", "zero context")}
    chk.update({"prose matched": ar.any_rate("prose cut to rollup length"),
                "long sonnet": ar.any_rate("sonnet long"),
                "short sonnet": ar.any_rate("sonnet short"),
                "matched sonnet": ar.any_rate("sonnet matched")})
    rep = {}
    for k, rate in chk.items():
        n, rec = len(S[k]), sum(y for _, y in S[k].values())
        ok = (n == rate.n and rec == rate.recoverable_n)
        rep[k] = {"n": n, "recoverable": rec, "pct": round(100 * rec / n, 1),
                  "arm_rates_n": rate.n, "arm_rates_recoverable": rate.recoverable_n, "ok": ok}
        if not ok:
            sys.exit(f"MISMATCH {k}: {n}/{rec} vs arm_rates {rate.n}/{rate.recoverable_n}")
    for label, key in RB_ARMS.items():
        s_ = S["rb:" + label]
        n, rec = len(s_), sum(y for _, y in s_.values())
        rate = ar.read_budget(key)
        rep["rb:" + label] = {"n": n, "recoverable": rec, "pct": round(100 * rec / n, 1),
                              "arm_rates_recoverable": rate.recoverable_n,
                              "ok": rec == rate.recoverable_n and n == rate.n}
        if not rep["rb:" + label]["ok"]:
            sys.exit(f"MISMATCH rb:{label}: {rec} vs arm_rates {rate.recoverable_n}")
    return rep


def pct(x):
    return round(100 * float(x), 2)


def ci(a, lo=2.5, hi=97.5):
    return [pct(np.percentile(a, lo)), pct(np.percentile(a, hi))]


class Boot:
    def __init__(self, S, sessions):
        self.sessions = sessions
        self.k = len(sessions)
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, self.k, size=(B, self.k))
        self.W = np.zeros((B, self.k))
        for j in range(self.k):  # multiplicity of each session per resample
            self.W[:, j] = (idx == j).sum(axis=1)
        self.S = S

    def agg(self, ids_by_arm_series, ids):
        """Per-session (successes, count) over the record id set `ids`."""
        pos = {s: i for i, s in enumerate(self.sessions)}
        y = np.zeros(self.k)
        c = np.zeros(self.k)
        for rid in ids:
            sess, v = ids_by_arm_series[rid]
            y[pos[sess]] += v
            c[pos[sess]] += 1
        return y, c

    def rate(self, name):
        s = self.S[name]
        y, c = self.agg(s, s.keys())
        num, den = self.W @ y, self.W @ c
        return y, c, num / den

    def diff(self, a, b, ids=None):
        sa, sb = self.S[a], self.S[b]
        shared = sorted(set(sa) & set(sb)) if ids is None else ids
        ya, c = self.agg(sa, shared)
        yb, _ = self.agg(sb, shared)
        num_a, num_b, den = self.W @ ya, self.W @ yb, self.W @ c
        d = (num_a - num_b) / den
        point = (ya.sum() - yb.sum()) / c.sum()
        # discordant pairs for the record
        disc_a = sum(1 for r in shared if sa[r][1] == 1 and sb[r][1] == 0)
        disc_b = sum(1 for r in shared if sa[r][1] == 0 and sb[r][1] == 1)
        return shared, point, d, (disc_a, disc_b)


def boot_p(d):
    le = (np.sum(d <= 0) + 1) / (len(d) + 1)
    ge = (np.sum(d >= 0) + 1) / (len(d) + 1)
    return float(min(1.0, 2 * min(le, ge)))


def icc_deff(y, c):
    m = c > 0
    y, c = y[m], c[m]
    N, k = c.sum(), len(c)
    p = y.sum() / N
    ssb = np.sum(y ** 2 / c) - N * p ** 2
    ssw = y.sum() - np.sum(y ** 2 / c)
    msb, msw = ssb / (k - 1), ssw / (N - k)
    if msb + msw == 0:  # no variance (all 0 or all 1): ICC undefined, no clustering to correct
        return {"n": int(N), "sessions": int(k), "mean_cluster_size": round(float(N / k), 2),
                "icc_anova": None, "design_effect": None, "n_effective": None}
    m0 = (N - np.sum(c ** 2) / N) / (k - 1)
    icc = (msb - msw) / (msb + (m0 - 1) * msw)
    mbar = N / k
    deff = 1 + (mbar - 1) * max(icc, 0.0)
    return {"n": int(N), "sessions": int(k), "mean_cluster_size": round(float(mbar), 2),
            "icc_anova": round(float(icc), 4), "design_effect": round(float(deff), 3),
            "n_effective": round(float(N / deff), 1)}


def main():
    S = build()
    repro = reproduce(S)
    sessions = sorted({s for ser in S.values() for s, _ in ser.values()})
    bt = Boot(S, sessions)

    arms = {}
    for name in S:
        y, c, r = bt.rate(name)
        p = y.sum() / c.sum()
        boot_var = float(np.var(r, ddof=1))
        bin_var = p * (1 - p) / c.sum()
        e = icc_deff(y, c)
        e["bootstrap_design_effect"] = round(boot_var / bin_var, 3) if bin_var > 0 else None
        k_n, n = int(y.sum()), int(c.sum())
        lo, hi = ar.wilson(k_n, n)
        arms[name] = {"n": n, "recoverable": k_n, "pct": round(100 * p, 1),
                      "wilson95_pct": [lo, hi], "cluster95_pct": ci(r),
                      "design": e}

    pairs_spec = [
        ("output-stripped - rollup (124)", "output-stripped", "rollup"),
        ("output-stripped - raw (89)", "output-stripped", "raw"),
        ("action record - output-stripped (124)", "action record", "output-stripped"),
        ("prose-only - call-only (123 shared)", "prose-only", "call-only"),
        ("prose matched - rollup (124)", "prose matched", "rollup"),
        ("read-budget chunk - rollup summary (89)", "rb:output-stripped chunk", "rb:rollup summary"),
        ("long sonnet - rollup (124)", "long sonnet", "rollup"),
        ("long sonnet - output-stripped (124)", "long sonnet", "output-stripped"),
        ("long sonnet - prose-only (124)", "long sonnet", "prose-only"),
        ("prose-only - long sonnet (124)", "prose-only", "long sonnet"),
        ("short sonnet - rollup (124)", "short sonnet", "rollup"),
        ("matched sonnet - rollup (124)", "matched sonnet", "rollup"),
        ("matched sonnet - prose matched (124)", "matched sonnet", "prose matched"),
        ("matched sonnet - output-stripped (124)", "matched sonnet", "output-stripped"),
        ("long sonnet - matched sonnet (124)", "long sonnet", "matched sonnet"),
        ("short sonnet - long sonnet (124)", "short sonnet", "long sonnet"),
    ]
    pairs = {}
    boots = {}
    for label, a, b in pairs_spec:
        shared, point, d, disc = bt.diff(a, b)
        boots[label] = d
        pairs[label] = {"a": a, "b": b, "n_shared": len(shared),
                        "diff_pts": round(100 * point, 3),
                        "cluster95_pts": ci(d), "cluster90_pts": ci(d, 5, 95),
                        "cluster_boot_p_two_sided": round(boot_p(d), 5),
                        "p_floor_2_over_B_plus_1": round(2 / (B + 1), 6),
                        "discordant_a_only_b_only": list(disc)}

    from scipy.stats import binomtest
    for label, v in pairs.items():
        a_only, b_only = v["discordant_a_only_b_only"]
        v["naive_exact_binomial_p"] = (float(binomtest(a_only, a_only + b_only, 0.5).pvalue)
                                      if a_only + b_only else 1.0)

    tost = {}
    for tag, label in (("output-stripped - raw (89)", "output-stripped - raw (89)"),):
        d = boots[label]
        lo_, hi_ = float(np.percentile(d, 5)) * 100, float(np.percentile(d, 95)) * 100
        tost = {"diff_pts": pairs[label]["diff_pts"], "cluster90_pts": [round(lo_, 3), round(hi_, 3)],
                "equivalent_pm5": bool(lo_ > -5 and hi_ < 5),
                "equivalent_pm10": bool(lo_ > -10 and hi_ < 10)}

    # Matched length: prose cut to the rollup's length against the rollup. A tie
    # in point estimate is not equivalence; TOST at +-10 and +-5 on the 90%
    # session-clustered CI.
    d = boots["prose matched - rollup (124)"]
    lo_, hi_ = float(np.percentile(d, 5)) * 100, float(np.percentile(d, 95)) * 100
    tost_matched = {"diff_pts": pairs["prose matched - rollup (124)"]["diff_pts"],
                    "cluster95_pts": pairs["prose matched - rollup (124)"]["cluster95_pts"],
                    "cluster90_pts": [round(lo_, 3), round(hi_, 3)],
                    "equivalent_pm5": bool(lo_ > -5 and hi_ < 5),
                    "equivalent_pm10": bool(lo_ > -10 and hi_ < 10)}

    out = {
        "what_this_is": "session-clustered bootstrap (resample sessions, keep all their questions) over the RQ3 retention arms; no model calls",
        "B": B, "seed": SEED, "n_sessions_all": len(sessions),
        "sessions_in_124_set": len({s for s, _ in S["output-stripped"].values()}),
        "sessions_in_raw_89_set": len({s for s, _ in S["raw"].values()}),
        "p_method": "2*min(P(d*<=0),P(d*>=0)), +1 smoothed, percentile inversion",
        "reproduction_check": repro,
        "arms": arms, "paired_differences": pairs, "tost_output_stripped_vs_raw": tost,
        "tost_prose_matched_vs_rollup": tost_matched,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT}")
    for k, v in arms.items():
        print(f"{k:34s} n={v['n']:3d} {v['pct']:5.1f}  wilson {v['wilson95_pct']}  cluster {v['cluster95_pct']}"
              f"  deff {v['design']['design_effect']} neff {v['design']['n_effective']}")
    for k, v in pairs.items():
        print(f"{k:60s} {v['diff_pts']:+6.2f} {v['cluster95_pts']} p={v['cluster_boot_p_two_sided']}"
              f" naive={v['naive_exact_binomial_p']:.3g} disc={v['discordant_a_only_b_only']}")
    print(json.dumps(tost))
    print("matched:", json.dumps(tost_matched))


if __name__ == "__main__":
    main()
