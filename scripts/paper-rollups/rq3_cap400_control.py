#!/usr/bin/env python3
"""CONTROL for rq3 experiment 2 (cap400): a second independent Sonnet-400 generation, a
length-constrained 'sonnet400trim' arm, a second answer draw on generation 1, and a re-grade
gate on 10 stored generation-1 grades.

Design: data/paper-rollups/cap400/CONTROL_DESIGN.md (committed before any spend).
Everything reuses rq3_cap400_run.py byte for byte (prompts, models, grader, analysis helpers);
the control's own cache lives in data/paper-rollups/cap400/control/, and the first run's files
in data/paper-rollups/cap400/ are only read, never written.

Modes (run in this order):  --gate  --gen2  --answer-gen2  --trim  --answer-trim  --draw2  --analyse
Run: python3 scripts/paper-rollups/rq3_cap400_control.py --gate ...
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import rq3_cap400_run as R  # noqa: E402

BASE = R.DATA / "cap400"
CTRL = BASE / "control"
CEIL_USD, CEIL_CALLS = 9.0, 700
TRIM_RESERVE_USD = 0.40          # kept back under the ceiling for retries and the analysis margin
ANS_USD_PER_Q, ANS_CALLS_PER_Q = 0.0075, 2.0   # answer + judge per question, measured in the first run

# The sonnet400trim length instruction (CONTROL_DESIGN.md states it verbatim). No {placeholders}.
TRIM_INSTR = ("\n\nLength: this is a hard limit. Your entire reply, the whole JSON object including its keys, quotes "
              "and punctuation, must stay under 1,500 characters (about 250 words, roughly 350 to 400 tokens). "
              "A reply of 1,500 characters or more is a failure, so count as you write and stop well before the "
              "limit. Write the intent, thinking and outcome fields as two or three sentences each. Be selective: "
              "keep the decisive facts, numbers, identifiers, files and dead ends, and drop the rest.")
R.SUMMARY_ARMS["sonnet400trim"] = R.SONNET
R.RETUNE["sonnet400trim"] = TRIM_INSTR
assert "{" not in TRIM_INSTR


def ctrl_store():
    s = R.Store(CTRL)
    base = R.Store(BASE)
    # generation 1 is read from the first run (for the draw-2 unit); never written to CTRL
    for k, v in base.gens.items():
        if k[0] == "sonnet400" and k[2] == 1:
            s.gens.setdefault(k, v)
    return s, base


def spend(store):
    _, tot = R.ledger_summary(store)
    return tot["usd"], tot["calls"]


def args_ns():
    return argparse.Namespace(questions=None, sessions=None, only=None)


def gate(store, base, qs):
    """Re-grade 10 stored sonnet400 g1 d1 answers with fresh grader draws (arm label
    'sonnet400-regrade' so no cached draw is reused). Same grade() as the first run."""
    qmap = {q["record_id"]: q for q in qs}
    rows = sorted((r for (a, _, g, d), r in base.ans.items() if a == "sonnet400" and g == 1 and d == 1),
                  key=lambda r: r["record_id"])
    pick = random.Random(R.SEED).sample(rows, 10)
    out = []
    for r in pick:
        g, draws, fails = R.grade(store, qmap[r["record_id"]], "sonnet400-regrade", 1, 1,
                                  r["answer"], r["confidence"], r["raw_reply"])
        out.append({"record_id": r["record_id"], "stored": r["grade"], "regrade": g, "draws": draws,
                    "failed_calls": fails, "same_grade": g == r["grade"],
                    "same_recoverable": (g in R.REC) == (r["grade"] in R.REC)})
        print(out[-1])
    (CTRL / "regrade_gate.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out))


def trim_sessions(store, base, qs):
    """Pre-stated shrink rule: ascending stripped size, include a session while measured spend
    + projected trim cost (first run's per-session Sonnet generation cost + per-question answer
    cost) stays <= CEIL_USD - TRIM_RESERVE_USD and calls <= CEIL_CALLS."""
    usd, calls = spend(store)
    nq = {}
    for q in qs:
        nq[q["session_id"]] = nq.get(q["session_id"], 0) + 1
    g1 = sorted((v for k, v in base.gens.items() if k[0] == "sonnet400" and k[2] == 1),
                key=lambda r: r["stripped_tokens"])
    keep = []
    for r in g1:
        u = r["usage"]
        c = u["prompt_tokens"] * 3e-6 + 400 * 15e-6 + nq[r["session_id"]] * ANS_USD_PER_Q
        k = 1 + nq[r["session_id"]] * ANS_CALLS_PER_Q
        if usd + c > CEIL_USD - TRIM_RESERVE_USD or calls + k > CEIL_CALLS:
            break
        usd += c; calls += k
        keep.append(r["session_id"])
    print(f"trim sessions {len(keep)}/{len(g1)}, projected ${usd:.2f}, {calls:.0f} calls")
    return keep


def boot_pooled(sa, sb):
    """R.boot_diff's session-clustered bootstrap (same B, seed, resampling order) for
    non-binary per-record values; McNemar does not apply."""
    shared = sorted(set(sa) & set(sb))
    by = {}
    for rid in shared:
        by.setdefault(sa[rid][0], []).append((sa[rid][1], sb[rid][1]))
    ks = sorted(by)
    rng = random.Random(R.SEED)
    ds = []
    for _ in range(R.B):
        x = y = n = 0
        for k in (rng.choice(ks) for _ in ks):
            for a_, b_ in by[k]:
                x += a_; y += b_; n += 1
        ds.append(100 * (x - y) / n)
    ds.sort()
    point = 100 * sum(sa[r][1] - sb[r][1] for r in shared) / len(shared)
    p = 2 * min((sum(d <= 0 for d in ds) + 1) / (R.B + 1), (sum(d >= 0 for d in ds) + 1) / (R.B + 1))
    return {"n": len(shared), "n_sessions": len(ks), "diff_pts": round(point, 1),
            "cluster95_pts": [round(ds[int(0.025 * R.B)], 1), round(ds[int(0.975 * R.B) - 1], 1)],
            "boot_p": round(min(1.0, p), 4)}


def analyse(store, base):
    """Merge first-run answers (read-only) with control answers, then the pre-stated analysis."""
    ans = dict(base.ans)
    ans.update(store.ans)
    M = R.Store.__new__(R.Store)
    M.ans = ans
    gens = dict(base.gens)
    gens.update({k: v for k, v in store.gens.items()})
    M.gens = gens
    S = {}
    units = sorted({(a, g, d) for (a, _, g, d) in ans})
    for u in units:
        S[f"{u[0]}:g{u[1]}:d{u[2]}"] = R.series(M, *u)
    res = {"arms": {}, "generation_tokens": {}, "paired": {}, "spread": {}}
    for u in units:
        if u[0] in ("sonnet400", "sonnet400trim", "head1500", "rollup", "prose2k", "haiku400"):
            res["arms"][f"{u[0]}:g{u[1]}:d{u[2]}"] = R.arm_stats(M, *u)
    for arm, g in sorted({(a, g) for (a, _, g) in gens if a.startswith("sonnet400")}):
        rows = [v for (a, _, gg), v in gens.items() if a == arm and gg == g]
        t = [r["summary_cl100k"] for r in rows]
        c = [r["summary_chars"] for r in rows]
        res["generation_tokens"][f"{arm}:g{g}"] = {
            "n_sessions": len(rows), "median": statistics.median(t), "mean": round(statistics.mean(t)),
            "min_max": [min(t), max(t)], "median_chars": statistics.median(c), "max_chars": max(c),
            "n_over_1500_chars": sum(x >= 1500 for x in c),
            "pct_miss_of_400": round(100 * (statistics.median(t) - R.TARGET) / R.TARGET, 1)}
    # pooled: per-record mean of the two generations (0, 0.5, 1); stacked n for recall
    g1, g2 = S.get("sonnet400:g1:d1", {}), S.get("sonnet400:g2:d1", {})
    pooled = {r: (g1[r][0], (g1[r][1] + g2[r][1]) / 2) for r in set(g1) & set(g2)}
    if pooled:
        st = [x for r in pooled for x in (g1[r][1], g2[r][1])]
        lo, hi = R.wilson(sum(st), len(st))
        res["arms"]["sonnet400:pooled g1+g2 (stacked)"] = {"n": len(st), "recoverable": sum(st),
                                                         "pct": round(100 * sum(st) / len(st), 1),
                                                         "wilson95_naive": [lo, hi]}
        S["sonnet400:pooled"] = pooled
    for a_ in ("sonnet400:g1:d1", "sonnet400:g2:d1", "sonnet400:pooled", "sonnet400:g1:d2",
               "sonnet400trim:g1:d1"):
        for b_ in ("head1500:g0:d1", "rollup:g0:d1", "prose2k:g0:d1", "haiku400:g1:d1"):
            if a_ in S and b_ in S:
                res["paired"][f"{a_} - {b_}"] = (boot_pooled(S[a_], S[b_]) if a_ == "sonnet400:pooled"
                                                 else R.boot_diff(S[a_], S[b_]))
    for a_, b_ in (("sonnet400trim:g1:d1", "sonnet400:g1:d1"), ("sonnet400trim:g1:d1", "sonnet400:g2:d1")):
        if a_ in S and b_ in S:
            res["paired"][f"{a_} - {b_}"] = R.boot_diff(S[a_], S[b_])
    for lab, x, y in (("generation (g1 vs g2)", "sonnet400:g1:d1", "sonnet400:g2:d1"),
                      ("answer draw (g1 d1 vs d2)", "sonnet400:g1:d1", "sonnet400:g1:d2")):
        if x in S and y in S:
            sh = set(S[x]) & set(S[y])
            res["spread"][f"sonnet400 {lab}"] = {
                "n": len(sh), "pct_x": round(100 * sum(S[x][r][1] for r in sh) / len(sh), 1),
                "pct_y": round(100 * sum(S[y][r][1] for r in sh) / len(sh), 1),
                "x_only": sum(S[x][r][1] and not S[y][r][1] for r in sh),
                "y_only": sum(S[y][r][1] and not S[x][r][1] for r in sh),
                "paired": R.boot_diff(S[x], S[y])}
    gate_f = CTRL / "regrade_gate.jsonl"
    if gate_f.exists():
        gr = R.jl(gate_f)
        res["gate_regrade10"] = {"n": len(gr), "same_grade": sum(r["same_grade"] for r in gr),
                                 "same_recoverable": sum(r["same_recoverable"] for r in gr),
                                 "failed_calls": sum(r["failed_calls"] for r in gr)}
    res["failures"] = {f: len(R.jl(CTRL / f)) for f in
                       ("generation_failures.jsonl", "answer_failures.jsonl", "grade_failures.jsonl")}
    res["ledger"], res["ledger_total"] = R.ledger_summary(store)
    (CTRL / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    # graded records of the control, one file
    with (CTRL / "graded.jsonl").open("w") as f:
        for r in sorted(store.ans.values(), key=lambda r: (r["arm"], r["gen"], r["draw"], r["record_id"])):
            f.write(json.dumps(r) + "\n")
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    for f in ("gate", "gen2", "answer-gen2", "trim", "answer-trim", "draw2", "analyse", "spend"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    store, base = ctrl_store()
    qs, sess = R.load_inputs(args_ns())
    if a.spend:
        print(spend(store))
    if a.gate:
        gate(store, base, qs)
    if a.gen2:
        R.generate(store, qs, sess, ["sonnet400"], [2], a.workers)
    if a.answer_gen2:
        R.answer_all(store, qs, sess, [("sonnet400", 2, 1)], a.workers)
    if a.trim:
        t2 = [v["summary_cl100k"] for (arm, _, g), v in store.gens.items() if arm == "sonnet400" and g == 2]
        med = statistics.median(t2)
        print(f"gen2 median {med} cl100k, trigger is > {R.TARGET * 1.25:.0f}")
        if len(t2) < 25 or med <= R.TARGET * 1.25:
            print("trim arm not triggered")
            a.trim = a.answer_trim = False
    if a.trim or a.answer_trim:
        keep = set(trim_sessions(store, base, qs)) if a.trim else \
            {sid for (arm, sid, g) in store.gens if arm == "sonnet400trim"}
        tq = [q for q in qs if q["session_id"] in keep]
        if a.trim:
            R.generate(store, tq, sess, ["sonnet400trim"], [1], a.workers)
        if a.answer_trim:
            R.answer_all(store, tq, sess, [("sonnet400trim", 1, 1)], a.workers)
    if a.draw2:
        usd, calls = spend(store)
        left_q = int(min((CEIL_USD - usd) / ANS_USD_PER_Q, (CEIL_CALLS - calls) / ANS_CALLS_PER_Q))
        if left_q < len(qs):
            print(f"draw2 skipped: budget covers {left_q} of {len(qs)} questions (${usd:.2f}, {calls} calls spent)")
        else:
            R.answer_all(store, qs, sess, [("sonnet400", 1, 2)], a.workers)
    if a.analyse:
        analyse(store, base)


if __name__ == "__main__":
    main()
