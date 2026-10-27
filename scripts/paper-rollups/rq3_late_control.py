#!/usr/bin/env python3
"""EARLY-window control for the later-window experiment (late-window/CONTROL_DESIGN.md).

Runs rq3_late_window.py's own gen / answer / grade code, unchanged, on the SAME
sessions as MID and LATE, with one difference: the window is EARLY, the first
30,000 characters (offset 0) of the same harness-stripped generator text, with
the same EXCERPT_NOTE filled in for that position. Everything is written under
late-window/control/, so the MID/LATE files are only read, never written.

Stages (resumable, every reply cached on disk before use):
  gen    [--first K]       5 EARLY questions per session
  answer [--first K] [--arms rollup,prose,noop]
  grade  [--first K]
  regrade                  gate: re-grade 10 stored answers with fresh draws
  cost

Hard stop: $12 or 1,200 calls on the control ledger, checked before every call.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq3_late_common as lc  # noqa: E402
import rq3_late_window as lw  # noqa: E402

CTRL = lc.OUTDIR / "control"
CTRL.mkdir(exist_ok=True)
lw.OUT = CTRL
lw.QFILE = CTRL / "questions.jsonl"
lw.AFILE = CTRL / "answers.jsonl"
lw.GFILE = CTRL / "grades.jsonl"
lw.LEDGER = CTRL / "ledger.jsonl"
lw.GEN_FAIL = CTRL / "gen_failures.jsonl"
lw.ANS_ERR = CTRL / "answer_errors.jsonl"
lw.DRAWS = CTRL / "grade_draws.jsonl"
lw.WHERE = dict(lw.WHERE, early="beginning (the first 30,000 characters)")

MAX_USD = 12.0
MAX_CALLS = 1200


class Budget(Exception):
    pass


def spent():
    rows = lc.read_jsonl(lw.LEDGER)
    return sum(v["usd"] for v in lw.spend(rows).values()), len(rows)


def check():
    usd, n = spent()
    if usd >= MAX_USD or n >= MAX_CALLS:
        raise Budget(f"hard stop: ${usd:.2f}, {n} calls")


# answer stage already consults total_usd() before each call; point it at the
# control ledger and make the call ceiling trip it too.
lw.HARD_STOP_USD = MAX_USD


def _total_usd():
    usd, n = spent()
    return float("inf") if n >= MAX_CALLS else usd


lw.total_usd = _total_usd

_gen_call, _grade_call = lw.gen_call, lw.grade_call


def gen_call(prompt, model, max_tokens=2048):
    check()
    return _gen_call(prompt, model, max_tokens)


def grade_call(prompt):
    check()
    return _grade_call(prompt)


lw.gen_call, lw.grade_call = gen_call, grade_call

_manifest = lw.manifest


def manifest(first=0, only=None):
    """The late-window manifest, in its own order, with EARLY at offset 0."""
    return [dict(s, early_offset=0) for s in _manifest(first, only)]


lw.manifest = manifest


def cmd_regrade(a):
    """Gate: 10 stored control answers that needed a grader call (seeded), re-graded
    from scratch into a separate draw cache, compared with their stored grade."""
    rows = [r for r in lc.read_jsonl(lw.GFILE) if not r["rule_absent"] and r["grade"]]
    rows.sort(key=lambda r: (r["qid"], r["arm"]))
    sample = random.Random(20261011).sample(rows, min(10, len(rows)))
    lw.DRAWS = CTRL / "regrade_draws.jsonl"
    cache = {r["key"]: r for r in lc.read_jsonl(lw.DRAWS)}
    out = []
    for r in sample:
        q = {"qid": r["qid"], "question": r["question"], "ground_truth": r["ground_truth"]}
        ans = {"arm": r["arm"], "answer": r["answer"], "confidence": r["confidence"]}
        g = lw.grade_record(q, ans, cache)
        out.append({"qid": r["qid"], "arm": r["arm"], "stored": r["grade"], "regraded": g["grade"],
                    "draws": g["draws"], "match": g["grade"] == r["grade"]})
    rec = lambda g: g in ("correct", "partially_correct")
    res = {"n": len(out), "exact_match": sum(o["match"] for o in out),
           "recoverable_match": sum(rec(o["stored"]) == rec(o["regraded"]) for o in out),
           "seed": 20261011, "rows": out}
    (CTRL / "regrade_gate.json").write_text(json.dumps(res, indent=1))
    print(f"regrade gate: exact {res['exact_match']}/{res['n']}, recoverable {res['recoverable_match']}/{res['n']}")


def cmd_cost(a):
    s = lw.spend()
    for k, v in s.items():
        print(f"{k:7s} calls {v['calls']:5d} failed {v['failed']:4d} in {v['in']:>11,} out {v['out']:>9,} ${v['usd']:.2f}")
    usd, n = spent()
    print(f"total {n} calls ${usd:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--only", type=lambda s: s.split(","), default=None)
    ap.add_argument("--arms", type=lambda s: s.split(","), default=["rollup", "prose", "noop"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    a.windows = ["early"]
    try:
        {"gen": lw.cmd_gen, "answer": lw.cmd_answer, "grade": lw.cmd_grade,
         "regrade": cmd_regrade, "cost": cmd_cost}[a.stage](a)
    except Budget as e:
        print(e)
        sys.exit(2)


if __name__ == "__main__":
    main()
