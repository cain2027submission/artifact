#!/usr/bin/env python3
"""Analysis for the rollup-from-stripped spike. No model calls."""
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_raw_transcript_arm import DATA, wilson  # noqa: E402

NEW = DATA / "rq3_rollup_from_stripped_spike_graded.jsonl"
GEN = DATA / "rq3_rollup_from_stripped_spike.jsonl"
OLD = DATA / "rq3_rollup_uniform.jsonl"


def rows(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def corrected_grade(r):
    up = r.get("answer", "").upper()
    if r.get("confidence") != "NOT_ANSWERABLE" and ("NOT ANSWERABLE" in up or "NOT_ANSWERABLE" in up):
        return "absent"
    return r["grade"]


def rec(g):
    return g in ("correct", "partially_correct")


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


new = {r["record_id"]: r for r in rows(NEW)}
old = {r["record_id"]: r for r in rows(OLD)}
ids = sorted(new)
n = len(ids)
print(f"n new {n}  n old {len(old)}  matched {len(set(ids) & set(old))}")

for label, fn in (("uncorrected", lambda r: r["grade"]), ("corrected", corrected_grade)):
    k = sum(rec(fn(new[i])) for i in ids)
    lo, hi = wilson(k, n)
    ko = sum(rec(fn(old[i])) for i in ids if i in old)
    lo2, hi2 = wilson(ko, n)
    print(f"{label}: new {k}/{n} = {100*k/n:.1f}% [{lo}, {hi}]   old(production) {ko}/{n} = {100*ko/n:.1f}% [{lo2}, {hi2}]")
    both = sum(rec(fn(new[i])) and rec(fn(old[i])) for i in ids)
    new_only = sum(rec(fn(new[i])) and not rec(fn(old[i])) for i in ids)
    old_only = sum(not rec(fn(new[i])) and rec(fn(old[i])) for i in ids)
    neither = n - both - new_only - old_only
    print(f"  both-right {both}  new-only {new_only}  old-only {old_only}  both-wrong {neither}  "
          f"exact McNemar p = {mcnemar(new_only, old_only):.4g}")

chars = [len(json.loads(l)["rollup_raw"]) for l in GEN.read_text().splitlines() if l.strip()]
for d in (2.7, 4.17):
    t = [c / d for c in chars]
    print(f"rollup tokens at chars/{d}: median {statistics.median(t):.0f}  mean {statistics.mean(t):.0f}  (n={len(t)} sessions)")
# per-question weighting (each question sees its session's rollup)
qc = [new[i]["rollup_chars"] for i in ids]
for d in (2.7, 4.17):
    t = [c / d for c in qc]
    print(f"  per-question: median {statistics.median(t):.0f}  mean {statistics.mean(t):.0f}")

ins = outs = 0
for l in GEN.read_text().splitlines():
    if l.strip():
        for u in json.loads(l)["usage"]:
            ins += u.get("prompt_tokens", 0)
            outs += u.get("completion_tokens", 0)
print(f"generation usage: prompt {ins:,}  completion {outs:,}")
