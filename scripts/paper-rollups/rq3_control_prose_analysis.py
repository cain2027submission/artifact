#!/usr/bin/env python3
"""Tables for the control / length-matched prose / held-back-225 jobs. No model calls.
Wilson and exact McNemar as in rq3_rollup_from_stripped_spike_analysis.py."""
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_raw_transcript_arm import DATA, QUESTIONS, SUMMARY, load_records, record_id, wilson  # noqa: E402


def rows(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()] if Path(p).exists() else []


def rec(g):
    return g in ("correct", "partially_correct")


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def unc(r):
    return r["grade"]


def cor(r):
    conf = r.get("confidence") or r.get("rating") or r.get("rating")
    up = (r.get("answer") or r.get("judge_answer") or "").upper()
    if conf != "NOT_ANSWERABLE" and ("NOT ANSWERABLE" in up or "NOT_ANSWERABLE" in up):
        return "absent"
    return r["grade"]


def stat(d, fn, ids=None):
    ids = ids if ids is not None else sorted(d)
    k = sum(rec(fn(d[i])) for i in ids)
    n = len(ids)
    lo, hi = wilson(k, n)
    return k, n, 100 * k / n if n else 0, lo, hi


def paired(a, b, fn, ids):
    ids = [i for i in ids if i in a and i in b]
    ao = sum(rec(fn(a[i])) and not rec(fn(b[i])) for i in ids)
    bo = sum(rec(fn(b[i])) and not rec(fn(a[i])) for i in ids)
    return ao, bo, mcnemar(ao, bo), len(ids)


def main():
    prod = {r["record_id"]: r for r in rows(DATA / "rq3_rollup_uniform.jsonl")}
    ids124 = sorted(prod)
    arms = {}
    for name in ("prod_pipeline_control", "prose_matched_per_record", "prose_matched_median",
                 "prose_only_heldback225"):
        arms[name] = {r["record_id"]: r for r in rows(DATA / f"rq3_arm_{name}_graded.jsonl")}
    ctrl = arms["prod_pipeline_control"]
    print(f"{'arm':28s} {'n':>4s} {'unc %  [CI]':>24s} {'corr %  [CI]':>24s} {'med tok':>8s}  p vs prod (b/c)   p vs control (b/c)")
    ku, n, pu, lo, hi = stat(prod, unc, ids124)
    kc, _, pc, lo2, hi2 = stat(prod, cor, ids124)
    print(f"{'production_rollup (uniform)':28s} {n:4d} {pu:5.1f} [{lo},{hi}]  {pc:5.1f} [{lo2},{hi2}]")
    out = {}
    for name in ("prod_pipeline_control", "prose_matched_per_record", "prose_matched_median"):
        d = arms[name]
        if not d:
            continue
        ids = sorted(d)
        ku, n, pu, lo, hi = stat(d, unc)
        kc, _, pc, lo2, hi2 = stat(d, cor)
        med = statistics.median(d[i]["artifact_cl100k_tokens"] for i in ids)
        a, b, p, m = paired(d, prod, unc, ids)
        ac, bc, pcn, _ = paired(d, prod, cor, ids)
        s = f"{name:28s} {n:4d} {pu:5.1f} [{lo},{hi}]  {pc:5.1f} [{lo2},{hi2}] {med:8.0f}  unc p={p:.3g} ({a}/{b}) cor p={pcn:.3g} ({ac}/{bc})"
        if name != "prod_pipeline_control" and ctrl:
            a2, b2, p2, _ = paired(d, ctrl, unc, ids)
            a3, b3, p3, _ = paired(d, ctrl, cor, ids)
            s += f"  vs ctrl unc p={p2:.3g} ({a2}/{b2}) cor p={p3:.3g} ({a3}/{b3})"
        print(s)
    # production artifact tokens as graded
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    t = [len(enc.encode(Path(prod[i]["rollup"]).read_text(errors="replace"), disallowed_special=())) for i in ids124]
    print(f"production note cl100k tokens per question: median {statistics.median(t)} mean {statistics.mean(t):.0f}")

    # ---- held-back 225 ----
    h = arms["prose_only_heldback225"]
    if h:
        qs = {record_id(r): r for r in load_records(QUESTIONS)}
        g = set(json.loads(SUMMARY.read_text())["graded_record_ids"])
        ids225 = [i for i in qs if i not in g]
        hid = sorted(h)
        roll225 = {i: qs[i] for i in hid}  # published rollup grade/answer live in the questions file
        print(f"\nheld-back: prose graded {len(hid)} of {len(ids225)}")
        for lab, fn in (("uncorrected", unc), ("corrected", cor)):
            k, n, p, lo, hi = stat(h, fn)
            kr, _, pr, lo2, hi2 = stat(roll225, fn)
            a, b, pm, _ = paired(h, roll225, fn, hid)
            print(f"  225 {lab}: prose {k}/{n}={p:.1f}% [{lo},{hi}]  rollup(published) {kr}/{n}={pr:.1f}% [{lo2},{hi2}]  gap {p-pr:+.1f} pts  prose-only {a} rollup-only {b} McNemar p={pm:.3g}")
        # frozen 124 for comparison
        po = {r["record_id"]: r for r in rows(DATA / "rq3_prose_only_arm.jsonl")}
        pids = [i for i in ids124 if i in po]
        for lab, fn in (("uncorrected", unc), ("corrected", cor)):
            k, n, p, lo, hi = stat(po, fn, pids)
            kr, _, pr, lo2, hi2 = stat(prod, fn, pids)
            a, b, pm, _ = paired(po, prod, fn, pids)
            print(f"  124 {lab}: prose {k}/{n}={p:.1f}% [{lo},{hi}]  rollup(uniform) {kr}/{n}={pr:.1f}% [{lo2},{hi2}]  gap {p-pr:+.1f} pts  prose-only {a} rollup-only {b} McNemar p={pm:.3g}")
        print("  median prose tokens (225):", statistics.median(h[i]["artifact_cl100k_tokens"] for i in hid))
        # published rollup on 124 (mixed protocol) for reference
        q124 = {i: qs[i] for i in ids124 if i in qs}
        k, n, p, lo, hi = stat(q124, unc)
        print(f"  published (questions file) rollup on the 124: {k}/{n}={p:.1f}%")
    L = DATA / "rq3_control_prose_spend.json"
    if L.exists():
        print("\nspend:", json.dumps(json.loads(L.read_text())))


if __name__ == "__main__":
    main()
