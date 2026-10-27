#!/usr/bin/env python3
"""Is the frozen 124-record question set representative of the 349 available?

Why this exists. A torn JSONL write (32,343 NUL bytes) stopped the freezing
script's stream decode partway, so every retention figure in the paper rests on
a 124-record *prefix* of 349 graded records. Threats to Validity used to end
"the direction of any resulting bias is unestablished", which is the kind of
sentence a reviewer converts into "re-run it on all 349".

It cannot be fully re-run without new model calls: only the rollup arm carries
grades for all 349. Every comparison arm -- output-stripped, prose-only,
call-only, the action record, the dialogue record, answer-only, prompts-only,
zero context -- exists at 124 records over 25 sessions. Rebuilding them at 349
means answering and grading 8 arms x 225 questions, and the raw arm's median
artifact is 691 KB, which mostly will not fit a context window.

But the rollup arm alone is enough to bound the artifact, because it is the arm
the paper's central claim is measured *against*. If the prefix flatters the
rollup, the published gap is conservative. That is what this measures, and it
needs no model call: the grades are already on disk.

Two things this gets right that a first attempt did not:

1. It verifies the prefix really is a prefix, rather than assuming it. The
   frozen set is identified by (session_id, question) pairs taken from the
   stripped arm's file, and the script asserts the matched indices are exactly
   range(124). A silent mismatch here would compare two overlapping samples and
   report a meaningless difference.

2. It resolves sessions by the 8-hex UUID prefix carried in the rollup slug
   (`2026-09-15_claude-code_b766c7ce_...`), not by substring against the
   manifest's full-UUID paths. Substring matching scores 0/25 on the known-good
   sessions, so it reads as "every session pruned" when nothing is pruned at
   all. The 25-session control below is what catches that.

Usage:
    rq3_prefix_representativeness.py
    rq3_prefix_representativeness.py --json    # machine-readable
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paths import REPO_ROOT  # noqa: E402

DATA = pathlib.Path(REPO_ROOT) / "data" / "paper-rollups"

# The confidence-corrected output-stripped rate the paper's central gap uses.
OUTPUT_STRIPPED_CORRECTED = 90.3

RECOVERABLE = ("correct", "partially_correct")


def load(name):
    with open(DATA / name) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - margin) / denom * 100, (centre + margin) / denom * 100)


def fisher_two_sided(a, b, c, d):
    """Exact two-sided p by summing every table no more likely than observed."""
    comb, n = math.comb, a + b + c + d
    observed = comb(a + b, a) * comb(c + d, c) / comb(n, a + c)
    total = 0.0
    for i in range(0, min(a + b, a + c) + 1):
        j, k = a + b - i, a + c - i
        l = c + d - k
        if j < 0 or k < 0 or l < 0:
            continue
        p = comb(a + b, i) * comb(c + d, k) / comb(n, a + c)
        if p <= observed + 1e-12:
            total += p
    return min(1.0, total)


def rate(rows):
    n = len(rows)
    rec = sum(1 for r in rows if r["grade"] in RECOVERABLE)
    att = sum(1 for r in rows if r["grade"] != "absent")
    wrong = sum(1 for r in rows if r["grade"] == "wrong")
    lo, hi = wilson(rec, n)
    wlo, whi = wilson(wrong, att)
    return dict(n=n, recoverable=rec, recoverable_pct=rec / n * 100,
                recoverable_ci=[lo, hi], attempted=att, wrong=wrong,
                wrong_of_attempted_pct=(wrong / att * 100) if att else 0.0,
                wrong_ci=[wlo, whi])


def session_survival():
    """Control + check: do the 349 file's sessions resolve and still exist?"""
    snap = json.load(open(DATA / "corpus_snapshot.json"))
    by8 = collections.defaultdict(list)
    for path in snap["transcript_bytes"]:
        uuid = os.path.basename(path).split(".")[0]
        by8[uuid[:8]].append(path)

    def prefix_of(session_id):
        parts = session_id.split("_")
        return parts[2] if len(parts) > 2 else ""

    def tally(session_ids):
        resolved = on_disk = 0
        for sid in session_ids:
            paths = by8.get(prefix_of(sid), [])
            if not paths:
                continue
            resolved += 1
            on_disk += any(os.path.exists(p) for p in paths)
        return dict(n=len(session_ids), resolved=resolved, on_disk=on_disk)

    q, s = load("rq3_questions.jsonl"), load("rq3_questions_stripped_arm.jsonl")
    known = sorted({r["session_id"] for r in s})
    extra = sorted({r["session_id"] for r in q} - set(known))
    return {"control_graded_25": tally(known), "extra_sessions": tally(extra)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    q = load("rq3_questions.jsonl")
    s = load("rq3_questions_stripped_arm.jsonl")

    frozen = {(r["session_id"], r["question"]) for r in s}
    idx = [i for i, r in enumerate(q) if (r["session_id"], r["question"]) in frozen]
    if idx != list(range(len(frozen))):
        raise SystemExit(
            f"FAIL: the frozen set is not a contiguous prefix "
            f"(matched {len(idx)} of {len(frozen)}, max index "
            f"{max(idx) if idx else -1}). Every comparison below would be "
            f"between overlapping samples.")

    cut = len(idx)
    head, tail = q[:cut], q[cut:]
    res = {"prefix": rate(head), "held_back": rate(tail), "all": rate(q)}
    p = fisher_two_sided(res["prefix"]["recoverable"],
                         res["prefix"]["n"] - res["prefix"]["recoverable"],
                         res["held_back"]["recoverable"],
                         res["held_back"]["n"] - res["held_back"]["recoverable"])
    drift = res["prefix"]["recoverable_pct"] - res["all"]["recoverable_pct"]
    gap_124 = OUTPUT_STRIPPED_CORRECTED - res["prefix"]["recoverable_pct"]
    gap_349 = OUTPUT_STRIPPED_CORRECTED - res["all"]["recoverable_pct"]
    survival = session_survival()

    out = dict(fisher_p=p, prefix_drift_points=drift, gap_on_124=gap_124,
               gap_on_349=gap_349, rates=res, session_survival=survival,
               note="rollup arm only; it is the one arm graded on all 349")

    if args.json:
        print(json.dumps(out, indent=1, sort_keys=True))
        return

    print("the frozen set is a contiguous prefix of "
          f"{len(q)} records (checked, {cut} of {cut} matched)\n")
    print("rollup arm, the one arm graded on all 349:")
    for label, key in (("frozen 124 (published)", "prefix"),
                       ("held-back 225", "held_back"),
                       ("all 349", "all")):
        r = res[key]
        print(f"  {label:<24} n={r['n']:>3}  recoverable {r['recoverable_pct']:>5.1f}% "
              f"[{r['recoverable_ci'][0]:.1f}, {r['recoverable_ci'][1]:.1f}]"
              f"   wrong/att {r['wrong']}/{r['attempted']} "
              f"= {r['wrong_of_attempted_pct']:>4.1f}%")
    print(f"\n  Fisher exact two-sided p = {p:.3f}  "
          f"(prefix vs held-back, recoverable)")
    print(f"  prefix drift against the full set: {drift:+.1f} points, "
          f"favouring the rollup arm" if drift > 0 else
          f"  prefix drift against the full set: {drift:+.1f} points")
    print(f"  central gap on 124: {gap_124:.1f} points   "
          f"on 349: {gap_349:.1f} points")
    print("\nsession survival (control first; 0 resolved on the control means "
          "the matcher is broken, not that files are gone):")
    for label, t in survival.items():
        print(f"  {label:<20} n={t['n']:>3}  resolved={t['resolved']:>3}  "
              f"transcript_on_disk={t['on_disk']:>3}")


if __name__ == "__main__":
    main()
