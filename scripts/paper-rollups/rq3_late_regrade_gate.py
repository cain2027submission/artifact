#!/usr/bin/env python3
"""Gate (b): re-grade 20 stored published answers through this pipeline's grader.

A seeded sample of 10 rollup (rq3_rollup_uniform) and 10 output-stripped
(rq3_questions_stripped_arm) records from the frozen 124. Each is re-graded with
rq3_late_window.grade_record (same prompt, parser, model, majority protocol) and
compared with its stored 3-draw-majority `grade`. Draws are cached in
late-window/gate_b_draws.jsonl; the result goes to late-window/gate_b.json.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq3_late_common as lc  # noqa: E402
import rq3_late_window as lw  # noqa: E402

lw.DRAWS = lc.OUTDIR / "gate_b_draws.jsonl"
SEED = 20261010


def main():
    graded = set(json.loads(lc.SUMMARY.read_text())["graded_record_ids"])
    rng = random.Random(SEED)
    items = []
    for stem, arm in (("rq3_rollup_uniform", "rollup"), ("rq3_questions_stripped_arm", "stripped")):
        rows = [r for r in lc.read_jsonl(lc.DATA / f"{stem}.jsonl") if r["record_id"] in graded]
        rows.sort(key=lambda r: r["record_id"])
        for r in rng.sample(rows, 10):
            items.append((stem, arm, r))
    cache = {r["key"]: r for r in lc.read_jsonl(lw.DRAWS)}
    out = []
    for stem, arm, r in items:
        q = {"qid": "gate-" + r["record_id"], "question": r["question"],
             "ground_truth": r["ground_truth"]}
        ans = {"arm": arm, "answer": r.get("answer") or "",
               "confidence": r.get("rating") or r.get("confidence") or "ANSWERED"}
        g = lw.grade_record(q, ans, cache)
        out.append({"file": stem, "record_id": r["record_id"], "stored": r["grade"],
                    "regraded": g["grade"], "draws": g["draws"], "match": g["grade"] == r["grade"]})
    n_match = sum(o["match"] for o in out)
    # recoverable (correct+partial) agreement is the quantity the recall rates use
    rec = lambda g: g in ("correct", "partially_correct")
    n_rec = sum(rec(o["stored"]) == rec(o["regraded"]) for o in out)
    res = {"n": len(out), "exact_match": n_match, "recoverable_match": n_rec,
           "seed": SEED, "rows": out}
    (lc.OUTDIR / "gate_b.json").write_text(json.dumps(res, indent=1))
    print(f"gate (b): exact grade match {n_match}/{len(out)}, recoverable match {n_rec}/{len(out)}")
    for o in out:
        if not o["match"]:
            print("  mismatch", o["file"], o["record_id"], o["stored"], "->", o["regraded"], o["draws"])


if __name__ == "__main__":
    main()
