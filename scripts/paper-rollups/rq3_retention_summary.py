#!/usr/bin/env python3
"""Summarise judged question retention per question class, with Wilson intervals.

This reads the judged question set produced by part2_streamlined.py and reports
what a rollup can and cannot answer about the session it summarises.

Two things this script has to defend against, both learned the hard way:

1. The JSONL is written incrementally and the generator was killed mid-line, so
   the final record is truncated. Reading it with a plain `json.loads` per line
   raises and loses all 119 good records. It is parsed with a streaming decoder
   that stops cleanly at the first unparseable byte, and reports how many
   records were dropped, so a truncated tail degrades the sample by one instead
   of failing the analysis.

2. The generating agent reported "20 questions, 70% absent" from its own run.
   The file on disk holds 119 graded questions with a materially different
   distribution. The agent's summary was stale, not its data. Every number here
   is recomputed from the file rather than taken from any report.

Per-class n is about 24, so a bare percentage would overstate precision. Wilson
score intervals are used rather than normal-approximation ones because they stay
inside [0,1] at these counts, where the normal interval does not.

3. The question file grows. It is appended to as more sessions are graded, which
   is the same reproducibility hole that forced the corpus freeze: a number
   computed from it is only meaningful against one state of the file. Nothing
   here can stop the appending, so instead the exact input is recorded --
   `input_sha256`, `input_bytes` and the sorted record ids -- and the paper cites
   a number together with the input digest that produced it. If a figure in the
   draft disagrees with a rerun, compare digests before looking for a bug.

Sampling caveat carried into the output: part2_streamlined.py samples only
sessions whose transcript is >=100KB, stratified half 100KB-1MB and half >=1MB.
These rates therefore describe rollups of substantial sessions. They must not be
quoted as a corpus-wide retention rate -- most sessions are smaller than this
floor, and a short session's rollup has far less to lose.

Run with the indexer venv interpreter:
    python3 rq3_retention_summary.py
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "paper-rollups"
QUESTIONS = DATA / "rq3_questions.jsonl"
OUT = DATA / "rq3_retention_summary.json"

GRADES = ("correct", "partially_correct", "wrong", "absent")

# What each class probes, kept here so the paper's table caption and the code
# cannot drift apart.
CLASS_MEANING = {
    "decision": "why a choice was made",
    "dead_end": "an approach tried and abandoned",
    "outcome": "what the session concluded or produced",
    "artifact_detail": "a specific path, value or identifier",
    "sequence": "the order in which steps happened",
}


def load_records(path: Path) -> tuple[list[dict], int]:
    """Stream-decode concatenated JSON objects, tolerating a truncated tail."""
    raw = path.read_text(errors="replace")
    dec = json.JSONDecoder()
    rows: list[dict] = []
    i = 0
    while i < len(raw):
        while i < len(raw) and raw[i] in " \n\r\t":
            i += 1
        if i >= len(raw):
            break
        try:
            obj, i = dec.raw_decode(raw, i)
        except json.JSONDecodeError:
            break
        rows.append(obj)
    # Anything after the last successful decode is a partial write.
    dropped = 1 if i < len(raw) and raw[i:].strip() else 0
    return rows, dropped


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval; returns (lo, hi) as percentages."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * max(0.0, centre - half), 1),
            round(100 * min(1.0, centre + half), 1))


def rates(rows: list[dict]) -> dict:
    n = len(rows)
    c = Counter(r.get("grade") for r in rows)
    out = {"n": n}
    for g in GRADES:
        lo, hi = wilson(c[g], n)
        out[g] = {"n": c[g],
                  "pct": round(100.0 * c[g] / n, 1) if n else 0.0,
                  "ci95": [lo, hi]}
    # Two derived rates the paper argues from directly.
    #  - recoverable: the rollup said something right or partly right, i.e. the
    #    information survived compression in some usable form.
    #  - hallucination_of_attempted: of the questions the rollup did answer,
    #    how often it answered wrongly. This is the number that matters for
    #    trusting a rollup as context, because an absent answer is visibly
    #    absent whereas a wrong one is not.
    rec = c["correct"] + c["partially_correct"]
    lo, hi = wilson(rec, n)
    out["recoverable"] = {"n": rec,
                          "pct": round(100.0 * rec / n, 1) if n else 0.0,
                          "ci95": [lo, hi]}
    attempted = n - c["absent"]
    lo, hi = wilson(c["wrong"], attempted)
    out["hallucination_of_attempted"] = {
        "n_wrong": c["wrong"], "n_attempted": attempted,
        "pct": round(100.0 * c["wrong"] / attempted, 1) if attempted else None,
        "ci95": [lo, hi]}
    return out


def record_id(r: dict) -> str:
    """Stable identity for a graded question, for provenance and for sorting.

    There is no guaranteed id field across schema generations, so fall back to a
    hash of the session and the question text rather than to file order, which
    changes as the file is appended to.
    """
    for key in ("id", "question_id", "qid"):
        if r.get(key):
            return str(r[key])
    seed = f"{r.get('session_id','')}|{r.get('class','')}|{r.get('question','')}"
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def main() -> None:
    raw_bytes = QUESTIONS.read_bytes()
    input_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    rows, dropped = load_records(QUESTIONS)
    rows = [r for r in rows if r.get("grade") in GRADES]
    if not rows:
        raise SystemExit(f"no graded records in {QUESTIONS}")

    # Sort by record id so every aggregate is computed in a fixed order and the
    # output bytes do not depend on when a record was appended.
    rows.sort(key=record_id)

    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r.get("class", "unknown")].append(r)

    per_class = {k: {**rates(v), "probes": CLASS_MEANING.get(k, "")}
                 for k, v in sorted(by_class.items())}

    # Rank classes by the recoverable rate. The paper's claim is about the
    # ordering, not the absolute values, because the ordering is what survives
    # the sampling caveat above.
    ranking = sorted(per_class.items(), key=lambda kv: -kv[1]["recoverable"]["pct"])

    summary = {
        "note": "judged retention of session content in its rollup; "
                "ground truth read from the transcript, answers graded from the "
                "rollup alone",
        "sampling_caveat": "sessions sampled only from those with a transcript "
                           ">=100KB, stratified half 100KB-1MB half >=1MB, "
                           "seed 42; not a corpus-wide rate",
        "judge": "claude-sonnet-5 via local gateway, no temperature set",
        "input_file": QUESTIONS.name,
        "input_sha256": input_sha256,
        "input_bytes": len(raw_bytes),
        "graded_record_ids": sorted(record_id(r) for r in rows),
        "n_questions": len(rows),
        "n_sessions": len({r["session_id"] for r in rows}),
        "n_truncated_records_dropped": dropped,
        "overall": rates(rows),
        "per_class": per_class,
        "recoverable_ranking": [{"class": k,
                                 "recoverable_pct": v["recoverable"]["pct"],
                                 "correct_pct": v["correct"]["pct"]}
                                for k, v in ranking],
    }

    OUT.write_text(json.dumps(summary, indent=2))

    o = summary["overall"]
    print(f"questions {summary['n_questions']} over {summary['n_sessions']} sessions "
          f"(dropped {dropped} truncated)")
    print(f"input {QUESTIONS.name} sha256 {input_sha256[:16]} "
          f"({len(raw_bytes)} bytes)")
    for g in GRADES:
        print(f"  {g:20s} {o[g]['n']:4d}  {o[g]['pct']:5.1f}%  CI {o[g]['ci95']}")
    print(f"  {'recoverable':20s} {o['recoverable']['n']:4d}  "
          f"{o['recoverable']['pct']:5.1f}%  CI {o['recoverable']['ci95']}")
    h = o["hallucination_of_attempted"]
    print(f"  wrong / attempted    {h['n_wrong']}/{h['n_attempted']}  "
          f"{h['pct']}%  CI {h['ci95']}")
    print("\nby class, best-retained first:")
    for k, v in ranking:
        print(f"  {k:16s} n={v['n']:3d}  recoverable {v['recoverable']['pct']:5.1f}% "
              f"CI {v['recoverable']['ci95']}  correct {v['correct']['pct']:5.1f}%")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
