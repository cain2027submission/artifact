#!/usr/bin/env python3
"""Place the STRIPPED-TRANSCRIPT arm beside the rollup and raw arms.

rq3_paired_comparison.py hardcodes the uniform-rollup/raw contrast. This adds the
stripped arm without touching that file's published numbers. Every statistic here
is computed by the helpers that produced the other arms' numbers -- `rates` and
`wilson` are imported, and the paired test is the same exact two-sided binomial on
the discordant pairs -- so no new method is introduced.

The stripped arm covers all 124 graded records. The raw arm covers 89, because 5
sessions overflowed the context window even unstripped. Reporting the stripped arm
only on 124 would compare it against a raw arm measured on an easier subset, so
both the full set and the 89-record intersection are printed.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_raw_transcript_arm import (  # noqa: E402
    DATA, QUESTIONS, SUMMARY, load_records, rates, record_id,
)

REPORT = DATA / "rq3_stripped_paired.json"
RECOVERABLE = ("correct", "partially_correct")


def exact_binomial(b: int, c: int) -> float:
    """Two-sided exact binomial on discordant pairs, verbatim from the sibling."""
    n = b + c
    if not n:
        return 1.0
    k = min(b, c)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(p, 1.0)


def paired(a: dict, bd: dict, shared: list[str], la: str, lb: str) -> dict:
    a_ok = {i for i in shared if a[i]["grade"] in RECOVERABLE}
    b_ok = {i for i in shared if bd[i]["grade"] in RECOVERABLE}
    b = len(b_ok - a_ok)
    c = len(a_ok - b_ok)
    both = len(a_ok & b_ok)
    neither = len(shared) - len(a_ok | b_ok)
    p = exact_binomial(b, c)
    print(f"\npaired on 'recoverable' over {len(shared)} records: {la} vs {lb}")
    print(f"  both {both}   {lb} only {b}   {la} only {c}   neither {neither}")
    print(f"  discordant {b + c}, exact binomial two-sided p = {p:.4g}")
    return {"n": len(shared), "both": both, f"{lb}_only": b, f"{la}_only": c,
            "neither": neither, "discordant": b + c,
            "exact_binomial_two_sided_p": p}


def first_line_refusals(answers: dict, ans_key: str, rate_key: str
                        ) -> tuple[list[str], list[str]]:
    """Split replies that open with a refusal but were NOT recorded as one.

    A bare refusal (nothing but "NOT ANSWERABLE") is a genuine miscoding. A long
    reply that opens with the same words and then answers is not -- the last-line
    rule gets those right -- so the two are counted separately rather than pooled.
    """
    bare, hedged = [], []
    for rid, r in answers.items():
        a = (r.get(ans_key) or "").strip()
        if not a:
            continue
        head = a.split("\n")[0].upper()
        if ("NOT ANSWERABLE" in head or "NOT_ANSWERABLE" in head) \
                and r.get(rate_key) != "NOT_ANSWERABLE":
            (bare if len(a) < 40 else hedged).append(rid)
    return bare, hedged


def artifact_sensitivity(strip: dict, full: list[str], shared89: list[str]) -> dict:
    answers = {r["record_id"]: r
               for r in load_records(DATA / "rq3_questions_stripped_arm.jsonl")}
    raw_ans = {r["record_id"]: r
               for r in load_records(DATA / "rq3_questions_raw_arm.jsonl")}
    bare, hedged = first_line_refusals(answers, "answer", "rating")
    r_bare, r_hedged = first_line_refusals(raw_ans, "raw_answer", "raw_rating")
    print(f"\nlast-line confidence artifact -- replies opening with a refusal but "
          f"recorded otherwise:")
    print(f"  stripped arm: {len(bare) + len(hedged):3d}  "
          f"({len(bare)} bare refusal, {len(hedged)} hedged-then-answered)")
    print(f"  raw arm:      {len(r_bare) + len(r_hedged):3d}  "
          f"({len(r_bare)} bare refusal, {len(r_hedged)} hedged-then-answered)")

    out = {"stripped_bare": bare, "stripped_hedged": hedged,
           "raw_bare": r_bare, "raw_hedged": r_hedged, "sets": {}}
    for label, ids in (("full", full), ("raw_comparable_89", shared89)):
        variants = {
            "as_recorded": rates([strip[i] for i in ids]),
            "bare_to_absent": rates(
                [dict(strip[i], grade="absent") if i in bare else strip[i]
                 for i in ids]),
            "all_affected_to_absent": rates(
                [dict(strip[i], grade="absent") if i in bare + hedged else strip[i]
                 for i in ids]),
        }
        out["sets"][label] = variants
        print(f"  {label}:")
        for nm, d in variants.items():
            print(f"    {nm:24s} n={d['n']:3d}  recoverable "
                  f"{d['recoverable']['pct']:5.1f}% {d['recoverable']['ci95']}  "
                  f"wrong-of-att {d['hallucination_of_attempted']['pct']}%")
    return out


def line(label: str, d: dict) -> str:
    h = d["hallucination_of_attempted"]
    return (f"{label:28s} n={d['n']:3d}  {d['correct']['n']:3d} "
            f"{d['partially_correct']['n']:3d} {d['wrong']['n']:3d} "
            f"{d['absent']['n']:3d}   {d['recoverable']['pct']:5.1f}% "
            f"{str(d['recoverable']['ci95']):14s} "
            f"{h['pct']}% {h['ci95']}")


def main() -> None:
    graded = set(json.loads(SUMMARY.read_text())["graded_record_ids"])
    published = {record_id(r): r for r in load_records(QUESTIONS)
                 if record_id(r) in graded}
    uniform = {r["record_id"]: r for r in load_records(DATA / "rq3_majority_uniform.jsonl")}
    raw = {r["record_id"]: r for r in load_records(DATA / "rq3_majority_raw.jsonl")}
    strip = {r["record_id"]: r for r in load_records(DATA / "rq3_majority_stripped.jsonl")}

    # A record the grader left unresolved after 5 draws has grade None. Dropping it
    # beats inventing a grade the grader declined to give.
    unresolved = sorted({i for d in (uniform, raw, strip) for i, r in d.items()
                         if r.get("grade") is None})
    for d in (uniform, raw, strip):
        for i in unresolved:
            d.pop(i, None)

    full = sorted(set(strip) & set(published))
    shared89 = sorted(set(strip) & set(raw) & set(uniform) & set(published))
    print(f"published {len(published)}  uniform {len(uniform)}  raw {len(raw)}  "
          f"stripped {len(strip)}  (dropped unresolved: {len(unresolved)})")
    print(f"full stripped set {len(full)}   raw-comparable intersection {len(shared89)}")

    r = {
        "pub_full": rates([published[i] for i in full]),
        "uni_full": rates([uniform[i] for i in full if i in uniform]),
        "strip_full": rates([strip[i] for i in full]),
        "pub_89": rates([published[i] for i in shared89]),
        "uni_89": rates([uniform[i] for i in shared89]),
        "raw_89": rates([raw[i] for i in shared89]),
        "strip_89": rates([strip[i] for i in shared89]),
    }

    hdr = (f"{'arm':28s} {'n':>5s}  {'cor':>3s} {'par':>3s} {'wrg':>3s} "
           f"{'abs':>3s}   {'recoverable':>6s}                wrong-of-attempted")
    print("\n" + hdr)
    print("-" * len(hdr))
    print(line("published rollup, full", r["pub_full"]))
    print(line("uniform rollup, full", r["uni_full"]))
    print(line("STRIPPED, full", r["strip_full"]))
    print("-" * len(hdr))
    print(line("published rollup, 89", r["pub_89"]))
    print(line("uniform rollup, 89", r["uni_89"]))
    print(line("RAW TRANSCRIPT, 89", r["raw_89"]))
    print(line("STRIPPED, 89", r["strip_89"]))

    pairs = {
        "stripped_vs_raw_89": paired(strip, raw, shared89, "stripped", "raw"),
        "stripped_vs_uniform_89": paired(strip, uniform, shared89, "stripped", "uniform"),
        "stripped_vs_published_89": paired(strip, published, shared89, "stripped", "published"),
        "stripped_vs_uniform_full": paired(strip, uniform, [i for i in full if i in uniform],
                                           "stripped", "uniform"),
        "stripped_vs_published_full": paired(strip, published, full, "stripped", "published"),
    }

    # Confidence is read from the reply's LAST line, so a reply that opens
    # "NOT ANSWERABLE" and then answers anyway is recorded ANSWERED. Both arms
    # share the extraction verbatim, so the as-recorded rate is the comparable
    # one and stays the headline; these are the sensitivity bounds.
    sens = artifact_sensitivity(strip, full, shared89)

    print("\nper class, stripped arm on the full set (recoverable %):")
    per_class = {}
    for c in sorted({strip[i].get("class") for i in full if strip[i].get("class")}):
        ids = [i for i in full if strip[i].get("class") == c]
        s = rates([strip[i] for i in ids])
        p = rates([published[i] for i in ids])
        per_class[c] = {"n": s["n"], "published": p, "stripped": s}
        print(f"  {c:16s} n={s['n']:3d}  rollup {p['recoverable']['pct']:5.1f}% -> "
              f"stripped {s['recoverable']['pct']:5.1f}%")

    REPORT.write_text(json.dumps(
        {"n_full": len(full), "n_raw_comparable": len(shared89),
         "dropped_unresolved": unresolved, "rates": r, "paired": pairs,
         "last_line_confidence_artifact": sens,
         "per_class": per_class}, indent=2) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
