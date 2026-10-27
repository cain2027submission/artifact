#!/usr/bin/env python3
"""Per-class paired McNemar test, prose-only against the tool-inputs-only floor.

The per-class recall table compares five question classes across two arms by
reading their Wilson intervals and noting they do not overlap. That is the
wrong test for this data: the two arms answer the SAME 123 questions, so the
comparison is paired and an independent-sample interval throws away the
pairing. This script runs the paired test instead -- exact McNemar (two-sided
binomial on the discordant pairs) per class, with Holm correction across the
five classes.

What the test can and cannot claim. rq3_tool_inputs_only_arm_summary.json
records that the tool-inputs-only arm is cited in the paper as an explicit
negative/floor control and NOT as a rival comparison arm, because it deletes
the entire substrate (prose) where these questions' answers live. So a
significant result here says "prose-only clears the floor control in this
class", not "prose-only beats call-only as a substrate choice".

Positive outcome is the paper's "recoverable" = correct + partially_correct,
with the confidence correction applied: a record listed in its arm summary's
reclassified_records with corrected_rating NOT_ANSWERABLE is counted absent.
Strict correct-only is reported alongside as a sensitivity check.

No model calls. Deterministic over the two committed arm files.

Usage:
    rq3_per_class_mcnemar.py
    rq3_per_class_mcnemar.py --json-only
"""

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
DATA = REPO / "data" / "paper-rollups"

PROSE_ARM = DATA / "rq3_prose_only_arm.jsonl"
PROSE_SUMMARY = DATA / "rq3_prose_only_arm_summary.json"
CALL_ARM = DATA / "rq3_tool_inputs_only_arm.jsonl"
CALL_SUMMARY = DATA / "rq3_tool_inputs_only_arm_summary.json"
REPORT = DATA / "rq3_per_class_mcnemar.json"

RECOVERABLE = {"correct", "partially_correct"}
STRICT = {"correct"}

# published figures these arms' summaries report, reproduced before any new
# number is computed -- a control against grading the wrong field
PUBLISHED = {
    "prose_only": {"uncorrected": 89.5, "corrected": 86.3},
    "tool_inputs_only": {"uncorrected": 47.2, "corrected": 44.7},
}


def load_arm(path):
    rows = [json.loads(line) for line in path.open() if line.strip()]
    by_id = {}
    for r in rows:
        by_id[r["record_id"]] = r
    if len(by_id) != len(rows):
        sys.exit(f"{path.name}: duplicate record_id, {len(rows)} rows "
                 f"collapse to {len(by_id)}")
    return by_id


def corrections(summary_path):
    """record_ids the confidence correction moves to NOT_ANSWERABLE."""
    d = json.load(summary_path.open())
    out = set()
    for rec in d.get("reclassified_records", []):
        if rec.get("corrected_rating") == "NOT_ANSWERABLE":
            out.add(rec["record_id"])
    return out


def positive(row, record_id, corrected_ids, positive_grades, apply_correction):
    if apply_correction and record_id in corrected_ids:
        return False
    return row["grade"] in positive_grades


def binom_two_sided(b, c):
    """Exact two-sided binomial p on b successes of n = b + c at p = 0.5.

    This is exact McNemar. Returns 1.0 when there are no discordant pairs,
    which is the honest answer: the arms are indistinguishable on this class.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adjusted = [None] * m
    running = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running = max(running, adj)
        adjusted[idx] = min(1.0, running)
    return adjusted


def table(pairs, prose, call, prose_fix, call_fix, grades, apply_correction):
    """2x2 counts over paired record_ids."""
    both = prose_only = call_only = neither = 0
    for rid in pairs:
        p = positive(prose[rid], rid, prose_fix, grades, apply_correction)
        c = positive(call[rid], rid, call_fix, grades, apply_correction)
        if p and c:
            both += 1
        elif p:
            prose_only += 1
        elif c:
            call_only += 1
        else:
            neither += 1
    return both, prose_only, call_only, neither


def rate(rows, fix, grades, apply_correction):
    n = len(rows)
    hits = sum(
        1 for rid, r in rows.items()
        if positive(r, rid, fix, grades, apply_correction)
    )
    return hits, n, round(100.0 * hits / n, 1) if n else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-only", action="store_true",
                    help="write the report and print nothing else")
    args = ap.parse_args()

    prose = load_arm(PROSE_ARM)
    call = load_arm(CALL_ARM)
    prose_fix = corrections(PROSE_SUMMARY)
    call_fix = corrections(CALL_SUMMARY)

    pairs = sorted(set(prose) & set(call))
    unpaired_prose = sorted(set(prose) - set(call))
    unpaired_call = sorted(set(call) - set(prose))

    # control: reproduce each arm's published whole-arm recoverable rate on its
    # own full file, before the paired subset is touched
    repro = {}
    for name, rows, fix in (("prose_only", prose, prose_fix),
                            ("tool_inputs_only", call, call_fix)):
        unc = rate(rows, fix, RECOVERABLE, False)
        cor = rate(rows, fix, RECOVERABLE, True)
        repro[name] = {
            "n": unc[1],
            "uncorrected_pct": unc[2],
            "corrected_pct": cor[2],
            "published_uncorrected_pct": PUBLISHED[name]["uncorrected"],
            "published_corrected_pct": PUBLISHED[name]["corrected"],
            "uncorrected_matches_published":
                abs(unc[2] - PUBLISHED[name]["uncorrected"]) <= 0.1,
            "corrected_matches_published":
                abs(cor[2] - PUBLISHED[name]["corrected"]) <= 0.1,
        }

    classes = sorted({prose[rid]["class"] for rid in pairs})

    results = {}
    for definition, grades in (("recoverable", RECOVERABLE),
                               ("correct_only", STRICT)):
        per_class = {}
        raw_p = []
        for cls in classes:
            ids = [rid for rid in pairs if prose[rid]["class"] == cls]
            both, p_only, c_only, neither = table(
                ids, prose, call, prose_fix, call_fix, grades, True)
            p = binom_two_sided(p_only, c_only)
            raw_p.append(p)
            per_class[cls] = {
                "n_pairs": len(ids),
                "both_positive": both,
                "prose_only_positive": p_only,
                "call_only_positive": c_only,
                "neither_positive": neither,
                "discordant": p_only + c_only,
                "prose_pct": round(100.0 * (both + p_only) / len(ids), 1),
                "call_pct": round(100.0 * (both + c_only) / len(ids), 1),
                "gap_points": round(100.0 * (p_only - c_only) / len(ids), 1),
                "p_exact": round(p, 6),
            }
        for cls, adj in zip(classes, holm(raw_p)):
            per_class[cls]["p_holm"] = round(adj, 6)
            per_class[cls]["significant_holm_05"] = adj < 0.05

        both, p_only, c_only, neither = table(
            pairs, prose, call, prose_fix, call_fix, grades, True)
        overall_p = binom_two_sided(p_only, c_only)
        results[definition] = {
            "overall": {
                "n_pairs": len(pairs),
                "both_positive": both,
                "prose_only_positive": p_only,
                "call_only_positive": c_only,
                "neither_positive": neither,
                "discordant": p_only + c_only,
                "prose_pct": round(100.0 * (both + p_only) / len(pairs), 1),
                "call_pct": round(100.0 * (both + c_only) / len(pairs), 1),
                "p_exact": round(overall_p, 8),
            },
            "per_class": per_class,
            "n_classes_significant_holm_05": sum(
                1 for v in per_class.values() if v["significant_holm_05"]),
        }

    report = {
        "what_this_is": (
            "paired exact McNemar per question class, prose-only against the "
            "tool-inputs-only floor control, replacing a Wilson-interval "
            "overlap read of the same table"
        ),
        "claim_it_supports": (
            "prose-only clears the tool-inputs-only floor control in every "
            "question class, not only in aggregate -- which is a statement "
            "about the floor, not a substrate ranking against call-only, "
            "because the floor arm deletes the substrate the answers live in"
        ),
        "positive_outcome": "recoverable = correct + partially_correct",
        "correction": (
            "records listed in each arm summary's reclassified_records with "
            "corrected_rating NOT_ANSWERABLE counted absent"
        ),
        "model_calls": 0,
        "pairing": {
            "key": "record_id",
            "n_pairs": len(pairs),
            "n_prose_rows": len(prose),
            "n_call_rows": len(call),
            "unpaired_prose_only": unpaired_prose,
            "unpaired_call_only": unpaired_call,
        },
        "reproduction_control": repro,
        "corrections_applied": {
            "prose_only": sorted(prose_fix),
            "tool_inputs_only": sorted(call_fix),
        },
        "results": results,
    }

    REPORT.write_text(json.dumps(report, indent=2) + "\n")

    if args.json_only:
        return

    print(f"wrote {REPORT.relative_to(REPO)}")
    print()
    print("reproduction control (whole-arm recoverable, against published)")
    for name, v in repro.items():
        ok_u = "ok" if v["uncorrected_matches_published"] else "MISMATCH"
        ok_c = "ok" if v["corrected_matches_published"] else "MISMATCH"
        print(f"  {name:18s} n={v['n']:3d}  "
              f"uncorrected {v['uncorrected_pct']:5.1f} vs "
              f"{v['published_uncorrected_pct']:5.1f} {ok_u:8s}  "
              f"corrected {v['corrected_pct']:5.1f} vs "
              f"{v['published_corrected_pct']:5.1f} {ok_c}")
    print()
    print(f"paired on record_id: {len(pairs)} pairs "
          f"(prose {len(prose)} rows, call {len(call)} rows)")

    for definition in ("recoverable", "correct_only"):
        r = results[definition]
        print()
        print(f"=== {definition} ===")
        head = (f"{'class':18s} {'n':>4s} {'prose%':>7s} {'call%':>7s} "
                f"{'gap':>6s} {'b':>3s} {'c':>3s} {'p_exact':>9s} "
                f"{'p_holm':>9s}")
        print(head)
        for cls, v in r["per_class"].items():
            star = " *" if v["significant_holm_05"] else ""
            print(f"{cls:18s} {v['n_pairs']:4d} {v['prose_pct']:7.1f} "
                  f"{v['call_pct']:7.1f} {v['gap_points']:+6.1f} "
                  f"{v['prose_only_positive']:3d} "
                  f"{v['call_only_positive']:3d} {v['p_exact']:9.5f} "
                  f"{v['p_holm']:9.5f}{star}")
        o = r["overall"]
        print(f"{'OVERALL':18s} {o['n_pairs']:4d} {o['prose_pct']:7.1f} "
              f"{o['call_pct']:7.1f} {'':6s} {o['prose_only_positive']:3d} "
              f"{o['call_only_positive']:3d} {o['p_exact']:9.5f}")
        print(f"  classes significant after Holm at 0.05: "
              f"{r['n_classes_significant_holm_05']} of "
              f"{len(r['per_class'])}")
        print("  b = prose positive and call negative, "
              "c = call positive and prose negative")


if __name__ == "__main__":
    main()
