#!/usr/bin/env python3
"""Score human-vs-judge agreement on the blind grading sheet. Zero model calls.

    python3 scripts/paper-rollups/human_grading_score.py [sheet.csv] [key.json]

Prints n graded, exact agreement, Cohen's kappa (4-way), agreement on
wrong-vs-not, per-arm recoverable rate (correct + partial) for human and judge,
and the human-by-judge confusion matrix. Rows with a blank human_grade are skipped.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "paper-rollups"
GRADES = ("correct", "partial", "wrong", "absent")
JUDGE_MAP = {"correct": "correct", "partially_correct": "partial", "wrong": "wrong", "absent": "absent"}


def kappa(pairs, labels):
    n = len(pairs)
    po = sum(h == j for h, j in pairs) / n
    hc, jc = Counter(h for h, _ in pairs), Counter(j for _, j in pairs)
    pe = sum(hc[l] * jc[l] for l in labels) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def main():
    sheet = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "human_grading_sheet.csv"
    keyp = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA / "human_grading_key.json"
    key = json.loads(keyp.read_text())["items"]
    rows = list(csv.DictReader(open(sheet, newline="", encoding="utf-8")))
    items = []
    for r in rows:
        h = r["human_grade"].strip().lower()
        if not h:
            continue
        if h not in GRADES:
            raise SystemExit(f"{r['item_id']}: invalid human_grade {h!r}; allowed {'|'.join(GRADES)}")
        k = key[r["item_id"]]
        items.append((r["item_id"], k["arm"], h, JUDGE_MAP[k["judge_grade"]]))
    n = len(items)
    print(f"n graded: {n} of {len(rows)}")
    if n == 0:
        return
    pairs = [(h, j) for _, _, h, j in items]
    print(f"exact agreement: {sum(h == j for h, j in pairs)}/{n} = {sum(h == j for h, j in pairs)/n:.3f}")
    print(f"Cohen's kappa (4-way): {kappa(pairs, GRADES):.3f}")
    bp = [(h == "wrong", j == "wrong") for h, j in pairs]
    agree = sum(a == b for a, b in bp)
    print(f"agreement on wrong-vs-not: {agree}/{n} = {agree/n:.3f}; kappa {kappa(bp, (True, False)):.3f}")

    print("\nper-arm recoverable rate (correct + partial):")
    by = defaultdict(list)
    for _, arm, h, j in items:
        by[arm].append((h, j))
    print(f"  {'arm':<18}{'n':>3}  {'human':>7}  {'judge':>7}")
    for arm, ps in sorted(by.items()):
        hr = sum(h in ("correct", "partial") for h, _ in ps) / len(ps)
        jr = sum(j in ("correct", "partial") for _, j in ps) / len(ps)
        print(f"  {arm:<18}{len(ps):>3}  {hr:>7.1%}  {jr:>7.1%}")
    hr = sum(h in ("correct", "partial") for h, _ in pairs) / n
    jr = sum(j in ("correct", "partial") for _, j in pairs) / n
    print(f"  {'all':<18}{n:>3}  {hr:>7.1%}  {jr:>7.1%}")

    print("\nconfusion matrix (rows human, columns judge):")
    cm = Counter(pairs)
    print("  " + " " * 9 + "".join(f"{g:>9}" for g in GRADES))
    for h in GRADES:
        print(f"  {h:<9}" + "".join(f"{cm[(h, j)]:>9}" for j in GRADES))


if __name__ == "__main__":
    main()
