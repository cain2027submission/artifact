#!/usr/bin/env python3
"""Grading-rule sensitivity grid for the benchmark fragility paper.

Shows how the headline recall figure moves across every combination of:
- Denominator choice (attempted only vs all questions)
- Partial credit rule (count as correct, as partially_correct, or exclude)

This quantifies the grading-rule sensitivity finding.
"""

import json
import math
from pathlib import Path
from collections import Counter

DATA = Path(__file__).parent.parent.parent / "data" / "paper-rollups"
QUESTIONS = DATA / "rq3_questions.jsonl"
OUT = DATA / "grading_rule_grid.json"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * max(0.0, centre - half), 1),
            round(100 * min(1.0, centre + half), 1))


def load_records(path: Path) -> list[dict]:
    """Load JSONL, tolerating truncated tail."""
    raw = path.read_text(errors="replace")
    rows = []
    for line in raw.splitlines():
        if not line.strip() or "\x00" in line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def label(r: dict) -> str | None:
    """Confidence marker."""
    return r.get("rating") or r.get("confidence")


def compute_headline(rows: list[dict], partial_rule: str, denominator: str) -> dict:
    """Compute wrong-of-attempted under a specific grading rule.

    partial_rule: 'as_correct', 'as_partial', 'exclude'
    denominator: 'attempted', 'all'
    """
    # Apply partial-credit rule
    working = []
    for r in rows:
        r = dict(r)
        grade = r.get("grade")
        conf = label(r)

        if partial_rule == "as_correct":
            # PARTIALLY_ANSWERED with grade=correct stays correct (old behavior)
            pass
        elif partial_rule == "as_partial":
            # PARTIALLY_ANSWERED with grade=correct becomes partially_correct
            if conf == "PARTIALLY_ANSWERED" and grade == "correct":
                r["grade"] = "partially_correct"
        elif partial_rule == "exclude":
            # Exclude all PARTIALLY_ANSWERED from the sample
            if conf == "PARTIALLY_ANSWERED":
                continue

        working.append(r)

    # Count grades
    counts = Counter(r.get("grade") for r in working)
    n_total = len(working)
    n_wrong = counts["wrong"]
    n_absent = counts["absent"]
    n_attempted = n_total - n_absent

    # Compute by denominator
    if denominator == "attempted":
        numer = n_wrong
        denom = n_attempted
    elif denominator == "all":
        numer = n_wrong
        denom = n_total
    else:
        raise ValueError(f"Unknown denominator: {denominator}")

    pct = round(100 * numer / denom, 1) if denom > 0 else None
    ci = list(wilson(numer, denom)) if denom > 0 else None

    return {
        "n_total": n_total,
        "n_wrong": n_wrong,
        "n_absent": n_absent,
        "n_attempted": n_attempted,
        "numerator": numer,
        "denominator": denom,
        "pct": pct,
        "ci95": ci,
        "grade_distribution": dict(counts),
    }


def main():
    rows = load_records(QUESTIONS)
    print(f"Loaded {len(rows)} records")

    # Only use the published 124
    rows = rows[:124]
    print(f"Using first 124 (published sample)")

    print("\n" + "=" * 70)
    print("GRADING-RULE SENSITIVITY GRID")
    print("=" * 70)

    partial_rules = ["as_correct", "as_partial", "exclude"]
    denominators = ["attempted", "all"]

    results = {}
    table = []

    for p_rule in partial_rules:
        for denom in denominators:
            key = f"{p_rule}_{denom}"
            result = compute_headline(rows, p_rule, denom)
            results[key] = result

            rule_desc = {
                "as_correct": "partial→correct",
                "as_partial": "partial→partial",
                "exclude": "partial excluded",
            }[p_rule]

            denom_desc = {
                "attempted": "attempted",
                "all": "all questions",
            }[denom]

            table.append({
                "partial_rule": p_rule,
                "denominator": denom,
                "rule_label": rule_desc,
                "denom_label": denom_desc,
                **result
            })

            print(f"\n{rule_desc:20s} / {denom_desc:12s}:")
            print(f"  {result['numerator']}/{result['denominator']} = {result['pct']}% {result['ci95']}")

    # Compute range per arm
    all_pcts = [r["pct"] for r in results.values() if r["pct"] is not None]
    pct_min = min(all_pcts)
    pct_max = max(all_pcts)
    pct_range = pct_max - pct_min

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Headline moves across: {pct_min}% to {pct_max}%")
    print(f"  Range: {pct_range} percentage points")
    print(f"  Rules tested: {len(partial_rules)} partial-credit × {len(denominators)} denominator = {len(results)}")

    # Which was published?
    published = results["as_correct_attempted"]
    print(f"\n  Published figure (partial→correct, attempted denominator):")
    print(f"    {published['numerator']}/{published['denominator']} = {published['pct']}% {published['ci95']}")

    output = {
        "grid": table,
        "summary": {
            "min_pct": pct_min,
            "max_pct": pct_max,
            "range_pp": pct_range,
            "n_combinations": len(results),
        },
        "published_rule": {
            "partial_credit": "as_correct",
            "denominator": "attempted",
            **published
        },
        "note": "All figures computed over the published 124-question sample",
    }

    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nWrote {OUT.name}")


if __name__ == "__main__":
    main()
