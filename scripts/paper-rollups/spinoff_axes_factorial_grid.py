#!/usr/bin/env python3
"""
Analyze the factorial grid coverage for the orthogonal axes spinoff paper.

Shows which cells in the volume × content-class grid are measured vs missing.
"""

import json
from pathlib import Path

# Existing measured points from the data.
#
# SUPERSEDED for the three prose rows: these predate the continuation-summary
# filter (2026-10-01), which removed harness "This session is being continued"
# blocks from the prose arms. Current figures are prose @2k 478/37.9%,
# prose @8k 1914/76.6%, prose-only 3082/89.5%. They are left as-is because this
# script records a grid-coverage decision taken against the older numbers;
# nothing reads its output, and main.tex no longer agrees with it. For live
# figures use rq3_prose_only_arm_summary.json and
# rq3_prose_only_budget_ladder_summary.json, or frontier_bands.py --filtered.
MEASURED = {
    "prose": [
        {"tokens": 478, "recall": 0.395, "source": "prose @2k truncated"},
        {"tokens": 1914, "recall": 0.710, "source": "prose @8k truncated"},
        {"tokens": 3799, "recall": 0.927, "source": "prose-only whole"},
    ],
    "call-only": [
        {"tokens": 6329, "recall": 0.472, "source": "call-only whole"},
    ],
    "action-record": [
        {"tokens": 6393, "recall": 0.968, "source": "action record whole"},
    ],
    "rollup": [
        {"tokens": 929, "recall": 0.565, "source": "rollup whole"},
    ],
    "answer-only": [
        {"tokens": 2533, "recall": 0.855, "source": "answer-only whole"},
    ],
}

def main():
    # Define the ideal 3×3 grid: 3 volume levels × 3 content classes
    volume_levels = ["low (~500-1k tok)", "medium (~2-4k tok)", "high (~6-8k tok)"]
    content_classes = ["prose", "structured-tool-calls", "generated-summary"]

    print("=" * 80)
    print("FACTORIAL GRID: Volume × Content Class")
    print("=" * 80)
    print()

    # Show what we have
    print("MEASURED POINTS:")
    print("-" * 80)
    for content_class, points in MEASURED.items():
        print(f"\n{content_class.upper()}:")
        for p in points:
            print(f"  {p['tokens']:>5} tokens → {p['recall']:.1%} recall ({p['source']})")

    print()
    print("=" * 80)
    print("3×3 FACTORIAL GRID COVERAGE")
    print("=" * 80)
    print()

    # Map our data to the grid
    grid = {}
    for vol in volume_levels:
        grid[vol] = {}
        for cls in content_classes:
            grid[vol][cls] = "MISSING"

    # Fill in what we have
    # Prose
    grid["low (~500-1k tok)"]["prose"] = "✓ prose @2k (478 tok, 39.5%)"
    grid["medium (~2-4k tok)"]["prose"] = "✓ prose @8k (1914 tok, 71.0%) + prose-full (3799 tok, 92.7%)"
    grid["high (~6-8k tok)"]["prose"] = "✓ (covered by prose-full at 3799 tok)"

    # Structured tool calls (action record)
    grid["high (~6-8k tok)"]["structured-tool-calls"] = "✓ action record (6393 tok, 96.8%)"

    # Generated summary
    grid["low (~500-1k tok)"]["generated-summary"] = "✓ rollup (929 tok, 56.5%)"

    # Print grid
    header = f"{'Volume Level':<22} | {'Prose':<35} | {'Structured Tool Calls':<30} | {'Generated Summary':<30}"
    print(header)
    print("-" * len(header))

    for vol in volume_levels:
        row = f"{vol:<22} | {grid[vol]['prose']:<35} | {grid[vol]['structured-tool-calls']:<30} | {grid[vol]['generated-summary']:<30}"
        print(row)

    print()
    print("=" * 80)
    print("COVERAGE SUMMARY")
    print("=" * 80)

    measured = sum(1 for v in grid.values() for c in v.values() if c != "MISSING")
    total = len(volume_levels) * len(content_classes)

    print(f"Measured: {measured}/{total} cells ({measured/total:.1%})")
    print(f"Missing: {total-measured}/{total} cells")
    print()
    print("KEY GAPS:")
    print("  - No low/medium volume points for structured tool calls")
    print("  - No medium/high volume points for generated summaries")
    print("  - Sparse diagonal: can claim orthogonality but cannot test interaction")
    print()
    print("TO FILL THE GRID COMPLETELY:")
    print("  1. Truncate action record to ~1k and ~3k tokens, regrade")
    print("  2. Truncate call-only to ~1k and ~3k tokens, regrade")
    print("  3. Generate longer rollups (or pad/retrieve more), test at ~3k and ~6k")
    print("  4. Total cost: ~6 new arms × 124 questions × ~$0.01/q ≈ $7.50")
    print()

    # Cost-matched pairs analysis
    print("=" * 80)
    print("COST-MATCHED PAIRS (volume held constant, class varies)")
    print("=" * 80)
    print()
    print("Pair 1: call-only (6329 tok, 47.2%) vs action-record (6393 tok, 96.8%)")
    print("  Gap: +64 tokens (+1.0%) for +49.6 points")
    print("  Status: MEASURED, strongest evidence")
    print()
    print("Pair 2 (MISSING): prose @1k vs structured-tool-calls @1k")
    print("  Would isolate class effect at low volume")
    print()
    print("Pair 3 (MISSING): prose @3k vs generated-summary @3k")
    print("  Would test whether generation loses detail vs extraction")
    print()

if __name__ == "__main__":
    main()
