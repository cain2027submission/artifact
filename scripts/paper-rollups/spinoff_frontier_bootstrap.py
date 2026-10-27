#!/usr/bin/env python3
"""Bootstrap sensitivity analysis for the F-banded substrate frontier.

Resample the per-question grades and report how stable the 7 band boundaries are.

Usage:
    spinoff_frontier_bootstrap.py                    # 1000 bootstrap samples
    spinoff_frontier_bootstrap.py --samples 100      # smoke test with 100
    spinoff_frontier_bootstrap.py --smoke            # tiny test: 10 samples on 3 questions
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
REPO = HERE.parents[2]
DATA = REPO / "data" / "paper-rollups"

# Substrate definitions matching substrate_value.py
SUBSTRATES = [
    ("rollup", DATA / "rq3_rollup_uniform.jsonl", 932, 0.578),
    ("dialogue record", DATA / "rq3_verbatim_arm.jsonl", 3928, 0.0),
    ("call-only", DATA / "rq3_tool_inputs_only_arm.jsonl", 6329, 0.0),
    ("action record", DATA / "rq3_action_record_arm.jsonl", 6393, 0.0),  # corrected from 9757
    ("output-stripped", DATA / "rq3_stripped_transcript_arm.jsonl", 8880, 0.0),
    ("raw", DATA / "rq3_questions_raw_arm.jsonl", 137230, 0.0),
]

# Prose truncation arms from the paper.
#
# SUPERSEDED 2026-10-01 by the continuation-summary filter: the current figures
# are prose-2k 478/37.9%, prose-8k 1914/76.6%, prose-full 3082/89.5%, and on
# those prose-2k no longer wins a band. Left as a snapshot (this script is
# already a snapshot -- its rollup row is 932/57.8% where the paper says
# 929/56.5%) and nothing in main.tex reads its output. These three are held at
# fixed recall rather than resampled, so updating them would not yield a
# resampled interval for prose-2k; per-question grades for that now exist in
# rq3_prose_only_budget_ladder.jsonl if someone wants to resample them.
PROSE_TRUNCATIONS = [
    ("prose-2k", 478, 0.395, 0.0),   # from paper
    ("prose-8k", 1914, 0.710, 0.0),
    ("prose-full", 3799, 0.927, 0.0),
]

# Zero-context and prompts-only from paper
ZERO_PROMPT = [
    ("zero", 0, 0.0, 0.0),
    ("prompts-only", 202, 0.323, 0.0),
]


def load_jsonl(path: Path) -> List[dict]:
    """Load JSONL file."""
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def bootstrap_recall(rows: List[dict], grade_field: str = "grade") -> float:
    """Resample rows with replacement and compute recall."""
    n = len(rows)
    resampled = [random.choice(rows) for _ in range(n)]
    correct = sum(1 for r in resampled if r.get(grade_field) in ("correct", "partially_correct"))
    return correct / n if n > 0 else 0.0


def compute_frontier_bands(substrates: List[Tuple[str, int, float, float]],
                          f_range: List[int]) -> Dict[str, List[Tuple[int, int]]]:
    """For each F value, find which substrate is cheapest per correct answer.

    Returns dict mapping substrate name to list of (F_start, F_end) bands.
    """
    # For each F, compute cost per correct for each substrate
    cheapest_at_f = {}

    for f in f_range:
        min_cost = float('inf')
        cheapest = None

        for name, read_tokens, recall, gen_cost in substrates:
            if recall == 0:
                cost_per_correct = float('inf')
            else:
                # total(s) = read_tokens(s) + (1 - recall(s)) * F
                # cost_per_correct = total / recall
                total_tokens = read_tokens + (1 - recall) * f
                cost_per_correct = total_tokens / recall

            if cost_per_correct < min_cost:
                min_cost = cost_per_correct
                cheapest = name

        cheapest_at_f[f] = cheapest

    # Convert to bands
    bands = defaultdict(list)
    current_substrate = None
    band_start = None

    for f in sorted(f_range):
        substrate = cheapest_at_f[f]
        if substrate != current_substrate:
            if current_substrate is not None:
                bands[current_substrate].append((band_start, f - 1))
            current_substrate = substrate
            band_start = f

    # Close final band
    if current_substrate is not None:
        bands[current_substrate].append((band_start, f_range[-1]))

    return dict(bands)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=1000,
                    help="number of bootstrap samples (default 1000)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny test: 10 samples on 3 questions")
    args = ap.parse_args()

    n_samples = 10 if args.smoke else args.samples

    print(f"Bootstrap sensitivity analysis with {n_samples} samples")
    if args.smoke:
        print("SMOKE TEST: using first 3 questions only\n")

    # Load grades for each substrate
    substrate_data = []
    for name, path, read_tokens, gen_cost in SUBSTRATES:
        if not path.exists():
            print(f"  WARNING: {name} file not found, skipping")
            continue
        rows = load_jsonl(path)
        if args.smoke:
            rows = rows[:3]
        substrate_data.append((name, rows, read_tokens, gen_cost))

    # Add prose truncations and zero/prompts from paper (no resampling, fixed values)
    for name, tokens, recall, gen_cost in PROSE_TRUNCATIONS + ZERO_PROMPT:
        substrate_data.append((name, None, tokens, gen_cost))  # None = fixed recall

    # F range to test (0 to 100k in steps)
    f_range = list(range(0, 100001, 100))

    # Bootstrap loop
    print("Resampling...")
    band_boundaries = defaultdict(list)  # substrate -> list of boundary F values across samples
    band_survival = defaultdict(int)  # substrate -> count of samples where it wins any band

    for sample_idx in range(n_samples):
        if sample_idx % 100 == 0 and sample_idx > 0:
            print(f"  {sample_idx}/{n_samples} samples...")

        # Build substrate list for this sample
        sample_substrates = []
        for name, rows, read_tokens, gen_cost in substrate_data:
            if rows is None:
                # Fixed recall from paper (prose truncations, zero, prompts)
                recall = next(r for n, t, r, g in PROSE_TRUNCATIONS + ZERO_PROMPT if n == name)
                sample_substrates.append((name, read_tokens, recall, gen_cost))
            else:
                # Resample grades
                recall = bootstrap_recall(rows)
                sample_substrates.append((name, read_tokens, recall, gen_cost))

        # Compute bands for this sample
        bands = compute_frontier_bands(sample_substrates, f_range)

        # Record which substrates won any band
        for substrate in bands:
            band_survival[substrate] += 1

        # Record boundary F values
        for substrate, band_list in bands.items():
            for f_start, f_end in band_list:
                band_boundaries[substrate].append(f_start)
                band_boundaries[substrate].append(f_end)

    print(f"  {n_samples}/{n_samples} samples complete\n")

    # Report results
    print("=== BAND SURVIVAL RATE ===")
    print(f"(fraction of {n_samples} samples where substrate wins at least one band)")
    for substrate in sorted(band_survival.keys(), key=lambda s: band_survival[s], reverse=True):
        survival_rate = band_survival[substrate] / n_samples
        print(f"  {substrate:20s}  {survival_rate:5.1%}  ({band_survival[substrate]:4d}/{n_samples})")

    print("\n=== BOUNDARY STABILITY ===")
    print("(distribution of band boundary F values across bootstrap samples)")
    for substrate in sorted(band_boundaries.keys()):
        boundaries = band_boundaries[substrate]
        if len(boundaries) < 2:
            continue
        median_b = int(statistics.median(boundaries))
        p5 = int(sorted(boundaries)[int(0.05 * len(boundaries))])
        p95 = int(sorted(boundaries)[int(0.95 * len(boundaries))])
        print(f"  {substrate:20s}  median={median_b:6d}  [p5={p5:6d}, p95={p95:6d}]  n={len(boundaries)}")

    # Write JSON output
    if not args.smoke:
        output = {
            "n_samples": n_samples,
            "band_survival": dict(band_survival),
            "band_boundaries": {k: sorted(v) for k, v in band_boundaries.items()},
            "summary": {
                substrate: {
                    "survival_rate": band_survival[substrate] / n_samples,
                    "boundary_median": int(statistics.median(band_boundaries[substrate])) if band_boundaries[substrate] else None,
                    "boundary_p5": int(sorted(band_boundaries[substrate])[int(0.05 * len(band_boundaries[substrate]))]) if len(band_boundaries[substrate]) > 1 else None,
                    "boundary_p95": int(sorted(band_boundaries[substrate])[int(0.95 * len(band_boundaries[substrate]))]) if len(band_boundaries[substrate]) > 1 else None,
                }
                for substrate in band_survival.keys()
            }
        }
        out_path = DATA / "spinoff_frontier_bootstrap.json"
        out_path.write_text(json.dumps(output, indent=2))
        print(f"\nwrote {out_path}")
    else:
        print("\n--smoke: no output written")


if __name__ == "__main__":
    main()
