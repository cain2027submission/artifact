#!/usr/bin/env python3
"""Back-of-envelope probe-window coverage model for the benchmark fragility paper.

The probe-window bias claim: RQ3 questions were generated from the first 30,000
characters of transcripts whose median length is 691 KB. Verbatim-derived arms
retain that probed prefix nearly intact while a rollup summary spreads its
budget over the whole session, so arms differ in coverage of the region being
tested as well as in volume and content class.

This script estimates how much of the observed rollup-to-transcript accuracy gap
the coverage difference plausibly accounts for, with stated assumptions.
"""

import json
from pathlib import Path

# Constants from the paper
PROBE_WINDOW_CHARS = 30_000
MEDIAN_TRANSCRIPT_KB = 691
MEDIAN_TRANSCRIPT_CHARS = MEDIAN_TRANSCRIPT_KB * 1024  # 707,584 chars

# From the paper's measurements
ROLLUP_TOKENS = 3_799  # prose-only
ROLLUP_CHARS_APPROX = ROLLUP_TOKENS * 4  # ~15,196 chars at 4 chars/token

# Accuracy figures from the paper (from rq3_headline_corrected.json)
# Using recoverable (correct + partially_correct) from the published pooled set
ROLLUP_ACCURACY = 0.565  # 56.5%
PROSE_ONLY_ACCURACY = 0.927  # 92.7% (verbatim-derived, retains prefix)
GAP = PROSE_ONLY_ACCURACY - ROLLUP_ACCURACY  # 36.2 percentage points


def main():
    print("=" * 70)
    print("PROBE-WINDOW COVERAGE MODEL")
    print("=" * 70)

    print("\n## STATED FACTS")
    print(f"  Probe window:          {PROBE_WINDOW_CHARS:,} characters")
    print(f"  Median transcript:     {MEDIAN_TRANSCRIPT_KB} KB = {MEDIAN_TRANSCRIPT_CHARS:,} chars")
    print(f"  Probe as % of median:  {100 * PROBE_WINDOW_CHARS / MEDIAN_TRANSCRIPT_CHARS:.1f}%")
    print(f"  Rollup size:           ~{ROLLUP_CHARS_APPROX:,} chars ({ROLLUP_TOKENS} tokens)")
    print(f"  Prose-only accuracy:   {100 * PROSE_ONLY_ACCURACY:.1f}%")
    print(f"  Rollup accuracy:       {100 * ROLLUP_ACCURACY:.1f}%")
    print(f"  Gap to explain:        {100 * GAP:.1f} percentage points")

    print("\n## COVERAGE OF THE PROBED REGION")
    print("  Assumption: verbatim arms (prose-only, output-stripped, raw) retain")
    print("  the first 30k characters nearly intact.")
    verbatim_coverage = 1.0  # 100% of probed region
    print(f"    Verbatim coverage of probed region:  {100 * verbatim_coverage:.0f}%")

    print("\n  Assumption: rollup summary spreads its budget uniformly over the")
    print("  whole session (or concentrates on salient regions, which may not")
    print("  be the chronological start).")
    rollup_coverage = ROLLUP_CHARS_APPROX / MEDIAN_TRANSCRIPT_CHARS
    print(f"    Rollup coverage of probed region:    {100 * rollup_coverage:.1f}%")
    print(f"      (if uniformly distributed; {ROLLUP_CHARS_APPROX:,} / {MEDIAN_TRANSCRIPT_CHARS:,})")

    coverage_diff = verbatim_coverage - rollup_coverage
    print(f"\n  Coverage difference:  {100 * coverage_diff:.1f} percentage points")

    print("\n## BACK-OF-ENVELOPE ESTIMATE")
    print("  If accuracy tracks coverage linearly (a strong simplifying assumption),")
    print("  the coverage difference alone would account for:")

    # Linear model: accuracy ≈ k * coverage
    # At verbatim_coverage=1.0, accuracy=0.927
    # At rollup_coverage, accuracy would be:
    predicted_rollup_acc = PROSE_ONLY_ACCURACY * rollup_coverage
    explained_gap = PROSE_ONLY_ACCURACY - predicted_rollup_acc

    print(f"    Predicted rollup accuracy: {100 * predicted_rollup_acc:.1f}%")
    print(f"    Predicted gap:             {100 * explained_gap:.1f} percentage points")
    print(f"    Observed gap:              {100 * GAP:.1f} percentage points")
    print(f"    Explained fraction:        {100 * explained_gap / GAP:.0f}%")

    print("\n  CAVEAT: This is a lower bound if the rollup concentrates on important")
    print("  content rather than distributing uniformly. If the rollup captures")
    print("  ~50% of the probed region's content (e.g., key outcomes, decisions),")
    print("  the coverage model would explain less:")

    for rollup_coverage_alt in [0.5, 0.3, 0.1]:
        predicted = PROSE_ONLY_ACCURACY * rollup_coverage_alt
        gap_alt = PROSE_ONLY_ACCURACY - predicted
        print(f"    If rollup captures {100*rollup_coverage_alt:.0f}% of probed region:")
        print(f"      Explained gap: {100*gap_alt:.1f}pp ({100*gap_alt/GAP:.0f}% of observed)")

    print("\n## STATED ASSUMPTIONS")
    print("  1. Verbatim arms retain the probed prefix at ~100% fidelity.")
    print("  2. Rollup distributes content uniformly (or near-uniformly) over the")
    print("     whole session, not concentrating on the chronological start.")
    print("  3. Accuracy scales linearly with coverage of the tested region.")
    print("  4. The median transcript length (691 KB) is representative of the")
    print("     RQ3 sample, which was filtered to >=100 KB.")
    print("  5. No other factors (content class, compression artifacts) contribute")
    print("     to the gap — this isolates the coverage effect only.")

    print("\n## VERDICT")
    print(f"  Under the stated assumptions, the probe-window coverage difference")
    print(f"  ({100*coverage_diff:.1f} percentage points) plausibly accounts for")
    print(f"  {100*explained_gap/GAP:.0f}% of the observed {100*GAP:.1f}pp rollup-to-transcript gap.")
    print(f"  This is a back-of-envelope estimate, not a controlled measurement.")
    print("=" * 70)

    # Write output
    output = {
        "model": "probe_window_coverage",
        "probe_window_chars": PROBE_WINDOW_CHARS,
        "median_transcript_chars": MEDIAN_TRANSCRIPT_CHARS,
        "rollup_chars_approx": ROLLUP_CHARS_APPROX,
        "probe_fraction_of_median": round(PROBE_WINDOW_CHARS / MEDIAN_TRANSCRIPT_CHARS, 4),
        "verbatim_coverage_of_probed": verbatim_coverage,
        "rollup_coverage_of_probed_uniform": round(rollup_coverage, 4),
        "coverage_difference": round(coverage_diff, 4),
        "accuracy": {
            "prose_only": PROSE_ONLY_ACCURACY,
            "rollup": ROLLUP_ACCURACY,
            "observed_gap_pp": round(100 * GAP, 1),
        },
        "linear_model": {
            "predicted_rollup_accuracy": round(predicted_rollup_acc, 3),
            "predicted_gap_pp": round(100 * explained_gap, 1),
            "explained_fraction": round(explained_gap / GAP, 2),
        },
        "assumptions": [
            "Verbatim arms retain probed prefix at ~100% fidelity",
            "Rollup distributes content uniformly over whole session",
            "Accuracy scales linearly with coverage",
            "691 KB median is representative of >=100 KB sample",
            "No other factors contribute (content class, artifacts)",
        ],
        "verdict": f"Coverage difference plausibly accounts for {100*explained_gap/GAP:.0f}% of gap under linear model"
    }

    out_path = Path(__file__).parent.parent.parent / "data" / "paper-rollups" / "probe_window_coverage.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nWrote {out_path.name}")


if __name__ == "__main__":
    main()
