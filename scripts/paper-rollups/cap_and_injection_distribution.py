#!/usr/bin/env python3
"""
Analyze prompt-context telemetry for candidate C points 1 and 3.

POINT 1: Distribution of injected_chars and body_chars relative to caps
POINT 3: Per-artifact injection frequency distribution
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import TELEMETRY_LOG  # noqa: E402


def load_records(path):
    """Load JSONL records from file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def check_overlap(records1, records2):
    """Check if record sets are disjoint or overlapping based on ts."""
    ts1 = {r.get('ts') for r in records1 if 'ts' in r}
    ts2 = {r.get('ts') for r in records2 if 'ts' in r}
    overlap = ts1 & ts2
    return len(overlap), len(ts1), len(ts2)


def percentile(values, p):
    """Calculate percentile p (0-100) of sorted values."""
    if not values:
        return None
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[-1]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def histogram_near_ceiling(values, ceiling, window=50):
    """Count values in bins near the ceiling."""
    bins = defaultdict(int)
    for v in values:
        if v >= ceiling - window:
            bins[v] += 1
    return dict(sorted(bins.items()))


def point1_analysis(records):
    """Analyze injected_chars and body_chars distributions."""
    print("\n=== POINT 1: Cap distribution analysis ===\n")

    # Extract injected_chars (treating absent as distinct from 0)
    injected = [r['injected_chars'] for r in records if 'injected_chars' in r]
    print(f"injected_chars: n={len(injected)} records WITH field (out of {len(records)} total)")

    if injected:
        print(f"  min={min(injected)}, median={percentile(injected, 50):.0f}, "
              f"p90={percentile(injected, 90):.0f}, p95={percentile(injected, 95):.0f}, "
              f"max={max(injected)}")

        # Determine actual ceiling from data
        max_val = max(injected)
        print(f"  Actual ceiling in data: {max_val}")

        # Count how many are AT the cap (within 2 chars)
        at_cap = sum(1 for v in injected if v >= max_val - 2)
        print(f"  At cap (>= {max_val - 2}): {at_cap}/{len(injected)} = {100*at_cap/len(injected):.1f}%")

        # Histogram near ceiling
        hist = histogram_near_ceiling(injected, max_val, window=50)
        if hist:
            print(f"  Histogram near ceiling (last 50 chars):")
            for val, count in list(hist.items())[:10]:  # Show first 10 bins
                print(f"    {val}: {count}")

    # Extract body_chars
    body = [r['body_chars'] for r in records if 'body_chars' in r]
    print(f"\nbody_chars: n={len(body)} records WITH field (out of {len(records)} total)")

    if body:
        print(f"  min={min(body)}, median={percentile(body, 50):.0f}, "
              f"p90={percentile(body, 90):.0f}, p95={percentile(body, 95):.0f}, "
              f"max={max(body)}")

        # Test against 500 cap
        cap_500 = 500
        at_cap = sum(1 for v in body if v >= cap_500 - 2)
        print(f"  At cap (>= {cap_500 - 2}): {at_cap}/{len(body)} = {100*at_cap/len(body):.1f}%")

        # Histogram near 500
        hist = histogram_near_ceiling(body, cap_500, window=50)
        if hist:
            print(f"  Histogram near 500 ceiling:")
            for val, count in list(hist.items())[:6]:
                print(f"    {val}: {count}")

    # Check for pre-truncation size field
    print(f"\nDiscard fraction: ", end="")
    has_pretrunc = any('pre_truncation_chars' in r or 'original_chars' in r
                       or 'truncated_from' in r for r in records)
    if has_pretrunc:
        print("Found pre-truncation field, computing...")
    else:
        print("Not derivable from telemetry (no pre-truncation size field found)")


def point3_analysis(records):
    """Analyze per-artifact injection frequency."""
    print("\n\n=== POINT 3: Per-artifact injection frequency ===\n")

    # Find time window
    timestamps = [r['ts'] for r in records if 'ts' in r]
    if timestamps:
        print(f"Telemetry window: {min(timestamps)} to {max(timestamps)}")

    # Analyze hits field for vault/rollup notes
    hits_records = [r for r in records if 'hits' in r]
    print(f"\nRecords with 'hits' field: {len(hits_records)}/{len(records)}")

    # Sample hits structure
    if hits_records:
        sample_hits = hits_records[0]['hits']
        print(f"Sample 'hits' type: {type(sample_hits)}")
        if isinstance(sample_hits, list) and sample_hits:
            print(f"  First hit type: {type(sample_hits[0])}")
            print(f"  First hit sample: {str(sample_hits[0])[:200]}")

    # Check payload_keys (turns out to be metadata field names, not note identifiers)
    payload_records = [r for r in records if 'payload_keys' in r]
    print(f"\nRecords with 'payload_keys' field: {len(payload_records)}/{len(records)}")

    if payload_records:
        sample_payload = payload_records[0]['payload_keys']
        print(f"Sample 'payload_keys': {sample_payload}")
        print(f"  => These are metadata field names, NOT note identifiers")

    # Check if span field might identify individual notes
    span_records = [r for r in records if 'span' in r and 'memory_slug' not in r]
    print(f"\nVault note tracking: Records with 'span' (non-memory): {len(span_records)}/{len(records)}")
    print("  => Telemetry does not track individual vault/rollup note identities")
    print("  => Cannot compute per-vault-note injection frequency from available fields")

    # Memory notes (separate analysis)
    memory_records = [r for r in records if 'memory_slug' in r]
    print(f"\nMemory note records: {len(memory_records)}/{len(records)}")

    memory_counts = Counter()
    for r in memory_records:
        slug = r['memory_slug']
        memory_counts[slug] += 1

    if memory_counts:
        counts = list(memory_counts.values())
        print(f"Memory artifact injection distribution:")
        print(f"  Distinct memory artifacts: {len(memory_counts)}")
        print(f"  Median injections per artifact: {percentile(counts, 50):.0f}")
        print(f"  p90: {percentile(counts, 90):.0f}")
        print(f"  Max: {max(counts)}")
        injected_once = sum(1 for c in counts if c == 1)
        print(f"  Injected exactly once: {injected_once}/{len(memory_counts)} = "
              f"{100*injected_once/len(memory_counts):.1f}%")

    print("\nNote: Zero-injection count requires eligible artifact inventory, which is not in telemetry.")


def main():
    main_file = TELEMETRY_LOG
    real_file = TELEMETRY_LOG.parent / "prompt-context-telemetry.real-through-20260927T0915.jsonl"

    print("Loading telemetry files...")
    main_records = load_records(main_file)
    real_records = load_records(real_file)

    print(f"Loaded {len(main_records)} records from main file")
    print(f"Loaded {len(real_records)} records from real-through file")

    # Check overlap
    overlap_count, ts1_count, ts2_count = check_overlap(main_records, real_records)
    print(f"\nOverlap check: {overlap_count} timestamps appear in both files")
    print(f"  Main file unique timestamps: {ts1_count}")
    print(f"  Real-through file timestamps: {ts2_count}")

    if overlap_count == ts2_count and ts2_count > 0:
        print("  => Real-through file is a SUBSET of main file")
        records = main_records
    elif overlap_count == 0:
        print("  => Files are DISJOINT")
        records = main_records + real_records
    else:
        print("  => Files have PARTIAL overlap")
        # Use main file only to avoid double-counting
        records = main_records

    print(f"\nAnalyzing {len(records)} total records")

    # Smoke test on first 50
    print("\n--- SMOKE TEST on first 50 records ---")
    point1_analysis(records[:50])
    point3_analysis(records[:50])

    # Full analysis
    print("\n\n" + "="*60)
    print("FULL ANALYSIS")
    print("="*60)
    point1_analysis(records)
    point3_analysis(records)


if __name__ == "__main__":
    main()
