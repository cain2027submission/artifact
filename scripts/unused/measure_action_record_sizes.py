#!/usr/bin/env python3
"""Measure action record sizes for the 25 RQ3 sessions.

Compares action record size against verbatim and stripped baselines.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from action_record import build_action_record
from paths import JOBS_DIR  # noqa: E402
from rq3_raw_transcript_arm import index_transcripts, resolve

# The 25 RQ3 sessions
SESSIONS = [
    "1327c110", "37d5326d", "3b5fdb29", "4d4c46f6", "53f752ef",
    "57153a94", "6d7b783e", "714318cd", "757d3aaf", "88f820e3",
    "9576c5a8", "98a92be4", "9d4c165a", "a596be9f", "a8273208",
    "ab7bad8a", "b766c7ce", "b9c67733", "bc746e37", "c99a79d3",
    "d7aa4f50", "e7486f80", "f4a20cd4", "f5b8c8d0", "f8828ed8",
]

# Load verbatim and stripped sizes from existing arms
VERBATIM_PATH = Path(__file__).parent.parent.parent / "data" / "paper-rollups" / "rq3_verbatim_arm.jsonl"
STRIPPED_PATH = Path(__file__).parent.parent.parent / "data" / "paper-rollups" / "rq3_questions_stripped_arm.jsonl"


def load_baseline_sizes():
    """Load verbatim and stripped sizes per session from existing arms."""
    verbatim_sizes = {}
    if VERBATIM_PATH.exists():
        for line in VERBATIM_PATH.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            prefix = r.get("session_id", "").split("_")[2] if "_" in r.get("session_id", "") else ""
            if prefix and prefix in SESSIONS:
                verbatim_sizes[prefix] = r.get("verbatim_chars", 0)

    stripped_sizes = {}
    if STRIPPED_PATH.exists():
        for line in STRIPPED_PATH.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            sid = r.get("session_id", "")
            prefix = sid.split("_")[2] if "_" in sid else ""
            if prefix and prefix in SESSIONS:
                stripped_sizes[prefix] = r.get("stripped_chars", 0)

    # Deduplicate: take one record per session (they all have the same char count)
    verbatim_unique = {p: max((v for k, v in verbatim_sizes.items() if k == p), default=0) for p in SESSIONS}
    stripped_unique = {p: max((v for k, v in stripped_sizes.items() if k == p), default=0) for p in SESSIONS}

    return verbatim_unique, stripped_unique


def main():
    by_prefix = index_transcripts()
    verbatim_sizes, stripped_sizes = load_baseline_sizes()

    results = []
    for prefix in SESSIONS:
        path = resolve(prefix, by_prefix)
        if path is None:
            print(f"WARNING: {prefix} not found", file=sys.stderr)
            continue

        try:
            record, stats = build_action_record(str(path))
            action_chars = len(record)
            verbatim_chars = verbatim_sizes.get(prefix, 0)
            stripped_chars = stripped_sizes.get(prefix, 0)

            results.append({
                "prefix": prefix,
                "transcript_bytes": path.stat().st_size,
                "action_chars": action_chars,
                "verbatim_chars": verbatim_chars,
                "stripped_chars": stripped_chars,
                "stats": stats,
            })
            print(f"{prefix} -> {action_chars:,} chars "
                  f"(verbatim: {verbatim_chars:,}, stripped: {stripped_chars:,})",
                  file=sys.stderr)
        except Exception as e:
            print(f"ERROR building {prefix}: {e}", file=sys.stderr)

    # Sort by action_chars descending
    results.sort(key=lambda r: -r["action_chars"])

    # Print table
    print(f"\n{'prefix':10s} {'raw MB':>8s} {'action':>10s} {'verbatim':>10s} {'stripped':>10s} "
          f"{'action/verb':>10s} {'action/strip':>11s}")
    print("-" * 82)

    for r in results:
        raw_mb = r["transcript_bytes"] / 1e6
        act = r["action_chars"]
        verb = r["verbatim_chars"]
        strip = r["stripped_chars"]
        ratio_v = f"{act/verb:.2f}x" if verb > 0 else "n/a"
        ratio_s = f"{act/strip:.2f}x" if strip > 0 else "n/a"
        print(f"{r['prefix']:10s} {raw_mb:8.2f} {act:10,} {verb:10,} {strip:10,} "
              f"{ratio_v:>10s} {ratio_s:>11s}")

    # Summary stats
    action_vals = [r["action_chars"] for r in results]
    verbatim_vals = [r["verbatim_chars"] for r in results if r["verbatim_chars"] > 0]
    stripped_vals = [r["stripped_chars"] for r in results if r["stripped_chars"] > 0]

    action_vals.sort()
    verbatim_vals.sort()
    stripped_vals.sort()

    def median(vals):
        if not vals:
            return 0
        n = len(vals)
        if n % 2 == 1:
            return vals[n // 2]
        return (vals[n // 2 - 1] + vals[n // 2]) / 2

    def p90(vals):
        if not vals:
            return 0
        idx = int(len(vals) * 0.9)
        return vals[min(idx, len(vals) - 1)]

    print(f"\n{'':10s} {'median':>10s} {'p90':>10s} {'max':>10s}")
    print("-" * 42)
    print(f"{'action':10s} {median(action_vals):10,.0f} {p90(action_vals):10,.0f} {max(action_vals):10,}")
    print(f"{'verbatim':10s} {median(verbatim_vals):10,.0f} {p90(verbatim_vals):10,.0f} {max(verbatim_vals):10,}")
    print(f"{'stripped':10s} {median(stripped_vals):10,.0f} {p90(stripped_vals):10,.0f} {max(stripped_vals):10,}")

    print(f"\naction / verbatim median ratio: {median(action_vals) / median(verbatim_vals):.2f}x")
    print(f"action / stripped median ratio: {median(action_vals) / median(stripped_vals):.2f}x")

    # Save full example of ONE session to tmp/
    tmp_dir = JOBS_DIR / "2919c040" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Pick a medium-sized one for the example (not the smallest, not the largest)
    mid_idx = len(results) // 2
    example = results[mid_idx]
    example_path = resolve(example["prefix"], by_prefix)
    example_record, _ = build_action_record(str(example_path))

    example_file = tmp_dir / f"action_record_example_{example['prefix']}.md"
    example_file.write_text(example_record)
    print(f"\nFull example saved to: {example_file}")

    # Print ~40 lines of it
    lines = example_record.split("\n")
    sample = "\n".join(lines[:40])
    print(f"\nFirst 40 lines of example ({example['prefix']}):")
    print("=" * 80)
    print(sample)
    print("=" * 80)


if __name__ == "__main__":
    main()
