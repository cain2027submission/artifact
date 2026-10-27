#!/usr/bin/env python3
"""RQ4: Missing cell - BM25 over output-stripped transcripts.

This is the F_bm25_raw_no_tool arm: BM25 retrieval over chunked transcripts with
tool_use and tool_result blocks removed (same substrate as arm C, same retrieval
as arm E). The paper identified this as the most attackable methodological gap.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import freeze_corpus  # noqa: E402
import rq4_known_item as R  # noqa: E402

SEED = 42  # recorded in the output only; sessions come from the probes, not a sample


def load_probes(path: Path) -> list[dict]:
    """Load probes from jsonl file."""
    probes = []
    for line in path.read_text().splitlines():
        if line.strip():
            probes.append(json.loads(line))
    return probes


def main():
    start_time = time.time()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", required=True, help="probe jsonl; its target sessions ARE the candidate pool")
    ap.add_argument("--out", required=True, help="output json (refuses to overwrite an existing file)")
    ap.add_argument(
        "--snapshot", default=None,
        help="corpus snapshot json (e.g. corpus_snapshot_rq4_*.json, which archives rollup+transcript "
             "bytes). Unset: the shared corpus_snapshot.json, which drops sessions whose rollup "
             "note was renamed since it was frozen -- the postcondition then fails loudly.",
    )
    args = ap.parse_args()
    probes_path = Path(args.probes).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if args.snapshot:
        freeze_corpus.OUT = Path(args.snapshot).expanduser().resolve()
        freeze_corpus._CACHE = None

    print("=" * 80)
    print("RQ4 Missing Cell: BM25 over output-stripped transcripts")
    print("=" * 80)
    print(f"Probes:   {probes_path}")
    print(f"Snapshot: {freeze_corpus.OUT}")
    print(f"Output:   {out_path}")
    if out_path.exists():
        raise SystemExit(f"ABORT: {out_path} already exists; refusing to overwrite")

    probes = load_probes(probes_path)
    probe_sessions = sorted({p["target_session_id"] for p in probes})
    print(f"  {len(probes)} probes over {len(probe_sessions)} distinct sessions")
    if not probes:
        raise SystemExit("ABORT: no probes")

    manifest = R.load_manifest()
    not_in_snapshot = sorted(set(probe_sessions) - set(manifest["paired_session_ids"]))
    if not_in_snapshot:
        print(f"ABORT: {len(not_in_snapshot)} probe sessions not in snapshot: {not_in_snapshot}")
        return 1

    session_map = R.build_session_id_map()
    print(f"  Mapped {len(session_map)} rollups to session IDs")

    # Every probe session must load. A session that cannot load is an error, never a skip.
    sessions, failed = [], []
    for session_id in probe_sessions:
        sess = R.load_session(session_id, manifest, session_map)
        if sess is None:
            failed.append(session_id)
        else:
            sessions.append(sess)
    print(f"  Loaded {len(sessions)}/{len(probe_sessions)} sessions")
    if failed:
        print(f"ABORT: {len(failed)} sessions failed to load (no valid rollup pin / transcript): {failed}")
        return 1

    # Build arm F: BM25 over transcript_no_tool, chunked like arm E
    print("\n" + "="*80)
    print("Building F_bm25_raw_no_tool...")
    print("  Substrate: transcript_no_tool (tool_use/tool_result removed)")
    print("  Retrieval: BM25")
    print(f"  Chunking: {R.CHUNK_WORDS} words, overlap {R.CHUNK_OVERLAP}")
    print("="*80)

    t0 = time.time()
    arm_f = R.build_bm25_index(sessions, "transcript_no_tool", chunk=True)
    arm_f["build_time_s"] = time.time() - t0

    # Postcondition: the index must hold exactly the sessions the probes target.
    indexed = set(arm_f["chunk_to_session"])
    missing = sorted(set(probe_sessions) - indexed)
    extra = sorted(indexed - set(probe_sessions))
    print(f"  Sessions indexed: {len(indexed)} (probes target {len(probe_sessions)})")
    if missing or extra or len(indexed) != len(probe_sessions):
        print(f"ABORT: candidate pool mismatch. missing from index: {missing}; unexpected in index: {extra}")
        return 1

    print(f"\nArm F built in {arm_f['build_time_s']:.2f}s")
    print(f"  Chunks: {arm_f['num_chunks']:,}")
    print(f"  Tokens: {arm_f['total_tokens']:,} ({arm_f['total_tokens']/1e6:.2f}M)")

    # Evaluate arm F
    print("\nEvaluating F_bm25_raw_no_tool...")
    # Need a minimal cache dict for the retrieve functions
    cache = {}
    results_f = R.evaluate_arm(probes, arm_f, "F_bm25_raw_no_tool", None, cache)
    results_f["build_time_s"] = arm_f["build_time_s"]

    # Build output
    output = {
        "experiment": "rq4_bm25_no_tool",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "seed": SEED,
        "num_sessions": len(sessions),
        "num_sessions_indexed": len(indexed),
        "probes_file": str(probes_path),
        "snapshot": str(freeze_corpus.OUT),
        "num_probes": len(probes),
        "arm": {
            "F_bm25_raw_no_tool": results_f
        },
        "runtime_s": time.time() - start_time,
    }

    # Save results
    print(f"\nSaving results to {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True))

    # Print summary
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nF_bm25_raw_no_tool:")
    print(f"  Recall@1:  {results_f['recall@1']:.3f}")
    print(f"  Recall@5:  {results_f['recall@5']:.3f}")
    print(f"  Recall@10: {results_f['recall@10']:.3f}")
    print(f"  MRR@10:    {results_f['mrr@10']:.3f}")
    print(f"\nIndex:")
    print(f"  Chunks:     {results_f['num_chunks']:,}")
    print(f"  Tokens:     {results_f['total_tokens']:,} ({results_f['total_tokens']/1e6:.2f}M)")
    print(f"  Build time: {results_f['build_time_s']:.2f}s")

    # Compare with published arms D and E
    print("\n" + "="*80)
    print("COMPARISON WITH PUBLISHED ARMS")
    print("="*80)
    print("Recall@1, Recall@5, Recall@10, MRR@10, Tokens(M), Build(s)")
    print(f"D (BM25+rollup):     0.472 / 0.698 / 0.768 / 0.569, 0.22M, 0.1s")
    print(f"E (BM25+raw):        0.430 / 0.707 / 0.785 / 0.545, 23.3M, 9s")
    print(f"F (BM25+no_tool):    {results_f['recall@1']:.3f} / {results_f['recall@5']:.3f} / {results_f['recall@10']:.3f} / {results_f['mrr@10']:.3f}, {results_f['total_tokens']/1e6:.2f}M, {results_f['build_time_s']:.1f}s")

    # Compute differences
    print(f"\nF vs D (rollup):")
    print(f"  Recall@10: {results_f['recall@10']:.3f} vs 0.768 = {results_f['recall@10'] - 0.768:+.3f}")
    print(f"  MRR@10:    {results_f['mrr@10']:.3f} vs 0.569 = {results_f['mrr@10'] - 0.569:+.3f}")

    print(f"\nF vs E (raw):")
    print(f"  Recall@10: {results_f['recall@10']:.3f} vs 0.785 = {results_f['recall@10'] - 0.785:+.3f}")
    print(f"  MRR@10:    {results_f['mrr@10']:.3f} vs 0.545 = {results_f['mrr@10'] - 0.545:+.3f}")

    print(f"\nRuntime: {output['runtime_s']:.1f}s")
    print("\n" + "="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
