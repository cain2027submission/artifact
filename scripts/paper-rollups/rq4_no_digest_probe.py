#!/usr/bin/env python3
"""RQ4 probe over a NO-DIGEST ACTION RECORD substrate, next to the output-stripped one.

Reuses rq4_known_item's retrieval code unchanged (chunker, BM25, dense, scoring,
clustered bootstrap). The only new thing is one substrate arm:

  G_bm25_no_digest  - BM25 over spinoff_actionrecord_ablation.build_action_record_no_result_digests,
                      chunked like arm E/F (same candidate pool: the 550 probe sessions).
  G_dense_no_digest - dense (bge-small, local fastembed) over the same text; only with --dense-nodigest.

GATE 1: before any no-digest number is trusted, the existing arms are re-run here
(D_bm25_rollup, E_bm25_raw, F_bm25_no_tool, and dense A/B/C from the shard cache)
and must match the paper within TOL = 0.005. If not, the result file says so and
the no-digest numbers are marked not comparable.

No paid API is used: probes are read from disk, embeddings are local fastembed.

    rq4_no_digest_probe.py [--dense-control] [--dense-nodigest]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import freeze_corpus  # noqa: E402
import rq4_known_item as R  # noqa: E402
import rq4_batched as B  # noqa: E402
from spinoff_actionrecord_ablation import build_action_record_no_result_digests  # noqa: E402

DATA = HERE.parents[1] / "data" / "paper-rollups"
SNAPSHOT = DATA / "corpus_snapshot_rq4_2026-10-06.json"
PROBES = Path.home() / ".cache" / "paper-rollups" / "rq4_probes_rq4_2026-10-06.jsonl"
OUT = DATA / "rq4_no_digest_probe.json"
TOL = 0.005
POOL = 550
N_BOOT = 10_000
PAPER = {  # recall@10 as reported in paper/main.tex RQ4
    "D_bm25_rollup": 0.708, "E_bm25_raw": 0.728, "F_bm25_no_tool": 0.737,
}
PAPER_DENSE_GAP = {"A_rollup_minus_B_raw": -0.070, "A_rollup_minus_C_raw_no_tool": -0.076}
PAPER_OVERLAP = {"transcript": 0.908, "no_tool": 0.842, "rollup": 0.725}
SESSION_CACHE = Path(tempfile.gettempdir()) / "rq4_no_digest_texts"


def no_digest_text(session_id: str, manifest: dict) -> str:
    """Run the existing builder on the pinned (archived) transcript bytes."""
    tp = [p for p in manifest["transcripts"] if session_id in p][0]
    raw = freeze_corpus.read_pinned(tp, manifest)
    SESSION_CACHE.mkdir(parents=True, exist_ok=True)
    f = SESSION_CACHE / f"{session_id}.jsonl"
    f.write_text(raw)
    text, _stats = build_action_record_no_result_digests(str(f))
    return text


def ci(a, b, sids, n=N_BOOT):
    diffs = R._clustered_bootstrap_diffs(a, b, sids, n_bootstrap=n, seed=42)
    lo, hi = diffs[int(0.025 * n)], diffs[int(0.975 * n)]
    m = float(np.mean(a) - np.mean(b))
    verdict = ("no detected difference" if lo <= 0 <= hi
               else ("no-digest higher" if m > 0 else "no-digest lower"))
    return {"mean_diff": m, "ci_lower": float(lo), "ci_upper": float(hi),
            "n_bootstrap": n, "verdict": verdict}


def slim(r):
    return {k: v for k, v in r.items() if k not in ("hits_at_10", "target_session_ids")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dense-control", action="store_true", help="re-run dense A/B/C from cached shards")
    ap.add_argument("--dense-nodigest", action="store_true", help="embed the no-digest substrate (slow, local)")
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="embed only the first N no-digest chunks, print timing and shape, then exit (writes no results)")
    args = ap.parse_args()
    t_start = time.time()

    freeze_corpus.OUT = SNAPSHOT
    freeze_corpus._CACHE = None
    probes = B.load_existing_probes(PROBES)
    probe_targets = {p["target_session_id"] for p in probes}
    # Candidate pool = the 550 sessions the published run sampled (same seed and
    # sampler as rq4_batched), NOT just the probe targets: 6 pool sessions have no
    # probe, and dropping them changes the distractor set and the cached dense shards.
    probe_sessions = B.sample_sessions(POOL, 42)
    assert probe_targets <= set(probe_sessions)
    print(f"{len(probes)} probes over {len(probe_targets)} target sessions; pool {len(probe_sessions)}")
    manifest = R.load_manifest()
    session_map = R.build_session_id_map()
    sessions = []
    for sid in probe_sessions:
        s = R.load_session(sid, manifest, session_map)
        if s is None:
            raise SystemExit(f"ABORT: session {sid} failed to load")
        s["transcript_no_digest"] = no_digest_text(sid, manifest)
        sessions.append(s)
    sids = [p["target_session_id"] for p in probes]
    by_id = {s["session_id"]: s for s in sessions}

    if args.smoke:
        from fastembed import TextEmbedding
        emb = TextEmbedding(model_name=R.EMBEDDING_MODEL, providers=["CPUExecutionProvider"])
        allc = [c for s_ in sessions for c in R.chunk_text(s_["transcript_no_digest"],
                chunk_size=R.CHUNK_WORDS, overlap=R.CHUNK_OVERLAP)]
        sub = allc[:args.smoke]
        t0 = time.time()
        v = B.resumable_embed_texts(sub, emb, {}, "smoke_no_digest")
        dt = time.time() - t0
        print(f"SMOKE: {len(allc)} total chunks; embedded {len(sub)} -> shape {v.shape} dtype {v.dtype} "
              f"in {dt:.1f}s = {len(sub)/dt:.2f} chunks/s; projected full run {len(allc)/(len(sub)/dt)/60:.0f} min")
        return 0

    # ---- overlap (same definition as R.compute_fairness_stats) ----
    def overlap(key):
        vals = []
        for p in probes:
            pt = set(R.TOKEN_RE.findall(p["query"].lower()))
            tt = set(R.TOKEN_RE.findall(by_id[p["target_session_id"]][key].lower()))
            vals.append(len(pt & tt) / len(pt) if pt else 0.0)
        return float(np.mean(vals))
    overlaps = {k: overlap(v) for k, v in
                {"transcript": "transcript", "no_tool": "transcript_no_tool",
                 "rollup": "rollup_full", "no_digest": "transcript_no_digest"}.items()}
    print("overlap", overlaps)

    # ---- BM25 arms: control + new ----
    arms = {}
    for name, key, chunk in (("D_bm25_rollup", "rollup_full", False),
                             ("E_bm25_raw", "transcript", True),
                             ("F_bm25_no_tool", "transcript_no_tool", True),
                             ("G_bm25_no_digest", "transcript_no_digest", True)):
        t0 = time.time()
        idx = R.build_bm25_index(sessions, key, chunk=chunk)
        assert idx["num_distinct_sessions"] == len(probe_sessions)
        arms[name] = R.evaluate_arm(probes, idx, name, None, {})
        arms[name]["build_time_s"] = time.time() - t0
        print(f"{name}: chunks {idx['num_chunks']} recall@10 {arms[name]['recall@10']:.4f}")

    gate = {"tolerance": TOL, "bm25": {}, "dense": {}}
    ok = True
    for name, paper in PAPER.items():
        got = arms[name]["recall@10"]
        gate["bm25"][name] = {"paper": paper, "reproduced": got, "abs_err": abs(got - paper),
                              "pass": abs(got - paper) <= TOL}
        ok &= gate["bm25"][name]["pass"]

    # ---- dense ----
    embedder = None
    if args.dense_control or args.dense_nodigest:
        from fastembed import TextEmbedding
        embedder = TextEmbedding(model_name=R.EMBEDDING_MODEL, providers=["CPUExecutionProvider"])
        R.embed_texts = B.resumable_embed_texts
    dense_cache: dict = {}
    if args.dense_control:
        for name, key, prefix, chunk in (("A_rollup", "rollup_full", "arm_a_rollup", False),
                                         ("B_raw", "transcript", "arm_b_raw", True),
                                         ("C_raw_no_tool", "transcript_no_tool", "arm_c_no_tool", True)):
            idx = R.build_dense_index(sessions, key, embedder, dense_cache, prefix, chunk=chunk)
            arms[name] = R.evaluate_arm(probes, idx, name, embedder, dense_cache)
            print(f"{name}: recall@10 {arms[name]['recall@10']:.4f}")
        for gname, (a, b) in {"A_rollup_minus_B_raw": ("A_rollup", "B_raw"),
                              "A_rollup_minus_C_raw_no_tool": ("A_rollup", "C_raw_no_tool")}.items():
            got = arms[a]["recall@10"] - arms[b]["recall@10"]
            gate["dense"][gname] = {"paper": PAPER_DENSE_GAP[gname], "reproduced": got,
                                    "abs_err": abs(got - PAPER_DENSE_GAP[gname]),
                                    "pass": abs(got - PAPER_DENSE_GAP[gname]) <= TOL}
            ok &= gate["dense"][gname]["pass"]
    else:
        ok = False
        gate["dense"] = "not run (--dense-control not given)"
    if args.dense_nodigest:
        idx = R.build_dense_index(sessions, "transcript_no_digest", embedder, dense_cache,
                                  "arm_g_no_digest", chunk=True)
        arms["G_dense_no_digest"] = R.evaluate_arm(probes, idx, "G_dense_no_digest", embedder, dense_cache)
        print(f"G_dense_no_digest: recall@10 {arms['G_dense_no_digest']['recall@10']:.4f}")

    gate["passed"] = bool(ok)
    # overlap control: stripped/raw/rollup overlaps must also match
    gate["overlap"] = {k: {"paper": v, "reproduced": overlaps[k], "pass": abs(overlaps[k] - v) <= 0.005}
                       for k, v in PAPER_OVERLAP.items()}

    comps = {}
    for g, f in (("G_bm25_no_digest", "F_bm25_no_tool"), ("G_dense_no_digest", "C_raw_no_tool"),
                 ("G_dense_no_digest", "B_raw"), ("G_dense_no_digest", "A_rollup")):
        if g in arms and f in arms:
            comps[f"{g}_minus_{f}"] = ci(arms[g]["hits_at_10"], arms[f]["hits_at_10"], sids)
    if "G_bm25_no_digest" in arms:
        for o in ("E_bm25_raw", "D_bm25_rollup"):
            comps[f"G_bm25_no_digest_minus_{o}"] = ci(arms["G_bm25_no_digest"]["hits_at_10"],
                                                      arms[o]["hits_at_10"], sids)

    out = {
        "experiment": "rq4_no_digest_probe",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "num_probes": len(probes), "num_sessions": len(probe_targets), "candidate_pool_sessions": len(probe_sessions),
        "probes_file": str(PROBES), "snapshot": str(SNAPSHOT),
        "no_digest_builder": "spinoff_actionrecord_ablation.build_action_record_no_result_digests",
        "retrieval": "rq4_known_item (unchanged); BM25 k1=1.5 b=0.75, 280-word chunks, max-pool to session",
        "gate1": gate,
        "gate1_verdict": ("PASS: control reproduces paper within tolerance" if gate["passed"]
                          else "NOT PASSED or incomplete: no-digest numbers are not comparable"),
        "overlap": overlaps,
        "arms": {k: slim(v) for k, v in arms.items()},
        "paired_bootstrap_no_digest_minus_other": comps,
        "dense_nodigest_ran": bool(args.dense_nodigest),
        "dense_nodigest_note": (None if args.dense_nodigest else
            "skipped: 35,918 chunks at the measured local rate of ~6.4 chunks/s is ~94 min, over the "
            "20-minute ceiling set for this run. Local fastembed, no paid API; just slow."),
        "runtime_s": time.time() - t_start,
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps({"gate1": gate, "comparisons": comps, "overlap": overlaps}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
