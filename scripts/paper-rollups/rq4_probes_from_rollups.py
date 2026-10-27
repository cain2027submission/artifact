#!/usr/bin/env python3
"""RQ4 probe-bias check: known-item probes generated from the ROLLUP text, BM25 arms rerun.

Plan: data/paper-rollups/probes-from-rollups/analysis_plan.md (post hoc, exploratory).

  rq4_probes_from_rollups.py control                run D/E/F/G BM25 on the original 1,087 probes (gate)
  rq4_probes_from_rollups.py generate --n 10        smoke test, prints the spend projection
  rq4_probes_from_rollups.py generate               generate for the pool under the USD cap
  rq4_probes_from_rollups.py analyse                retrieval + CIs + overlap -> results.json
  rq4_probes_from_rollups.py analyse --dense        dense recall (cached shards, local) -> results.json `dense` key

Retrieval code is rq4_known_item / rq4_no_digest_probe, unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import freeze_corpus  # noqa: E402
import rq4_known_item as R  # noqa: E402
import rq4_batched as B  # noqa: E402
import rq4_no_digest_probe as ND  # noqa: E402

OUTDIR = HERE.parents[1] / "data" / "paper-rollups" / "probes-from-rollups"
PROBES = OUTDIR / "probes.jsonl"
COST_LOG = OUTDIR / "cost_log.jsonl"
RESULTS = OUTDIR / "results.json"
SKIPPED = OUTDIR / "skipped_sessions.json"
CAP_USD = 3.0
PRICE_IN, PRICE_OUT = 3.0 / 1e6, 15.0 / 1e6
BATCH = 25
POOL, SEED = 550, 42
ARMS = (("D_bm25_rollup", "rollup_full", False), ("E_bm25_raw", "transcript", True),
        ("F_bm25_no_tool", "transcript_no_tool", True), ("G_bm25_no_digest", "transcript_no_digest", True))

PROMPT = """You are helping build a retrieval benchmark. Below is a slice from an agent session rollup note (a written summary of the session).

Generate EXACTLY 2 search queries that a developer would TYPE LATER to find this session again.

Requirements:
- Queries should be 4-8 words, keyword-shaped
- Phrase them as a developer would naturally search, not quoting verbatim strings
- Capture the INTENT and TOPIC, not exact wording
- Think: "How would I search for this session if I remembered the problem but not the exact words?"

Examples of GOOD queries:
- "fix pytest import error macos"
- "git worktree remove classifier denied"
- "token burn measurement script"

Examples of BAD queries (too verbatim):
- "trace_path cost_usd repo path"
- "and sweep up only"
- exact code snippets or error messages copied verbatim

Transcript slice:
{slice}

Return ONLY a JSON array of 2 query strings, nothing else:"""


def log_cost(rec: dict) -> float:
    with COST_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec["usd"]


def spent() -> float:
    if not COST_LOG.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in COST_LOG.read_text().splitlines() if l.strip())


def paid_call(prompt: str, sid: str, tag: str) -> list[str]:
    """One gateway call; a failed call is retried once, every attempt logged."""
    for attempt in (1, 2):
        try:
            data = R.post_json(f"{R.GATEWAY}/v1/chat/completions",
                               {"model": R.PROBE_MODEL, "max_tokens": 1024,
                                "messages": [{"role": "user", "content": prompt}]}, timeout=120)
            u = data.get("usage", {})
            usd = u.get("prompt_tokens", 0) * PRICE_IN + u.get("completion_tokens", 0) * PRICE_OUT
            log_cost({"session_id": sid, "tag": tag, "attempt": attempt, "ok": True,
                      "prompt_tokens": u.get("prompt_tokens", 0),
                      "completion_tokens": u.get("completion_tokens", 0), "usd": usd})
            txt = data["choices"][0]["message"]["content"].strip()
            if "[" in txt and "]" in txt:
                try:
                    q = json.loads(txt[txt.index("["):txt.rindex("]") + 1])
                    return [str(x).strip() for x in q[:2] if str(x).strip()]
                except Exception:
                    return []
            return []
        except Exception as e:  # failed call: logged at est. cost 0, retried once
            log_cost({"session_id": sid, "tag": tag, "attempt": attempt, "ok": False,
                      "error": str(e)[:200], "usd": 0.0})
    return []


def load_pool():
    freeze_corpus.OUT = ND.SNAPSHOT
    freeze_corpus._CACHE = None
    manifest = R.load_manifest()
    smap = R.build_session_id_map()
    ids = B.sample_sessions(POOL, SEED)
    sessions, skipped = [], []
    for sid in ids:
        s = R.load_session(sid, manifest, smap)
        (sessions if s else skipped).append(s or sid)
    return manifest, sessions, skipped


def cmd_generate(a):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    manifest, sessions, skipped = load_pool()
    SKIPPED.write_text(json.dumps({"no_production_note_or_transcript": skipped}, indent=1))
    print(f"pool {POOL}: {len(sessions)} loadable, {len(skipped)} skipped (listed in {SKIPPED.name})")
    done = {json.loads(l)["target_session_id"] for l in PROBES.read_text().splitlines() if l.strip()} \
        if PROBES.exists() else set()
    todo = [s for s in sessions if s["session_id"] not in done]
    if a.n:
        todo = todo[:a.n]
    n_gen = 0
    per_session_cost = None
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        est = (per_session_cost or 0.005) * len(batch)
        print(f"batch {i // BATCH + 1}: est +USD {est:.3f}, running total USD {spent():.3f} (cap {CAP_USD})")
        if spent() + est > CAP_USD:
            print("STOP: next batch would exceed the cap; generated subset only")
            break
        for s in batch:
            sid, rollup, tr = s["session_id"], s["rollup_full"], s["transcript"]
            slice_ = rollup[:6000]
            before = spent()
            cands = paid_call(PROMPT.format(slice=slice_[:3000]), sid, "gen")
            kept = []
            for c in cands:
                for _ in range(1 + 3):  # first try + up to 3 regenerations, as in the original
                    if R.validate_probe(c, tr, rollup):
                        kept.append(c)
                        break
                    if _ == 3:
                        break
                    if spent() >= CAP_USD:
                        break
                    new = paid_call(PROMPT.format(slice=slice_[:3000]), sid, "regen")
                    if not new:
                        break
                    c = new[0]
            with PROBES.open("a") as f:
                for c in kept:
                    t, r = R.compute_fairness_stats(c, tr, rollup)
                    f.write(json.dumps({"query": c, "target_session_id": sid,
                                        "fairness_transcript": t, "fairness_rollup": r}) + "\n")
            n_gen += 1
            per_session_cost = ((per_session_cost or 0) * (n_gen - 1) + (spent() - before)) / n_gen
    n_sess = len(done) + n_gen
    print(f"sessions generated this run {n_gen}, total spend USD {spent():.4f}, "
          f"avg USD {per_session_cost or 0:.5f}/session, projection for {len(sessions)} sessions: "
          f"USD {(per_session_cost or 0) * len(sessions):.2f}")
    return 0


def paired(a, b, sids):
    d = R._clustered_bootstrap_diffs(a, b, sids, n_bootstrap=10_000, seed=42)
    return {"mean_diff": float(np.mean(a) - np.mean(b)), "ci_lower": float(d[250]), "ci_upper": float(d[9750])}


def run_arms(probes, sessions):
    by_id = {s["session_id"]: s for s in sessions}
    def ov(key):
        v = []
        for p in probes:
            pt = set(R.TOKEN_RE.findall(p["query"].lower()))
            tt = set(R.TOKEN_RE.findall(by_id[p["target_session_id"]][key].lower()))
            v.append(len(pt & tt) / len(pt) if pt else 0.0)
        return float(np.mean(v))
    overlap = {"transcript": ov("transcript"), "no_tool": ov("transcript_no_tool"),
               "rollup": ov("rollup_full"), "no_digest": ov("transcript_no_digest")}
    arms = {}
    for name, key, chunk in ARMS:
        idx = R.build_bm25_index(sessions, key, chunk=chunk)
        arms[name] = R.evaluate_arm(probes, idx, name, None, {})
        print(f"  {name}: recall@10 {arms[name]['recall@10']:.4f}")
    sids = [p["target_session_id"] for p in probes]
    names = [n for n, _, _ in ARMS]
    diffs = {f"{x}_minus_{y}": paired(arms[x]["hits_at_10"], arms[y]["hits_at_10"], sids)
             for i, x in enumerate(names) for y in names[i + 1:]}
    slim = {k: {a: b for a, b in v.items() if a not in ("hits_at_10", "target_session_ids")} for k, v in arms.items()}
    return slim, diffs, overlap


def load_sessions_with_nd(manifest, sessions):
    for s in sessions:
        s["transcript_no_digest"] = ND.no_digest_text(s["session_id"], manifest)
    return sessions


def cmd_control(a):
    manifest, sessions, skipped = load_pool()
    assert not skipped, skipped
    load_sessions_with_nd(manifest, sessions)
    probes = B.load_existing_probes(ND.PROBES)
    arms, diffs, overlap = run_arms(probes, sessions)
    paper = {**ND.PAPER, "G_bm25_no_digest": 0.745}
    gate = {k: {"paper": v, "got": arms[k]["recall@10"], "pass": abs(arms[k]["recall@10"] - v) <= ND.TOL}
            for k, v in paper.items()}
    out = {"n_probes": len(probes), "arms": arms, "overlap": overlap, "gate": gate,
           "passed": all(g["pass"] for g in gate.values())}
    (OUTDIR / "control_original_probes.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps({"gate": gate, "overlap": overlap}, indent=1))
    return 0 if out["passed"] else 1


def cmd_analyse(a):
    manifest, sessions, skipped = load_pool()
    load_sessions_with_nd(manifest, sessions)
    probes = B.load_existing_probes(PROBES)
    ctrl = json.loads((OUTDIR / "control_original_probes.json").read_text())
    arms, diffs, overlap = run_arms(probes, sessions)
    order_new = sorted(arms, key=lambda k: -arms[k]["recall@10"])
    order_old = sorted(ctrl["arms"], key=lambda k: -ctrl["arms"][k]["recall@10"])
    gaps = {k: v["mean_diff"] for k, v in diffs.items()}
    max_gap = max(abs(v) for v in gaps.values())
    dense = {"A_rollup": 0.574, "B_raw": 0.644, "C_raw_no_tool": 0.650}
    method_gaps = {"bm25_rollup_minus_dense_rollup(old probes)": 0.708 - 0.574,
                   "bm25_raw_minus_dense_raw(old probes)": 0.728 - 0.644,
                   "bm25_stripped_minus_dense_stripped(old probes)": 0.737 - 0.650}
    out = {"experiment": "rq4_probes_from_rollups", "post_hoc_exploratory": True,
           "n_probes": len(probes), "n_sessions": len({p["target_session_id"] for p in probes}),
           "pool_sessions": len(sessions), "skipped_sessions": skipped, "spend_usd": spent(),
           "recall_arms_new_probes": arms, "recall_arms_original_probes": ctrl["arms"],
           "paired_differences_new_probes": diffs, "overlap_new_probes": overlap,
           "overlap_original_probes": ctrl["overlap"],
           "ordering_new": order_new, "ordering_original": order_old,
           "prediction": "rollup overlap highest; D-E >= 0 or narrower than -0.020, D-F narrower than -0.029; rollup first or tied first",
           "observed": {"D_minus_E": gaps["D_bm25_rollup_minus_E_bm25_raw"],
                        "D_minus_F": gaps["D_bm25_rollup_minus_F_bm25_no_tool"],
                        "D_minus_E_original": ctrl["arms"]["D_bm25_rollup"]["recall@10"] - ctrl["arms"]["E_bm25_raw"]["recall@10"],
                        "D_minus_F_original": ctrl["arms"]["D_bm25_rollup"]["recall@10"] - ctrl["arms"]["F_bm25_no_tool"]["recall@10"],
                        "rollup_overlap_highest": overlap["rollup"] == max(overlap.values())},
           "reading_rule": {"largest_substrate_gap_new_probes": max_gap, "method_gaps_old_probes": method_gaps,
                            "dense_recall10_old_probes": dense,
                            "method_conclusion_survives": bool(min(method_gaps.values()) > max_gap)
                            if False else bool(method_gaps["bm25_rollup_minus_dense_rollup(old probes)"] > max_gap),
                            "caveat": "dense not remeasured on the new probes"}}
    RESULTS.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps({k: out[k] for k in ("ordering_new", "ordering_original", "overlap_new_probes", "observed", "reading_rule")}, indent=1))
    return 0


DENSE_ARMS = (("A_dense_rollup", "rollup_full", "arm_a_rollup", False),
              ("B_dense_raw", "transcript", "arm_b_raw", True),
              ("C_dense_no_tool", "transcript_no_tool", "arm_c_no_tool", True),
              ("G_dense_no_digest", "transcript_no_digest", "arm_g_no_digest", True))
DENSE_ORIG_REF = {"A_dense_rollup": 0.5741, "B_dense_raw": 0.6440,
                  "C_dense_no_tool": 0.6504, "G_dense_no_digest": 0.6587}
BM25_OF = {"A_dense_rollup": "D_bm25_rollup", "B_dense_raw": "E_bm25_raw",
           "C_dense_no_tool": "F_bm25_no_tool", "G_dense_no_digest": "G_bm25_no_digest"}


def _cached_only_embed(texts, embedder, cache, cache_key):
    """B.resumable_embed_texts, but refuses to compute: every shard must already be on disk."""
    fp = B.texts_fingerprint(texts)
    d = B.SCRATCH / "embeddings" / f"{cache_key}_b{B.EMBED_BATCH}_{fp}"
    n = (len(texts) + B.EMBED_BLOCK - 1) // B.EMBED_BLOCK
    missing = [i for i in range(n) if not (d / f"{i:06d}.npy").exists()]
    if missing:
        raise SystemExit(f"ABORT: {len(missing)}/{n} shards missing for {cache_key} ({d.name}); not embedding chunks")
    return B.resumable_embed_texts(texts, embedder, cache, cache_key)


def cmd_dense(a):
    """Dense recall@10 (local fastembed, cached chunk shards only) on original + new probes."""
    t0 = time.time()
    from fastembed import TextEmbedding
    manifest, sessions, skipped = load_pool()
    assert not skipped, skipped
    load_sessions_with_nd(manifest, sessions)
    embedder = TextEmbedding(model_name=R.EMBEDDING_MODEL, providers=["CPUExecutionProvider"])
    R.embed_texts = _cached_only_embed
    old = B.load_existing_probes(ND.PROBES)
    new = B.load_existing_probes(PROBES)
    qcache: dict = {}
    res = {"old": {}, "new": {}}
    for name, key, prefix, chunk in DENSE_ARMS:
        idx = R.build_dense_index(sessions, key, embedder, qcache, prefix, chunk=chunk)
        res["old"][name] = R.evaluate_arm(old, idx, name, embedder, qcache)
        res["new"][name] = R.evaluate_arm(new, idx, name, embedder, qcache)
        print(f"  {name}: old {res['old'][name]['recall@10']:.4f} new {res['new'][name]['recall@10']:.4f}")
    gate = {k: {"reference": v, "got": res["old"][k]["recall@10"],
                "abs_err": abs(res["old"][k]["recall@10"] - v), "pass": abs(res["old"][k]["recall@10"] - v) <= 0.005}
            for k, v in DENSE_ORIG_REF.items()}
    passed = all(g["pass"] for g in gate.values())
    print(json.dumps(gate, indent=1))
    if not passed:
        print("GATE FAILED: not writing results")
        return 1
    sids = [p["target_session_id"] for p in new]
    r = json.loads(RESULTS.read_text())
    bm25_hits = {}
    # BM25 hits are not stored in results.json; recompute (BM25 is deterministic, local)
    for bname, bkey, bchunk in ARMS:
        bidx = R.build_bm25_index(sessions, bkey, chunk=bchunk)
        bm25_hits[bname] = R.evaluate_arm(new, bidx, bname, None, {})
        assert abs(bm25_hits[bname]["recall@10"] - r["recall_arms_new_probes"][bname]["recall@10"]) < 1e-9
    d = res["new"]
    names = [n for n, *_ in DENSE_ARMS]
    sub = {f"{x}_minus_{y}": paired(d[x]["hits_at_10"], d[y]["hits_at_10"], sids)
           for i, x in enumerate(names) for y in names[i + 1:]}
    method = {k: paired(bm25_hits[BM25_OF[k]]["hits_at_10"], d[k]["hits_at_10"], sids) for k in names}
    largest_sub = max(sub, key=lambda k: abs(sub[k]["mean_diff"]))
    smallest_method = min(method.values(), key=lambda v: v["mean_diff"])
    r["dense"] = {
        "note": "dense (bge-small, local fastembed, cached chunk shards) on the same 550-session pool; max-pool to session; "
                "bootstrap: session-clustered paired, 10,000 resamples, seed 42",
        "gate_original_probes": {"tolerance": 0.005, "passed": passed, "arms": gate},
        "recall_original_probes": {k: {m: v[m] for m in ("recall@1", "recall@5", "recall@10", "mrr@10")} for k, v in res["old"].items()},
        "recall_new_probes": {k: {m: v[m] for m in ("recall@1", "recall@5", "recall@10", "mrr@10")} for k, v in d.items()},
        "paired_substrate_differences_new_probes_dense": sub,
        "bm25_minus_dense_new_probes": method,
        "reading_rule_same_probe_set": {
            "largest_dense_substrate_gap": {"pair": largest_sub, **sub[largest_sub]},
            "largest_bm25_substrate_gap_new_probes": r["reading_rule"]["largest_substrate_gap_new_probes"],
            "smallest_bm25_minus_dense_gap": smallest_method["mean_diff"],
            "bm25_minus_dense_gap_rollup": method["A_dense_rollup"]["mean_diff"],
            "largest_substrate_gap_abs": abs(sub[largest_sub]["mean_diff"]),
            "method_gap_larger_than_largest_dense_substrate_gap": bool(method["A_dense_rollup"]["mean_diff"] > abs(sub[largest_sub]["mean_diff"])),
        },
        "runtime_s": time.time() - t0,
    }
    RESULTS.write_text(json.dumps(r, indent=2, sort_keys=True))
    print(json.dumps(r["dense"], indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--n", type=int, default=0)
    sub.add_parser("control"); an = sub.add_parser("analyse")
    an.add_argument("--dense", action="store_true", help="dense recall on original+new probes from cached shards; adds a `dense` key to results.json, BM25 content untouched")
    a = ap.parse_args()
    if a.cmd == "analyse" and a.dense:
        return cmd_dense(a)
    return {"generate": cmd_generate, "control": cmd_control, "analyse": cmd_analyse}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
