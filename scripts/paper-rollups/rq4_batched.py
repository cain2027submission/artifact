#!/usr/bin/env python3
"""RQ4 in small resumable batches, so a crash costs one batch and not the run.

Why this exists rather than calling rq4_known_item.main() directly: that entry
point generates probes for every sampled session in one pass, and `call_llm`
returns "" on any exception. A gateway outage therefore produced zero probes and
still wrote a results file reporting num_probes: 0 with every metric NaN, which
is indistinguishable from a legitimately empty corpus. Two sessions were spent
before anyone read the cause. This runner splits the fragile phase from the
deterministic one and refuses to continue when a batch fails wholesale.

Phase 1 (fragile, network-bound): generate probes in batches of BATCH_SIZE
sessions, appending each batch to PROBES_FILE as it lands. Re-running skips
sessions already present, so an interrupted run resumes instead of restarting.
A batch that yields zero probes aborts the run rather than being averaged in.

Phase 2 (deterministic, CPU-bound): build the six arms over the full session set
and evaluate. Only runs once phase 1 has a complete probe set.

The embedding cache is written outside the repository, because chunked raw
transcripts for the full corpus serialize to hundreds of megabytes and must
never reach a commit.

    rq4_batched.py --sessions 30 --probes-only     # phase 1 only, 30 sessions
    rq4_batched.py --sessions 30                   # both phases
    rq4_batched.py --sessions 30 --eval-only       # phase 2 over existing probes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import freeze_corpus
import rq4_known_item as R

# Kept out of the repo: chunked-transcript embeddings for the full corpus run to
# hundreds of MB as JSON, and a committed cache would dwarf the whole vault.
SCRATCH = Path.home() / ".cache" / "paper-rollups"
BATCH_SIZE = 10
# A run where this many consecutive batches produce nothing is an outage, not a
# corpus with nothing to say. Stop and say so rather than writing NaN.
MAX_EMPTY_BATCHES = 2
# Vectors per checkpoint shard in phase 2. The full corpus is ~52k chunks, so an
# uncheckpointed phase 2 is a multi-hour job that a crash returns to zero. 256
# keeps a lost block under a minute at the rate below.
EMBED_BLOCK = 256
# How many chunks go to the model at once. Deliberately NOT EMBED_BLOCK: this was
# originally passed as batch_size=EMBED_BLOCK, and handing fastembed a batch of
# 256 x 512 tokens is 4.42x slower than batching 32 (1.02 vs 4.53 chunks/s,
# interleaved on an otherwise identical workload). That single conflation was the
# difference between a ~19 hour phase 2 and a ~3 hour one.
#
# It must stay fixed across every arm. Batch size perturbs the vectors slightly
# (4.2e-4 max abs difference between 32 and 256, from padding and reduction order
# inside onnxruntime) which is negligible within one index and NOT negligible
# between two arms whose retrieval scores are being compared to each other.
EMBED_BATCH = 32


def texts_fingerprint(texts: list[str]) -> str:
    """Identify an exact text set, so shards are never reused for other text.

    Covers content, order and count. A changed chunk width or a regrown corpus
    therefore produces a different fingerprint and recomputes, rather than
    silently pairing last run's vectors with this run's chunks.
    """
    h = hashlib.sha256()
    h.update(str(len(texts)).encode())
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode())
    return h.hexdigest()[:16]


def resumable_embed_texts(texts: list[str], embedder, cache: dict, cache_key: str) -> np.ndarray:
    """Block-checkpointed stand-in for R.embed_texts.

    Each block of EMBED_BLOCK vectors is written to its own .npy shard as soon as
    it is computed, so an interrupted phase 2 resumes from the last whole block
    instead of restarting. Deliberately does NOT write into `cache`: the original
    put every arm's vectors in one JSON blob, which for the full corpus is ~800MB
    of indented floats serialized at the very end of the run -- slow, and lost
    entirely on any crash before that point.
    """
    fp = texts_fingerprint(texts)
    # The batch size is part of the shard identity, not just the text. Batch size
    # perturbs the vectors (4.2e-4 max abs between 32 and 256), so a directory keyed
    # on text alone would hand a resumed run last setting's vectors for this
    # setting's chunks -- one index holding two embedding spaces, invisibly. Shards
    # written under a different batch size are simply not found, and can be deleted
    # by hand whenever convenient.
    shard_dir = SCRATCH / "embeddings" / f"{cache_key}_b{EMBED_BATCH}_{fp}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    n_blocks = (len(texts) + EMBED_BLOCK - 1) // EMBED_BLOCK
    done = sum(1 for i in range(n_blocks) if (shard_dir / f"{i:06d}.npy").exists())
    if done:
        print(f"      resuming: {done}/{n_blocks} blocks already on disk")

    out = []
    t_start = time.time()
    computed = 0
    for i in range(n_blocks):
        shard = shard_dir / f"{i:06d}.npy"
        if shard.exists():
            out.append(np.load(shard))
            continue
        block = texts[i * EMBED_BLOCK:(i + 1) * EMBED_BLOCK]
        vecs = np.array(list(embedder.embed(block, batch_size=EMBED_BATCH)))
        # Write to a temp name then rename, so a crash mid-write cannot leave a
        # truncated shard that a later run would load as if it were complete.
        # Written through a handle because np.save appends ".npy" to any path that
        # does not already end in it, which silently produced "<name>.part.npy"
        # and made the rename below fail.
        tmp = shard.with_name(shard.name + ".part")
        with tmp.open("wb") as fh:
            np.save(fh, vecs)
        tmp.replace(shard)
        out.append(vecs)
        computed += len(block)
        rate = computed / max(time.time() - t_start, 1e-9)
        remaining = len(texts) - (i + 1) * EMBED_BLOCK
        eta = remaining / rate if rate > 0 else 0
        print(f"      block {i + 1}/{n_blocks}  {rate:.1f} chunks/s  "
              f"eta {eta / 60:.0f}m", flush=True)

    embeddings = np.vstack(out) if out else np.zeros((0, 384))
    if embeddings.shape[0] != len(texts):
        raise SystemExit(
            f"ABORT: {cache_key} embedded {embeddings.shape[0]} vectors for {len(texts)} "
            f"chunks. Shards in {shard_dir} are inconsistent; delete that directory and re-run."
        )
    return embeddings


def load_existing_probes(path: Path) -> list[dict]:
    """Read probes from a previous run so an interrupted run can resume."""
    if not path.exists():
        return []
    probes = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            probes.append(json.loads(line))
        except json.JSONDecodeError:
            # A half-written final record is the expected shape of a crash here.
            print(f"  warning: dropping malformed probe record in {path.name}")
    return probes


def sample_sessions(n: int, seed: int) -> list[str]:
    """The same sample rq4_known_item would take, so results stay comparable."""
    manifest = R.load_manifest()
    paired = manifest["paired_session_ids"]
    random.seed(seed)
    sampled = random.sample(paired, min(n, len(paired)))
    return sorted(sampled)


def generate_probes_batched(session_ids: list[str], probes_file: Path, cache_file: Path, seed: int) -> list[dict]:
    """Phase 1. Generate probes batch by batch, checkpointing after each."""
    manifest = R.load_manifest()
    session_map = R.build_session_id_map()

    existing = load_existing_probes(probes_file)
    done_ids = {p["target_session_id"] for p in existing}
    todo = [s for s in session_ids if s not in done_ids]

    print(f"phase 1: {len(existing)} probes already on disk covering {len(done_ids)} sessions")
    print(f"         {len(todo)} sessions still to do, in batches of {BATCH_SIZE}")
    if not todo:
        return existing

    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    probes_file.parent.mkdir(parents=True, exist_ok=True)

    all_probes = list(existing)
    empty_streak = 0
    load_failures = 0

    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start:start + BATCH_SIZE]
        n_batch = start // BATCH_SIZE + 1
        n_total = (len(todo) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n  batch {n_batch}/{n_total}: {len(batch)} sessions")

        sessions = []
        for sid in batch:
            sess = R.load_session(sid, manifest, session_map)
            if sess is None:
                load_failures += 1
            else:
                sessions.append(sess)

        if not sessions:
            print(f"    no session in this batch could be loaded; skipping")
            continue

        t0 = time.time()
        batch_probes = R.generate_probes_for_corpus(sessions, cache)
        elapsed = time.time() - t0

        # Append before anything else can fail, so the batch is banked.
        with probes_file.open("a") as f:
            for probe in batch_probes:
                f.write(json.dumps(probe) + "\n")
        cache_file.write_text(json.dumps(cache, indent=2, sort_keys=True))

        all_probes.extend(batch_probes)
        print(f"    +{len(batch_probes)} probes in {elapsed:.0f}s (total {len(all_probes)})")

        if not batch_probes:
            empty_streak += 1
            print(f"    WARNING: batch produced no probes ({empty_streak}/{MAX_EMPTY_BATCHES})")
            if empty_streak >= MAX_EMPTY_BATCHES:
                print(
                    f"\nABORT: {empty_streak} consecutive batches produced no probes.\n"
                    f"  This is what a gateway outage looks like, not an empty corpus.\n"
                    f"  Check: curl -s -o /dev/null -w '%{{http_code}}' -X POST {R.GATEWAY}/v1/chat/completions \\\n"
                    f"    -H 'Content-Type: application/json' \\\n"
                    f"    -d '{{\"model\":\"{R.PROBE_MODEL}\",\"max_tokens\":8,\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}'\n"
                    f"  {len(all_probes)} probes are banked in {probes_file}; re-run to resume."
                )
                raise SystemExit(2)
        else:
            empty_streak = 0

    if load_failures:
        print(f"\n  {load_failures} sessions could not be loaded and were skipped")

    return all_probes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=30, help="how many paired sessions to sample")
    ap.add_argument("--seed", type=int, default=R.SEED)
    ap.add_argument("--probes-only", action="store_true", help="phase 1 only")
    ap.add_argument("--eval-only", action="store_true", help="phase 2 only, over probes already on disk")
    ap.add_argument("--tag", default=None, help="suffix for output files, to keep runs side by side")
    ap.add_argument(
        "--corpus-snapshot", default=None,
        help="override freeze_corpus.OUT, e.g. to point at RQ4's own "
             "corpus_snapshot_rq4_*.json (with its archived transcript/rollup "
             "content) instead of the shared corpus_snapshot.json every other "
             "RQ depends on. Unset: default shared-snapshot behavior, unchanged.",
    )
    ap.add_argument("--with-f", action="store_true",
                    help="also build F_bm25_no_tool (BM25 over the output-stripped text) and its tests")
    ap.add_argument("--results-file", default=None,
                    help="write results here instead of data/paper-rollups/rq4_known_item_<tag>.json")
    args = ap.parse_args()

    if args.corpus_snapshot:
        # Module-attribute override, same pattern as R.TARGET_SESSIONS / R.SEED
        # below: freeze_corpus.load_manifest()/read_pinned()/read_pinned_rollup()
        # all resolve against freeze_corpus's own globals at call time, no matter
        # which module imported their names, so this is the only line needed to
        # redirect every one of rq4_known_item's manifest reads. Reset _CACHE too,
        # in case load_manifest() already ran once in this process under the old
        # OUT. Every other RQ script never sets this flag, so freeze_corpus.OUT
        # keeps pointing at the shared corpus_snapshot.json for them, unaffected.
        freeze_corpus.OUT = Path(args.corpus_snapshot).expanduser()
        freeze_corpus._CACHE = None

    tag = args.tag or f"n{args.sessions}"
    SCRATCH.mkdir(parents=True, exist_ok=True)
    probes_file = SCRATCH / f"rq4_probes_{tag}.jsonl"
    cache_file = SCRATCH / f"rq4_cache_{tag}.json"
    results_file = R.REPO_ROOT_LOCAL / "data" / "paper-rollups" / f"rq4_known_item_{tag}.json"
    if args.results_file:
        results_file = Path(args.results_file).expanduser().resolve()
    if args.with_f:
        R.EXTRA_ARMS = ("F_bm25_no_tool",)

    print(f"RQ4 batched run: {args.sessions} sessions, seed {args.seed}, tag {tag}")
    print(f"  probes  {probes_file}")
    print(f"  cache   {cache_file}  (outside the repo on purpose)")
    print(f"  results {results_file}")

    session_ids = sample_sessions(args.sessions, args.seed)

    if args.eval_only:
        probes = load_existing_probes(probes_file)
        if not probes:
            raise SystemExit(f"--eval-only but no probes in {probes_file}")
    else:
        probes = generate_probes_batched(session_ids, probes_file, cache_file, args.seed)

    print(f"\nphase 1 complete: {len(probes)} probes over {len({p['target_session_id'] for p in probes})} sessions")

    # Checked before the --probes-only return, not after it. A run that generated
    # nothing is a failure whichever phases were asked for, and exiting 0 here is
    # the same silent success that hid the gateway outage in the first place.
    if not probes:
        raise SystemExit("ABORT: phase 1 produced no probes at all; refusing to report success")

    if args.probes_only:
        return 0

    # Phase 2 reuses rq4_known_item's own arm construction and evaluation by
    # pointing its module-level paths at this run's files, so there is one
    # implementation of the six arms rather than two that can drift apart.
    R.PROBES_FILE = probes_file
    R.CACHE_FILE = cache_file
    R.RESULTS_FILE = results_file
    R.TARGET_SESSIONS = args.sessions
    R.SEED = args.seed
    # Make the expensive phase resumable. Swapped at the module level so every
    # arm built by R.main() gets checkpointing without duplicating arm logic.
    R.embed_texts = resumable_embed_texts

    print(f"\nphase 2: building and evaluating six arms over {len(session_ids)} sessions")
    return R.main()


if __name__ == "__main__":
    sys.exit(main())
