#!/usr/bin/env python3
"""Characterise the real agent query workload against the vault.

The paper needs to argue that its retrieval evaluation reflects how agents
actually query a memory substrate, not how a paper author imagines they do.
This reads the production query telemetry log and reports the distribution of
query length, shape (keyword vs question), hit rate and origin.

Run:
    python3 rq0_query_workload.py
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freeze_corpus import load_manifest
from paths import REPO_ROOT

LOGS = [
    REPO_ROOT / "data" / "telemetry" / "queries" / "queries_mac.jsonl",
    REPO_ROOT / "data" / "telemetry" / "pkm_queries.jsonl",
]
OUT = Path(__file__).resolve().parents[2] / "data" / "paper-rollups" / "rq0_query_workload.json"

QUESTION_WORDS = ("what", "why", "how", "when", "where", "which", "who", "did", "does",
                  "is", "are", "can", "should", "was", "were")

# The raw log is not all organic traffic. Two sources have to come out before any
# claim about "how agents query the vault" is defensible:
#   - origins prefixed `eval-` are the A/B benchmark harness replaying fixed
#     question sets (eval_bench.py rejects an origin that does not start so),
#   - the literal query `heartbeat` is the daemon health probe: 969 identical
#     rows, always limit 20, always the same vault.
# Both are synthetic and would otherwise dominate the repetition statistics.
# Filtering on the origin tag alone is not enough, and assuming it was would have
# put fabricated "rediscovery" numbers in the paper. The benchmark question sets
# also appear with origin unset: `how does the search daemon rank results` has 95
# occurrences all on 2026-09-25, and `why not use a vector database for this` has
# 60 all on 2026-09-26 of which 10 are tagged `eval-rerank` outright. Same-day
# bursts of an identical question are a harness replaying a fixed set, not an
# agent asking twice. So any query text ever seen under an eval-* origin is
# treated as a benchmark question wherever else it appears.
SYNTHETIC_QUERIES = {"heartbeat"}


def benchmark_question_set(rows: list[dict]) -> set[str]:
    return {r["_q"].strip().lower() for r in rows
            if (r.get("origin") or "").startswith("eval-")}


def is_organic(rec: dict, bench: set[str]) -> bool:
    if (rec.get("origin") or "").startswith("eval-"):
        return False
    q = rec["_q"].strip().lower()
    return q not in SYNTHETIC_QUERIES and q not in bench


def load(cutoff_utc: str | None = None) -> list[dict]:
    """Load query records, optionally filtering by timestamp <= cutoff_utc."""
    cutoff_dt = None
    if cutoff_utc:
        cutoff_dt = datetime.fromisoformat(cutoff_utc.replace("Z", "+00:00"))
    rows = []
    for log in LOGS:
        if not log.exists():
            continue
        for line in log.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") != "search":
                continue
            q = (rec.get("q") or "").strip()
            if not q:
                continue
            # Filter by timestamp if cutoff is set
            if cutoff_dt and "t" in rec:
                try:
                    rec_dt = datetime.fromisoformat(rec["t"].replace("Z", "+00:00"))
                    # Ensure both are timezone-aware for comparison
                    if rec_dt.tzinfo is None:
                        rec_dt = rec_dt.replace(tzinfo=timezone.utc)
                    if rec_dt > cutoff_dt:
                        continue
                except (ValueError, AttributeError):
                    pass  # Keep records with unparseable timestamps
            rec["_q"] = q
            rec["_log"] = log.name
            rows.append(rec)
    return rows


def shape(q: str) -> str:
    """Question-shaped queries hit the reranker; keyword queries hit expansion."""
    words = re.findall(r"[a-z0-9']+", q.lower())
    if not words:
        return "empty"
    if q.rstrip().endswith("?") or words[0] in QUESTION_WORDS:
        return "question"
    return "keyword"


def pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 1) if whole else 0.0


def describe(rows: list[dict]) -> dict:
    n = len(rows)
    words = [len(re.findall(r"[a-z0-9']+", r["_q"].lower())) for r in rows]
    chars = [len(r["_q"]) for r in rows]
    shapes = Counter(shape(r["_q"]) for r in rows)
    origins = Counter(r.get("origin") or "unknown" for r in rows)
    vaults = Counter(r.get("vault") or "unknown" for r in rows)

    # A logged `results` list is what the deployed system actually returned.
    # Empty means the agent asked and got nothing back -- the miss rate is the
    # honest denominator for any claim that the substrate is useful.
    with_results = [r for r in rows if isinstance(r.get("results"), list)]
    empty = [r for r in with_results if len(r["results"]) == 0]

    # Backfilled rows were reconstructed from transcripts and carry took_ms 0.0,
    # so latency must be read only from live rows.
    live_latency = [r["took_ms"] for r in rows
                    if isinstance(r.get("took_ms"), (int, float)) and r["took_ms"] > 0]

    # How many distinct queries repeat? A repeated query is a rediscovery signal.
    counts = Counter(r["_q"].lower() for r in rows)
    repeated = {q: c for q, c in counts.items() if c > 1}

    summary = {
        "n_queries": n,
        "n_distinct": len(counts),
        "logs": dict(Counter(r["_log"] for r in rows)),
        "vaults": dict(vaults),
        "origins": dict(origins.most_common(10)),
        "words": {
            "mean": round(statistics.mean(words), 2),
            "median": statistics.median(words),
            "p10": statistics.quantiles(words, n=10)[0] if len(words) > 10 else None,
            "p90": statistics.quantiles(words, n=10)[8] if len(words) > 10 else None,
            "max": max(words),
        },
        "chars": {
            "mean": round(statistics.mean(chars), 1),
            "median": statistics.median(chars),
        },
        "shape": {k: {"n": v, "pct": pct(v, n)} for k, v in shapes.items()},
        # MAX_ENTITY_TOKENS = 8 in entity_expansion.py: queries longer than this
        # bypass expansion entirely, so this split says which code path the real
        # workload actually exercises.
        "over_8_tokens": {"n": sum(1 for w in words if w > 8),
                          "pct": pct(sum(1 for w in words if w > 8), n)},
        "results_logged": {
            "n": len(with_results),
            "empty_result_queries": len(empty),
            "empty_pct_of_logged": pct(len(empty), len(with_results)),
        },
        "live_latency_ms": {
            "n": len(live_latency),
            "median": round(statistics.median(live_latency), 1) if live_latency else None,
            "p95": round(sorted(live_latency)[int(0.95 * len(live_latency))], 1)
            if len(live_latency) > 20 else None,
        } if live_latency else {"n": 0},
        "repetition": {
            "distinct_queries_asked_more_than_once": len(repeated),
            "max_repeats": max(repeated.values()) if repeated else 0,
            "top_repeated": sorted(repeated.items(), key=lambda kv: -kv[1])[:10],
        },
        "sample_queries": [r["_q"] for r in rows[:15]],
    }
    return summary


def main() -> None:
    manifest = load_manifest()
    rows = load(cutoff_utc=manifest["snapshot_utc"])
    if not rows:
        raise SystemExit("no query records found; check the telemetry log paths")

    bench = benchmark_question_set(rows)
    organic = [r for r in rows if is_organic(r, bench)]
    summary = {
        "corpus_snapshot_utc": manifest["snapshot_utc"],
        "note": "organic excludes eval-* benchmark origins and the daemon "
                "heartbeat probe; the paper should quote organic figures only",
        "all_traffic": describe(rows),
        "organic": describe(organic),
        "excluded": {
            "eval_harness": sum(1 for r in rows if (r.get("origin") or "").startswith("eval-")),
            "heartbeat_probe": sum(1 for r in rows
                                   if r["_q"].strip().lower() in SYNTHETIC_QUERIES),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    o = summary["organic"]
    print(f"organic queries: {o['n_queries']} ({o['n_distinct']} distinct)")
    print(f"words: median {o['words']['median']}, mean {o['words']['mean']}, p90 {o['words']['p90']}")
    print(f"shape: {json.dumps(o['shape'])}")
    print(f"over 8 tokens (bypasses entity expansion): {json.dumps(o['over_8_tokens'])}")
    print(f"empty-result rate: {json.dumps(o['results_logged'])}")
    print(f"live latency ms: {json.dumps(o['live_latency_ms'])}")
    print(f"repetition: {json.dumps(o['repetition']['top_repeated'][:8])}")
    print(f"excluded: {json.dumps(summary['excluded'])}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
