#!/usr/bin/env python3
"""RQ5 addendum: what a WHOLE-ARTIFACT read costs, in tokens, on the same sessions.

Why this exists. The paper's recommendation turns on what it costs an agent to
pull an entire memory artifact into context, and the figure being quoted for that
is ~6x -- which is RQ2's `inflation_serialized_over_payload`, a ratio between two
ways of counting *bytes* of a transcript, and not a read cost at all. A read is
billed in tokens, and the three readable artifacts tokenise at very different
rates: a rollup is markdown prose, a raw transcript is minified JSONL dense in
braces, quoted keys and hex ids, and the stripped rendering sits between them.
So a byte ratio and a token ratio are not the same number, and this script
measures the token one and reports the byte one beside it so the gap is explicit.

What is held fixed, and why it matters. The three arms have to be the same
artifacts RQ3 judged, or a ratio printed next to RQ3's accuracy numbers does not
apply to them:

  * the stripped arm is `strip_transcript()` IMPORTED from
    `rq3_stripped_transcript_arm.py`, not a second implementation of the same
    idea. L2: authored user prose, assistant prose, surviving reasoning text and
    every tool call's input capped at 2,000 chars, with tool RESULTS dropped.
  * the raw arm is the transcript `freeze_corpus.read_pinned()` returns, the same
    pinned bytes RQ1 and RQ2 read, truncated at the manifest's snapshot length so
    a file that has been appended to since still yields the snapshot's text.
  * the population is `corpus_snapshot.json`'s `paired_session_ids` and nothing
    else. `sessions/` has grown from 499 notes to 1,114 since the snapshot, so a
    `sessions/*.md` glob is a reproducibility defect rather than a convenience.

Rollup content is read from the pinned git commit rather than from the working
tree. `freeze_corpus.read_pinned_rollup()` refuses any note whose path changed
since the snapshot -- correctly, since a rename invalidates the pin -- and on
this corpus the rollup drain has renamed or rewritten enough notes that the
working-tree route loses sessions for a reason that has nothing to do with the
measurement. `content_pinning` in the manifest records that `rollup_note_bytes`
was backfilled from commit 27c1de0, so that commit's blob IS the pinned content:
every one of the 276 paired notes with a pin has a blob there whose length equals
its pinned byte count exactly, which is checked per session here and is a drop
rather than a fallback when it fails. The working-tree route is still run as a
cross-check and its yield is reported.

Ratios are formed per session and then aggregated, never as a quotient of two
corpus totals. One 55 MB transcript would otherwise decide the headline.

Usage:
    rq5_wholeread_token_ratio.py --smoke     # 3 real sessions + degenerate cases
    rq5_wholeread_token_ratio.py --limit 20  # small resumable-sized batch
    rq5_wholeread_token_ratio.py             # full corpus, writes the JSON
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
REPO = HERE.parents[2]
DATA = REPO / "data" / "paper-rollups"
OUT = DATA / "rq5_wholeread_token_ratio.json"

from freeze_corpus import (  # noqa: E402
    load_manifest, read_pinned, read_pinned_rollup,
)
from jsonl_loader import load_jsonl_strict  # noqa: E402
from rq3_stripped_transcript_arm import strip_transcript  # noqa: E402

# The commit the manifest's rollup_note_bytes was backfilled from, per its own
# `content_pinning` field. Named here rather than derived, because guessing a
# commit would silently change what "the rollup" means.
PIN_COMMIT = "27c1de0"

ENCODING = "cl100k_base"
BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 20260929

_ENC = None


def encoder():
    """cl100k_base, loaded once per process."""
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding(ENCODING)
    return _ENC


def count_tokens(text: str) -> int:
    """`encode_ordinary` so a literal `<|endoftext|>` in a transcript is text.

    The special-token-aware `encode` raises on those strings, and transcripts of
    sessions that discuss tokenisers contain them.
    """
    return len(encoder().encode_ordinary(text))


# --------------------------------------------------------------------------
# population
# --------------------------------------------------------------------------

def manifest_population(manifest: dict) -> List[str]:
    """The only sessions this measurement may touch: the manifest's paired set."""
    return list(manifest["paired_session_ids"])


def transcript_path_for(sid: str, manifest: dict) -> Optional[str]:
    """The manifest transcript whose stem is this session id, or None.

    Keyed off the manifest list rather than a filesystem glob so a transcript
    written since the snapshot cannot enter the corpus.
    """
    for p in manifest["transcripts"]:
        if Path(p).stem == sid:
            return p
    return None


def git_blob(rel_path: str) -> Optional[bytes]:
    """`<PIN_COMMIT>:<rel_path>` as bytes, or None if it is not in that tree."""
    r = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{PIN_COMMIT}:{rel_path}"],
        capture_output=True,
    )
    return r.stdout if r.returncode == 0 else None


def pinned_rollup_text(sid: str, manifest: dict) -> Tuple[Optional[str], str]:
    """Rollup note content at snapshot time, from the pinned commit.

    Returns (text, reason). `reason` is the drop reason when text is None, and
    "pinned_commit_blob" when it is not.
    """
    path = manifest.get("rollup_note_paths", {}).get(sid)
    pinned_bytes = manifest.get("rollup_note_bytes", {}).get(sid)
    if path is None or pinned_bytes is None:
        # 24 of the 300 paired sessions: the retroactive backfill could not
        # match a note to them, so they have no valid pin at all.
        return None, "no_rollup_pin_in_manifest"
    blob = git_blob(f"sessions/{Path(path).name}")
    if blob is None:
        return None, "rollup_absent_from_pinned_commit"
    if len(blob) != pinned_bytes:
        # The pin and the blob disagree, so one of them is not the snapshot's
        # content and there is no way to tell which. Drop rather than guess.
        return None, "rollup_pin_bytes_mismatch"
    return blob.decode(errors="replace"), "pinned_commit_blob"


# --------------------------------------------------------------------------
# per-session measurement
# --------------------------------------------------------------------------

def session_measurement(sid: str, rollup_text: str, raw_text: str) -> dict:
    """Token and byte counts for the three arms of one session.

    Pure in its inputs: the two texts are supplied, so the degenerate cases
    (an empty rollup, a one-record transcript) are testable without touching
    the corpus. A zero-token rollup is returned as a drop rather than as a
    ratio, because dividing by it would invent an infinity.
    """
    stripped_text, strip_stats = strip_transcript(raw_text)

    rollup_tok = count_tokens(rollup_text)
    stripped_tok = count_tokens(stripped_text)
    raw_tok = count_tokens(raw_text)

    # Bytes are taken from the same text objects that were tokenised, so the
    # two denominators describe the same artifact. The manifest's pinned byte
    # length is carried alongside for the raw arm as a provenance check.
    rollup_b = len(rollup_text.encode("utf-8"))
    stripped_b = len(stripped_text.encode("utf-8"))
    raw_b = len(raw_text.encode("utf-8"))

    if rollup_tok == 0 or rollup_b == 0:
        return {"session_id": sid, "drop": "rollup_empty",
                "rollup_tokens": rollup_tok, "rollup_bytes": rollup_b}

    return {
        "session_id": sid,
        "rollup_tokens": rollup_tok,
        "stripped_tokens": stripped_tok,
        "raw_tokens": raw_tok,
        "rollup_bytes": rollup_b,
        "stripped_bytes": stripped_b,
        "raw_bytes": raw_b,
        "raw_over_rollup_tokens": raw_tok / rollup_tok,
        "stripped_over_rollup_tokens": stripped_tok / rollup_tok,
        "raw_over_rollup_bytes": raw_b / rollup_b,
        "stripped_over_rollup_bytes": stripped_b / rollup_b,
        "rollup_bytes_per_token": rollup_b / rollup_tok,
        "stripped_bytes_per_token": stripped_b / max(stripped_tok, 1),
        "raw_bytes_per_token": raw_b / max(raw_tok, 1),
        "tool_calls": strip_stats["tool_calls"],
        "dropped_results": strip_stats["dropped_results"],
    }


def measure_one(job: Tuple[str, str, str]) -> dict:
    """Worker entry point: (sid, transcript_path, rollup_text) -> measurement."""
    sid, tpath, rollup_text = job
    manifest = load_manifest()
    try:
        raw_text = read_pinned(tpath, manifest)
    except OSError as exc:
        return {"session_id": sid, "drop": "transcript_unreadable",
                "detail": f"{type(exc).__name__}: {exc}"}
    row = session_measurement(sid, rollup_text, raw_text)
    row["transcript"] = tpath
    row["manifest_transcript_bytes"] = manifest["transcript_bytes"].get(tpath)
    return row


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def quantiles(vals: List[float]) -> dict:
    """Median, mean and tails of a per-session ratio distribution."""
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return {"n": 0}
    def q(p: float) -> float:
        return s[min(n - 1, max(0, int(round(p * (n - 1)))))]
    return {
        "n": n,
        "median": statistics.median(s),
        "mean": statistics.fmean(s),
        "p10": q(0.10),
        "p25": q(0.25),
        "p75": q(0.75),
        "p90": q(0.90),
        "min": s[0],
        "max": s[-1],
    }


def bootstrap_median_ci(vals: List[float], iters: int = BOOTSTRAP_ITERS,
                        seed: int = BOOTSTRAP_SEED) -> List[float]:
    """Session-level bootstrap 95% CI on the median of a per-session ratio.

    Resamples sessions with replacement, which is the unit of independence
    here: two questions from one session are not two observations of it.
    """
    n = len(vals)
    if n == 0:
        return [0.0, 0.0]
    rng = random.Random(seed)
    meds = []
    for _ in range(iters):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        meds.append(statistics.median(sample))
    meds.sort()
    lo = meds[int(0.025 * (iters - 1))]
    hi = meds[int(0.975 * (iters - 1))]
    return [round(lo, 4), round(hi, 4)]


def aggregate(rows: List[dict], drops: List[dict], manifest: dict,
              extra: Optional[dict] = None, iters: int = BOOTSTRAP_ITERS) -> dict:
    """Build the report. Refuses to aggregate a dropped session.

    The guard is the point: a drop absorbed as a zero, or as a ratio of 1,
    would move the median silently, and this corpus has at least one session
    (`bb26f10d`) whose transcript has been pruned since the snapshot.
    """
    for r in rows:
        if "drop" in r:
            raise ValueError(f"dropped session in the kept set: {r['session_id']}")
        for k in ("rollup_tokens", "stripped_tokens", "raw_tokens"):
            if not isinstance(r.get(k), int):
                raise ValueError(f"{r.get('session_id')} missing {k}")

    def col(key: str) -> List[float]:
        return [r[key] for r in rows]

    report: dict = {
        "corpus_snapshot_utc": manifest["snapshot_utc"],
        "question": ("what a whole-artifact read costs in TOKENS: rollup vs "
                     "tool-stripped transcript vs raw transcript, on the same "
                     "sessions, with the byte ratios beside them"),
        "method": {
            "tokeniser": (f"tiktoken {ENCODING}, encode_ordinary (special "
                          "tokens treated as literal text); exact counts, not "
                          "an estimate"),
            "population": ("corpus_snapshot.json paired_session_ids only; "
                           "sessions/*.md is never globbed"),
            "rollup_arm": (f"content of the pinned note at git {PIN_COMMIT}, "
                           "verified byte-length-equal to the manifest's "
                           "rollup_note_bytes for every kept session"),
            "stripped_arm": ("strip_transcript() imported from "
                             "rq3_stripped_transcript_arm.py (L2), the same "
                             "renderer the RQ3 judged stripped arm read"),
            "raw_arm": ("freeze_corpus.read_pinned(), truncated at the "
                        "manifest's snapshot byte length"),
            "bytes": ("utf-8 length of the same three text objects that were "
                      "tokenised, so tokens and bytes describe one artifact"),
            "ratio_construction": ("per-session pair ratios, then aggregated; "
                                   "never a quotient of two corpus totals"),
            "ci": (f"session-level bootstrap, {iters} resamples, "
                   f"seed {BOOTSTRAP_SEED}, percentile 2.5/97.5 on the median"),
        },
        "n_paired_in_manifest": manifest["n_paired"],
        "n_measured": len(rows),
        "n_dropped": len(drops),
        "drops": drops,
        "token_ratios": {
            "raw_over_rollup": quantiles(col("raw_over_rollup_tokens")),
            "stripped_over_rollup": quantiles(col("stripped_over_rollup_tokens")),
        },
        "byte_ratios": {
            "raw_over_rollup": quantiles(col("raw_over_rollup_bytes")),
            "stripped_over_rollup": quantiles(col("stripped_over_rollup_bytes")),
        },
        "ci95_median": {
            "raw_over_rollup_tokens":
                bootstrap_median_ci(col("raw_over_rollup_tokens"), iters),
            "stripped_over_rollup_tokens":
                bootstrap_median_ci(col("stripped_over_rollup_tokens"), iters),
            "raw_over_rollup_bytes":
                bootstrap_median_ci(col("raw_over_rollup_bytes"), iters),
            "stripped_over_rollup_bytes":
                bootstrap_median_ci(col("stripped_over_rollup_bytes"), iters),
        },
        "absolute_tokens": {
            "rollup": quantiles([float(x) for x in col("rollup_tokens")]),
            "stripped": quantiles([float(x) for x in col("stripped_tokens")]),
            "raw": quantiles([float(x) for x in col("raw_tokens")]),
        },
        "bytes_per_token": {
            "rollup": quantiles(col("rollup_bytes_per_token")),
            "stripped": quantiles(col("stripped_bytes_per_token")),
            "raw": quantiles(col("raw_bytes_per_token")),
        },
    }

    # The divergence, stated two ways. The ratio of medians is what a reader
    # comparing two published tables computes; the median of per-session
    # ratio-of-ratios is the paired version and is the defensible one.
    div = {}
    for arm in ("raw_over_rollup", "stripped_over_rollup"):
        tok = report["token_ratios"][arm]["median"]
        byt = report["byte_ratios"][arm]["median"]
        paired = [r[f"{arm}_tokens"] / r[f"{arm}_bytes"] for r in rows]
        div[arm] = {
            "median_token_ratio": tok,
            "median_byte_ratio": byt,
            "token_over_byte_ratio_of_medians": tok / byt,
            "token_over_byte_paired_median": statistics.median(paired),
            "token_over_byte_paired_ci95": bootstrap_median_ci(paired, iters),
            "pct_token_ratio_above_byte_ratio": 100.0 * (tok / byt - 1),
        }
    report["token_vs_byte_divergence"] = div

    # The direction is computed, never asserted. The first draft of this line
    # stated that raw JSONL tokenises worse per byte than prose and that the
    # byte ratio therefore overstates the read -- which is backwards on this
    # corpus, and a hardcoded sentence would have published it.
    bpt = report["bytes_per_token"]
    raw_dir = div["raw_over_rollup"]["token_over_byte_paired_median"]
    verb = "UNDERSTATES" if raw_dir > 1 else "OVERSTATES"
    report["divergence_mechanism"] = (
        f"a rollup is markdown prose at {bpt['rollup']['median']:.2f} "
        f"bytes/token median, the raw JSONL is {bpt['raw']['median']:.2f} and "
        f"the stripped rendering {bpt['stripped']['median']:.2f}; a transcript "
        "packs more tokens into a byte than prose does because its ids, paths "
        "and punctuation fragment, so on the raw arm the byte ratio "
        f"{verb} the read by a paired median factor of {raw_dir:.3f}"
    )
    if extra:
        report.update(extra)
    return report


# --------------------------------------------------------------------------
# tokeniser validation against real billed token counts
# --------------------------------------------------------------------------

def _jsonl(path: Path, required_keys: set[str] | None = None) -> List[dict]:
    """Load a JSONL file, strictly raising on any parse failure."""
    result = load_jsonl_strict(path, required_keys=required_keys)
    if result.failed > 0:
        raise ValueError(
            f"{path.name}: {result.failed} unparseable lines detected. "
            f"Parsed {result.parsed} of {result.attempted} attempted."
        )
    return result.records


def _ols(pairs: List[Tuple[float, float]]) -> Tuple[float, float]:
    """billed = slope * cl100k + intercept. The intercept is the fixed prompt
    wrapper, which a bare billed/cl100k mean would otherwise attribute to the
    tokeniser and would inflate worst on the smallest artifact -- the rollup."""
    n = len(pairs)
    sx = sum(p[0] for p in pairs)
    sy = sum(p[1] for p in pairs)
    sxx = sum(p[0] * p[0] for p in pairs)
    sxy = sum(p[0] * p[1] for p in pairs)
    den = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / den
    return slope, (sy - slope * sx) / n


def tokeniser_validation(manifest: dict) -> dict:
    """Check cl100k against the gateway's own prompt_tokens, per artifact class.

    cl100k_base is not Claude's tokeniser, so its absolute counts are wrong by
    some factor. That would not matter for a ratio IF the factor were the same
    for markdown prose and for minified JSONL -- and whether it is, is an
    empirical question this corpus can answer, because the RQ3 arms recorded
    real `usage.prompt_tokens` for all three artifact classes. Rows are used
    only where the artifact can be reproduced byte-exactly today (pinned
    transcript length equal to the recorded length, and the recorded character
    count equal to the reproduction's), so a file that grew since RQ3 ran is
    excluded rather than compared against the wrong text.
    """
    tb = manifest["transcript_bytes"]
    classes: Dict[str, List[Tuple[float, float]]] = {}

    raw_rows = _jsonl(DATA / "rq3_questions_raw_arm.jsonl")
    cache: Dict[str, int] = {}
    pairs = []
    for r in raw_rows:
        t = r.get("transcript")
        if t is None or tb.get(t) != r.get("transcript_bytes"):
            continue
        if t not in cache:
            cache[t] = count_tokens(read_pinned(t, manifest))
        pairs.append((float(cache[t]), float(r["usage"]["prompt_tokens"])))
    classes["raw"] = pairs

    strip_rows = _jsonl(DATA / "rq3_questions_stripped_arm.jsonl")
    scache: Dict[str, str] = {}
    pairs = []
    for r in strip_rows:
        t = r.get("transcript")
        if t is None or tb.get(t) != r.get("raw_bytes"):
            continue
        if t not in scache:
            scache[t] = strip_transcript(read_pinned(t, manifest))[0]
        if len(scache[t]) != r.get("stripped_chars"):
            continue
        pairs.append((float(count_tokens(scache[t])),
                      float(r["usage"]["prompt_tokens"])))
    classes["stripped"] = pairs

    roll_rows = _jsonl(DATA / "rq3_rollup_uniform.jsonl")
    pairs = []
    for r in roll_rows:
        name = Path(r.get("rollup", "")).name
        if not name:
            continue
        blob = git_blob(f"sessions/{name}")
        if blob is None:
            continue
        text = blob.decode(errors="replace")
        if len(text) != r.get("rollup_chars"):
            continue
        pairs.append((float(count_tokens(text)),
                      float(r["usage"]["prompt_tokens"])))
    classes["rollup"] = pairs

    fits = {}
    for name, pairs in classes.items():
        if len(pairs) < 3:
            fits[name] = {"n": len(pairs)}
            continue
        slope, intercept = _ols(pairs)
        fits[name] = {
            "n": len(pairs),
            "billed_per_cl100k_slope": round(slope, 4),
            "fixed_prompt_tokens_intercept": round(intercept, 1),
        }

    out = {
        "why": ("cl100k_base is not Claude's tokeniser; this checks whether its "
                "error is the same for all three artifact classes, because only "
                "then is a cl100k RATIO unbiased even though every cl100k "
                "absolute count is low"),
        "source": ("usage.prompt_tokens recorded by the RQ3 arms, restricted to "
                   "rows whose artifact reproduces byte-exactly today"),
        "per_class": fits,
    }
    if all("billed_per_cl100k_slope" in fits[k] for k in ("raw", "stripped", "rollup")):
        sr = fits["raw"]["billed_per_cl100k_slope"]
        ss = fits["stripped"]["billed_per_cl100k_slope"]
        sp = fits["rollup"]["billed_per_cl100k_slope"]
        out["ratio_bias"] = {
            "raw_over_rollup": round(sr / sp, 4),
            "stripped_over_rollup": round(ss / sp, 4),
            "reading": ("multiply a cl100k ratio by this to get the "
                        "billed-token ratio; 1.0 means no bias"),
        }
        out["absolute_undercount"] = (
            f"cl100k undercounts billed tokens by {sp:.2f}x on rollups, "
            f"{ss:.2f}x on stripped renderings and {sr:.2f}x on raw "
            "transcripts, so the absolute token counts here are low by about "
            "half again, while the ratios are within "
            f"{100*max(abs(sr/sp-1), abs(ss/sp-1)):.1f}% of the billed ones"
        )
    return out


# --------------------------------------------------------------------------
# smoke tests
# --------------------------------------------------------------------------

ONE_RECORD = json.dumps({
    "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]}
})


def smoke(manifest: dict, n_real: int = 3) -> None:
    """Prove the pipeline on a few real sessions and on degenerate inputs.

    A full-corpus first run cannot distinguish "the renderer works" from "the
    renderer returned an empty string 300 times", so this runs first.
    """
    pop = manifest_population(manifest)
    print("=== smoke: real sessions ===")
    picked, i = [], 0
    while len(picked) < n_real and i < len(pop):
        sid = pop[i]
        i += 1
        text, reason = pinned_rollup_text(sid, manifest)
        tpath = transcript_path_for(sid, manifest)
        if text is None or tpath is None or not Path(tpath).exists():
            continue
        picked.append((sid, tpath, text))
    for sid, tpath, text in picked:
        row = measure_one((sid, tpath, text))
        print(f"  {sid[:8]}  rollup {row['rollup_tokens']:>7,}tok  "
              f"stripped {row['stripped_tokens']:>9,}tok  "
              f"raw {row['raw_tokens']:>9,}tok  "
              f"raw/rollup {row['raw_over_rollup_tokens']:8.1f}x  "
              f"stripped/rollup {row['stripped_over_rollup_tokens']:7.1f}x")
    assert len(picked) == n_real, "could not resolve enough real sessions"

    print("=== smoke: degenerate inputs ===")
    empty = session_measurement("EMPTY-ROLLUP", "", ONE_RECORD)
    print(f"  empty rollup             -> drop={empty.get('drop')!r} "
          f"(rollup_tokens={empty['rollup_tokens']})")
    assert empty.get("drop") == "rollup_empty", empty

    one = session_measurement("ONE-RECORD", "# a rollup with some prose\n", ONE_RECORD)
    print(f"  one-record transcript    -> raw {one['raw_tokens']}tok "
          f"stripped {one['stripped_tokens']}tok "
          f"raw/rollup {one['raw_over_rollup_tokens']:.2f}x")
    assert one["raw_tokens"] > 0 and one["stripped_tokens"] > 0
    assert "drop" not in one

    missing = [s for s in pop
               if (p := transcript_path_for(s, manifest)) and not Path(p).exists()]
    print(f"  transcripts pruned since snapshot: {len(missing)} "
          f"({', '.join(s[:8] for s in missing) or 'none'})")
    for sid in missing:
        row = measure_one((sid, transcript_path_for(sid, manifest), "x"))
        print(f"    {sid[:8]} -> drop={row.get('drop')!r}")
        assert row.get("drop") == "transcript_unreadable", row

    no_pin = [s for s in pop if pinned_rollup_text(s, manifest)[0] is None]
    print(f"  paired sessions with no valid rollup pin: {len(no_pin)}")

    try:
        aggregate([{"session_id": "D", "drop": "transcript_unreadable"}], [], manifest)
    except ValueError as exc:
        print(f"  aggregate() refuses a dropped session: {exc}")
    else:  # pragma: no cover
        raise AssertionError("aggregate accepted a dropped session")
    print("=== smoke passed ===\n")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="prove the pipeline on 3 real + degenerate inputs, then stop")
    ap.add_argument("--limit", type=int, default=0,
                    help="measure only the first N paired sessions")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the cl100k-vs-billed-tokens cross-check")
    args = ap.parse_args()

    manifest = load_manifest()
    print(f"snapshot {manifest['snapshot_utc']}  paired {manifest['n_paired']}")

    smoke(manifest, n_real=3)
    if args.smoke:
        return

    pop = manifest_population(manifest)
    if args.limit:
        pop = pop[:args.limit]

    jobs: List[Tuple[str, str, str]] = []
    drops: List[dict] = []
    live_pin_ok = 0
    for sid in pop:
        rollup, reason = pinned_rollup_text(sid, manifest)
        tp = transcript_path_for(sid, manifest)
        if rollup is None:
            # The transcript state is recorded even when the rollup is what
            # dropped the session, so a session that fails BOTH checks -- as
            # bb26f10d does -- is not filed under the first one and lost.
            drops.append({"session_id": sid, "reason": reason,
                          "transcript_also_missing":
                              tp is None or not Path(tp).exists()})
            continue
        if read_pinned_rollup(sid, manifest) is not None:
            live_pin_ok += 1
        if tp is None:
            drops.append({"session_id": sid, "reason": "no_manifest_transcript"})
            continue
        if not Path(tp).exists():
            drops.append({"session_id": sid,
                          "reason": "transcript_pruned_since_snapshot",
                          "transcript": tp})
            continue
        jobs.append((sid, tp, rollup))

    total_mb = sum(manifest["transcript_bytes"].get(j[1], 0) for j in jobs) / 1e6
    print(f"measuring {len(jobs)} sessions ({total_mb:,.0f} MB of pinned "
          f"transcript), {len(drops)} dropped before reading")

    rows: List[dict] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, row in enumerate(pool.map(measure_one, jobs, chunksize=1), 1):
            if "drop" in row:
                drops.append({"session_id": row["session_id"],
                              "reason": row["drop"],
                              **({"detail": row["detail"]} if "detail" in row else {})})
            else:
                rows.append(row)
            if i % 25 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] {time.time()-t0:.0f}s")

    extra = {
        "tokeniser_validation": (
            {"skipped": "--no-validate"} if args.no_validate
            else tokeniser_validation(manifest)),
        "rollup_source_crosscheck": {
            "pinned_commit_blob_ok": len(jobs),
            "working_tree_read_pinned_rollup_ok": live_pin_ok,
            "note": ("read_pinned_rollup() reads the live sessions/ path and "
                     "returns None once the drain has renamed or rewritten the "
                     "note; the pinned commit is byte-identical where both "
                     "agree and survives the rename, which is why it is the "
                     "rollup source here"),
        },
    }
    report = aggregate(rows, drops, manifest, extra=extra)

    t = report["token_ratios"]
    b = report["byte_ratios"]
    c = report["ci95_median"]
    print(f"\nn={report['n_measured']}  dropped={report['n_dropped']}")
    print(f"  raw/rollup      tokens median {t['raw_over_rollup']['median']:8.1f}x "
          f"CI {c['raw_over_rollup_tokens']}   "
          f"mean {t['raw_over_rollup']['mean']:.1f}x")
    print(f"                  bytes  median {b['raw_over_rollup']['median']:8.1f}x "
          f"CI {c['raw_over_rollup_bytes']}   "
          f"mean {b['raw_over_rollup']['mean']:.1f}x")
    print(f"  stripped/rollup tokens median {t['stripped_over_rollup']['median']:8.1f}x "
          f"CI {c['stripped_over_rollup_tokens']}   "
          f"mean {t['stripped_over_rollup']['mean']:.1f}x")
    print(f"                  bytes  median {b['stripped_over_rollup']['median']:8.1f}x "
          f"CI {c['stripped_over_rollup_bytes']}   "
          f"mean {b['stripped_over_rollup']['mean']:.1f}x")
    for arm, d in report["token_vs_byte_divergence"].items():
        pct = d["pct_token_ratio_above_byte_ratio"]
        print(f"  {arm}: ratio-of-medians puts tokens {abs(pct):.1f}% "
              f"{'ABOVE' if pct >= 0 else 'BELOW'} bytes; paired per-session "
              f"median {d['token_over_byte_paired_median']:.3f} "
              f"CI {d['token_over_byte_paired_ci95']}")
    print(f"  {report['divergence_mechanism']}")
    tv = report.get("tokeniser_validation", {})
    if "ratio_bias" in tv:
        print(f"  tokeniser check: {tv['absolute_undercount']}")
        print(f"  ratio bias vs billed tokens: {tv['ratio_bias']}")

    if args.no_write:
        print("\n--no-write: nothing written")
        return
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
