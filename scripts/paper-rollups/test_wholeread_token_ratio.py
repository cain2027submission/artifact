#!/usr/bin/env python3
"""Gate for rq5_wholeread_token_ratio.py.

Four things are checked, and each one is here because getting it wrong would
produce a number that looks fine:

  1. the ratio arithmetic -- per-session pairs aggregated, never a quotient of
     two corpus totals, which is the failure mode where one 55 MB transcript
     decides the headline;
  2. the manifest filter -- the population is `paired_session_ids` and nothing
     else, because `sessions/` has grown from 499 notes to 1,114 since the
     snapshot and a glob would silently widen the corpus;
  3. the degenerate inputs -- an empty rollup, a one-record transcript, a
     session whose transcript is gone;
  4. that no ratio is ever computed from a dropped session, including the case
     where a drop is handed to the aggregator by mistake.

Run:
    python3 scripts/paper-rollups/test_wholeread_token_ratio.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import rq5_wholeread_token_ratio as M  # noqa: E402
from paths import PROJECTS_DIR  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(f"{name}{(' -- ' + detail) if detail else ''}")
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))


def fake_manifest(rows: dict) -> dict:
    return {
        "snapshot_utc": "2026-09-27T09:29:21+00:00",
        "n_paired": len(rows),
        "paired_session_ids": sorted(rows),
        "transcripts": list(rows.values()),
        "transcript_bytes": {v: 10 for v in rows.values()},
        "rollup_note_paths": {},
        "rollup_note_bytes": {},
    }


# --------------------------------------------------------------------------
# 1. ratio arithmetic
# --------------------------------------------------------------------------

def synth(sid: str, rollup_tok: int, stripped_tok: int, raw_tok: int) -> dict:
    """A measurement row with token counts chosen so the ratios are exact."""
    return {
        "session_id": sid,
        "rollup_tokens": rollup_tok,
        "stripped_tokens": stripped_tok,
        "raw_tokens": raw_tok,
        "rollup_bytes": rollup_tok * 4,
        "stripped_bytes": stripped_tok * 4,
        "raw_bytes": raw_tok * 2,
        "raw_over_rollup_tokens": raw_tok / rollup_tok,
        "stripped_over_rollup_tokens": stripped_tok / rollup_tok,
        "raw_over_rollup_bytes": (raw_tok * 2) / (rollup_tok * 4),
        "stripped_over_rollup_bytes": (stripped_tok * 4) / (rollup_tok * 4),
        "rollup_bytes_per_token": 4.0,
        "stripped_bytes_per_token": 4.0,
        "raw_bytes_per_token": 2.0,
        "tool_calls": 0,
        "dropped_results": 0,
    }


rows = [
    synth("aaa", 1_000, 10_000, 100_000),    # raw/rollup 100x, stripped 10x
    synth("bbb", 1_000, 20_000, 200_000),    # 200x, 20x
    synth("ccc", 1_000, 30_000, 300_000),    # 300x, 30x
    # A whale: 100x the tokens of the others, and a much SMALLER ratio. A
    # total-over-total computation would let it set the answer.
    synth("ddd", 1_000_000, 2_000_000, 4_000_000),   # 4x, 2x
]
man = fake_manifest({s["session_id"]: f"/t/{s['session_id']}.jsonl" for s in rows})
rep = M.aggregate(rows, [], man, iters=200)

check("median raw/rollup token ratio is the per-session median",
      rep["token_ratios"]["raw_over_rollup"]["median"] == statistics.median(
          [100.0, 200.0, 300.0, 4.0]),
      f"got {rep['token_ratios']['raw_over_rollup']['median']}")

pooled = sum(r["raw_tokens"] for r in rows) / sum(r["rollup_tokens"] for r in rows)
check("per-session median is not the pooled total-over-total ratio",
      abs(rep["token_ratios"]["raw_over_rollup"]["median"] - pooled) > 1.0,
      f"median {rep['token_ratios']['raw_over_rollup']['median']:.2f} vs pooled {pooled:.2f}")

check("median stripped/rollup token ratio",
      rep["token_ratios"]["stripped_over_rollup"]["median"] == statistics.median(
          [10.0, 20.0, 30.0, 2.0]),
      f"got {rep['token_ratios']['stripped_over_rollup']['median']}")

check("mean is reported alongside the median",
      abs(rep["token_ratios"]["raw_over_rollup"]["mean"]
          - statistics.fmean([100.0, 200.0, 300.0, 4.0])) < 1e-9)

# bytes here are constructed at half the tokens for raw and 4x for rollups, so
# every byte ratio must be exactly half its token ratio.
check("byte ratios are computed from the byte columns, not rescaled tokens",
      rep["byte_ratios"]["raw_over_rollup"]["median"]
      == rep["token_ratios"]["raw_over_rollup"]["median"] / 2,
      f"bytes {rep['byte_ratios']['raw_over_rollup']['median']}")

check("divergence factor is tokens over bytes",
      abs(rep["token_vs_byte_divergence"]["raw_over_rollup"][
          "token_over_byte_paired_median"] - 2.0) < 1e-9)

check("the ratio-of-medians divergence percentage is signed the same way",
      abs(rep["token_vs_byte_divergence"]["raw_over_rollup"][
          "pct_token_ratio_above_byte_ratio"] - 100.0) < 1e-9,
      "token ratio is exactly double the byte ratio in this fixture")

check("divergence direction is read off the data, not asserted",
      "UNDERSTATES" in rep["divergence_mechanism"],
      "tokens exceed bytes in this fixture, so the prose must say UNDERSTATES")

lo, hi = rep["ci95_median"]["raw_over_rollup_tokens"]
check("bootstrap CI brackets the median", lo <= 150.0 <= hi, f"[{lo}, {hi}]")
check("bootstrap CI is deterministic under a fixed seed",
      M.bootstrap_median_ci([100.0, 200.0, 300.0, 4.0], 500)
      == M.bootstrap_median_ci([100.0, 200.0, 300.0, 4.0], 500))

single = M.bootstrap_median_ci([7.0], 100)
check("bootstrap on one session collapses to that session",
      single == [7.0, 7.0], str(single))
check("bootstrap on an empty list does not raise",
      M.bootstrap_median_ci([], 10) == [0.0, 0.0])

# --------------------------------------------------------------------------
# 2. no ratio from a dropped session
# --------------------------------------------------------------------------

try:
    M.aggregate(rows + [{"session_id": "zzz", "drop": "transcript_unreadable"}],
                [], man, iters=10)
except ValueError as exc:
    check("aggregate refuses a dropped session in the kept set", True, str(exc))
else:
    check("aggregate refuses a dropped session in the kept set", False,
          "a drop was aggregated")

try:
    M.aggregate(rows + [{"session_id": "yyy", "rollup_tokens": 5}], [], man, iters=10)
except ValueError as exc:
    check("aggregate refuses a row missing a token count", True, str(exc))
else:
    check("aggregate refuses a row missing a token count", False, "accepted")

drops = [{"session_id": "bb26f10d-14a4-4318-aead-512909db8405",
          "reason": "transcript_pruned_since_snapshot"}]
rep2 = M.aggregate(rows, drops, man, iters=50)
check("drops are recorded rather than absorbed as zeros",
      rep2["n_dropped"] == 1 and rep2["n_measured"] == len(rows)
      and rep2["token_ratios"]["raw_over_rollup"]["n"] == len(rows),
      f"n_measured {rep2['n_measured']} n_dropped {rep2['n_dropped']} "
      f"ratio n {rep2['token_ratios']['raw_over_rollup']['n']}")
check("a drop carries its reason into the report",
      rep2["drops"][0]["reason"] == "transcript_pruned_since_snapshot")
check("the report states the manifest denominator beside n",
      rep2["n_paired_in_manifest"] == len(rows))

# --------------------------------------------------------------------------
# 3. manifest filter
# --------------------------------------------------------------------------

m3 = {
    "snapshot_utc": "x",
    "n_paired": 2,
    "paired_session_ids": ["sid-a", "sid-b"],
    # a third transcript exists in the manifest but is NOT paired, and a fourth
    # exists nowhere in the manifest at all -- the growth case.
    "transcripts": ["/p/sid-a.jsonl", "/p/sid-b.jsonl", "/p/sid-c.jsonl"],
    "transcript_bytes": {"/p/sid-a.jsonl": 1, "/p/sid-b.jsonl": 1, "/p/sid-c.jsonl": 1},
    "rollup_note_paths": {"sid-a": "/v/sessions/a.md", "sid-b": "/v/sessions/b.md"},
    "rollup_note_bytes": {"sid-a": 10, "sid-b": 10},
}
pop = M.manifest_population(m3)
check("population is exactly paired_session_ids", pop == ["sid-a", "sid-b"], str(pop))
check("an unpaired manifest transcript is not in the population",
      "sid-c" not in pop)
check("transcript_path_for resolves by manifest stem",
      M.transcript_path_for("sid-b", m3) == "/p/sid-b.jsonl")
check("transcript_path_for returns None for a session outside the manifest",
      M.transcript_path_for("sid-newer-than-snapshot", m3) is None)

check("a session with no rollup pin is dropped, not defaulted",
      M.pinned_rollup_text("sid-z", m3) == (None, "no_rollup_pin_in_manifest"),
      str(M.pinned_rollup_text("sid-z", m3)))

# The real manifest: the population must be the frozen 300 and the script must
# never widen it to the live sessions/ folder.
real = M.load_manifest()
check("real manifest population is the frozen paired set",
      len(M.manifest_population(real)) == real["n_paired"] == 300,
      f"n_paired {real['n_paired']}")
live_sessions = len(list((M.REPO / "sessions").glob("*.md")))
check("the live sessions/ folder is larger than the frozen corpus, so the "
      "filter is load-bearing",
      live_sessions > real["n_rollups"],
      f"{live_sessions} notes on disk vs {real['n_rollups']} frozen")
src = HERE.parent.joinpath("rq5_wholeread_token_ratio.py").read_text()
check("the measurement never globs sessions/*.md",
      'glob("*.md")' not in src and "glob('*.md')" not in src)
check("the stripping function is imported from the RQ3 arm, not redefined",
      "from rq3_stripped_transcript_arm import strip_transcript" in src
      and "def strip_transcript" not in src)

# --------------------------------------------------------------------------
# 4. degenerate inputs
# --------------------------------------------------------------------------

one_record = json.dumps({"message": {"role": "user",
                                     "content": [{"type": "text", "text": "hello"}]}})

d = M.session_measurement("EMPTY", "", one_record)
check("an empty rollup is a drop, not a division by zero",
      d.get("drop") == "rollup_empty", str(d))
check("an empty rollup carries no ratio at all",
      "raw_over_rollup_tokens" not in d)

d = M.session_measurement("ONE", "# rollup prose\n", one_record)
check("a one-record transcript measures without raising",
      "drop" not in d and d["raw_tokens"] > 0 and d["stripped_tokens"] > 0,
      f"raw {d['raw_tokens']}tok stripped {d['stripped_tokens']}tok")
check("a one-record transcript's stripped rendering keeps the user prose",
      d["stripped_tokens"] > 0)

d = M.session_measurement("EMPTYTRANSCRIPT", "# rollup prose\n", "")
check("an empty transcript still measures, with the rollup as denominator",
      "drop" not in d and d["raw_tokens"] == 0
      and d["raw_over_rollup_tokens"] == 0.0,
      f"raw/rollup {d.get('raw_over_rollup_tokens')}")

d = M.session_measurement("GARBAGE", "# rollup prose\n", "not json\n{oops\n")
check("an unparseable transcript is counted, not skipped",
      "drop" not in d and d["raw_tokens"] > 0)

missing = M.measure_one(("GONE", str(PROJECTS_DIR / "does-not-exist.jsonl"), "x"))
check("a missing transcript is a drop with a reason",
      missing.get("drop") == "transcript_unreadable", str(missing)[:120])
check("a missing transcript produces no token counts",
      "raw_tokens" not in missing)

# The real pruned session named in the brief, end to end.
pruned = "bb26f10d-14a4-4318-aead-512909db8405"
check("bb26f10d is in the frozen population", pruned in M.manifest_population(real))
p = M.transcript_path_for(pruned, real)
check("bb26f10d's manifest transcript no longer exists on disk",
      p is not None and not Path(p).exists(), str(p))
row = M.measure_one((pruned, p, "# some rollup\n"))
check("bb26f10d is dropped rather than measured",
      row.get("drop") == "transcript_unreadable", str(row)[:120])

# --------------------------------------------------------------------------
# tokeniser
# --------------------------------------------------------------------------

check("cl100k_base counts a literal special token as text rather than raising",
      M.count_tokens("<|endoftext|> in a transcript") > 0)
check("cl100k_base is the declared encoding", M.ENCODING == "cl100k_base")
check("token counts are exact, not a chars-per-token estimate",
      M.count_tokens("aaaa") != M.count_tokens("the quick brown fox jumped"))

# --------------------------------------------------------------------------

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print(f"  FAILED: {f}")
    sys.exit(1)
print("gate: ok")
