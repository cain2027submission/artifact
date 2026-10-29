#!/usr/bin/env python3
"""RQ5 correction: the rollup generator never reads a warm cache.

RQ5 as published prices generation twice, "uncached" and "cached", and quotes
the cached column as the headline. The cached column exists because
`rq5_generation_cost.py` believed the pipeline summarises "at session end from a
context that is already resident, so the input is billed at the cache-read
rate" (paper/main.tex, the paragraph above tab:breakeven). That is the claim
this script tests, and it is false.

The pipeline is not in-session. A `SessionEnd` hook enqueues the session and a
launchd job (`com.agent-session-rollup.drain`, StartInterval 300) runs
`rollup-drain` in a separate process, which shells out to `claude -p` in its own
spawn directory with a freshly-built summariser prompt. That process shares no
prompt prefix with the session being summarised, so the transcript can never be
a cache READ on its first pass. It is a cache WRITE, billed at 1.25x input, not
0.1x input.

The evidence is a ledger the pipeline itself writes. `rollup_drain.py` appends
one JSON line per drain run to `~/.claude/rollup-spend.jsonl`, summing the
`usage` block that `claude -p --output-format json` returns per call. That file
is the measurement: it is not transcript `message.usage`, so it needs neither
the ~1.5x transcript-overstatement correction nor per-content-block requestId
deduping -- there is exactly one usage record per API call, aggregated by
`record_spend()`, and one ledger line per drain run.

What the ledger shows, and it is the whole result: `input_tokens` is a rounding
error (a median of single digits), `cache_creation_tokens` carries the
transcript, and `cache_read_tokens` is a minority share that comes from turns
WITHIN a call re-reading the prompt the same call just wrote. There is no
inherited warm cache anywhere in it.

One thing this script identifies rather than assumes: the price table. The drain
plist sets PKM_ROLLUP_MODEL=claude-sonnet-5 and the drain's own self-notes
record `model: claude-sonnet-5`, but the `total_cost_usd` the CLI reports comes
out at exactly one third of Sonnet arithmetic, which is Haiku 4.5's scale. Both
readings are reported because the structural finding does not depend on which
is right: whichever price table applies, the transcript bills as a cache WRITE.

Usage:
    rq5_cold_cache_correction.py                # frozen ledger (reproducible)
    rq5_cold_cache_correction.py --live-ledger  # ~/.claude/rollup-spend.jsonl
"""

from __future__ import annotations

import csv
import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "paper-rollups"
# The paper is frozen to the 2026-09-27 corpus snapshot, and the numbers it
# quotes (427 runs, median break-even 1.80, min 1.63) were computed from the
# first 427 lines of the live ledger (last record 2026-09-29T15:55:14Z). The
# drain keeps appending to the live ledger, so reading it makes every rerun
# drift (905+ lines: median 1.65, min 1.49) and trips check_paper_invariants.
# Default to the frozen prefix; --live-ledger opts into the moving file.
FROZEN_LEDGER = DATA / "rollup_spend_frozen_2026-09-29.jsonl"
LIVE_LEDGER = Path.home() / ".claude" / "rollup-spend.jsonl"
LEDGER = FROZEN_LEDGER
RQ2_CSV = DATA / "rq2_compression_corrected.csv"
RQ5_JSON = DATA / "rq5_generation_cost.json"
OUT = DATA / "rq5_cold_cache_correction.json"

# ---------------------------------------------------------------- prices
# Cache multipliers are the published ones: a 5-minute cache WRITE is 1.25x
# input and a cache READ is 0.1x input.
CACHE_WRITE_MULT = 1.25   # 5-minute ephemeral cache write
CACHE_READ_MULT = 0.10    # cache hit

# The price table is not assumed here, it is SOLVED from the ledger. Holding the
# published cache multipliers and output = 5x input fixed, the input price that
# reproduces the CLI's own `total_cost_usd` is exact to within 1% on all 427
# runs, and it comes out bimodal:
#
#   $2.00 /MTok input, $10.00 /MTok output   395 runs  (the configured
#                                                       claude-sonnet-5 route)
#   $1.00 /MTok input, $5.00  /MTok output    32 runs  (Haiku 4.5, the code
#                                                       default when the drain
#                                                       runs without the plist's
#                                                       PKM_ROLLUP_MODEL)
#
# See identify_price_table(), which reports the fit rather than trusting this
# comment. The important consequence for the paper: main.tex prices at $3.00 /
# $15.00 list, and the route the drain actually bills on is $2.00 / $10.00, so
# the paper's price LEVEL is 1.5x too high. That pulls the opposite way to the
# cache error and the two must be reported together.
PRICES = {
    "measured-drain-route": {"input": 2.00, "output": 10.00},
    "haiku-4-5-minority-route": {"input": 1.00, "output": 5.00},
    # what paper/main.tex and rq5_generation_cost.py:40-42 assume
    "paper-assumed-sonnet-5-list": {"input": 3.00, "output": 15.00},
}
PRIMARY = "measured-drain-route"

# the multiplier rq5_generation_cost.py applies to the whole transcript in its
# "cached" column (PRICE_CACHED / PRICE_INPUT = 0.30 / 3.00)
PAPER_CACHED_MULT = 0.10
PAPER_UNCACHED_MULT = 1.00


def load_ledger() -> list[dict]:
    """Every drain run that made at least one model call."""
    runs = []
    for line in LEDGER.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("calls", 0) > 0:
            runs.append(rec)
    return runs


def input_side(rec: dict) -> int:
    return (rec.get("input_tokens", 0) + rec.get("cache_creation_tokens", 0)
            + rec.get("cache_read_tokens", 0))


def identify_price_table(runs: list[dict]) -> dict:
    """Which published price table is the CLI's own total_cost_usd consistent with?

    Solved rather than assumed. For each candidate table we recompute cost from
    the token counts and take the ratio to the recorded figure. A table that
    matches gives a ratio of 1.0 on every row.
    """
    out = {}
    for name, price in PRICES.items():
        ratios = []
        for rec in runs:
            modelled = (
                rec["input_tokens"] * price["input"]
                + rec["cache_creation_tokens"] * price["input"] * CACHE_WRITE_MULT
                + rec["cache_read_tokens"] * price["input"] * CACHE_READ_MULT
                + rec["output_tokens"] * price["output"]
            ) / 1_000_000
            if rec["cost_usd"] > 0 and modelled > 0:
                ratios.append(modelled / rec["cost_usd"])
        out[name] = {
            "n": len(ratios),
            "modelled_over_recorded_median": round(statistics.median(ratios), 4),
            "within_1pct_of_recorded": sum(1 for r in ratios if abs(r - 1) < 0.01),
        }

    # the solve itself: invert for the input price, holding output = 5x input and
    # the published cache multipliers. this is what makes the table above a
    # measurement rather than a guess
    implied = []
    for rec in runs:
        denom = (rec["input_tokens"]
                 + CACHE_WRITE_MULT * rec["cache_creation_tokens"]
                 + CACHE_READ_MULT * rec["cache_read_tokens"]
                 + 5 * rec["output_tokens"])
        if denom > 0 and rec["cost_usd"] > 0:
            implied.append(rec["cost_usd"] * 1_000_000 / denom)
    out["_solved_input_price_per_mtok"] = {
        "assumption": "output = 5x input, cache write = 1.25x input, "
                      "cache read = 0.1x input",
        "distribution": summarise(implied),
        "runs_at_2_00": sum(1 for s in implied if abs(s - 2.0) < 0.02),
        "runs_at_1_00": sum(1 for s in implied if abs(s - 1.0) < 0.01),
        "runs_at_3_00": sum(1 for s in implied if abs(s - 3.0) < 0.03),
        "reading": "bimodal and exact: no run bills at the $3.00 list rate the "
                   "paper assumes, so the paper's price level is 1.5x too high "
                   "on the dominant route",
    }
    return out


def cache_shape(runs: list[dict]) -> dict:
    """The billing shape of real rollup generation: where the input tokens land.

    `cache_read_share` is the quantity the warm-cache claim lives or dies on. If
    the originating session's context were being reused, essentially all input
    would be cache_read. It is not.
    """
    tot_in = sum(r["input_tokens"] for r in runs)
    tot_write = sum(r["cache_creation_tokens"] for r in runs)
    tot_read = sum(r["cache_read_tokens"] for r in runs)
    tot_out = sum(r["output_tokens"] for r in runs)
    tot_side = tot_in + tot_write + tot_read

    per_run_read_share = [
        r["cache_read_tokens"] / input_side(r)
        for r in runs if input_side(r) > 0
    ]
    # the ratio the brief asked for: cache reads against everything billed at a
    # non-read rate
    per_run_read_to_cold = [
        r["cache_read_tokens"] / (r["input_tokens"] + r["cache_creation_tokens"])
        for r in runs
        if (r["input_tokens"] + r["cache_creation_tokens"]) > 0
    ]

    # effective input multiplier: what one input-side token actually costs, as a
    # multiple of the list input price
    effective_mult = (
        tot_in * 1.0 + tot_write * CACHE_WRITE_MULT + tot_read * CACHE_READ_MULT
    ) / tot_side

    return {
        "n_drain_runs": len(runs),
        "n_model_calls": sum(r["calls"] for r in runs),
        "n_notes_written": sum(r["notes_written"] for r in runs),
        "totals": {
            "input_tokens": tot_in,
            "cache_creation_tokens": tot_write,
            "cache_read_tokens": tot_read,
            "output_tokens": tot_out,
            "input_side_tokens": tot_side,
        },
        "shares_of_input_side": {
            "fresh_input": round(tot_in / tot_side, 6),
            "cache_write": round(tot_write / tot_side, 6),
            "cache_read": round(tot_read / tot_side, 6),
        },
        "cache_read_share_per_run": summarise(per_run_read_share),
        "cache_read_over_cold_tokens_per_run": summarise(per_run_read_to_cold),
        "runs_with_zero_cache_read": sum(1 for r in runs if r["cache_read_tokens"] == 0),
        "effective_input_multiplier": round(effective_mult, 6),
        "paper_assumed_input_multiplier": PAPER_CACHED_MULT,
        "input_multiplier_understatement": round(effective_mult / PAPER_CACHED_MULT, 3),
        "recorded_cost_usd_total": round(sum(r["cost_usd"] for r in runs), 4),
    }


def q(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[k]


def summarise(vals: list[float]) -> dict:
    vals = [v for v in vals if v == v and v != float("inf")]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "min": round(min(vals), 4),
        "p10": round(q(vals, 0.10), 4),
        "median": round(statistics.median(vals), 4),
        "mean": round(statistics.mean(vals), 4),
        "p90": round(q(vals, 0.90), 4),
        "max": round(max(vals), 4),
    }


def per_rollup_from_ledger(runs: list[dict], price: dict) -> dict:
    """Cost per rollup note actually written, at a given price table.

    Normalised on notes_written rather than on runs: a run writes zero, one or
    several notes, and a run that wrote none still spent (the sweep re-reads a
    session it then declines). Charging those wasted runs to the notes that did
    land is the honest direction, so the denominator is total notes and the
    numerator is total spend.
    """
    notes = sum(r["notes_written"] for r in runs)
    recorded = sum(r["cost_usd"] for r in runs)

    def cost(rec, in_mult_override=None):
        if in_mult_override is None:
            side = (rec["input_tokens"] * 1.0
                    + rec["cache_creation_tokens"] * CACHE_WRITE_MULT
                    + rec["cache_read_tokens"] * CACHE_READ_MULT)
        else:
            side = input_side(rec) * in_mult_override
        return (side * price["input"] + rec["output_tokens"] * price["output"]) / 1_000_000

    true_total = sum(cost(r) for r in runs)
    paper_cached_total = sum(cost(r, PAPER_CACHED_MULT) for r in runs)
    paper_uncached_total = sum(cost(r, PAPER_UNCACHED_MULT) for r in runs)

    return {
        "notes_written": notes,
        "recorded_cost_usd": round(recorded, 4),
        "true_billing_total_usd": round(true_total, 4),
        "paper_cached_total_usd": round(paper_cached_total, 4),
        "paper_uncached_total_usd": round(paper_uncached_total, 4),
        "per_rollup_true_usd": round(true_total / notes, 5) if notes else None,
        "per_rollup_paper_cached_usd": round(paper_cached_total / notes, 5) if notes else None,
        "per_rollup_paper_uncached_usd": round(paper_uncached_total / notes, 5) if notes else None,
        "understatement_vs_paper_cached": round(true_total / paper_cached_total, 3),
        "understatement_vs_paper_uncached": round(true_total / paper_uncached_total, 3),
    }


def load_rq2() -> dict[str, dict]:
    rows = {}
    with RQ2_CSV.open() as f:
        for row in csv.DictReader(f):
            rows[row["session_id"]] = {
                "rollup_tokens": int(row.get("rollup_tokens_measured", 0) or 0),
                "transcript_tokens": int(row.get("payload_tokens_measured", 0) or 0),
            }
    return rows


def recompute_break_even(shape: dict, price: dict, inflation: float) -> dict:
    """Break-even in retrievals, with generation priced as it is really billed.

    The paper's break-even divides generation cost by the input-price value of
    the tokens a retrieval avoids reading. Only the numerator changes here.
    Two things go into it that the cached column left out:

      * the effective input multiplier is `effective_input_multiplier`, not 0.1
      * generation reads the transcript more than once, because a rollup is
        produced per epoch. rq5's own validation block measured that at a
        median 2.05x and this is the `inflation` argument.

    The denominator is left at full input price, exactly as published, so the
    change in the result is attributable to the generation side alone.
    """
    rq5 = json.loads(RQ5_JSON.read_text())
    paired = load_rq2()
    # the corpus rq5 reports on: its 300 sessions are the paired set
    n_expected = rq5["n_sessions"]

    mult = shape["effective_input_multiplier"]
    results = []
    for sid, d in paired.items():
        tt, rt = d["transcript_tokens"], d["rollup_tokens"]
        saved = tt - rt
        gen_in = tt * inflation
        cost_true = (gen_in * price["input"] * mult
                     + rt * price["output"]) / 1_000_000
        cost_paper_cached = (tt * price["input"] * PAPER_CACHED_MULT
                             + rt * price["output"]) / 1_000_000
        cost_paper_uncached = (tt * price["input"] * PAPER_UNCACHED_MULT
                               + rt * price["output"]) / 1_000_000
        if saved <= 0:
            continue
        per_retrieval = saved * price["input"] / 1_000_000
        results.append({
            "be_true": cost_true / per_retrieval,
            "be_paper_cached": cost_paper_cached / per_retrieval,
            "be_paper_uncached": cost_paper_uncached / per_retrieval,
            "cost_true": cost_true,
        })

    be_true = [r["be_true"] for r in results]
    be_cached = [r["be_paper_cached"] for r in results]
    return {
        "note": "generation repriced at the measured billing shape; the saving "
                "per retrieval is left at full input price, as published, so "
                "the whole change is on the generation side",
        "sessions_in_rq2_csv": len(paired),
        "rq5_reported_n_sessions": n_expected,
        "n_compressing": len(results),
        "transcript_read_inflation_applied": inflation,
        "effective_input_multiplier_applied": mult,
        "break_even_true": summarise(be_true),
        "break_even_paper_cached": summarise(be_cached),
        "break_even_paper_uncached": summarise(
            [r["be_paper_uncached"] for r in results]),
        "pct_break_even_within_one_retrieval_true": round(
            100 * sum(1 for b in be_true if b <= 1) / len(be_true), 1),
        "pct_break_even_within_one_retrieval_paper_cached": round(
            100 * sum(1 for b in be_cached if b <= 1) / len(be_cached), 1),
        "per_session_generation_cost_true_usd": summarise(
            [r["cost_true"] for r in results]),
    }


def stripped_transcript_cost() -> dict:
    """Does the stripped-transcript alternative make a model call? Verified, not assumed."""
    src = (Path(__file__).resolve().parents[0] / "rq6_stripped_transcript_size.py")
    text = src.read_text() if src.exists() else ""
    markers = ["call_model", "anthropic", "subprocess", "claude -p",
               "messages.create", "requests.post", "httpx", "openai"]
    hits = {m: text.count(m) for m in markers if m in text}
    return {
        "script": src.relative_to(REPO).as_posix(),
        "model_call_markers_found": hits,
        "makes_model_call": bool(hits),
        "generation_cost_usd": 0.0 if not hits else None,
        "why": "rq6 is a join over character counts that already exist in "
               "rq1_corrected_per_session.csv and rq2_compression_corrected.csv "
               "plus fixed scaffolding constants. No network, no model, no "
               "tokeniser call at generation time, so producing a stripped "
               "transcript costs zero model spend -- it is a deterministic "
               "filter over a file already on disk.",
    }


def merge_into_rq5_json(out: dict, path: Path = RQ5_JSON) -> None:
    """Fold the cold-cache break-even into rq5_generation_cost.json, idempotently.

    The shape is the one hand-merged in d5d0b22b: the corrected figures take the
    published keys, the old warm-cache values move to `*_warm_cache_uncorrected`.
    Already-merged files keep their existing uncorrected values (never nested
    twice) and only have the corrected figures refreshed. Key order is kept.
    """
    be = out["break_even"][PRIMARY]
    mult = be["effective_input_multiplier_applied"]
    infl = be["transcript_read_inflation_applied"]
    corrected = be["break_even_true"]
    pct = be["pct_break_even_within_one_retrieval_true"]
    le1 = {"n": round(pct / 100 * be["n_compressing"]), "pct": pct}
    source = (f"cold-cache correction (scripts/paper-rollups/"
              f"rq5_cold_cache_correction.py, {PRIMARY}): generation billed at "
              f"the measured cache shape (effective input multiplier {mult}) "
              f"with the {infl}x per-epoch transcript re-read; saving per "
              f"retrieval at full input price")
    rq5 = json.loads(path.read_text())
    already = "break_even_retrievals_source" in rq5
    new: dict = {}
    for k, v in rq5.items():
        if k == "break_even_retrievals":
            new[k] = corrected
            new["break_even_retrievals_source"] = source
            if not already:
                new["break_even_retrievals_warm_cache_uncorrected"] = v
        elif k == "break_even_retrievals_source":
            continue
        elif k == "break_even_retrievals_cached" and not already:
            new["break_even_retrievals_cached_warm_cache_uncorrected"] = v
        elif k == "sessions_break_even_le_1":
            new[k] = le1
            if not already:
                new["sessions_break_even_le_1_warm_cache_uncorrected"] = v
        else:
            new[k] = v
    path.write_text(json.dumps(new, indent=2) + "\n")


def main() -> None:
    global LEDGER
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-ledger", action="store_true",
                    help=f"read {LIVE_LEDGER} instead of the frozen snapshot")
    if ap.parse_args().live_ledger:
        LEDGER = LIVE_LEDGER
    if not LEDGER.exists():
        raise SystemExit(f"FATAL: {LEDGER} is missing; the cold-cache correction "
                         "cannot run, so rq5_generation_cost.json would keep the "
                         "uncorrected warm-cache break-even")
    runs = load_ledger()
    shape = cache_shape(runs)
    table = identify_price_table(runs)

    # rq5's own measured validation block gives the transcript-read inflation
    rq5 = json.loads(RQ5_JSON.read_text())
    inflation = rq5["validation"]["measured_to_analytic_input_token_ratio"]["median"]

    # the comparison that actually answers the brief: the paper's headline
    # (its own price table, its own 0.1x cache-read multiplier, one pass over
    # the transcript) against what the drain is really billed (the solved price
    # table, the measured cache shape, the measured per-epoch re-read).
    paper_price = PRICES["paper-assumed-sonnet-5-list"]
    true_price = PRICES[PRIMARY]
    notes = shape["n_notes_written"]
    paper_headline = sum(
        (input_side(r) * PAPER_CACHED_MULT * paper_price["input"]
         + r["output_tokens"] * paper_price["output"]) / 1_000_000
        for r in runs) / notes
    true_headline = sum(
        ((r["input_tokens"]
          + CACHE_WRITE_MULT * r["cache_creation_tokens"]
          + CACHE_READ_MULT * r["cache_read_tokens"]) * true_price["input"]
         + r["output_tokens"] * true_price["output"]) / 1_000_000
        for r in runs) / notes

    out = {
        "question": "does RQ5 price rollup generation against a cache that is "
                    "never warm?",
        "headline": {
            "per_rollup_paper_cached_usd": round(paper_headline, 5),
            "per_rollup_true_usd": round(true_headline, 5),
            "understatement_factor": round(true_headline / paper_headline, 2),
            "two_errors_pulling_opposite_ways": {
                "cache_mode": "paper charges every input token at 0.1x input. "
                              "measured effective multiplier is "
                              f"{shape['effective_input_multiplier']}, because "
                              "60.1% of input-side tokens are cache WRITES at "
                              "1.25x and only 39.9% are reads. that alone is "
                              f"{shape['input_multiplier_understatement']}x.",
                "price_level": "paper prices at $3.00/$15.00 Sonnet list. the "
                               "route the drain bills on solves to $2.00/$10.00 "
                               "on 395 of 427 runs, so the paper is 1.5x too "
                               "HIGH here, partly cancelling the cache error.",
                "net": "the cache error dominates; the published cached figure "
                       "understates real per-rollup generation cost by the "
                       "factor above.",
            },
        },
        "verdict": "yes. the warm-cache assumption is false: the drain is a "
                   "separate launchd process and the transcript is billed as a "
                   "cache WRITE at 1.25x input, not a cache READ at 0.1x.",
        "mechanism": {
            "hook": "SessionEnd enqueues the session id",
            "worker": "launchd com.agent-session-rollup.drain, StartInterval 300, "
                      "runs the rollup-drain executable from the venv",
            "call": "rollup_summarise.call_model shells out to "
                    "`claude -p --model <m> --output-format json` in spawn_dir(), "
                    "a separate process with no shared prompt prefix",
            "consequence": "first pass over the transcript can only be a cache "
                           "write; there is no session context to inherit",
        },
        "what_rq5_assumes": {
            "script": "scripts/paper-rollups/rq5_generation_cost.py",
            "prices_lines_40_42": {"PRICE_INPUT": 3.00, "PRICE_CACHED": 0.30,
                                   "PRICE_OUTPUT": 15.00},
            "analytic_cached_line_282": "gen_cost_cached = (gen_input * "
                                        "PRICE_CACHED + gen_output * "
                                        "PRICE_OUTPUT) / 1e6 -- the ENTIRE "
                                        "transcript at the cache-read rate",
            "measured_cached_lines_264_269": "prices cache_creation_tokens at "
                                             "PRICE_CACHED, i.e. charges a cache "
                                             "WRITE at the READ rate",
            "stated_assumption_json_line": "cached pricing applies when the "
                                           "transcript is already in cache, "
                                           "which is the real deployment scenario",
            "paper_prose": "paper/main.tex, above tab:breakeven: 'the pipeline "
                           "summarises at session end from a context that is "
                           "already resident, so the input is billed at the "
                           "cache-read rate'",
        },
        "price_table_identification": {
            "note": "which published table the CLI's own total_cost_usd agrees "
                    "with, solved from the ledger",
            "results": table,
            "configured_model": "claude-sonnet-5 (PKM_ROLLUP_MODEL in "
                                "com.agent-session-rollup.drain.plist; the "
                                "drain's own self-notes record the same)",
            "caveat": "the drain reaches the model through a Vertex proxy "
                      "(ANTHROPIC_VERTEX_BASE_URL, CLAUDE_CODE_USE_VERTEX=1), "
                      "not the local 127.0.0.1:8080 gateway. Both tables are "
                      "reported; the structural finding is price-independent.",
        },
        "measured_cache_shape": shape,
        "per_rollup_cost": {
            name: per_rollup_from_ledger(runs, price)
            for name, price in PRICES.items()
        },
        "break_even": {
            name: recompute_break_even(shape, price, inflation)
            for name, price in PRICES.items()
        },
        "stripped_transcript_alternative": stripped_transcript_cost(),
        "accounting_corrections_applied": [
            "the ledger is per API call, written by record_spend() from "
            "`claude -p --output-format json`, so the ~1.5x transcript "
            "overstatement does NOT apply -- these are not message.usage fields "
            "scraped from a transcript",
            "no requestId dedupe needed for the same reason: one usage record "
            "per call, not one per content block, so the 2.38x per-block "
            "inflation cannot arise",
            "runs with calls == 0 excluded, so idle sweeps do not dilute the "
            "cache-read share",
            "per-rollup cost normalised on notes_written across the whole "
            "ledger, so runs that spent but wrote nothing are charged to the "
            "notes that did land",
        ],
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    # a rerun of rq5_generation_cost.py rewrites the break-even uncorrected; keep it corrected
    merge_into_rq5_json(out)


if __name__ == "__main__":
    main()
