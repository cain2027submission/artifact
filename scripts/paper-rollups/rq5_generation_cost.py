#!/usr/bin/env python3
"""RQ5: Generation cost of trajectory rollups and break-even analysis.

Answers: what does it cost to generate a rollup, and after how many retrievals
does it pay for itself?

CRITICAL: The `cost_usd` and `tokens` fields in a rollup note's YAML frontmatter
are the SESSION's own cost and token usage, NOT the cost of generating the rollup.
This script extracts measured rollup generation usage from the rollup generation
transcripts themselves.

Run:
    python3 rq5_generation_cost.py
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freeze_corpus import load_manifest
from paths import PROJECTS_DIR, SESSIONS_DIR, project_slug

# Rollup generation transcripts
ROLLUP_GEN = PROJECTS_DIR / project_slug(Path.home() / ".cache/agent-session-rollup-spawn")
# Vault session rollups
ROLLUPS = SESSIONS_DIR
# Project transcripts
PROJECTS = PROJECTS_DIR
OUTDIR = Path(__file__).resolve().parents[2] / "data" / "paper-rollups"

# Load existing rq2 and rq3 data
RQ2_CSV = OUTDIR / "rq2_compression_corrected.csv"
RQ3_JSON = OUTDIR / "rq3_structural_loss.json"

# Claude Sonnet 5 pricing (per million tokens)
# https://www.anthropic.com/pricing (as of 2026-09)
PRICE_INPUT = 3.00  # per 1M input tokens
PRICE_CACHED = 0.30  # per 1M cached input tokens
PRICE_OUTPUT = 15.00  # per 1M output tokens

TOKENIZER_PATH = (str(Path.home() / ".cache" / "huggingface" / "hub") + "/"
                  "models--mlx-community--Qwen2.5-Coder-7B-Instruct-4bit/"
                  "snapshots/019cc73c45c770444708a6dd8690c66243cc5c80/"
                  "tokenizer.json")


def load_tokenizer():
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(TOKENIZER_PATH)
    except Exception as exc:
        print(f"warning: tokenizer unavailable ({exc})")
        return None


def count_tokens(tok, text: str) -> int:
    """Token count, chunked for long inputs."""
    if tok is None or not text:
        return 0
    total = 0
    step = 200_000
    for i in range(0, len(text), step):
        total += len(tok.encode(text[i:i + step], add_special_tokens=False).ids)
    return total


def parse_rollup_gen_usage_from_file(transcript_path: Path, usage_by_request: dict[str, dict]) -> None:
    """Extract usage data from a rollup generation transcript into the shared dict.

    Mutates usage_by_request to dedupe requestIds across multiple files for a session.
    """
    try:
        raw = transcript_path.read_text(errors="replace")
    except OSError:
        return

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg = rec.get("message")
        if not isinstance(msg, dict):
            continue

        # Dedupe on requestId to avoid per-block overcounting
        rid = rec.get("requestId")
        usage = msg.get("usage")
        if rid and isinstance(usage, dict):
            usage_by_request[rid] = usage


def compute_usage_totals(usage_by_request: dict[str, dict]) -> dict | None:
    """Compute total tokens and cost from deduped usage records."""
    if not usage_by_request:
        return None

    # Sum across all API calls
    total_input = 0
    total_cache_read = 0
    total_cache_creation = 0
    total_output = 0

    for u in usage_by_request.values():
        total_input += (u.get("input_tokens", 0) or 0)
        total_cache_read += (u.get("cache_read_input_tokens", 0) or 0)
        total_cache_creation += (u.get("cache_creation_input_tokens", 0) or 0)
        total_output += (u.get("output_tokens", 0) or 0)

    # Compute cost at Sonnet 5 pricing
    cost_usd = (
        (total_input * PRICE_INPUT / 1_000_000) +
        (total_cache_read * PRICE_CACHED / 1_000_000) +
        (total_cache_creation * PRICE_INPUT / 1_000_000) +
        (total_output * PRICE_OUTPUT / 1_000_000)
    )

    return {
        "input_tokens": total_input,
        "cache_read_tokens": total_cache_read,
        "cache_creation_tokens": total_cache_creation,
        "output_tokens": total_output,
        "total_tokens": total_input + total_cache_read + total_cache_creation + total_output,
        "cost_usd": cost_usd,
        "requests": len(usage_by_request),
    }


def load_rq3_data():
    """Load aggregate token counts from rq3."""
    with RQ3_JSON.open() as f:
        return json.load(f)


def q(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[k]


def summarise(vals: list[float]) -> dict:
    vals = [v for v in vals if v == v and v not in (float("inf"),)]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "min": round(min(vals), 2),
        "p10": round(q(vals, 0.10), 2),
        "median": round(statistics.median(vals), 2),
        "mean": round(statistics.mean(vals), 2),
        "p90": round(q(vals, 0.90), 2),
        "max": round(max(vals), 2),
    }


def main() -> None:
    manifest = load_manifest()
    paired_ids = set(manifest["paired_session_ids"])
    tok = load_tokenizer()
    rq3 = load_rq3_data()

    # Build prefix->UUID lookup for 8-hex matching
    prefix_to_uuid = {uuid[:8]: uuid for uuid in paired_ids}

    # Load rq2 data
    import csv
    rq2_sessions = {}
    with RQ2_CSV.open() as f:
        for row in csv.DictReader(f):
            sid = row["session_id"]
            if sid in paired_ids:
                rq2_sessions[sid] = {
                    "file": row["file"],
                    "rollup_tokens": int(row.get("rollup_tokens_measured", 0) or 0),
                    "transcript_tokens": int(row.get("payload_tokens_measured", 0) or 0),
                    "rollup_bytes": int(row.get("rollup_bytes", 0) or 0),
                    "transcript_bytes": int(row.get("payload_chars", 0) or 0),
                }

    # Attribute each generation transcript to the session it rolled up.
    #
    # The signal is the transcript PATH the drain was given, `<uuid>.jsonl`. That
    # is the run's input, so it names the subject. Three weaker signals were
    # tried and rejected, and the reason matters because each looks reasonable:
    #
    #   * a bare session UUID anywhere in the file (42 sessions, 35 files naming
    #     two or more)
    #   * a rollup note stem `YYYY-MM-DD_agent_<8hex>_slug` (20 sessions, 87
    #     self-ambiguous files)
    #   * the enqueue record's `content` rather than the whole file -- no
    #     narrower at all, because that field holds the entire summariser prompt
    #     with the transcript embedded in it
    #
    # The note stem is the one to be careful of: where it and the path are each
    # unique within a file they name DIFFERENT sessions 32 times out of 33. A
    # transcript quotes other sessions' note names freely, so the stem usually
    # belongs to something the session talked about rather than to the session.
    # Taking the union of the two signals inflates the sample to 93 sessions, and
    # most of those attributions are wrong. This path signal is the only one that
    # never names two paired sessions in one file.
    print(f"Scanning {len(list(ROLLUP_GEN.glob('*.jsonl')))} rollup generation transcripts...")

    transcript_path_pattern = re.compile(
        r"/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.jsonl")

    rollup_gen_by_session: dict[str, set[Path]] = defaultdict(set)
    n_unattributed = 0        # no paired session's transcript path present
    n_ambiguous = 0           # two paired transcripts named; unsplittable

    for gen_path in ROLLUP_GEN.glob("*.jsonl"):
        try:
            raw = gen_path.read_text(errors="replace")
        except OSError:
            continue
        if not raw:
            continue

        named = {u for u in transcript_path_pattern.findall(raw) if u in paired_ids}
        if not named:
            n_unattributed += 1
        elif len(named) > 1:
            n_ambiguous += 1
        else:
            rollup_gen_by_session[named.pop()].add(gen_path)

    matched_sessions = len([s for s in paired_ids if s in rollup_gen_by_session])
    print(f"Attributed {matched_sessions} of {len(paired_ids)} paired sessions "
          f"({n_ambiguous} files naming two paired transcripts, "
          f"{n_unattributed} naming none)")

    # Now compute generation costs
    results = []
    measured_count = 0
    analytic_count = 0

    for session_id, session_data in rq2_sessions.items():
        rollup_tokens = session_data["rollup_tokens"]
        transcript_tokens = session_data["transcript_tokens"]

        # Try to find measured generation usage (sum across all epochs)
        gen_usage = None
        if session_id in rollup_gen_by_session:
            # Cross-file requestId deduping: one shared dict per session
            usage_by_request: dict[str, dict] = {}
            for gen_path in rollup_gen_by_session[session_id]:
                parse_rollup_gen_usage_from_file(gen_path, usage_by_request)
            gen_usage = compute_usage_totals(usage_by_request)

        if gen_usage:
            # MEASURED
            measured_count += 1
            gen_input = gen_usage["input_tokens"] + gen_usage["cache_read_tokens"] + gen_usage["cache_creation_tokens"]
            gen_output = gen_usage["output_tokens"]
            gen_cost = gen_usage["cost_usd"]
            gen_cost_cached = (
                (gen_usage["input_tokens"] * PRICE_INPUT / 1_000_000) +
                (gen_usage["cache_read_tokens"] * PRICE_CACHED / 1_000_000) +
                (gen_usage["cache_creation_tokens"] * PRICE_CACHED / 1_000_000) +
                (gen_usage["output_tokens"] * PRICE_OUTPUT / 1_000_000)
            )
            data_source = "measured"
        else:
            # ANALYTIC estimate
            analytic_count += 1
            # Whole-transcript input and rollup output. This was annotated
            # "conservative" on the assumption that charging for the entire
            # transcript must overstate the cost. On the sessions where both are
            # available it understates it instead, because a rollup is generated
            # per epoch and so reads the transcript more than once.
            gen_input = transcript_tokens
            gen_output = rollup_tokens
            gen_cost = (gen_input * PRICE_INPUT + gen_output * PRICE_OUTPUT) / 1_000_000
            gen_cost_cached = (gen_input * PRICE_CACHED + gen_output * PRICE_OUTPUT) / 1_000_000
            data_source = "analytic"

        # Break-even: how many retrievals before generation pays for itself?
        # One retrieval = reading rollup (rollup_tokens) vs reading transcript (transcript_tokens)
        # Cost saved per retrieval = (transcript_tokens - rollup_tokens) * input_price
        tokens_saved_per_retrieval = transcript_tokens - rollup_tokens

        if tokens_saved_per_retrieval > 0:
            cost_saved_per_retrieval = tokens_saved_per_retrieval * PRICE_INPUT / 1_000_000
            break_even = gen_cost / cost_saved_per_retrieval if cost_saved_per_retrieval > 0 else float("inf")
            break_even_cached = gen_cost_cached / cost_saved_per_retrieval if cost_saved_per_retrieval > 0 else float("inf")
        else:
            # Rollup is LARGER than transcript - never pays for itself
            cost_saved_per_retrieval = 0
            break_even = float("inf")
            break_even_cached = float("inf")

        # For a measured session, record what the analytic model WOULD have said
        # about the same session. Without this the two arms can only be compared
        # across disjoint session sets, which measures the difference between the
        # sets -- measured sessions run about 2.5x larger -- rather than the
        # difference between the methods.
        if data_source == "measured":
            counterfactual_cost = (
                transcript_tokens * PRICE_INPUT + rollup_tokens * PRICE_OUTPUT
            ) / 1_000_000
            counterfactual_input = transcript_tokens
        else:
            counterfactual_cost = None
            counterfactual_input = None

        results.append({
            "session_id": session_id,
            "data_source": data_source,
            "analytic_counterfactual_cost_usd": counterfactual_cost,
            "analytic_counterfactual_input_tokens": counterfactual_input,
            "generation_requests": gen_usage["requests"] if gen_usage else None,
            "rollup_tokens": rollup_tokens,
            "transcript_tokens": transcript_tokens,
            "generation_input_tokens": gen_input,
            "generation_output_tokens": gen_output,
            "generation_cost_usd": gen_cost,
            "generation_cost_usd_cached": gen_cost_cached,
            "tokens_saved_per_retrieval": tokens_saved_per_retrieval,
            "break_even_retrievals": break_even,
            "break_even_retrievals_cached": break_even_cached,
            "expands": 1 if tokens_saved_per_retrieval <= 0 else 0,
        })

    print(f"\nData sources: {measured_count} measured, {analytic_count} analytic")

    # Compute distributions
    measured_results = [r for r in results if r["data_source"] == "measured"]
    analytic_results = [r for r in results if r["data_source"] == "analytic"]
    compresses = [r for r in results if r["expands"] == 0]
    expands = [r for r in results if r["expands"] == 1]

    # Bias analysis: compare measured vs unmeasured sessions
    measured_session_ids = {r["session_id"] for r in measured_results}

    # For bias analysis, we need session dates and sizes
    # Extract dates from file names (format: YYYY-MM-DD_agent_prefix_slug.md)
    def extract_date(session_id: str) -> str:
        """Extract date from session file name."""
        if session_id in rq2_sessions:
            filename = rq2_sessions[session_id].get("file", "")
            parts = filename.split("_")
            return parts[0] if parts else ""
        return ""

    measured_dates = [extract_date(r["session_id"]) for r in measured_results]
    unmeasured_dates = [extract_date(r["session_id"]) for r in analytic_results]
    measured_sizes = [r["transcript_tokens"] for r in measured_results]
    unmeasured_sizes = [r["transcript_tokens"] for r in analytic_results]

    # Validation: on the measured sessions only, how far is the analytic model
    # from the measurement for those same sessions?
    #
    # Both variants are reported because they differ and the choice is not
    # obvious. The uncached ratio compares like with like -- the measured cost
    # and the analytic estimate both at full input price. The cached ratio
    # divides the cached measured cost by a cached analytic estimate, which is
    # the deployment scenario but mixes a measured cache-hit pattern with an
    # assumed one. Quoting one without saying which is how this number came out
    # as 1.8x once and 1.2x on recount.
    #
    # A ratio above 1 means the analytic model UNDERSTATES generation cost. The
    # sign is the result: it was assumed to be an over-estimate.
    def ratios(cost_key: str, analytic_input_price: float) -> list[float]:
        out = []
        for r in measured_results:
            estimate = (r["transcript_tokens"] * analytic_input_price
                        + r["rollup_tokens"] * PRICE_OUTPUT) / 1_000_000
            if estimate > 0:
                out.append(r[cost_key] / estimate)
        return sorted(out)

    ratio_uncached = ratios("generation_cost_usd", PRICE_INPUT)
    ratio_cached = ratios("generation_cost_usd_cached", PRICE_CACHED)
    input_ratios = sorted(
        r["generation_input_tokens"] / r["transcript_tokens"]
        for r in measured_results if r["transcript_tokens"] > 0
    )
    measured_to_analytic_ratios = ratio_uncached

    # Sessions whose "transcript" is too small to be a real session. These make
    # the analytic arm's break-even meaningless -- a rollup trivially expands a
    # 2-token transcript -- so they are counted and reported rather than left to
    # sit inside the distribution unnoticed.
    degenerate = {
        f"transcript_tokens_under_{t}": sum(
            1 for r in results if r["transcript_tokens"] < t)
        for t in (10, 100, 1000)
    }

    summary = {
        "corpus_snapshot_utc": manifest["snapshot_utc"],
        "method": "MEASURED: parsed actual rollup generation transcript usage; "
                  "ANALYTIC: estimated as whole-transcript input + rollup output",
        "pricing": {
            "model": "claude-sonnet-5",
            "input_per_million": PRICE_INPUT,
            "cached_per_million": PRICE_CACHED,
            "output_per_million": PRICE_OUTPUT,
            "note": "cached pricing applies when the transcript is already in cache, "
                    "which is the real deployment scenario",
        },
        "n_sessions": len(results),
        "n_measured": measured_count,
        "n_analytic": analytic_count,
        "attribution": {
            "rule": "the session whose <uuid>.jsonl transcript path appears in the "
                    "generation transcript -- that path is the run's input, so it "
                    "names the subject",
            "files_naming_two_paired_transcripts": n_ambiguous,
            "files_naming_none": n_unattributed,
            "rejected_signals": {
                "bare_session_uuid_anywhere": "42 sessions but 35 files name two "
                    "or more paired sessions; transcripts quote other sessions",
                "rollup_note_stem": "20 sessions, 87 self-ambiguous files. Where "
                    "the note stem and the transcript path are each unique within "
                    "a file they name DIFFERENT sessions 32 times out of 33, so "
                    "the stem usually belongs to a session being discussed rather "
                    "than to the session being rolled up",
                "enqueue_record_content": "no narrower than the whole file: that "
                    "field holds the entire summariser prompt with the transcript "
                    "embedded in it",
                "union_of_signals": "inflates the sample to 93 sessions, and most "
                    "of the added attributions are wrong",
            },
        },
        "generation_cost_usd": summarise([r["generation_cost_usd"] for r in results]),
        "generation_cost_usd_cached": summarise([r["generation_cost_usd_cached"] for r in results]),
        "generation_input_tokens": summarise([r["generation_input_tokens"] for r in results]),
        "generation_output_tokens": summarise([r["generation_output_tokens"] for r in results]),
        "break_even_retrievals": summarise([r["break_even_retrievals"] for r in compresses]),
        "break_even_retrievals_cached": summarise([r["break_even_retrievals_cached"] for r in compresses]),
        "sessions_break_even_le_1": {
            "n": len([r for r in compresses if r["break_even_retrievals"] <= 1]),
            "pct": round(100 * len([r for r in compresses if r["break_even_retrievals"] <= 1]) / len(compresses), 1) if compresses else 0,
        },
        "sessions_break_even_le_1_cached": {
            "n": len([r for r in compresses if r["break_even_retrievals_cached"] <= 1]),
            "pct": round(100 * len([r for r in compresses if r["break_even_retrievals_cached"] <= 1]) / len(compresses), 1) if compresses else 0,
        },
        "sessions_rollup_expands": {
            "n": len(expands),
            "pct": round(100 * len(expands) / len(results), 1),
            "note": "rollup is larger than transcript; break-even never arrives",
        },
        "measured_vs_analytic": {
            "caveat": "these two blocks describe DISJOINT session sets, and the "
                      "measured set is the larger-session tail (see bias_analysis: "
                      "its median transcript is several times the unmeasured "
                      "median). The gap between the two medians is therefore mostly "
                      "a size difference, not evidence about the analytic model. "
                      "Only the same-session ratios under 'validation' test that.",
            "measured": {
                "n": measured_count,
                "generation_cost_usd": summarise([r["generation_cost_usd"] for r in measured_results]),
                "generation_cost_usd_cached": summarise([r["generation_cost_usd_cached"] for r in measured_results]),
                "break_even_retrievals": summarise([r["break_even_retrievals"] for r in measured_results if r["expands"] == 0]),
                "break_even_retrievals_cached": summarise([r["break_even_retrievals_cached"] for r in measured_results if r["expands"] == 0]),
            },
            "analytic": {
                "n": analytic_count,
                "generation_cost_usd": summarise([r["generation_cost_usd"] for r in analytic_results]),
                "generation_cost_usd_cached": summarise([r["generation_cost_usd_cached"] for r in analytic_results]),
                "break_even_retrievals": summarise([r["break_even_retrievals"] for r in analytic_results if r["expands"] == 0]),
                "break_even_retrievals_cached": summarise([r["break_even_retrievals_cached"] for r in analytic_results if r["expands"] == 0]),
            },
        },
        "bias_analysis": {
            "note": "Measured sessions may be biased (gen transcripts are pruned over time)",
            "measured_dates": {
                "min": min(measured_dates) if measured_dates else None,
                "max": max(measured_dates) if measured_dates else None,
            },
            "unmeasured_dates": {
                "min": min(unmeasured_dates) if unmeasured_dates else None,
                "max": max(unmeasured_dates) if unmeasured_dates else None,
            },
            "measured_transcript_tokens": summarise(measured_sizes),
            "unmeasured_transcript_tokens": summarise(unmeasured_sizes),
        },
        "validation": {
            "note": "On the measured sessions only, measured cost divided by the "
                    "analytic estimate for those same sessions. Above 1 means the "
                    "analytic model UNDERSTATES generation cost.",
            "n": len(measured_results),
            "measured_to_analytic_ratio_uncached": summarise(ratio_uncached),
            "measured_to_analytic_ratio_cached": summarise(ratio_cached),
            "measured_to_analytic_input_token_ratio": summarise(input_ratios),
            "sessions_where_measured_exceeds_analytic": {
                "n": sum(1 for r in ratio_uncached if r > 1),
                "of": len(ratio_uncached),
            },
            "why_above_one": "a rollup is generated per epoch, so the transcript "
                             "is read more than once; the analytic model charges "
                             "for a single pass",
            "generation_requests_per_session": summarise(
                [float(r["generation_requests"]) for r in measured_results
                 if r["generation_requests"]]),
        },
        "degenerate_sessions": {
            "note": "paired sessions whose transcript is too small to be a real "
                    "session; their break-even figures are not meaningful",
            **degenerate,
        },
    }

    # Save results
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "rq5_generation_cost.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))

    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    if measured_count > 0:
        print(f"\nMEASURED (n={measured_count}):")
        print(f"  Generation cost (cached): ${summary['measured_vs_analytic']['measured']['generation_cost_usd_cached']['median']:.4f} median")
        print(f"  Break-even (cached): {summary['measured_vs_analytic']['measured']['break_even_retrievals_cached']['median']:.1f} retrievals median")
    if analytic_count > 0:
        print(f"\nANALYTIC (n={analytic_count}):")
        print(f"  Generation cost (cached): ${summary['measured_vs_analytic']['analytic']['generation_cost_usd_cached']['median']:.4f} median")
        print(f"  Break-even (cached): {summary['measured_vs_analytic']['analytic']['break_even_retrievals_cached']['median']:.1f} retrievals median")
    print(f"\nOVERALL (n={len(results)}):")
    print(f"  Break-even ≤ 1 retrieval (cached): {summary['sessions_break_even_le_1_cached']['pct']}%")
    print(f"  Sessions where rollup expands: {summary['sessions_rollup_expands']['pct']}%")

    # this script writes the warm-cache break-even; paper/main.tex cites the cold-cache one, so re-apply it
    import rq5_cold_cache_correction
    rq5_cold_cache_correction.main()


if __name__ == "__main__":
    main()
