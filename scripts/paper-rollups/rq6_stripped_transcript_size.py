#!/usr/bin/env python3
"""How big is a git-friendly stripped transcript, against the rollup and the raw log?

The RQ3 raw-transcript arm recovers 95.5% of ground truth against the rollup's
68.5%, which raises the obvious follow-up: if the transcript is the better
memory, what does it cost to keep one in a form a git repo will tolerate? A raw
`.jsonl` is not that form -- it is one line per content block, minified JSON,
with tool payloads inline, so every session rewrites as one enormous single-line
diff and `git log -p` is unreadable.

So this measures the middle ground: strip the mechanical I/O, render what is
left as markdown, and ask how much bigger that is than the rollup.

Three strip levels, in increasing fidelity:

  L1 prose      user prose + assistant prose + whatever reasoning survived
  L2 +intents   L1 plus every tool_use INPUT (the command, the path, the patch)
                but no tool results. This is the level the RQ4 tool-stripped
                ablation indexes, and it is the one that keeps "what was done"
                while dropping "what came back".
  L3 +heads     L2 plus the first 200 chars of each tool result, which is where
                an error message or a row count lives.

Nothing is re-scanned or re-generated: every character count already exists
per session in rq1_corrected_per_session.csv, and every rollup size in
rq2_compression_corrected.csv. This script is a join and some arithmetic, so it
is deterministic and free.

Two honesty notes that belong with the numbers rather than under them:

1. Markdown scaffolding is not free. A rendered transcript needs a heading per
   turn and a fenced block per tool call. That is charged explicitly at
   SCAFFOLD_PER_TOOL_CALL and SCAFFOLD_PER_ENTRY rather than quietly omitted.
2. The ratios are medians of per-session ratios, never ratios of medians. The
   two differ here because the size distribution is heavily skewed.

Usage:
    rq6_stripped_transcript_size.py
"""

import csv
import json
import statistics
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "paper-rollups"
RQ1 = DATA / "rq1_corrected_per_session.csv"
RQ2 = DATA / "rq2_compression_corrected.csv"
REPORT = DATA / "rq6_stripped_size.json"

# A markdown rendering pays for its own structure. Charged, not hidden.
SCAFFOLD_PER_ENTRY = 12        # "\n### USER\n" style heading per message
SCAFFOLD_PER_TOOL_CALL = 40    # fence + tool name + newlines per call
TOOL_RESULT_HEAD = 200         # chars of each tool result kept at L3


def num(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def pct(vals, p):
    s = sorted(vals)
    if not s:
        return float("nan")
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]


def describe(name, vals):
    return {"metric": name, "n": len(vals),
            "p25": round(pct(vals, 0.25), 3),
            "median": round(pct(vals, 0.50), 3),
            "p75": round(pct(vals, 0.75), 3),
            "mean": round(statistics.fmean(vals), 3) if vals else None}


def main() -> None:
    rq1 = {}
    with RQ1.open() as fh:
        for r in csv.DictReader(fh):
            # Parents only. Subagent transcripts are separate files with their
            # own ids and are not paired to a rollup, so including them would
            # compare a different population to the one rq2 measures.
            if r.get("is_subagent") not in ("0", "", None):
                continue
            rq1[r["session_id"]] = r

    rows = []
    with RQ2.open() as fh:
        for r in csv.DictReader(fh):
            sid = r["session_id"]
            a = rq1.get(sid)
            if a is None:
                continue
            rollup = num(r["rollup_bytes"])
            raw = num(a["serialized_bytes"])
            if rollup <= 0 or raw <= 0:
                continue

            entries = num(a["entries"])
            calls = num(a["tool_calls"])
            scaffold = entries * SCAFFOLD_PER_ENTRY + calls * SCAFFOLD_PER_TOOL_CALL

            prose = (num(a["user_text_chars"]) + num(a["assistant_text_chars"])
                     + num(a["thinking_chars"]))
            l1 = prose + entries * SCAFFOLD_PER_ENTRY
            l2 = prose + num(a["tool_use_chars"]) + scaffold
            # A result shorter than the head is charged at its real length, so
            # min() per call is not available from aggregates -- but capping the
            # total at the true tool_result_chars keeps L3 from exceeding the raw
            # content it samples.
            l3 = l2 + min(calls * TOOL_RESULT_HEAD, num(a["tool_result_chars"]))

            # Tokens: convert with this session's own measured chars-per-token
            # rather than a global 4.0, because tool I/O tokenises differently
            # from prose and the ratio varies 2x across the corpus.
            pc, pt = num(r["payload_chars"]), num(r["payload_tokens_measured"])
            cpt = pc / pt if pt > 0 else None

            rows.append({
                "session_id": sid,
                "rollup_bytes": rollup,
                "raw_bytes": raw,
                "l1": l1, "l2": l2, "l3": l3,
                "chars_per_token": cpt,
                "rollup_tokens": num(r["rollup_tokens_measured"]),
                "payload_tokens": pt,
            })

    print(f"paired sessions: {len(rows)}\n")

    out = {"n_sessions": len(rows),
           "scaffold_charged": {"per_entry": SCAFFOLD_PER_ENTRY,
                                "per_tool_call": SCAFFOLD_PER_TOOL_CALL,
                                "tool_result_head_chars": TOOL_RESULT_HEAD},
           "levels": {}}

    hdr = f"{'level':12s} {'x rollup (bigger)':>22s} {'x raw (smaller)':>20s} {'% of raw':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for lvl, label in (("l1", "L1 prose"), ("l2", "L2 +intents"), ("l3", "L3 +heads")):
        vs_rollup = [r[lvl] / r["rollup_bytes"] for r in rows]
        vs_raw = [r["raw_bytes"] / r[lvl] for r in rows if r[lvl] > 0]
        share = [100.0 * r[lvl] / r["raw_bytes"] for r in rows]
        out["levels"][lvl] = {
            "label": label,
            "bytes": describe("stripped_bytes", [r[lvl] for r in rows]),
            "x_bigger_than_rollup": describe("stripped/rollup", vs_rollup),
            "x_smaller_than_raw": describe("raw/stripped", vs_raw),
            "pct_of_raw_bytes": describe("100*stripped/raw", share),
        }
        print(f"{label:12s} {pct(vs_rollup,0.5):16.1f}x     "
              f"{pct(vs_raw,0.5):14.1f}x  {pct(share,0.5):9.1f}%")

    out["absolute_bytes"] = {
        "rollup": describe("rollup_bytes", [r["rollup_bytes"] for r in rows]),
        "raw_jsonl": describe("raw_bytes", [r["raw_bytes"] for r in rows]),
    }
    print(f"\nmedian rollup {pct([r['rollup_bytes'] for r in rows],0.5):,.0f} B   "
          f"median raw jsonl {pct([r['raw_bytes'] for r in rows],0.5):,.0f} B   "
          f"median L2 {pct([r['l2'] for r in rows],0.5):,.0f} B")

    # Corpus totals: what a repo holding all of these would weigh.
    tot = {k: sum(r[k] for r in rows)
           for k in ("rollup_bytes", "raw_bytes", "l1", "l2", "l3")}
    out["corpus_totals_bytes"] = {k: int(v) for k, v in tot.items()}
    out["corpus_totals_mb"] = {k: round(v / 1e6, 2) for k, v in tot.items()}
    print("\ncorpus totals over these sessions (MB): "
          + "  ".join(f"{k}={v/1e6:.1f}" for k, v in tot.items()))
    print(f"  L2 is {tot['l2']/tot['rollup_bytes']:.1f}x the rollup corpus and "
          f"{tot['raw_bytes']/tot['l2']:.1f}x smaller than the raw corpus")

    # Token view, for the context-window question rather than the disk question.
    tok = [(r["l2"] / r["chars_per_token"], r["rollup_tokens"], r["payload_tokens"])
           for r in rows if r["chars_per_token"] and r["rollup_tokens"] > 0]
    if tok:
        out["l2_tokens"] = {
            "n": len(tok),
            "median_l2_tokens": round(pct([t[0] for t in tok], 0.5), 1),
            "median_rollup_tokens": round(pct([t[1] for t in tok], 0.5), 1),
            "x_bigger_than_rollup": describe(
                "l2_tokens/rollup_tokens", [t[0] / t[1] for t in tok]),
        }
        print(f"\ntokens: median L2 {pct([t[0] for t in tok],0.5):,.0f} vs rollup "
              f"{pct([t[1] for t in tok],0.5):,.0f}  "
              f"({pct([t[0]/t[1] for t in tok],0.5):.1f}x bigger, median of ratios)")

    # How many sessions would a 1M-token window now hold that it did not before?
    fits_raw = sum(1 for r in rows if r["payload_tokens"] and r["payload_tokens"] <= 1e6)
    fits_l2 = sum(1 for r in rows
                  if r["chars_per_token"] and r["l2"] / r["chars_per_token"] <= 1e6)
    out["fits_1m_window"] = {"raw_payload": fits_raw, "l2_stripped": fits_l2,
                             "n": len(rows)}
    print(f"\nfits a 1M-token window: raw payload {fits_raw}/{len(rows)}, "
          f"L2 stripped {fits_l2}/{len(rows)}")

    REPORT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
