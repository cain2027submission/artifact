#!/usr/bin/env python3
"""RQ3 BUDGET LADDER: how do rollup_hookshape and stripped_chunk perform across budgets?

The base RQ3 read-budget experiment measured 5 arms at ONE budget (~667 chars, the
production hook budget). This script sweeps SEVEN budgets from 125 to 8000 chars
across the TWO arms that scored above zero: rollup_hookshape and stripped_chunk.

Budgets (log-spaced doubling): [125, 250, 500, 1000, 2000, 4000, 8000]
Arms: rollup_hookshape, stripped_chunk
Total: 7 x 2 x 124 = 1736 answering calls (resumable)

Context-building functions are imported from rq3_read_budget_arms.py rather than
copied, so changes to the production hook's rendering stay in one place.

Usage:
    rq3_budget_ladder.py --degenerate           # smoke test edge cases
    rq3_budget_ladder.py --dry-run              # build contexts, no API calls
    rq3_budget_ladder.py --limit 3              # smoke test with API calls
    rq3_budget_ladder.py                        # full run (resumable)
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

# Import context builders from the base script
from rq3_read_budget_arms import (
    build_rollup_contexts, build_stripped_context, INDEX_DB,
    ANSWER_PROMPT, answer, SESSIONS_DIR, resolve_rollup,
)

# Import shared utilities
from rq3_raw_transcript_arm import (
    call_llm, grade_answer, load_records, rates, record_id, resolve,
    index_transcripts, SLUG, DATA, QUESTIONS, SUMMARY,
)

from rq3_stripped_transcript_arm import strip_transcript
from paths import REPO_ROOT

OUT = DATA / "rq3_budget_ladder.jsonl"
REPORT = DATA / "rq3_budget_ladder_summary.json"
JUDGE_MODEL = "claude-sonnet-5"

BUDGETS = [125, 250, 500, 1000, 2000, 4000, 8000]
ARMS = ["rollup_hookshape", "stripped_chunk"]


def build_items(limit: int = 0) -> list:
    """The 89-subset, identical logic to rq3_read_budget_arms.build_items()."""
    graded = set(json.loads(SUMMARY.read_text())["graded_record_ids"])
    published = {record_id(r): r for r in load_records(QUESTIONS)
                 if record_id(r) in graded}
    uniform = {r["record_id"]: r for r in load_records(DATA / "rq3_majority_uniform.jsonl")}
    raw = {r["record_id"]: r for r in load_records(DATA / "rq3_majority_raw.jsonl")}
    strip = {r["record_id"]: r for r in load_records(DATA / "rq3_majority_stripped.jsonl")}
    unresolved = sorted({i for d in (uniform, raw, strip) for i, r in d.items()
                         if r.get("grade") is None})
    for d in (uniform, raw, strip):
        for i in unresolved:
            d.pop(i, None)
    shared89 = sorted(set(strip) & set(raw) & set(uniform) & set(published))
    print("89-subset: %d records, %d unresolved dropped upstream"
          % (len(shared89), len(unresolved)))

    by_prefix = index_transcripts()
    con = sqlite3.connect(INDEX_DB, uri=True)
    cache = {}
    items, skipped = [], []
    for rid in shared89:
        r = published[rid]
        sid = r["session_id"]
        note = resolve_rollup(sid)
        m = SLUG.match(sid)
        tpath = resolve(m.group(1), by_prefix) if m else None
        if note is None or tpath is None:
            skipped.append({"record_id": rid, "session_id": sid,
                            "reason": "no rollup note" if note is None
                                      else "no transcript"})
            continue
        vault_rel = "sessions/" + note.name
        if tpath not in cache:
            cache[str(tpath)] = strip_transcript(tpath.read_text(errors="replace"))[0]
        stripped = cache[str(tpath)]

        items.append({
            "record_id": rid, "session_id": sid, "class": r.get("class"),
            "question": r["question"], "ground_truth": r.get("ground_truth", ""),
            "note": vault_rel, "transcript": str(tpath),
            "stripped_chars": len(stripped),
            "stripped": stripped,
        })
    con.close()
    if skipped:
        print("SKIPPED %d: %s" % (len(skipped), skipped))
    if limit:
        items = items[:limit]
    return items, skipped


def build_contexts_at_budget(item: dict, budget: int) -> dict:
    """Build contexts for both arms at the given budget."""
    con = sqlite3.connect(INDEX_DB, uri=True)
    ctx = build_rollup_contexts(con, item["note"], item["question"])
    con.close()

    # Build rollup_hookshape context at this budget
    rollup_base = ctx["rollup_hookshape"]
    if len(rollup_base) <= budget:
        rollup_ctx = rollup_base
    else:
        # Truncate at word boundary
        rollup_ctx = rollup_base[:budget].rsplit(" ", 1)[0] + "…"

    # Build stripped_chunk context at this budget
    stripped_ctx = build_stripped_context(
        item["transcript"], item["stripped"], item["question"], budget)

    return {
        "rollup_hookshape": rollup_ctx,
        "stripped_chunk": stripped_ctx,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--degenerate", action="store_true",
                    help="smoke the context builders on empty/garbage inputs")
    args = ap.parse_args()

    if args.degenerate:
        con = sqlite3.connect(INDEX_DB, uri=True)
        # Test with empty notes, empty questions
        for note, q in [("sessions/does-not-exist.md", "what happened?"),
                        ("sessions/does-not-exist.md", ""),
                        ("", "?????")]:
            c = build_rollup_contexts(con, note, q)
            for budget in [0, 10, 125, 500, 8000]:
                if len(c["rollup_hookshape"]) <= budget:
                    rollup_ctx = c["rollup_hookshape"]
                else:
                    rollup_ctx = c["rollup_hookshape"][:budget].rsplit(" ", 1)[0] + "…"
                print("note=%r q=%r budget=%d -> rollup=%d chars %r"
                      % (note, q, budget, len(rollup_ctx), rollup_ctx[:40]))
        # Test stripped context with extreme cases
        for stripped, budget in [("", 500), ("abc", 500), ("x" * 10, 3),
                                 ("y" * 50, 0), ("z" * 10000, 125)]:
            s = build_stripped_context("/tmp/t.jsonl", stripped, "what failed", budget)
            print("stripped=%d budget=%d -> %d chars %r"
                  % (len(stripped), budget, len(s), s[:40]))
        con.close()
        return

    items, skipped = build_items(args.limit)
    print(f"\nLoaded {len(items)} items")

    # Report context sizes per budget
    if items and args.dry_run:
        print("\nContext sizes per budget (first 3 items):")
        for budget in BUDGETS:
            sizes = {arm: [] for arm in ARMS}
            for item in items[:3]:
                ctx = build_contexts_at_budget(item, budget)
                for arm in ARMS:
                    sizes[arm].append(len(ctx[arm]))
            print(f"  budget={budget:5d}:", end="")
            for arm in ARMS:
                print(f"  {arm:17s} {sizes[arm]}", end="")
            print()

        # Verify context_chars never exceeds budget
        print("\nVerifying context_chars <= budget for all items...")
        violations = []
        for item in items:
            for budget in BUDGETS:
                ctx = build_contexts_at_budget(item, budget)
                for arm in ARMS:
                    if len(ctx[arm]) > budget:
                        violations.append({
                            "record_id": item["record_id"],
                            "arm": arm,
                            "budget": budget,
                            "actual": len(ctx[arm]),
                            "excess": len(ctx[arm]) - budget,
                        })
        if violations:
            print(f"VIOLATIONS: {len(violations)} contexts exceeded budget")
            for v in violations[:10]:
                print(f"  {v}")
        else:
            print(f"OK: all {len(items) * len(BUDGETS) * len(ARMS)} contexts fit their budgets")

        print("\n--dry-run: no API calls made")
        return

    # Resumable: skip already-graded cells
    done = set()
    if OUT.exists():
        for r in load_records(OUT):
            done.add((r.get("record_id"), r.get("arm"), r.get("budget")))
        print(f"resuming: {len(done)} (record, arm, budget) cells already graded")

    # Build todo list
    todo = [(i, a, b) for i in items for a in ARMS for b in BUDGETS
            if (i["record_id"], a, b) not in done]
    print(f"to grade: {len(todo)} cells (out of {len(items)} x {len(ARMS)} x {len(BUDGETS)})\n")

    t0 = time.time()
    tokens_in = tokens_out = 0
    errors = 0

    for n, (item, arm, budget) in enumerate(todo, 1):
        ctx = build_contexts_at_budget(item, budget)
        ans, conf, usage = answer(item["question"], ctx[arm])
        tokens_in += usage.get("prompt_tokens") or 0
        tokens_out += usage.get("completion_tokens") or 0

        if conf == "API_ERROR":
            errors += 1
            print("[%d/%d] %s %s budget=%d -> API_ERROR (not recorded)"
                  % (n, len(todo), item["record_id"], arm, budget))
            continue

        grade = grade_answer(ans, item["ground_truth"], conf)
        rec = {
            "record_id": item["record_id"],
            "arm": arm,
            "budget": budget,
            "session_id": item["session_id"],
            "class": item["class"],
            "context_chars": len(ctx[arm]),
            "confidence": conf,
            "grade": grade,
            "answer": ans,
            "usage": usage,
            "model": JUDGE_MODEL,
        }
        with OUT.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print("[%d/%d] %-16s %-17s budget=%5d %-10s %s  (%.0fs, in=%d out=%d)"
              % (n, len(todo), item["record_id"], arm, budget, conf, grade,
                 time.time() - t0, tokens_in, tokens_out))

    print("\nAPI errors (unrecorded, retryable): %d" % errors)
    print("tokens: prompt=%d completion=%d" % (tokens_in, tokens_out))

    # Compute spend
    INPUT_PRICE = 2.00 / 1_000_000   # $2.00 per 1M
    OUTPUT_PRICE = 10.00 / 1_000_000  # $10.00 per 1M
    spend_usd = tokens_in * INPUT_PRICE + tokens_out * OUTPUT_PRICE
    print(f"estimated cost: ${spend_usd:.2f}")

    summarise(items, skipped, {
        "prompt_tokens": tokens_in,
        "completion_tokens": tokens_out,
        "api_errors": errors,
        "spend_usd": round(spend_usd, 2),
    })


def summarise(items: list, skipped: list, spend: dict) -> None:
    rows = load_records(OUT)

    # Group by (arm, budget)
    by_arm_budget = {}
    for r in rows:
        key = (r.get("arm"), r.get("budget"))
        if key not in by_arm_budget:
            by_arm_budget[key] = {}
        by_arm_budget[key][r["record_id"]] = r

    attempted = {i["record_id"] for i in items}

    print(f"\nattempted {len(attempted)} records x {len(ARMS)} arms x {len(BUDGETS)} budgets = {len(attempted) * len(ARMS) * len(BUDGETS)} cells")

    # Summary table
    hdr = "%-17s %6s  %5s  %3s %3s %3s %3s  %-8s %-14s %s" % (
        "arm", "budget", "n", "cor", "par", "wrg", "abs",
        "recover", "wilson95", "median chars")
    print("\n" + hdr + "\n" + "-" * len(hdr))

    out = {
        "n_attempted_records": len(attempted),
        "skipped_records": skipped,
        "spend": spend,
        "cells": {},
    }

    for arm in ARMS:
        for budget in BUDGETS:
            key = (arm, budget)
            cell = by_arm_budget.get(key, {})
            complete = sorted(set(cell) & attempted)
            rs = [cell[i] for i in complete]

            if not rs:
                continue

            d = rates(rs)
            chars = sorted(r["context_chars"] for r in rs)
            d["median_context_chars"] = chars[len(chars) // 2] if chars else 0
            out["cells"][f"{arm}@{budget}"] = d

            print("%-17s %6d  %5d  %3d %3d %3d %3d  %7.1f%% %-14s %d"
                  % (arm, budget, d["n"], d["correct"]["n"],
                     d["partially_correct"]["n"], d["wrong"]["n"],
                     d["absent"]["n"], d["recoverable"]["pct"],
                     str(d["recoverable"]["ci95"]), d["median_context_chars"]))

    REPORT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
