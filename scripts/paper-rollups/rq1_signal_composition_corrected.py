#!/usr/bin/env python3
"""RQ1, corrected: what real coding-agent transcripts are actually made of.

Three corrections over the first pass, all of which changed the headline.

0. Reasoning is not in the parent transcript at all. The first pass reported
   thinking at 0.09% of parent volume against 8.4% of subagent volume and that
   asymmetry looked like a parser bug. It is not. A parent-transcript thinking
   block is persisted as `{"type": "thinking", "thinking": "", "signature":
   "<1216 chars>"}`: the block survives, the reasoning text does not. Measured
   over the frozen corpus, 97.40% of 18,096 parent thinking blocks are empty
   against 2.95% of 17,584 subagent blocks, leaving 1,101,450 chars of retained
   reasoning text in parent files against 9,984,812 in subagent files. (These
   supersede an earlier snapshot of 98.1%/17,135 and 670KB that a previous
   version of this docstring carried; `rq1_corrected_summary.json`
   -> `thinking_persistence` is the authority, not this comment.)

   The subagent column is the control that makes this a finding rather than a
   guess, and it also bounds the claim: one client, one API, one machine, the
   same models within a single session -- and the reasoning text is present in
   97.05% of subagent blocks. So the loss is not the model declining to emit
   reasoning and not the API declining to return it. It is where this client
   writes parent turns to disk, which is an unversioned implementation detail
   and could change in a patch release.

   The consequence for this paper is load-bearing and is measured here rather
   than asserted: for a parent session, no retrieval system operating on the
   archived trace can recover the model's reasoning, because it was never
   written to disk. A generative rollup produced while the session is live is
   the only durable record of it.

1. `user_text` was contaminated. A first pass attributed 38% of corpus volume to
   human-authored turns, and 67-81% in the smallest size deciles, which is not
   credible for a developer typing prompts. Inspection showed the smallest
   sessions consist of harness scaffolding: `<local-command-caveat>` blocks,
   `<command-name>/clear</command-name>` wrappers, `<task-notification>` frames
   and `isMeta` entries. None of that is authored by anyone; it is injected by
   the client. It is now stripped into `system_injected`, span by span, because
   these markers are frequently appended to a real prompt inside one block
   rather than occupying a block of their own.

2. Trivial sessions were included. A transcript containing one `/clear` and
   nothing else is not a session any memory system would want to summarise, yet
   it counts once in every median. Sessions are now partitioned into substantive
   and trivial, with the composition claim made over the substantive subset and
   the excluded count reported.

3. `has_subagents` was read from `isSidechain`, which is only ever written inside
   a subagent transcript. So it was true for every `agent-*.jsonl` and false for
   every one of the 1,477 parent sessions, duplicating `is_subagent` and leaving
   `by_subagent_use` with with_subagents n=0 against without n=249 -- a block
   that said, wrongly, that no session in this corpus delegated. Delegation is a
   structural fact about the directory layout, not a field, and is now read
   through `delegation.py`, which is also what the RQ3 reweighting imports so the
   two cannot disagree. 158 of the 1,477 parent sessions delegated, and all 158
   of them are substantive.

Run:
    python3 rq1_signal_composition_corrected.py --from-csv

`--from-csv` rebuilds every aggregate from the committed per-session CSV, which
is the snapshot-time scan. It is the correct mode for any rerun, because the
frozen corpus is only frozen in the manifest: the client's retention sweep had
deleted 813 of the 1,477 parent transcripts two days after the 2026-09-27
snapshot, and a rescan silently rebuilds the paper's numbers over what is left
(retained parent reasoning drops 1,101,450 -> 508,841 chars). Without the flag
this script now refuses to write rather than doing that quietly.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from delegation import delegating_ids_from_manifest
from freeze_corpus import load_manifest, read_pinned
from paths import PROJECTS_DIR

PROJECTS = PROJECTS_DIR
OUTDIR = Path(__file__).resolve().parents[2] / "data" / "paper-rollups"

# Client-injected wrappers observed in the corpus. Their contents are harness
# scaffolding, not conversation, and each is removed with its enclosed text.
META_TAGS = (
    "system-reminder", "local-command-caveat", "local-command-stdout",
    "command-name", "command-message", "command-args",
    "task-notification", "task-id", "tool-use-id", "output-file", "status",
    "summary", "note", "bash-input", "bash-stdout", "bash-stderr",
    "user-prompt-submit-hook", "session-start-hook", "env",
)
META_RE = re.compile(
    r"<(" + "|".join(META_TAGS) + r")>.*?</\1>|<(" + "|".join(META_TAGS) + r")>",
    re.DOTALL,
)

CATEGORIES = ("tool_result", "tool_use", "thinking", "assistant_text",
              "user_text", "system_injected", "other")


def split_meta(text: str) -> tuple[str, str]:
    """Return (authored_text, injected_text) for one user text block."""
    injected = "".join(m.group(0) for m in META_RE.finditer(text))
    authored = META_RE.sub("", text)
    return authored, injected


def scan(path: Path, manifest: dict, delegating_ids: set[str]) -> dict | None:
    chars: dict[str, int] = defaultdict(int)
    usage_by_request: dict[str, dict] = {}
    entries = 0
    tool_calls = 0
    compactions = 0
    think_blocks = 0
    think_empty = 0

    # Read through the pin, not path.read_text(): this transcript may still be
    # open and growing (the analysis session is itself a coding-agent session
    # in the same corpus), and an unpinned read moves this script's own numbers
    # between runs the same way the determinism gate caught for the parent
    # reasoning-persistence script (main.tex Sec. "The corpus is live").
    try:
        raw = read_pinned(str(path), manifest)
    except OSError:
        return None

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        entries += 1
        if rec.get("type") in ("summary", "compact_boundary") \
                or rec.get("subtype") == "compact_boundary":
            compactions += 1

        msg = rec.get("message")
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        is_meta = bool(rec.get("isMeta"))
        content = msg.get("content")
        blocks = content if isinstance(content, list) else [content]

        for b in blocks:
            if isinstance(b, str):
                btype, text = "text", b
            elif isinstance(b, dict):
                btype = b.get("type", "other")
                if btype in ("text", "thinking", "redacted_thinking"):
                    text = b.get("text") or b.get("thinking") or ""
                elif btype == "tool_use":
                    text = json.dumps(b.get("input", ""))
                elif btype == "tool_result":
                    c = b.get("content")
                    if isinstance(c, str):
                        text = c
                    elif isinstance(c, list):
                        text = "".join(x.get("text", "") for x in c
                                       if isinstance(x, dict))
                    else:
                        text = json.dumps(c) if c is not None else ""
                else:
                    btype, text = "other", ""
            else:
                continue

            # Counted before the empty-text guard below, because an empty
            # thinking block is the finding rather than something to skip.
            if btype in ("thinking", "redacted_thinking"):
                think_blocks += 1
                if not text:
                    think_empty += 1

            if not text:
                continue

            if btype == "tool_result":
                chars["tool_result"] += len(text)
            elif btype == "tool_use":
                chars["tool_use"] += len(text)
                tool_calls += 1
            elif btype in ("thinking", "redacted_thinking"):
                chars["thinking"] += len(text)
            elif btype == "text" and role == "assistant":
                chars["assistant_text"] += len(text)
            elif btype == "text" and role == "user":
                if is_meta:
                    chars["system_injected"] += len(text)
                else:
                    authored, injected = split_meta(text)
                    chars["user_text"] += len(authored.strip())
                    chars["system_injected"] += len(injected)
            else:
                chars["other"] += len(text)

        rid = rec.get("requestId")
        usage = msg.get("usage")
        if rid and isinstance(usage, dict):
            usage_by_request[rid] = usage

    total = sum(chars.values())
    if total == 0:
        return None
    return {
        "serialized_bytes": len(raw),
        "payload_chars": total,
        "entries": entries,
        "requests": len(usage_by_request),
        "tool_calls": tool_calls,
        # Read from the corpus manifest's file list, not from any field in this
        # transcript: nothing inside a parent transcript records that it
        # delegated. `isSidechain`, which this used to read, is written only
        # inside a subagent file, so the flag was 0 for every parent session and
        # `by_subagent_use` below came out with n=0 on one side. See
        # delegation.py for the single definition.
        "has_subagents": int(path.stem in delegating_ids),
        "compactions": compactions,
        "thinking_blocks": think_blocks,
        "thinking_blocks_empty": think_empty,
        **{f"{c}_chars": chars.get(c, 0) for c in CATEGORIES},
    }


def q(vals: list[float], p: float) -> float:
    s = sorted(vals)
    if not s:
        return float("nan")
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]


def shares(rows: list[dict]) -> dict:
    out = {}
    for c in CATEGORIES:
        vals = [r[f"{c}_chars"] / r["payload_chars"] for r in rows]
        out[c] = {
            "mean": round(statistics.mean(vals), 4),
            "median": round(statistics.median(vals), 4),
            "p10": round(q(vals, 0.10), 4),
            "p90": round(q(vals, 0.90), 4),
        }
    # Pooled share: one number over the whole corpus rather than an average of
    # per-session ratios. Small sessions cannot dominate it, so it is the figure
    # to quote when describing the corpus as a body of text.
    total = sum(r["payload_chars"] for r in rows)
    out["_pooled"] = {
        c: round(sum(r[f"{c}_chars"] for r in rows) / total, 4)
        for c in CATEGORIES
    } if total else {}
    return out


CSV_INT_FIELDS = ("serialized_bytes", "payload_chars", "entries", "requests",
                  "tool_calls", "has_subagents", "compactions", "thinking_blocks",
                  "thinking_blocks_empty", "is_subagent",
                  *(f"{c}_chars" for c in CATEGORIES))


def rows_from_csv(path: Path) -> list[dict]:
    """Re-read the committed per-session CSV instead of rescanning transcripts.

    This exists because the frozen corpus is not actually frozen on disk. The
    manifest pins path and byte length, which makes a file that GREW readable at
    its snapshot state, but it cannot resurrect a file that was DELETED -- and
    the client's retention sweep deletes them. Two days after the 2026-09-27
    snapshot, 813 of the 1,477 parent transcripts were gone. Rescanning then
    silently drops them and moves published numbers by up to 54% (retained parent
    reasoning 1,101,450 -> 508,841 chars), which looks like a bug in whatever
    change prompted the rerun and is not.

    The per-session CSV is the snapshot-time scan, committed, so every aggregate
    below can be rebuilt from it byte-for-byte without touching the corpus. Use
    --from-csv for any rerun that is not deliberately re-freezing the corpus.
    """
    with path.open(newline="") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    for r in rows:
        for k in CSV_INT_FIELDS:
            r[k] = int(r[k])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true",
                    help="rebuild the summary from the committed per-session CSV "
                         "rather than rescanning transcripts; required once the "
                         "retention sweep has deleted part of the frozen corpus")
    args = ap.parse_args()

    # The glob must recurse. A non-recursive `*/*.jsonl` sees 1,136 files and
    # misses 1,074 more: subagent transcripts are written to
    # `<project>/<session-id>/subagents/agent-*.jsonl`, one file per delegated
    # agent, and are NOT inlined into the parent transcript. Any volume claim
    # about "what a session contains" that reads only parent files understates
    # delegated tool churn by roughly half the corpus, and only parent sessions
    # receive a rollup note.
    manifest = load_manifest()
    transcript_paths = {Path(p) for p in manifest["transcripts"]}
    # Classified from the manifest rather than by globbing the live filesystem so
    # the flag is pinned to the snapshot: two of the delegating parents in the
    # 2026-09-27 snapshot have since had their subagent directory pruned, and a
    # live glob would move this number between runs.
    delegating_ids = delegating_ids_from_manifest(manifest["transcripts"])

    if args.from_csv:
        rows = rows_from_csv(OUTDIR / "rq1_corrected_per_session.csv")
        # The one field the CSV cannot be trusted on: it was written when
        # has_subagents came from isSidechain. Everything else in the CSV is a
        # count of snapshot-time bytes and is re-derived from nothing.
        for r in rows:
            r["has_subagents"] = int(r["session_id"] in delegating_ids
                                     and not r["is_subagent"])
        present = sum(1 for p in transcript_paths if p.exists())
        print(f"--from-csv: {len(rows)} rows from the snapshot-time scan "
              f"({present}/{len(transcript_paths)} manifest transcripts still on "
              f"disk, so a rescan would drop "
              f"{len(transcript_paths) - present})")
    else:
        rows = []
        gone = 0
        for p in sorted(transcript_paths):
            s = scan(p, manifest, delegating_ids)
            if s:
                s["session_id"] = p.stem
                s["project"] = p.parts[len(PROJECTS.parts)] if len(p.parts) > len(PROJECTS.parts) else ""
                s["is_subagent"] = int("subagents" in p.parts)
                rows.append(s)
            else:
                gone += 1
        if gone:
            # Loud, not a warning buried in a log: a rerun that quietly loses a
            # third of the corpus produces a plausible summary that disagrees
            # with the paper for reasons nothing in the output explains.
            raise SystemExit(
                f"REFUSING TO WRITE: {gone} of {len(transcript_paths)} frozen "
                f"transcripts could not be read (deleted by the retention sweep "
                f"since {manifest['snapshot_utc']}). Rescanning now would rebuild "
                f"every aggregate over a corpus that no longer matches the "
                f"snapshot. Re-run with --from-csv to rebuild from the "
                f"snapshot-time per-session scan, or re-freeze deliberately.")

    # A session is substantive if an agent actually did work in it: at least one
    # tool call and enough text to be worth summarising. The thresholds are
    # deliberately low; the purpose is to exclude `/clear` stubs and
    # opened-then-abandoned sessions, not to curate for a flattering result.
    #
    # An earlier version also required >=2 deduplicated API requests and that
    # rejected 896 sessions, which looked like a corpus of abandoned launches
    # until checked: `requestId` is absent on older entries, so the criterion was
    # measuring transcript vintage rather than session substance. Dropped.
    def substantive(r: dict) -> bool:
        return r["tool_calls"] >= 1 and r["payload_chars"] >= 5000

    parents = [r for r in rows if not r["is_subagent"]]
    subagent_files = [r for r in rows if r["is_subagent"]]
    sub = [r for r in parents if substantive(r)]
    triv = [r for r in parents if not substantive(r)]

    by_decile = []
    ordered = sorted(sub, key=lambda r: r["payload_chars"])
    if ordered:
        size = max(1, len(ordered) // 10)
        for i in range(10):
            chunk = ordered[i * size:(i + 1) * size] if i < 9 else ordered[9 * size:]
            if chunk:
                by_decile.append({
                    "decile": i + 1,
                    "n": len(chunk),
                    "median_payload_chars": statistics.median(
                        [c["payload_chars"] for c in chunk]),
                    **{c: round(statistics.mean(
                        [x[f"{c}_chars"] / x["payload_chars"] for x in chunk]), 4)
                       for c in ("tool_result", "tool_use", "thinking",
                                 "assistant_text", "user_text")},
                })

    with_sub = [r for r in sub if r["has_subagents"]]
    without_sub = [r for r in sub if not r["has_subagents"]]

    def think(rs: list[dict]) -> dict:
        n = sum(r["thinking_blocks"] for r in rs)
        e = sum(r["thinking_blocks_empty"] for r in rs)
        return {
            "blocks": n,
            "empty": e,
            "empty_pct": round(100.0 * e / n, 2) if n else None,
            "retained_chars": sum(r["thinking_chars"] for r in rs),
        }

    summary = {
        "corpus_snapshot_utc": manifest["snapshot_utc"],
        "n_transcripts_parsed": len(rows),
        # See the module docstring, correction 0. This is the paper's strongest
        # single number: the reasoning a rollup preserves is absent from the
        # archived parent trace, so no amount of retrieval engineering over raw
        # logs can recover it.
        "thinking_persistence": {
            "parent": think(parents),
            "subagent": think(subagent_files),
            "note": "a parent thinking block persists as an empty string plus a "
                    "signature; the reasoning text is not written to disk",
        },
        "n_parent_transcripts": len(parents),
        "n_subagent_transcripts": len(subagent_files),
        "n_substantive": len(sub),
        "n_trivial_excluded": len(triv),
        "substantive_filter": ">=1 tool call and >=5000 payload chars, parent sessions only",
        "shares_subagent_transcripts": shares(subagent_files) if subagent_files else {},
        "subagent_volume": {
            "payload_chars": sum(r["payload_chars"] for r in subagent_files),
            "share_of_total_corpus_payload": round(
                sum(r["payload_chars"] for r in subagent_files)
                / max(1, sum(r["payload_chars"] for r in rows)), 4),
        },
        "totals_substantive": {
            "serialized_bytes": sum(r["serialized_bytes"] for r in sub),
            "payload_chars": sum(r["payload_chars"] for r in sub),
            "api_requests": sum(r["requests"] for r in sub),
            "tool_calls": sum(r["tool_calls"] for r in sub),
            "sessions_with_subagents": len(with_sub),
        },
        "shares_substantive": shares(sub),
        "shares_all_including_trivial": shares(rows),
        "by_size_decile_substantive": by_decile,
        "n_delegating_parents": sum(r["has_subagents"] for r in parents),
        "delegation_detection": (
            "a parent session delegated iff <project>/<session-id>/subagents/"
            "agent-*.jsonl exists in the frozen manifest; see delegation.py. NOT "
            "isSidechain, which is written only inside subagent files and was 0 "
            "for every parent session in an earlier version of this script"),
        "by_subagent_use": {
            "with_subagents": {"n": len(with_sub), **(shares(with_sub) if with_sub else {})},
            "without_subagents": {"n": len(without_sub),
                                  **(shares(without_sub) if without_sub else {})},
        },
        # The paper's core framing: reasoning plus prose against mechanical I/O.
        "signal_vs_noise_substantive": {
            "reasoning_and_prose_pooled": round(
                sum(r["thinking_chars"] + r["assistant_text_chars"] for r in sub)
                / sum(r["payload_chars"] for r in sub), 4) if sub else None,
            "tool_io_pooled": round(
                sum(r["tool_result_chars"] + r["tool_use_chars"] for r in sub)
                / sum(r["payload_chars"] for r in sub), 4) if sub else None,
        },
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with (OUTDIR / "rq1_corrected_per_session.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUTDIR / "rq1_corrected_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
