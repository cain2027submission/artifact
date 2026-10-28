#!/usr/bin/env python3
"""Preflight for the later-window experiment. No API calls.

For every paired (developer) session of the pinned corpus, record whether its
transcript survives, its pinned rollup resolves, how long the generator's text
view is, whether MID and LATE each hold >= 30,000 characters, and the sizes of
the four answer contexts (for the cost estimate). Writes
data/paper-rollups/late-window/preflight.json and prints the counts.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq3_late_common as lc  # noqa: E402
from rq3_stripped_transcript_arm import strip_transcript  # noqa: E402
from rq3_prose_only_arm import prose_only_context  # noqa: E402
from rq3_tool_inputs_only_arm import tool_inputs_only_context  # noqa: E402


def one(args):
    uuid, tpath, nbytes, note = args
    row = {"uuid": uuid, "prefix": uuid[:8], "transcript": tpath, "pinned_bytes": nbytes,
           "rollup_note": note}
    row["transcript_exists"] = bool(tpath) and os.path.exists(tpath)
    rollup = lc.pinned_rollup(note) if note else None
    row["rollup_pinned"] = rollup is not None
    row["rollup_chars"] = len(rollup) if rollup else 0
    if note and os.path.exists(note) and rollup is not None:
        row["rollup_current_equals_pinned"] = open(note, errors="replace").read() == rollup
    if not row["transcript_exists"]:
        return row
    actual = os.path.getsize(tpath)
    row["transcript_shorter_than_pin"] = actual < nbytes
    raw = lc.pinned_raw(tpath, nbytes)
    all_entries = lc.parse_entries(raw)
    entries = lc.window_entries(all_entries)
    text = lc.transcript_to_text(entries)
    row["harness_entries_dropped"] = len(all_entries) - len(entries)
    row["gen_text_chars_unfiltered"] = len(lc.transcript_to_text(all_entries))
    row["gen_text_chars"] = len(text)
    row["delegating"] = lc.is_delegating(all_entries)
    spans, total = lc.entry_spans(entries)
    assert total == len(text), (uuid, total, len(text))
    for w in lc.WINDOWS:
        off = lc.window_offset(uuid, w, len(text))
        row[f"{w}_offset"] = off
        row[f"{w}_dominance"] = None if off is None else round(lc.window_dominance(spans, off), 3)
    row["stripped_chars"] = len(strip_transcript(raw)[0])
    row["prose_chars"] = len(prose_only_context(raw)[0])
    row["callonly_chars"] = len(tool_inputs_only_context(raw)[0])
    return row


def main():
    snap = lc.snapshot()
    by_uuid = {Path(p).stem: p for p in snap["transcripts"]}
    paired = snap["paired_session_ids"]
    existing = lc.existing_sessions()
    jobs = [(u, by_uuid.get(u), snap["transcript_bytes"].get(by_uuid.get(u, ""), 0),
             snap["rollup_note_paths"].get(u)) for u in paired]
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(one, jobs, chunksize=4))
    for r in rows:
        e = existing.get(r["prefix"])
        r["in_existing_68"] = e is not None
        r["in_graded_25"] = bool(e and e["graded"])
        r["slug"] = e["slug"] if e else None
        r["eligible_mid"] = r.get("mid_offset") is not None
        r["eligible_late"] = r.get("late_offset") is not None
        r["eligible_length"] = (r["transcript_exists"] and r["rollup_pinned"]
                                and r["eligible_mid"] and r["eligible_late"])
        r["dominated"] = bool(r["eligible_length"] and max(
            r["mid_dominance"], r["late_dominance"]) > 0.5)
        r["eligible"] = r["eligible_length"] and not r["dominated"]
    unmatched = sorted(set(existing) - {r["prefix"] for r in rows})
    lc.OUTDIR.mkdir(parents=True, exist_ok=True)
    (lc.OUTDIR / "preflight.json").write_text(json.dumps(
        {"rows": rows, "existing_prefixes_not_in_paired": unmatched}, indent=1))

    def cnt(f):
        return sum(1 for r in rows if f(r))
    print(f"paired sessions in snapshot: {len(rows)}")
    print(f"  transcript survives: {cnt(lambda r: r['transcript_exists'])}")
    print(f"  pinned rollup resolves: {cnt(lambda r: r['rollup_pinned'])}")
    print(f"  pinned rollup == current file: {cnt(lambda r: r.get('rollup_current_equals_pinned'))}")
    print(f"  long enough (both windows >= 30k, transcript + rollup): {cnt(lambda r: r['eligible_length'])}")
    print(f"  of those, a window slice >50% one entry (excluded): {cnt(lambda r: r['dominated'])}")
    print(f"  eligible (both windows >= 30k, transcript + rollup): {cnt(lambda r: r['eligible'])}")
    print(f"     of which delegating: {cnt(lambda r: r['eligible'] and r['delegating'])}")
    print(f"existing 68 matched to paired: {cnt(lambda r: r['in_existing_68'])}, "
          f"unmatched prefixes: {unmatched}")
    print(f"  existing 68 eligible: {cnt(lambda r: r['in_existing_68'] and r['eligible'])}")
    print(f"  graded 25 matched: {cnt(lambda r: r['in_graded_25'])}, eligible: "
          f"{cnt(lambda r: r['in_graded_25'] and r['eligible'])}")
    el = sorted([r for r in rows if r["eligible"]], key=lambda r: r["gen_text_chars"])
    if el:
        import statistics as st
        for k in ("gen_text_chars", "stripped_chars", "prose_chars", "callonly_chars", "rollup_chars"):
            v = [r[k] for r in el]
            print(f"  eligible {k}: median {st.median(v):,.0f}  p90 {v[int(.9*len(v))-1] if k=='gen_text_chars' else sorted(v)[int(.9*len(v))-1]:,}  max {max(v):,}  sum {sum(v):,}")


if __name__ == "__main__":
    main()
