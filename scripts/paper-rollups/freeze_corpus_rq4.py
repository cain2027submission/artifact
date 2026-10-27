#!/usr/bin/env python3
"""Freeze a SEPARATE corpus snapshot for RQ4's paired-session pool, with archived bytes.

Why a separate snapshot, rather than regenerating corpus_snapshot.json: that
file is read by rq0, rq1, rq2, three rq3 scripts, rq5 (x2) and
substrate_value (grep-confirmed), and the paper quotes its own
``snapshot_utc`` for every one of those. RQ4 is being re-run on an updated,
larger paired-session pool (300 -> 550+ since the 2026-09-27 freeze) with a
methodology fix landed in rq4_known_item.py, and regenerating the SHARED
snapshot to pick that up would silently move every other RQ's
already-published denominator. So RQ4 gets its own file --
``data/paper-rollups/corpus_snapshot_rq4_2026-10-06.json`` -- built here.
This script never touches ``freeze_corpus.build()``, ``load_manifest()`` or
``main()``, and never writes ``freeze_corpus.OUT``; those keep working
unchanged for every other RQ.

Why this one also archives bytes, unlike the shared snapshot: freeze_corpus's
``read_pinned`` / ``read_pinned_rollup`` only pin a path + byte-length and
re-read the LIVE file at use time, truncated to that length -- they do not
archive content. That is exactly why 968 of the original 2,602 pinned
transcripts are unreadable today: ``~/.claude/settings.json``'s
``cleanupPeriodDays: 180`` reclaims old session files out from under the pin,
and a second, independent gap -- the rollup drain renaming notes in place --
drops rollup pins the same way. Re-reading the live filesystem at analysis
time cannot recover from either failure after the fact, no matter how
careful the byte-length bookkeeping is. For RQ4's new snapshot specifically,
we archive the actual pinned bytes once, at freeze time, into their own
files under ``~/.cache/paper-rollups/rq4_archive_2026-10-06/`` -- outside the
repo and outside any committed JSON, the same pattern rq4_batched.py already
uses for its own SCRATCH dir, because a corpus of this size serialized into a
commit would dwarf the vault. Only the archive directory's *path* is recorded
in the manifest; the manifest itself stays small enough to commit.

Only sessions in the paired set (transcript AND rollup both exist) are
archived: RQ4 never looks at the other ~2,300 unpaired transcripts, so there
is nothing to gain archiving them.

Session discovery is NOT reimplemented here -- this calls freeze_corpus.build()
directly and only post-processes its output, so RQ4's notion of "paired"
can never drift from the shared one (same PROJECTS/ROLLUPS globs, same
parse_session_id frontmatter parsing, same transcript_bytes/rollup_note_bytes
pinning logic). read_pinned/read_pinned_rollup are reused the same way, to
read each paired session's content at the moment of this freeze.

Run:
    python3 freeze_corpus_rq4.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import freeze_corpus as FC
from paths import (
    DATA_DIR,
    home_path_pattern,
)

ARCHIVE_DIR = Path.home() / ".cache" / "paper-rollups" / "rq4_archive_2026-10-06"
OUT = DATA_DIR / "corpus_snapshot_rq4_2026-10-06.json"


def build() -> dict:
    """Build an RQ4-only manifest over the paired set, with content archived.

    ``live`` is exactly what freeze_corpus.build() reports right now -- full
    discovery, including the ~2,300 unpaired transcripts this script never
    touches. Only the paired subset gets archived and carried into the
    returned manifest.
    """
    live = FC.build()
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # sid -> its own main transcript path, keyed by exact stem rather than by
    # substring. freeze_corpus.build() glob is recursive over EVERY project
    # directory, so a session_id can appear as a substring of more than one
    # path -- its own "<sid>.jsonl" in one project dir, AND a subagent's
    # "<other-id>.jsonl" nested under "<sid>/subagents/" in another, possibly
    # a *different* project directory entirely. Sorting those candidates and
    # taking the first is NOT safe across project directories;
    # one pattern sorts before another (a dash is less than a slash in ASCII), so a
    # subagent file from an unrelated worktree's project dir can sort ahead of
    # the session's own transcript. That was caught here by a failing
    # round-trip check (the archived file came back empty on read because the
    # path's stem wasn't the session id) -- picking by exact stem match is
    # unambiguous and is already guaranteed to exist for every sid in
    # paired_session_ids, because that is exactly build()'s own pairing test
    # two lines up its chain: `stems = {Path(p).stem for p in transcripts};
    # paired = sid in stems`.
    stem_to_path = {Path(p).stem: p for p in live["transcripts"]}

    transcript_path_for: dict[str, str] = {}   # session_id -> transcript path
    transcript_bytes: dict[str, int] = {}
    rollup_paths: dict[str, str] = {}
    rollup_bytes: dict[str, int] = {}
    skipped_no_transcript_candidate = 0
    skipped_no_rollup_pin = 0

    for sid in live["paired_session_ids"]:
        path = stem_to_path.get(sid)
        if path is None:
            skipped_no_transcript_candidate += 1
            continue

        rollup_text = FC.read_pinned_rollup(sid, live)
        if rollup_text is None:
            # No valid pin at freeze time (missing rollup_note_bytes, or the
            # pinned path is already gone) -- drop rather than guess, same
            # policy read_pinned_rollup itself documents.
            skipped_no_rollup_pin += 1
            continue
        transcript_text = FC.read_pinned(path, live)

        (ARCHIVE_DIR / f"{sid}_transcript.txt").write_text(transcript_text)
        (ARCHIVE_DIR / f"{sid}_rollup.md").write_text(rollup_text)

        transcript_path_for[sid] = path
        transcript_bytes[path] = live["transcript_bytes"].get(path, 0)
        rollup_paths[sid] = live["rollup_note_paths"].get(sid, "")
        rollup_bytes[sid] = live["rollup_note_bytes"].get(sid, 0)

    paired = sorted(transcript_path_for)

    return {
        "snapshot_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "RQ4's OWN corpus snapshot -- separate from corpus_snapshot.json, "
                "which rq0/rq1/rq2/rq3 (x3)/rq5 (x2)/substrate_value depend on and "
                "which this script never writes. Scoped to the paired-session pool "
                "only. Unlike the shared snapshot, this one archives each paired "
                "session's pinned transcript+rollup bytes to archive_dir at freeze "
                "time (not just a path+length pin re-read live), so a later run is "
                "immune to transcript reclamation (cleanupPeriodDays in "
                "~/.claude/settings.json) and to rollup-drain renames.",
        "built_from": "freeze_corpus.build() (live discovery, not reimplemented here)",
        "source_live_snapshot_utc": live["snapshot_utc"],
        "source_live_n_transcripts": live["n_transcripts"],
        "source_live_n_rollups": live["n_rollups"],
        "source_live_n_paired": live["n_paired"],
        "skipped_no_transcript_candidate": skipped_no_transcript_candidate,
        "skipped_no_rollup_pin": skipped_no_rollup_pin,
        "archive_dir": str(ARCHIVE_DIR),
        "content_pinning": "transcript_bytes/rollup_note_bytes are kept for parity "
                           "with corpus_snapshot.json's shape, but read_pinned and "
                           "read_pinned_rollup prefer the archived copy in "
                           "archive_dir whenever this manifest is the one in use, "
                           "so the byte-length pin is a fallback here, not the "
                           "primary mechanism.",
        "n_transcripts": len(transcript_path_for),
        "n_rollups": len(rollup_paths),
        "n_paired": len(paired),
        "paired_session_ids": paired,
        "transcripts": sorted(transcript_path_for.values()),
        "transcript_bytes": transcript_bytes,
        "total_transcript_bytes": sum(transcript_bytes.values()),
        "rollup_files": sorted(Path(p).name for p in rollup_paths.values() if p),
        "rollup_note_paths": rollup_paths,
        "rollup_note_bytes": rollup_bytes,
    }


def main() -> None:
    m = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(m, indent=2))
    print(f"snapshot_utc  {m['snapshot_utc']}")
    print(f"transcripts   {m['n_transcripts']}")
    print(f"rollups       {m['n_rollups']}")
    print(f"paired        {m['n_paired']}")
    if m["skipped_no_transcript_candidate"] or m["skipped_no_rollup_pin"]:
        print(f"skipped       {m['skipped_no_transcript_candidate']} (no transcript candidate), "
              f"{m['skipped_no_rollup_pin']} (no valid rollup pin)")
    print(f"archive_dir   {m['archive_dir']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
