#!/usr/bin/env python3
"""Freeze the corpus so every RQ reports numbers against the same snapshot.

Why this exists. The rollup pipeline writes a new `sessions/` note every few
minutes and `~/.claude/projects` grows with every session, so the corpus is
live. Re-running RQ2 an hour apart moved n_pairs from 286 to 287 and the median
token compression from 33.01 to 33.38 -- small, but it means no number in the
paper is reproducible and two RQs run at different times silently disagree about
their own denominator. A reviewer who re-ran the scripts would get a third set.

So the analysis is pinned to an explicit manifest: the session ids that had both
a rollup and a transcript at the snapshot instant, and the transcript paths that
existed then. Scripts import `load_manifest()` and filter to it. Regenerating the
snapshot is a deliberate act that changes one committed file, and the paper
quotes that file's `snapshot_utc`.

Run:
    python3 freeze_corpus.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from paths import DATA_DIR, PROJECTS_DIR, REPO_ROOT, SESSIONS_DIR

VAULT = REPO_ROOT
ROLLUPS = SESSIONS_DIR
PROJECTS = PROJECTS_DIR
OUT = DATA_DIR / "corpus_snapshot.json"


def parse_session_id(text: str) -> str | None:
    """Pull session_id out of the note's YAML frontmatter without a yaml dep."""
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    for line in text[3:end].splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        if k.strip() == "session_id":
            return v.strip().strip("\"'") or None
    return None


def build() -> dict:
    # Recursive: subagent transcripts live at <project>/<sid>/subagents/*.jsonl
    # and are 1,075 of the 2,219 files. A non-recursive glob misses them, which
    # is how an earlier run undercounted retained reasoning by 93%.
    transcripts = sorted(str(p) for p in PROJECTS.glob("**/*.jsonl"))
    stems = {Path(p).stem for p in transcripts}

    rollups: dict[str, str] = {}
    for f in sorted(ROLLUPS.glob("*.md")):
        sid = parse_session_id(f.read_text(errors="replace"))
        if sid:
            rollups[sid] = f.name

    paired = sorted(sid for sid in rollups if sid in stems)

    # Pinning the path list is not enough, and the determinism gate is how we
    # found out. A transcript already in the manifest keeps growing while it is
    # being written, so two runs an hour apart read different bytes from the same
    # pinned file and RQ1 moved 17,236 parent reasoning blocks to 17,306. The
    # sessions doing the analysis are themselves in the projects directory, so
    # the study perturbs its own corpus: measuring harder makes the corpus
    # bigger. Recording each file's length at snapshot time and truncating reads
    # there makes the corpus immutable in content as well as in membership.
    sizes = {}
    for p in transcripts:
        try:
            sizes[p] = Path(p).stat().st_size
        except OSError:
            sizes[p] = 0  # vanished between glob and stat; treated as empty

    # Rollup notes are not append-only like transcripts, but they are just as
    # live: the rollup drain re-rolls a note in place -- rewriting its body,
    # renaming its file to match a new title slug, or patching a frontmatter
    # field -- on a cadence of minutes, and does so whether or not the note is
    # committed yet. Recording each rollup's path and byte length at snapshot
    # time, the same way transcript_bytes pins transcripts, lets a reader
    # truncate to the pinned length and get the same text on every re-run --
    # *if* the file is still at that path. A rename changes the path, not the
    # content at the old one, so a rollup pin does not survive a rename the way
    # a transcript pin survives being appended to. See rollup_note_bytes and the
    # 2026-09-28 investigation note for why that gap could not be closed
    # retroactively for the corpus already frozen on 2026-09-27.
    rollup_paths = {sid: str(ROLLUPS / name) for sid, name in rollups.items()}
    rollup_sizes = {}
    for sid, path in rollup_paths.items():
        try:
            rollup_sizes[sid] = Path(path).stat().st_size
        except OSError:
            rollup_sizes[sid] = 0

    return {
        "snapshot_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "the frozen corpus every RQ in the paper is computed over; "
                "regenerate deliberately, never as a side effect of a rerun",
        "content_pinning": "transcript_bytes gives each file's length at snapshot "
                           "time; readers must truncate at it, because these files "
                           "are appended to while the study runs. rollup_note_bytes "
                           "does the same for sessions/*.md, keyed by session_id "
                           "rather than path, because the rollup drain can rename "
                           "the file in place; a session_id whose rollup path "
                           "changes after the snapshot has no valid pin and "
                           "readers must treat it as missing rather than guess.",
        "n_transcripts": len(transcripts),
        "n_rollups": len(rollups),
        "n_rollups_with_session_id": len(rollups),
        "n_paired": len(paired),
        "paired_session_ids": paired,
        "transcripts": transcripts,
        "transcript_bytes": sizes,
        "total_transcript_bytes": sum(sizes.values()),
        "rollup_files": sorted(rollups.values()),
        "rollup_note_paths": rollup_paths,
        "rollup_note_bytes": rollup_sizes,
    }


def read_pinned(path: str, manifest: dict | None = None) -> str:
    """Read a transcript as it stood at snapshot time.

    Truncates at the pinned byte length so a file that has grown since the
    snapshot yields the same text it did then. Paths absent from the manifest are
    read whole and are the caller's problem: a file not in the manifest is not in
    the corpus.

    The cut is then pulled back to the last newline, because these files are
    JSONL and a byte offset lands mid-record about as often as not. Returning a
    half-record would make every caller's `json.loads` raise on the final line,
    which is the failure that cost this project 119 records once already. A
    snapshot taken mid-write therefore drops that record rather than corrupting
    it; the loss is one line per actively-growing file.

    If the manifest carries an ``archive_dir`` (only RQ4's own snapshot does;
    the shared corpus_snapshot.json never sets this key) and that directory
    holds an archived copy of this path's content, that copy is returned
    instead of re-reading the live file. The archive was already truncated to
    the pin at freeze time, so no further truncation is needed, and it is
    immune to the live file being reclaimed or rewritten after the fact. Every
    caller whose manifest has no ``archive_dir`` -- i.e. every RQ but RQ4 --
    falls straight through to the live-read path below, unchanged.
    """
    m = manifest if manifest is not None else load_manifest()
    archive_dir = m.get("archive_dir")
    if archive_dir:
        archived = Path(archive_dir) / f"{Path(path).stem}_transcript.txt"
        if archived.exists():
            return archived.read_text(errors="replace")
    limit = m.get("transcript_bytes", {}).get(path)
    raw = Path(path).read_bytes()
    if limit is not None and len(raw) > limit:
        raw = raw[:limit]
        cut = raw.rfind(b"\n")
        # No newline in the truncated buffer means the limit landed inside the
        # first record. Drop it rather than return a half-written line.
        raw = raw[:cut + 1] if cut != -1 else b""
    return raw.decode(errors="replace")


def read_pinned_rollup(session_id: str, manifest: dict | None = None) -> str | None:
    """Read a rollup note as it stood at snapshot time, keyed by session_id.

    Mirrors read_pinned above, but keyed by session_id rather than path: a
    rollup note can be renamed by the drain between snapshot and read time,
    where a transcript's path never changes underneath it. So this checks the
    pinned path is still the path on disk (via rollup_note_paths) before
    trusting the pinned byte length at all -- a renamed note has no valid pin
    and this returns None rather than truncating the wrong file's bytes to a
    length that was never its own. Callers must treat None as "drop this
    session from the corpus", the same as load_session already does for a
    missing transcript, rather than silently falling back to live content.

    Same ``archive_dir`` preference as ``read_pinned`` above: if the manifest
    sets it and an archived copy of this session's rollup exists there, that
    copy wins and the rename/rewrite checks below never run. Absent that key
    (every manifest but RQ4's), behaviour is exactly what it was before.
    """
    m = manifest if manifest is not None else load_manifest()
    archive_dir = m.get("archive_dir")
    if archive_dir:
        archived = Path(archive_dir) / f"{session_id}_rollup.md"
        if archived.exists():
            return archived.read_text(errors="replace")
    pinned_path = m.get("rollup_note_paths", {}).get(session_id)
    limit = m.get("rollup_note_bytes", {}).get(session_id)
    if pinned_path is None or limit is None:
        return None
    if not Path(pinned_path).exists():
        return None  # renamed or deleted since snapshot; no valid pin
    raw = Path(pinned_path).read_bytes()
    if len(raw) < limit:
        # shrank below its pinned length: it was rewritten, not just grown.
        # truncating cannot recover the snapshot-time content, so refuse
        # rather than return a shorter text that silently under-represents it.
        return None
    if len(raw) > limit:
        raw = raw[:limit]
    return raw.decode(errors="replace")


_CACHE: dict | None = None


def load_manifest() -> dict:
    """Read the frozen manifest. Importable by the RQ scripts."""
    global _CACHE
    if _CACHE is None:
        if not OUT.exists():
            raise SystemExit(f"no corpus snapshot at {OUT}; run freeze_corpus.py first")
        _CACHE = json.loads(OUT.read_text())
    return _CACHE


def main() -> None:
    m = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(m, indent=2))
    print(f"snapshot_utc  {m['snapshot_utc']}")
    print(f"transcripts   {m['n_transcripts']}")
    print(f"rollups       {m['n_rollups']}")
    print(f"paired        {m['n_paired']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
