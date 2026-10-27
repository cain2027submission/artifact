#!/usr/bin/env python3
"""One definition of "this session delegated", imported by every script that needs it.

Why this module exists. `rq1_signal_composition_corrected.py` set its
`has_subagents` flag from `rec.get("isSidechain")`. That field is written only
inside a subagent transcript, so it is true for every `agent-*.jsonl` file and
false for every parent file. The flag therefore duplicated `is_subagent` and was
0 for all 1,477 parent sessions, which made `by_subagent_use` in
`rq1_corrected_summary.json` degenerate: with_subagents n=0, without n=249. A
reviewer reading that block would have concluded no session in the corpus ever
delegated, when 158 of them did.

The actual on-disk signal is structural, not a field: a delegated agent gets its
own transcript at

    <project>/<parent-session-id>/subagents/agent-<id>.jsonl

so a parent session delegated exactly when that directory holds at least one
`agent-*.jsonl`. Nothing in the parent transcript records it. The shape is the
definition; the directory name alone is not, which is why `agent_files` matches
the glob inside the `subagents/` directory rather than accepting any path that
happens to contain the word.

Two callers need this (RQ1's composition split and RQ3's delegation reweighting)
and they must not drift, so the rule lives here and only here.

Determinism. `delegating_ids_from_manifest` is the form every paper number
should use: it classifies from the frozen corpus manifest's file list, so the
answer cannot move when the retention sweep prunes an old subagent file. The
live-filesystem form `is_delegating` exists for sessions outside the manifest
(the 2026-09-21 transcript backup holds sessions that predate the live retention
window) and for testing the detector, and callers that use it must say so in
their output.
"""

from __future__ import annotations

from pathlib import Path

# The directory a delegated agent's transcript is written into, and the filename
# shape it is written under. Both halves are load-bearing: a stray `agent-*.jsonl`
# beside the parent transcript is not a delegation, and neither is an empty or
# differently-named directory.
SUBAGENT_DIR = "subagents"
AGENT_GLOB = "agent-*.jsonl"


def subagent_dir(parent_transcript: Path | str) -> Path:
    """The directory delegated-agent transcripts for this parent would live in."""
    p = Path(parent_transcript)
    return p.with_suffix("") / SUBAGENT_DIR


def agent_files(parent_transcript: Path | str) -> list[Path]:
    """Delegated-agent transcripts on disk for this parent transcript, sorted.

    Empty if the parent transcript itself is not a file. An orphaned
    `subagents/` directory whose parent has been pruned is not a delegating
    session -- the session is gone -- and returning files for it would make this
    disagree with `delegating_ids_from_manifest`, which requires the parent
    transcript to be in the corpus.
    """
    if not Path(parent_transcript).is_file():
        return []
    p = Path(parent_transcript)
    found: list[Path] = []
    # The obvious place: beside the parent transcript, under its own directory.
    d = subagent_dir(p)
    if d.is_dir():
        found += [f for f in d.glob(AGENT_GLOB) if f.is_file()]
    # And the non-obvious one. A project directory is named after the session's
    # working directory, so a session that changes cwd mid-run writes its
    # subagent transcripts under a DIFFERENT project directory than its own
    # transcript. One of the 158 delegating sessions in the frozen corpus does
    # exactly this (it enters a git worktree), and a search that only looks
    # beside the parent transcript scores it as solo.
    projects_root = p.parent.parent
    if projects_root.is_dir():
        found += [f for f in projects_root.glob(f"*/{p.stem}/{SUBAGENT_DIR}/{AGENT_GLOB}")
                  if f.is_file()]
    return sorted(set(found))


def is_delegating(parent_transcript: Path | str) -> bool:
    """True if this parent session spawned at least one subagent, read from disk.

    Prefer `delegating_ids_from_manifest` for any number that goes in the paper;
    this reads the live filesystem and so is not pinned to the corpus snapshot.
    """
    return bool(agent_files(parent_transcript))


def is_subagent_path(path: Path | str) -> bool:
    """True if this path is itself a delegated-agent transcript, not a parent."""
    return SUBAGENT_DIR in Path(path).parts


def delegating_ids_from_manifest(transcript_paths) -> set[str]:
    """Parent session ids that delegated, derived from a frozen path list.

    Takes the manifest's `transcripts` list (strings or Paths) and returns the
    set of parent session ids that own at least one `subagents/agent-*.jsonl`
    entry *in that list*. Nothing is read from disk, so this is stable across
    reruns even as the retention sweep deletes old subagent files -- which it
    does: two of the 158 delegating parents in the 2026-09-27 snapshot no longer
    have their subagent directory on disk, and a live glob would silently
    reclassify them as solo.

    The shape is checked, not searched for. An earlier version took the component
    before the first `subagents` part anywhere in the path, and a fixture with an
    `<sid>/nested/subagents/agent-y.jsonl` layout made it invent a delegating
    session called "nested". So the directory must be the immediate parent of the
    agent file, and the name it sits under must be the stem of a parent
    transcript somewhere in the same list -- which is what makes a session id a
    session id rather than just a directory name.

    That second check deliberately does not require the parent transcript to be a
    SIBLING of the `subagents/` directory, which a stricter first attempt did and
    which silently dropped one of the 158 delegating sessions. Project directories
    are named after the session's working directory, so a session that switches
    cwd mid-run -- entering a git worktree, in the one real case -- has its
    subagent transcripts written under one project directory and its own
    transcript under another. The session id is the join key, not the path.
    """
    paths = [Path(p) for p in transcript_paths]
    parent_stems = {p.stem for p in paths if p.parent.name != SUBAGENT_DIR}
    out: set[str] = set()
    for p in paths:
        if p.parent.name != SUBAGENT_DIR or not p.match(AGENT_GLOB):
            continue
        sid = p.parent.parent.name
        if sid in parent_stems:
            out.add(sid)
    return out


def parent_ids_from_manifest(transcript_paths) -> set[str]:
    """Session ids of the parent (non-subagent) transcripts in a frozen path list."""
    return {Path(p).stem for p in transcript_paths if not is_subagent_path(p)}
