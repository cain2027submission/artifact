#!/usr/bin/env python3
"""Shared loaders for the later-window experiment (rq3_late_*.py).

Everything a session needs is resolved from the pinned corpus
(data/paper-rollups/corpus_snapshot.json):
  * the transcript, truncated at its pinned byte length, because the files are
    appended to while the study runs;
  * the rollup note as it stood at commit 27c1de0, the commit the snapshot's
    rollup_note_paths were backfilled from.

Windows are character positions in part2_question_retention.transcript_to_text(),
the exact view the published question generator read its first 30,000
characters from, so EARLY / MID / LATE are three slices of one string.
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from part2_question_retention import transcript_to_text  # noqa: E402
from rq3_raw_transcript_arm import (  # noqa: E402
    DATA, QUESTIONS, SUMMARY, SLUG, load_records, record_id,
)

REPO = HERE.parents[1]
SNAPSHOT = DATA / "corpus_snapshot.json"
PIN_COMMIT = "27c1de0"
OUTDIR = DATA / "late-window"
WINDOW_CHARS = 30_000
# MID = [0.30, 0.60) and LATE = [0.60, 1.00] of len(transcript_to_text).
WINDOWS = {"mid": (0.30, 0.60), "late": (0.60, 1.00)}
CLASSES = ["decision", "dead_end", "outcome", "artifact_detail", "sequence"]


def snapshot() -> dict:
    return json.loads(SNAPSHOT.read_text())


def pinned_raw(path: str, nbytes: int) -> str:
    with open(path, "rb") as f:
        data = f.read(nbytes)
    return data.decode("utf-8", errors="replace")


def parse_entries(raw: str) -> List[dict]:
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a partial last line at the pinned byte length
    return out


CONTINUATION_OPENER = "This session is being continued from a previous conversation"


def _is_harness_entry(e: dict) -> bool:
    """A compaction summary (isCompactSummary, or the continuation opener) or a
    harness-injected meta message (isMeta: skill bodies, command expansions).
    Neither is activity at its position in the session: a compaction summary
    restates EARLIER work, so a question drawn from it would be an early-session
    question placed in a late window."""
    if e.get("isMeta") or e.get("isCompactSummary"):
        return True
    msg = e.get("message")
    if isinstance(msg, dict) and msg.get("role") == "user":
        c = msg.get("content")
        first = c if isinstance(c, str) else next(
            (b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"), "") \
            if isinstance(c, list) else ""
        if isinstance(first, str) and first.lstrip().startswith(CONTINUATION_OPENER):
            return True
    return False


def window_entries(entries: List[dict]) -> List[dict]:
    """Entries the MID/LATE window positions are measured over."""
    return [e for e in entries if not _is_harness_entry(e)]


def generator_text(raw: str) -> str:
    """The generator's text view (transcript_to_text) over window_entries()."""
    return transcript_to_text(window_entries(parse_entries(raw)))


def pinned_rollup(note_path: str) -> Optional[str]:
    rel = "sessions/" + Path(note_path).name
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{PIN_COMMIT}:{rel}"],
                       capture_output=True)
    if r.returncode != 0:
        return None
    return r.stdout.decode("utf-8", errors="replace")


def is_delegating(entries: List[dict]) -> bool:
    for e in entries:
        msg = e.get("message")
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        c = msg.get("content")
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "tool_use" \
                        and b.get("name") in ("Task", "Agent"):
                    return True
    return False


def window_offset(session_uuid: str, window: str, text_len: int) -> Optional[int]:
    """Start of the 30,000-char slice the generator sees for this window, or
    None when the window holds fewer than 30,000 characters. The slice is placed
    uniformly at random inside the window, seeded by session and window, so the
    choice is reproducible and not tuned."""
    lo_f, hi_f = WINDOWS[window]
    lo, hi = int(lo_f * text_len), int(hi_f * text_len)
    if hi - lo < WINDOW_CHARS:
        return None
    seed = int(hashlib.sha256(f"{session_uuid}|{window}".encode()).hexdigest()[:12], 16)
    return random.Random(seed).randint(lo, hi - WINDOW_CHARS)


def existing_sessions() -> Dict[str, dict]:
    """prefix -> {slug, graded(bool)} for the 68 sessions of rq3_questions.jsonl."""
    graded = set(json.loads(SUMMARY.read_text())["graded_record_ids"])
    out: Dict[str, dict] = {}
    for r in load_records(QUESTIONS):
        m = SLUG.match(r.get("session_id", ""))
        if not m:
            continue
        d = out.setdefault(m.group(1), {"slug": r["session_id"], "graded": False,
                                         "n_questions": 0})
        d["n_questions"] += 1
        if record_id(r) in graded:
            d["graded"] = True
    return out


def entry_spans(entries: List[dict]):
    """(start, end) of each entry's text inside generator_text(); the join is
    reproduced exactly (asserted by the caller against the full string)."""
    spans, pos = [], 0
    for e in entries:
        t = transcript_to_text([e])
        if not t:
            continue
        if spans:
            pos += 2
        spans.append((pos, pos + len(t)))
        pos += len(t)
    return spans, pos


def window_dominance(spans, offset: int) -> float:
    """Largest fraction of the 30,000-char slice covered by ONE transcript
    entry. A slice that is mostly one pasted or injected blob (a schema, a
    CLAUDE.md) is not session activity, so DESIGN.md excludes slices above 0.5."""
    a, b = offset, offset + WINDOW_CHARS
    return max((max(0, min(b, y) - max(a, x)) for x, y in spans), default=0) / WINDOW_CHARS


def qid(session_uuid: str, window: str, cls: str, question: str) -> str:
    return hashlib.sha256(f"{session_uuid}|{window}|{cls}|{question}".encode()).hexdigest()[:16]


def read_jsonl(p: Path) -> List[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
