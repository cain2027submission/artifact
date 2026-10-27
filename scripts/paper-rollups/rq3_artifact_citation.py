#!/usr/bin/env python3
"""Count schema generations and artifact citation across the frozen rollup set.

Three different figures for "how often does a rollup name a file" were in
circulation, and all three were defensible under a different denominator, which
is exactly why this is computed once, here, over the frozen manifest:

  - 0.6%  -- a parser that matched only the `## Files` heading, which is the
             minority schema generation. A parser artifact, not a finding.
  - 44.5% -- pooled path *instances* recovered, transcript paths vs rollup
             paths. Some sessions scored above 100% under it, which is
             impossible for a recall and means the transcript-side extractor
             misses paths. Treated as unreliable and not reported.
  - 49.3% -- share of rollups citing at least one path, over all notes in
             sessions/ including hand-written ones with no session_id.

The last is the honest one but its denominator was wrong: it counted every note
in the folder, and the folder holds notes the pipeline did not write. This script
reports it over both the frozen paired set (the population every other RQ uses)
and the full pipeline-written set, and states which is which.

Run with the indexer venv interpreter:
    python3 rq3_artifact_citation.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from freeze_corpus import ROLLUPS, load_manifest, parse_session_id

OUT = Path(__file__).resolve().parents[2] / "data" / "paper-rollups" / "rq3_artifact_citation.json"

# A path-like token: at least one slash and a plausible filename, or a bare
# filename with a known code/config extension. Deliberately conservative --
# overcounting here would inflate the citation rate, which is the number the
# paper leans on.
PATH_RE = re.compile(
    r"(?:(?<![\w/])(?:[~.]?/)?(?:[\w.\-]+/){1,}[\w.\-]+"
    r"|(?<![\w/])[\w.\-]+\.(?:py|js|ts|tsx|jsx|md|json|ya?ml|sh|toml|tex|"
    r"plist|sql|txt|csv|html|css|rs|go|java|rb|c|h|cpp)(?![\w]))"
)

# Bare directory-ish false positives that PATH_RE will otherwise pick up from
# prose ("and/or", dates like 2026/09). Checked against the whole match.
NOISE_RE = re.compile(r"^(?:and/or|n/a|[\d/]+|https?://.*)$", re.I)

SCHEMA_PATTERNS = {
    "heading_files": re.compile(r"^##\s*Files\b", re.M | re.I),
    "label_files": re.compile(r"^\s*\*\*files edited:?\*\*", re.M | re.I),
    "heading_intent": re.compile(r"^##\s*Intent\b", re.M | re.I),
    "label_intent": re.compile(r"^\s*\*\*(?:intent|set out to):?\*\*", re.M | re.I),
    "label_dead_ends": re.compile(r"^\s*\*\*thinking and dead ends:?\*\*", re.M | re.I),
    "heading_dead_ends": re.compile(r"^##\s*Thinking\b", re.M | re.I),
}

# Two sections of a rollup are machine-appended boilerplate rather than summary,
# and counting paths in them is what produced an impossible 100% citation rate on
# the first run of this script:
#   - the trace backlink section always contains the literal trace path, so every
#     note "cites a path" whether or not it names anything it touched,
#   - the `## Recaps` tail is written by a separate mid-session mechanism, not by
#     the rollup summariser, so it is not evidence about what summarisation
#     retains. It is measured separately below because its existence matters.
TRACE_SECTION_RE = re.compile(r"^##\s*the full trace is at.*?(?=^##\s|\Z)",
                              re.M | re.S | re.I)
RECAPS_SECTION_RE = re.compile(r"^##\s*Recaps\b.*?(?=\Z)", re.M | re.S | re.I)


def split_note(text: str) -> tuple[str, str, str]:
    """Return (frontmatter, summary_body, recaps_body).

    Frontmatter is excluded from path counting: it carries `project_dir` and a
    `touched:` list, which are structured metadata the pipeline emits
    mechanically, not artifacts the summary chose to name.
    """
    fm = ""
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            fm = text[3:end]
            body = text[end + 4:]
    recaps = ""
    m = RECAPS_SECTION_RE.search(body)
    if m:
        recaps = m.group(0)
        body = body[:m.start()]
    body = TRACE_SECTION_RE.sub("", body)
    return fm, body, recaps


def paths_in(text: str) -> list[str]:
    out = []
    for m in PATH_RE.finditer(text):
        s = m.group(0)
        if NOISE_RE.match(s):
            continue
        out.append(s)
    return out


def describe(notes: list[tuple[str, str]], label: str) -> dict:
    """notes: list of (filename, text)."""
    n = len(notes)
    cites = 0
    mentions = 0
    distinct: set[str] = set()
    per_note_counts = []
    schema = Counter()
    has_recaps = 0
    for _, text in notes:
        _, body, recaps = split_note(text)
        if recaps.strip():
            has_recaps += 1

        ps = paths_in(body)
        if ps:
            cites += 1
        mentions += len(ps)
        distinct |= set(ps)
        per_note_counts.append(len(ps))

        # Which schema generation wrote this note. A note may carry a files
        # section under either convention or neither; "neither" is itself a
        # result, because a rollup with no files section cannot cite artifacts
        # structurally even when it mentions one in prose.
        has_h = bool(SCHEMA_PATTERNS["heading_files"].search(body))
        has_l = bool(SCHEMA_PATTERNS["label_files"].search(body))
        schema["files_heading" if has_h else
               "files_label" if has_l else "files_absent"] += 1
        if SCHEMA_PATTERNS["heading_intent"].search(body):
            schema["intent_heading"] += 1
        elif SCHEMA_PATTERNS["label_intent"].search(body):
            schema["intent_label"] += 1
        else:
            schema["intent_absent"] += 1
        if SCHEMA_PATTERNS["label_dead_ends"].search(body) or \
                SCHEMA_PATTERNS["heading_dead_ends"].search(body):
            schema["dead_ends_present"] += 1

    per_note_counts.sort()
    median = (per_note_counts[n // 2] if n % 2 else
              (per_note_counts[n // 2 - 1] + per_note_counts[n // 2]) / 2) if n else 0
    return {
        "population": label,
        "n_notes": n,
        "notes_citing_at_least_one_path": cites,
        "pct_citing": round(100.0 * cites / n, 1) if n else 0.0,
        "total_path_mentions": mentions,
        "distinct_paths": len(distinct),
        "path_mentions_per_note": {
            "median": median,
            "max": per_note_counts[-1] if per_note_counts else 0,
        },
        "schema": dict(schema),
        # A mid-session recap tail means the note carries text written while the
        # session was live, not only the retrospective summary. Any claim about
        # what "a rollup" preserves has to say whether this was included.
        "notes_with_recaps_tail": has_recaps,
        "pct_with_recaps_tail": round(100.0 * has_recaps / n, 1) if n else 0.0,
        "byte_shares": byte_shares(notes),
    }


def byte_shares(notes: list[tuple[str, str]]) -> dict:
    """How much of a rollup is actually the summary.

    This matters for RQ2, not just RQ3. Every compression ratio in this paper
    divides the transcript by the *whole note*, and the whole note includes
    frontmatter telemetry, a fixed trace-backlink paragraph, and in half the
    corpus a `## Recaps` tail written mid-session by a different mechanism. To
    the extent the note is not summary, the compression ratio understates how
    much the summariser compressed -- so this is a conservative bias, but it has
    to be stated with a number rather than waved at.
    """
    tot = fm_c = sum_c = trace_c = recap_c = 0
    fixed = []  # per-note frontmatter + trace boilerplate, the floor a note cannot go below
    for _, text in notes:
        fm, body, recaps = split_note(text)
        # split_note already removed the trace section from body; recover its
        # size by measuring it on the pre-strip text.
        pre = text
        if pre.startswith("---"):
            e = pre.find("\n---", 3)
            if e != -1:
                pre = pre[e + 4:]
        m = RECAPS_SECTION_RE.search(pre)
        pre_no_recaps = pre[:m.start()] if m else pre
        trace = "".join(TRACE_SECTION_RE.findall(pre_no_recaps))

        tot += len(text)
        fm_c += len(fm)
        sum_c += len(body)
        trace_c += len(trace)
        recap_c += len(recaps)
        fixed.append(len(fm) + len(trace))
    if not tot:
        return {}
    fixed.sort()
    return {
        "total_chars": tot,
        "summary_pct": round(100.0 * sum_c / tot, 1),
        "frontmatter_pct": round(100.0 * fm_c / tot, 1),
        "trace_backlink_pct": round(100.0 * trace_c / tot, 1),
        "recaps_tail_pct": round(100.0 * recap_c / tot, 1),
        # The floor. A rollup of a two-turn session still pays this, which is the
        # mechanism behind the sessions whose rollup is larger than the session.
        "fixed_overhead_chars": {
            "median": fixed[len(fixed) // 2],
            "p10": fixed[int(0.10 * (len(fixed) - 1))],
            "p90": fixed[int(0.90 * (len(fixed) - 1))],
        },
        # If the ratio were computed against summary text alone -- the
        # summariser's intrinsic performance rather than the deployed note's
        # size -- every compression figure would rise by this factor.
        "summary_only_ratio_multiplier": round(tot / sum_c, 2) if sum_c else None,
    }


def main() -> None:
    m = load_manifest()
    paired = set(m["paired_session_ids"])
    manifest_files = set(m["rollup_files"])

    all_pipeline: list[tuple[str, str]] = []
    no_session_id = 0
    by_sid: dict[str, list[str]] = {}
    # Paired stats are per session, not per note. Two sessions have two notes
    # each, so appending per file reported 302 units for the manifest's 300 and
    # put this file out of step with every other RQ. Their notes are concatenated
    # -- a session's artifact is all of its notes -- matching part1_structural_loss.
    paired_text: dict[str, list[str]] = {}
    for f in sorted(ROLLUPS.glob("*.md")):
        text = f.read_text(errors="replace")
        sid = parse_session_id(text)
        if not sid:
            no_session_id += 1
            continue
        # Only include files that are in the frozen manifest
        if f.name not in manifest_files:
            continue
        all_pipeline.append((f.name, text))
        by_sid.setdefault(sid, []).append(f.name)
        if sid in paired:
            paired_text.setdefault(sid, []).append(text)

    paired_notes: list[tuple[str, str]] = [
        (by_sid[sid][0], "\n".join(texts)) for sid, texts in sorted(paired_text.items())
    ]

    # More than one rollup can exist for the same session. This is why RQ2
    # iterating notes reports 287 pairs where the manifest holds 285 distinct
    # paired session ids: the duplicated sessions are counted twice in any
    # note-indexed statistic. Reported so the two-pair gap is documented rather
    # than rediscovered.
    duplicates = {sid: names for sid, names in by_sid.items() if len(names) > 1}

    summary = {
        "corpus_snapshot_utc": m["snapshot_utc"],
        "note": "pct_citing is the share of rollups naming at least one path in "
                "the summary body, excluding frontmatter, the trace-backlink "
                "section and the mid-session recaps tail. It is not a recall: "
                "the transcript-side path extractor is known to miss paths "
                "(some sessions scored above 100%) and cannot serve as a "
                "denominator",
        "notes_in_folder_without_session_id": no_session_id,
        "duplicate_rollups": {
            "n_sessions_with_more_than_one_rollup": len(duplicates),
            "n_excess_notes": sum(len(v) - 1 for v in duplicates.values()),
            "examples": dict(list(duplicates.items())[:5]),
        },
        "frozen_paired": describe(paired_notes, "frozen paired set (every other RQ)"),
        "all_pipeline_written": describe(all_pipeline, "all notes carrying a session_id"),
    }
    OUT.write_text(json.dumps(summary, indent=2))

    for key in ("frozen_paired", "all_pipeline_written"):
        d = summary[key]
        print(f"{d['population']}: n={d['n_notes']}")
        print(f"  cite >=1 path : {d['notes_citing_at_least_one_path']} "
              f"({d['pct_citing']}%)")
        print(f"  mentions      : {d['total_path_mentions']} total, "
              f"{d['distinct_paths']} distinct, median {d['path_mentions_per_note']['median']}/note")
        print(f"  schema        : {json.dumps(d['schema'])}")
        print(f"  recaps tail   : {d['notes_with_recaps_tail']} "
              f"({d['pct_with_recaps_tail']}%)")
    dup = summary["duplicate_rollups"]
    print(f"\nsessions with >1 rollup: {dup['n_sessions_with_more_than_one_rollup']} "
          f"({dup['n_excess_notes']} excess notes)")
    print(f"notes in sessions/ with no session_id (not pipeline-written): "
          f"{no_session_id}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
