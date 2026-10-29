#!/usr/bin/env python3
"""RQ3 path recall, recomputed. Replaces the 44.5% figure the paper withdrew.

The withdrawn number came from `part1_structural_loss.py`, which computed

    files_recall = len(rollup_files) / len(transcript_files)

-- a ratio of two set *cardinalities* with no intersection in it. 35 of 300
sessions scored above 100% under it (up to 400%), and on the four worst the true
intersection of the two sets was empty, so their real recall was 0%. The paper
attributed the impossibility to the transcript-side extractor missing paths.
That is true and it is fixed here, but it was the second cause, not the first:
the metric was never measuring overlap. See `path_extract.py` for all four
defects and the single shared normalisation that closes them.

What is reported. Path recall is
|{transcript paths the rollup also names}| / |{transcript paths}|, with the
numerator counted over transcript paths, which makes recall <= 1.0 by
construction rather than by luck. Three denominators are reported because the
denominator of a recall is a choice and has to be stated:

  acted     -- paths in tool-call arguments. The primary figure, and the
               repaired version of what the old code was reaching for.
  authored  -- also paths in assistant prose, thinking, user text and file
               content written by the session.
  all       -- also paths in tool *results*. One `ls` of a large directory adds
               hundreds of paths nobody named, so this is a sensitivity arm.

Run with the indexer venv interpreter:
    python3 rq3_path_recall.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import path_extract as px
from freeze_corpus import ROLLUPS, load_manifest, parse_session_id, read_pinned

OUT = (Path(__file__).resolve().parents[2] / "data" / "paper-rollups"
       / "rq3_path_recall_corrected.json")

# The figure being replaced, from data/paper-rollups/FINAL_REPORT.md. The
# committed rq3_structural_loss.json holds 46.3% (1,313/2,837) from a later run
# of the same broken statistic; both are the same defect and neither is a recall.
OLD_POOLED = 0.445
OLD_POOLED_FRACTION = "1,256/2,824"

BOOTSTRAP_N = 2000
SEED = 20260929


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_pooled(per_session: list[tuple[int, int]], reps: int = BOOTSTRAP_N
                     ) -> tuple[float, float]:
    """Percentile CI for the pooled ratio, resampling *sessions*.

    Wilson treats the 2,800-odd paths as independent Bernoulli draws, which they
    are not: paths cluster hard within a session, so Wilson is too narrow for the
    pooled figure. Both are reported -- Wilson for comparability with the other
    rates in the paper, the session bootstrap because it is the honest one here.
    """
    if not per_session:
        return (0.0, 0.0)
    rng = random.Random(SEED)
    n = len(per_session)
    vals = []
    for _ in range(reps):
        num = den = 0
        for _ in range(n):
            a, b = per_session[rng.randrange(n)]
            num += a
            den += b
        vals.append(num / den if den else 0.0)
    vals.sort()
    return (vals[int(0.025 * (reps - 1))], vals[int(0.975 * (reps - 1))])


def load_pairs() -> list[tuple[str, list[Path], Path]]:
    """(session_id, rollup notes, transcript) over the frozen paired set.

    Grouped by session rather than by note, because two sessions in this corpus
    have two rollup notes each; a session's generated artifact is all of its
    notes, so they are concatenated. This matches part1_structural_loss.py and
    rq3_artifact_citation.py, and is why 300 sessions come from 302 notes.
    """
    m = load_manifest()
    paired = set(m["paired_session_ids"])
    transcript_by_sid = {Path(p).stem: Path(p) for p in m["transcripts"]}

    by_session: dict[str, list[Path]] = {}
    for note in sorted(ROLLUPS.glob("*.md")):
        sid = parse_session_id(note.read_text(errors="replace"))
        if sid and sid in paired and sid in transcript_by_sid:
            by_session.setdefault(sid, []).append(note)
    return [(sid, sorted(notes), transcript_by_sid[sid])
            for sid, notes in sorted(by_session.items())]


def old_recall(rollup_text: str, entries: list[dict]) -> tuple[int, int, float]:
    """The withdrawn statistic, recomputed so old and new sit in one table."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import part1_structural_loss as p1
    rf, _ = p1.extract_files_from_rollup(rollup_text)
    tf = p1.extract_files_from_transcript(entries)
    return len(rf), len(tf), (len(rf) / len(tf) if tf else 0.0)


def main() -> None:
    pairs = load_pairs()
    # A small resumable sample first, never the whole corpus: `--limit N` exists
    # so this can be proved on 10 sessions before it is trusted on 300. With a
    # limit set, nothing is written -- a partial run must not overwrite the
    # committed figure.
    limit = 0
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
        pairs = pairs[:limit]
    print(f"frozen paired set: {len(pairs)} sessions"
          + (f" (LIMITED to {limit}, will not write)" if limit else ""))

    rows = []
    # A transcript can be pruned after the snapshot was frozen -- the raw
    # transcripts are cleaned up on a 180-day window and the rollup note outlives
    # the .jsonl it was written from. One of the 300 (3a0d132f) has gone since
    # 2026-09-27, so it is recorded here and dropped rather than crashing the run
    # or being silently absorbed into the denominator as zero paths.
    unreadable: list[str] = []
    for i, (sid, notes, tpath) in enumerate(pairs, 1):
        if i % 50 == 0:
            print(f"  {i}/{len(pairs)}")
        if not tpath.exists():
            unreadable.append(sid)
            continue
        rollup_text = "\n".join(p.read_text(errors="replace") for p in notes)
        entries = []
        for line in read_pinned(str(tpath)).splitlines():
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        r_paths, root = px.rollup_paths(rollup_text)
        r_paths_recaps, _ = px.rollup_paths(rollup_text, include_recaps=True)

        row: dict = {
            "session_id": sid,
            "rollup_notes": [p.name for p in notes],
            "project_dir": root,
            "n_rollup_paths": len(r_paths),
        }
        for surface in px.SURFACES:
            t_paths = px.transcript_paths(entries, surface=surface)
            num, den, rec = px.path_recall(r_paths, t_paths)
            row[surface] = {"numerator": num, "denominator": den, "recall": rec}
        # recaps tail included, primary surface only
        num, den, rec = px.path_recall(r_paths_recaps,
                                       px.transcript_paths(entries, px.SURFACE_ACTED))
        row["acted_with_recaps"] = {"numerator": num, "denominator": den, "recall": rec}

        o_num, o_den, o_rec = old_recall(rollup_text, entries)
        row["old_metric"] = {"rollup_set_size": o_num, "transcript_set_size": o_den,
                             "value": o_rec}
        rows.append(row)

    # --- pooled figures -----------------------------------------------------
    arms = {}
    for surface in list(px.SURFACES) + ["acted_with_recaps"]:
        per = [(r[surface]["numerator"], r[surface]["denominator"]) for r in rows]
        num = sum(a for a, _ in per)
        den = sum(b for _, b in per)
        pooled = num / den if den else 0.0
        wl, wh = wilson(num, den)
        bl, bh = bootstrap_pooled(per)
        scored = [r[surface]["recall"] for r in rows if r[surface]["denominator"] > 0]
        scored.sort()
        arms[surface] = {
            "numerator": num,
            "denominator": den,
            "pooled_recall": pooled,
            "wilson95": [wl, wh],
            "bootstrap95_by_session": [bl, bh],
            "n_sessions": len(rows),
            "n_sessions_with_paths": len(scored),
            "n_sessions_above_1": sum(
                1 for r in rows if r[surface]["recall"] > 1.0),
            "per_session_recall": {
                "mean": (sum(scored) / len(scored)) if scored else 0.0,
                "median": (scored[len(scored) // 2] if scored else 0.0),
                "p10": (scored[int(0.10 * (len(scored) - 1))] if scored else 0.0),
                "p90": (scored[int(0.90 * (len(scored) - 1))] if scored else 0.0),
                "max": (scored[-1] if scored else 0.0),
            },
        }

    old_per = [(r["old_metric"]["rollup_set_size"],
                r["old_metric"]["transcript_set_size"]) for r in rows]
    old_num = sum(a for a, _ in old_per)
    old_den = sum(b for _, b in old_per)
    old_above = [r for r in rows if r["old_metric"]["value"] > 1.0]

    primary = px.SURFACE_ACTED
    changed = sum(1 for r in rows
                  if abs(r[primary]["recall"] - r["old_metric"]["value"]) > 0.10)

    out = {
        "corpus_snapshot_utc": load_manifest()["snapshot_utc"],
        "n_sessions_in_manifest_paired_set": len(pairs),
        "n_sessions_measured": len(rows),
        "sessions_dropped_transcript_pruned": unreadable,
        "metric": "path recall = |transcript paths the rollup also names| / "
                  "|transcript paths|. Numerator is counted over transcript "
                  "paths, so recall <= 1.0 by construction.",
        "primary_arm": primary,
        "withdrawn_figure": {
            "value": OLD_POOLED,
            "fraction_as_reported": OLD_POOLED_FRACTION,
            "source": "data/paper-rollups/FINAL_REPORT.md; the committed "
                      "rq3_structural_loss.json holds 46.3% (1313/2837) from a "
                      "later run of the same statistic",
            "defect": "len(rollup_files)/len(transcript_files) -- a ratio of two "
                      "set sizes, not an intersection, so it was never a recall",
            "recomputed_here": {
                "rollup_set_total": old_num,
                "transcript_set_total": old_den,
                "pooled_value": (old_num / old_den if old_den else 0.0),
                "n_sessions_above_100pct": len(old_above),
                "worst_sessions": [
                    {"session_id": r["session_id"],
                     "value": r["old_metric"]["value"],
                     "rollup_set_size": r["old_metric"]["rollup_set_size"],
                     "transcript_set_size": r["old_metric"]["transcript_set_size"],
                     "true_recall_acted": r[primary]["recall"]}
                    for r in sorted(old_above,
                                    key=lambda x: -x["old_metric"]["value"])[:10]
                ],
            },
        },
        "sessions_changed_by_more_than_10_points": changed,
        "arms": arms,
        "per_session": rows,
    }
    if not limit:
        OUT.write_text(json.dumps(out, indent=2))

    print(f"\nmeasured {len(rows)} of {len(pairs)} manifest sessions"
          + (f"; dropped (transcript pruned): {unreadable}" if unreadable else ""))
    print(f"withdrawn metric recomputed: {old_num}/{old_den} = "
          f"{100 * old_num / old_den:.1f}%  ({len(old_above)} sessions > 100%)")
    for surface in list(px.SURFACES) + ["acted_with_recaps"]:
        a = arms[surface]
        print(f"\n{surface}:")
        print(f"  pooled     {a['numerator']}/{a['denominator']} = "
              f"{100 * a['pooled_recall']:.1f}%")
        print(f"  wilson95   [{100 * a['wilson95'][0]:.1f}, "
              f"{100 * a['wilson95'][1]:.1f}]")
        print(f"  boot95     [{100 * a['bootstrap95_by_session'][0]:.1f}, "
              f"{100 * a['bootstrap95_by_session'][1]:.1f}]  (session resample)")
        print(f"  per-session median {100 * a['per_session_recall']['median']:.1f}% "
              f"mean {100 * a['per_session_recall']['mean']:.1f}% "
              f"max {100 * a['per_session_recall']['max']:.1f}%")
        print(f"  sessions > 100%: {a['n_sessions_above_1']}")
    print(f"\nsessions changed by >10 points ({primary} vs old): {changed}")
    print(f"wrote {OUT}" if not limit else "(limited run; nothing written)")


if __name__ == "__main__":
    main()
