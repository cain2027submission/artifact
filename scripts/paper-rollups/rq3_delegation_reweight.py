#!/usr/bin/env python3
"""Post-stratify RQ3's retention rates on whether the session delegated.

Why this script exists. RQ3 grades 124 questions over 25 sessions, and the
sessions were not drawn uniformly from anything: `part2_streamlined.py` sampled
only transcripts >=100KB, stratified half 100KB-1MB and half >=1MB. Sessions that
delegate to subagents are 6.5x larger at the median than sessions that do not, so
a size-stratified sample is also, silently, a delegation-stratified sample. If a
rollup loses more on a delegating session -- and it does, by 29.5 points of
recoverable rate -- then RQ3's headline is partly a statement about the sample's
composition rather than about rollups.

A prior session measured this with shell heredocs that were never saved. The
numbers were quotable and unreproducible, which is the same defect the corpus
freeze was built to fix, so they could not go in the paper. This script derives
them from the committed graded records and writes one JSON.

Three things it does NOT do, each for a reason:

1. It does not regrade anything. Every grade is read from the majority-vote files
   `rq3_majority_uniform.jsonl` and `rq3_majority_raw.jsonl`, and `recoverable`
   and `wrong-of-attempted` come from `rq3_raw_transcript_arm.rates`, the same
   function the paper's other RQ3 tables are computed with. Reproducing the
   published 56.5% / 13.6% baseline exactly is the correctness gate: the script
   refuses to write if it does not.

2. It does not pick a single reweighted headline. The reweighted estimate is
   whatever the target population says it is, and the choice of target changes
   the SIGN of the correction. Against all 1,477 parent sessions, delegating is
   10.70% of the corpus and reweighting raises recoverable from 56.5% to 68.5%.
   Against the repo's own definition of a substantive session it is 63.45% and
   reweighting LOWERS it to 53.0%. Against the population RQ3 actually sampled
   from -- paired sessions with a transcript >=100KB -- it is 56.03%, the sample's
   own 52.0% is within sampling error of it, and there is nothing to correct.
   Every one of those is in the output, and the one the paper should quote is the
   sampling frame, because that is the population the estimate generalises to.

3. It does not use `isSidechain`. That field is true only inside a subagent
   transcript file, never in the parent that spawned it, which is exactly the bug
   this change also fixes in RQ1. Delegation is read through `delegation.py`.

Run:
    python3 rq3_delegation_reweight.py
    python3 rq3_delegation_reweight.py --limit 8   # smoke
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from delegation import (  # noqa: E402
    agent_files, delegating_ids_from_manifest, parent_ids_from_manifest,
)
from freeze_corpus import load_manifest  # noqa: E402
from rq3_raw_transcript_arm import (  # noqa: E402
    DATA, SLUG, SOURCES, load_records, rates, wilson,
)

OUT = DATA / "rq3_delegation_reweight.json"
RQ1_CSV = DATA / "rq1_corrected_per_session.csv"

# The arm the paper's retention headline is computed over: every question
# re-answered from the full untruncated rollup under one protocol, graded by
# majority vote over five draws.
ROLLUP_ARM = DATA / "rq3_majority_uniform.jsonl"
RAW_ARM = DATA / "rq3_majority_raw.jsonl"

# The published figures this script must reproduce before it is allowed to write.
# From rq3_paired_comparison.json -> uniform_all_124.
PUBLISHED_RECOVERABLE_PCT = 56.5
PUBLISHED_WRONG_OF_ATTEMPTED_PCT = 13.6

# The repo's existing definition of a session worth summarising, lifted verbatim
# from rq1_signal_composition_corrected.main. Not re-chosen here: the point of
# reporting a rate against this denominator is that the denominator is already
# committed to elsewhere in the paper.
SUBSTANTIVE_MIN_TOOL_CALLS = 1
SUBSTANTIVE_MIN_PAYLOAD_CHARS = 5000
SUBSTANTIVE_FILTER = (">=1 tool call and >=5000 payload chars, parent sessions "
                      "only (rq1_signal_composition_corrected.substantive)")

# Bootstrap settings follow rq4_known_item.bootstrap_ci: 1000 resamples, a
# np.random.RandomState seeded at 42, percentile interval at int(0.025*n) and
# int(0.975*n). Resampling is over SESSIONS, not questions, because questions
# from one session share a rollup and a ground truth and are not independent --
# a question-level bootstrap would report an interval several points too narrow.
SEED = 42
N_BOOTSTRAP = 1000

RECOVERED = ("correct", "partially_correct")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def session_prefix(session_id: str) -> str | None:
    """The 8-hex transcript id embedded in a rollup-note slug, or None."""
    m = SLUG.match(session_id or "")
    return m.group(1) if m else None


def classify_sessions(session_ids: list[str], manifest: dict) -> dict:
    """Split RQ3's sessions into delegating / solo, and say what could not be told.

    Manifest first, live disk only as a fallback. The manifest is the frozen
    corpus and so is reproducible; the fallback exists because RQ3 resolves
    transcripts against a 2026-09-21 backup as well as the live projects
    directory, and a session found only there is absent from the manifest.

    `unclassifiable` is emitted rather than folded into `solo` on purpose. A
    delegating session whose subagent files were pruned by the retention sweep
    looks exactly like a session that never delegated, and that error is
    one-directional: it can only move sessions from delegating to solo, i.e. it
    can only understate the delegating stratum. Hiding it inside the solo count
    would make the bias invisible, so a session whose transcript cannot be found
    at all is counted here and excluded from both strata.
    """
    man_delegating = delegating_ids_from_manifest(manifest["transcripts"])
    man_parents = parent_ids_from_manifest(manifest["transcripts"])
    by_prefix_manifest: dict[str, list[str]] = defaultdict(list)
    for sid in man_parents:
        if len(sid) >= 8:
            by_prefix_manifest[sid[:8]].append(sid)

    out = {"delegating": [], "solo": [], "unclassifiable": [],
           "from_manifest": 0, "from_disk_fallback": 0,
           "pruned_since_snapshot": 0, "detail": {}}

    for sid in sorted(session_ids):
        prefix = session_prefix(sid)
        if prefix is None:
            out["unclassifiable"].append(sid)
            out["detail"][sid] = {"reason": "session_id is not a rollup slug"}
            continue

        hits = sorted(by_prefix_manifest.get(prefix, []))
        if hits:
            deleg = any(h in man_delegating for h in hits)
            out["from_manifest"] += 1
            # Recorded, not acted on: whether the evidence still exists on disk.
            # A delegating session whose agent files have since been deleted is
            # still delegating in the frozen corpus, and this is the count that
            # tells a reader how much a live-disk rerun would disagree.
            live = any(agent_files(p) for p in _live_paths(prefix))
            if deleg and not live:
                out["pruned_since_snapshot"] += 1
            out["delegating" if deleg else "solo"].append(sid)
            out["detail"][sid] = {"prefix": prefix, "source": "manifest",
                                  "delegating": deleg,
                                  "agent_files_still_on_disk": live}
            continue

        paths = _live_paths(prefix)
        if not paths:
            out["unclassifiable"].append(sid)
            out["detail"][sid] = {"prefix": prefix,
                                  "reason": "no transcript in the manifest and "
                                            "none on disk; pruned"}
            continue
        deleg = any(agent_files(p) for p in paths)
        out["from_disk_fallback"] += 1
        out["delegating" if deleg else "solo"].append(sid)
        out["detail"][sid] = {"prefix": prefix, "source": "live_disk",
                              "delegating": deleg}
    return out


def _live_paths(prefix: str) -> list[Path]:
    """Parent transcripts on disk whose stem starts with this 8-hex prefix."""
    hits: list[Path] = []
    for root in SOURCES:
        if root.is_dir():
            hits += [p for p in root.glob(f"*/{prefix}*.jsonl") if p.is_file()]
    return sorted(hits)


def corpus_rates(manifest: dict) -> dict:
    """Delegation rate corpus-wide, under every denominator the paper could mean.

    Both of the two the paper needs are here and they differ by a factor of six,
    which is the whole point: 10.70% of all parent sessions delegated, but 63.45%
    of substantive ones did, because all 158 delegating sessions clear the
    substantive bar and 1,228 of the 1,477 parent transcripts are `/clear` stubs
    that do not. A "corpus delegation rate" quoted without its denominator is
    unusable, and reweighting with the wrong one inverts the correction.
    """
    delegating = delegating_ids_from_manifest(manifest["transcripts"])
    parents = parent_ids_from_manifest(manifest["transcripts"])

    if not RQ1_CSV.exists():
        raise SystemExit(f"missing {RQ1_CSV}; run "
                         f"rq1_signal_composition_corrected.py --from-csv first")
    with RQ1_CSV.open(newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["is_subagent"] == "0"]
    scanned = {r["session_id"] for r in rows}
    missing = parents - scanned
    if missing:
        raise SystemExit(
            f"{len(missing)} manifest parent sessions are absent from "
            f"{RQ1_CSV.name}; the two artifacts describe different corpora and "
            f"no rate computed across them would be meaningful")

    # Cross-check against the other artifact rather than trusting it: the CSV's
    # has_subagents column was written from isSidechain before this change, so a
    # stale CSV disagrees with delegation.py on all 158 sessions. Failing here is
    # the whole reason the helper is shared.
    csv_delegating = {r["session_id"] for r in rows if r["has_subagents"] == "1"}
    if csv_delegating != (delegating & parents):
        raise SystemExit(
            f"{RQ1_CSV.name} disagrees with delegation.py: it flags "
            f"{len(csv_delegating)} delegating parents, the manifest gives "
            f"{len(delegating & parents)}. Regenerate with "
            f"`rq1_signal_composition_corrected.py --from-csv` so there is one "
            f"definition of delegating in the repo.")

    substantive = {r["session_id"] for r in rows
                   if int(r["tool_calls"]) >= SUBSTANTIVE_MIN_TOOL_CALLS
                   and int(r["payload_chars"]) >= SUBSTANTIVE_MIN_PAYLOAD_CHARS}
    paired = {s for s in manifest["paired_session_ids"] if s in scanned}
    size = {r["session_id"]: int(r["serialized_bytes"]) for r in rows}
    # The frame part2_streamlined.py actually sampled from: a session with both a
    # rollup and a transcript, transcript >=100KB.
    frame = {s for s in paired if size.get(s, 0) >= 100 * 1024}

    def rate(ids: set[str], label: str, note: str) -> dict:
        d = ids & delegating
        return {"denominator": label, "n": len(ids), "n_delegating": len(d),
                "pct": round(100.0 * len(d) / len(ids), 2) if ids else None,
                "ci95": list(wilson(len(d), len(ids))), "note": note}

    med = lambda ids: (round(float(np.median([size[s] for s in ids])) / 1e6, 3)
                       if ids else None)
    return {
        "all_parent_sessions": rate(
            parents, "all parent transcripts in the frozen corpus",
            "includes 1,228 trivial sessions (a /clear and nothing else). This "
            "is the denominator that yields the ~12% figure a prior session "
            "quoted, and it is not the population RQ3 sampled from."),
        "substantive_sessions": rate(
            substantive, SUBSTANTIVE_FILTER,
            "the repo's own definition, taken from RQ1. Every one of the 158 "
            "delegating sessions is substantive, so this rate is six times the "
            "all-sessions rate and reweighting against it moves RQ3's numbers "
            "the opposite way."),
        "paired_sessions": rate(
            paired, "sessions with both a rollup note and a transcript",
            "the corpus freeze's paired set."),
        "rq3_sampling_frame": rate(
            frame, "paired sessions with a transcript >=100KB",
            "the population part2_streamlined.py sampled from, and therefore the "
            "only one RQ3's estimate generalises to."),
        "median_transcript_mb": {
            "delegating_all_parents": med(parents & delegating),
            "solo_all_parents": med(parents - delegating),
            "delegating_substantive": med(substantive & delegating),
            "solo_substantive": med(substantive - delegating),
        },
        "n_delegating_parents": len(delegating & parents),
        "detection": ("<project>/<session-id>/subagents/agent-*.jsonl present in "
                      "the frozen manifest; see delegation.py"),
        "unreproduced_figures": {
            "12.3_pct_substantive": (
                "not reproducible under any denominator defined in this repo. The "
                "nearest match is a payload-only floor of 1,000 chars (158/1,293 "
                "= 12.22%), which is not the repo's substantive filter and admits "
                "the /clear stubs that filter exists to exclude."),
            "37_pct_of_sessions_delegate": (
                "a figure documented outside this repo. It lies between the two "
                "denominators here (10.70% of all parent sessions, 63.45% of "
                "substantive ones) and is reproduced by neither, so it should be "
                "read as a rate over some third population rather than as "
                "agreeing or disagreeing with either."),
        },
    }


def stratum_stats(rows: list[dict]) -> dict:
    """recoverable and wrong-of-attempted for one stratum, via the repo's rates()."""
    r = rates(rows)
    return {
        "n": r["n"],
        "n_sessions": len({x["session_id"] for x in rows}),
        "recoverable_pct": r["recoverable"]["pct"],
        "recoverable_ci95": r["recoverable"]["ci95"],
        "recoverable_n": r["recoverable"]["n"],
        "wrong_of_attempted_pct": r["hallucination_of_attempted"]["pct"],
        "wrong_of_attempted_ci95": r["hallucination_of_attempted"]["ci95"],
        "n_wrong": r["hallucination_of_attempted"]["n_wrong"],
        "n_attempted": r["hallucination_of_attempted"]["n_attempted"],
        "grades": {g: r[g]["n"] for g in ("correct", "partially_correct",
                                          "wrong", "absent")},
    }


def _point(recs_d: list[dict], recs_s: list[dict], w: float) -> tuple:
    """Reweighted (recoverable, wrong-of-attempted x2) for one resample."""
    def rec(rs):
        return sum(1 for r in rs if r.get("grade") in RECOVERED) / len(rs)

    def att(rs):
        return sum(1 for r in rs if r.get("grade") != "absent") / len(rs)

    def wrong(rs):
        return sum(1 for r in rs if r.get("grade") == "wrong") / len(rs)

    recoverable = w * rec(recs_d) + (1 - w) * rec(recs_s)
    # Two estimators, both reported, because they differ by half a point and the
    # published 7.0% was the first one.
    #  - mean_of_ratios: average the two strata's wrong/attempted percentages.
    #    Simple, and what the unsaved heredoc computed, but it weights a stratum
    #    by its population share rather than by how many questions it answered.
    #  - ratio_of_means: reweight the wrong count and the attempted count
    #    separately, then divide. This is the design-consistent estimator for a
    #    ratio under post-stratification and is the one to prefer.
    a_d, a_s = att(recs_d), att(recs_s)
    r_d = wrong(recs_d) / a_d if a_d else None
    r_s = wrong(recs_s) / a_s if a_s else None
    mean_of_ratios = (w * r_d + (1 - w) * r_s
                      if r_d is not None and r_s is not None else None)
    denom = w * a_d + (1 - w) * a_s
    ratio_of_means = ((w * wrong(recs_d) + (1 - w) * wrong(recs_s)) / denom
                      if denom else None)
    return recoverable, mean_of_ratios, ratio_of_means


def reweight(by_session_d: dict, by_session_s: dict, w: float) -> dict:
    """Post-stratified estimates with a session-level bootstrap interval."""
    recs_d = [r for rs in by_session_d.values() for r in rs]
    recs_s = [r for rs in by_session_s.values() for r in rs]
    point = _point(recs_d, recs_s, w)

    rng = np.random.RandomState(SEED)
    ids_d, ids_s = sorted(by_session_d), sorted(by_session_s)
    draws: list[list[float]] = [[], [], []]
    degenerate = 0
    for _ in range(N_BOOTSTRAP):
        pick_d = [ids_d[i] for i in rng.choice(len(ids_d), len(ids_d), replace=True)]
        pick_s = [ids_s[i] for i in rng.choice(len(ids_s), len(ids_s), replace=True)]
        b_d = [r for s in pick_d for r in by_session_d[s]]
        b_s = [r for s in pick_s for r in by_session_s[s]]
        vals = _point(b_d, b_s, w)
        if any(v is None for v in vals):
            degenerate += 1
            continue
        for i, v in enumerate(vals):
            draws[i].append(v)

    def ci(vals: list[float]) -> list[float] | None:
        if not vals:
            return None
        s = sorted(vals)
        return [round(100 * s[int(0.025 * len(s))], 1),
                round(100 * s[int(0.975 * len(s))], 1)]

    names = ("recoverable", "wrong_of_attempted_mean_of_ratios",
             "wrong_of_attempted_ratio_of_means")
    out = {"weight_delegating": round(w, 4),
           "bootstrap": {"n_resamples": N_BOOTSTRAP, "seed": SEED,
                         "unit": "session",
                         "n_degenerate_resamples_dropped": degenerate}}
    for name, p, d in zip(names, point, draws):
        out[name] = {"pct": round(100 * p, 1) if p is not None else None,
                     "ci95_bootstrap": ci(d)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N graded records (smoke test); "
                         "skips the published-baseline gate and marks the output")
    ap.add_argument("--rollup-arm", type=Path, default=ROLLUP_ARM)
    ap.add_argument("--raw-arm", type=Path, default=RAW_ARM)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    for p in (args.rollup_arm, args.raw_arm):
        if not p.exists():
            raise SystemExit(f"missing arm file {p}")

    rollup = load_records(args.rollup_arm)
    raw = load_records(args.raw_arm)
    if not rollup:
        raise SystemExit(f"{args.rollup_arm} holds no parseable records; refusing "
                         f"to report a rate over an empty sample")
    # A majority-vote record with no plurality has grade None. rq3_paired_
    # comparison.py drops those from both arms rather than inventing a grade;
    # the same rule applies here so the two scripts agree on the denominator.
    unresolved = {r["record_id"] for r in rollup + raw if r.get("grade") is None}
    rollup = [r for r in rollup if r["record_id"] not in unresolved]
    raw = [r for r in raw if r["record_id"] not in unresolved]
    if not rollup:
        raise SystemExit(f"every record in {args.rollup_arm.name} has grade None "
                         f"(no majority after 5 draws); there is nothing to "
                         f"stratify and no rate to report")

    rollup.sort(key=lambda r: r["record_id"])
    raw.sort(key=lambda r: r["record_id"])
    if args.limit:
        keep = {r["record_id"] for r in rollup[:args.limit]}
        rollup = [r for r in rollup if r["record_id"] in keep]
        raw = [r for r in raw if r["record_id"] in keep]
        print(f"--limit {args.limit}: {len(rollup)} rollup / {len(raw)} raw records")

    manifest = load_manifest()
    session_ids = sorted({r["session_id"] for r in rollup})
    cls = classify_sessions(session_ids, manifest)
    deleg, solo = set(cls["delegating"]), set(cls["solo"])
    unclassifiable = cls["unclassifiable"]

    baseline = stratum_stats(rollup)
    print(f"baseline: {baseline['n']} records over {baseline['n_sessions']} "
          f"sessions  recoverable {baseline['recoverable_pct']}%  "
          f"wrong-of-attempted {baseline['wrong_of_attempted_pct']}%")

    gate_ok = (baseline["recoverable_pct"] == PUBLISHED_RECOVERABLE_PCT
               and baseline["wrong_of_attempted_pct"]
               == PUBLISHED_WRONG_OF_ATTEMPTED_PCT)
    if not args.limit and not gate_ok:
        raise SystemExit(
            f"BASELINE MISMATCH: recomputed {baseline['recoverable_pct']}% "
            f"recoverable / {baseline['wrong_of_attempted_pct']}% "
            f"wrong-of-attempted against the published "
            f"{PUBLISHED_RECOVERABLE_PCT}% / "
            f"{PUBLISHED_WRONG_OF_ATTEMPTED_PCT}%. Every reweighted number below "
            f"would be built on a baseline the paper does not contain, so "
            f"nothing is written. Compare the arm files' digests before looking "
            f"for a bug in the reweighting.")

    if unclassifiable and not args.limit:
        print(f"WARNING: {len(unclassifiable)} of {len(session_ids)} sessions "
              f"could not be classified and are excluded from both strata")

    by_d = {s: [r for r in rollup if r["session_id"] == s] for s in sorted(deleg)}
    by_s = {s: [r for r in rollup if r["session_id"] == s] for s in sorted(solo)}
    strata = {"delegating": stratum_stats([r for rs in by_d.values() for r in rs]),
              "solo": stratum_stats([r for rs in by_s.values() for r in rs])}
    print(f"sample: {len(deleg)} delegating / {len(solo)} solo sessions "
          f"({round(100*len(deleg)/max(1, len(deleg)+len(solo)), 1)}% delegating), "
          f"{len(unclassifiable)} unclassifiable")
    for k, v in strata.items():
        print(f"  {k:12s} n={v['n']:3d} recoverable {v['recoverable_pct']:5.1f}% "
              f"{v['recoverable_ci95']}  wrong-of-att {v['wrong_of_attempted_pct']}%")

    corpus = corpus_rates(manifest) if not args.limit else None

    reweighted = {}
    if by_d and by_s and corpus:
        for key in ("all_parent_sessions", "substantive_sessions",
                    "paired_sessions", "rq3_sampling_frame"):
            w = corpus[key]["pct"] / 100.0
            reweighted[key] = reweight(by_d, by_s, w)
            r = reweighted[key]
            print(f"  reweight to {key:22s} w={w:.4f}  recoverable "
                  f"{r['recoverable']['pct']}% {r['recoverable']['ci95_bootstrap']}"
                  f"  wrong-of-att "
                  f"{r['wrong_of_attempted_ratio_of_means']['pct']}% "
                  f"{r['wrong_of_attempted_ratio_of_means']['ci95_bootstrap']}")
        # The weight the unsaved heredoc used, kept so the published pair is
        # checkable even though no denominator in this repo produces it.
        reweighted["as_published_w_0.123"] = reweight(by_d, by_s, 0.123)
        reweighted["as_published_w_0.123"]["note"] = (
            "the 12.3% weight a prior session quoted as 'the corpus rate for "
            "substantive sessions'. No denominator defined in this repo yields "
            "it: all parent sessions give 10.70% and the repo's substantive "
            "filter gives 63.45%. Reproduced here only so the published 68.1% / "
            "7.0% pair can be checked; do not cite the weight.")

    # Per-stratum raw-vs-rollup gap, on the records both arms hold.
    shared = sorted(set(r["record_id"] for r in rollup)
                    & set(r["record_id"] for r in raw))
    rollup_by_id = {r["record_id"]: r for r in rollup}
    raw_by_id = {r["record_id"]: r for r in raw}
    per_stratum_gap = {"n_shared": len(shared)}
    print(f"\nraw transcript vs rollup on the {len(shared)} shared records:")
    for label, ids in (("delegating", deleg), ("solo", solo)):
        rs_u = [rollup_by_id[i] for i in shared
                if rollup_by_id[i]["session_id"] in ids]
        rs_r = [raw_by_id[i] for i in shared if raw_by_id[i]["session_id"] in ids]
        if not rs_u:
            per_stratum_gap[label] = None
            continue
        u, w = stratum_stats(rs_u), stratum_stats(rs_r)
        gap = round(w["recoverable_pct"] - u["recoverable_pct"], 1)
        per_stratum_gap[label] = {"n": u["n"], "rollup": u, "raw": w,
                                 "recoverable_gap_pp": gap}
        print(f"  {label:12s} n={u['n']:3d}  rollup {u['recoverable_pct']:5.1f}% "
              f"-> raw {w['recoverable_pct']:5.1f}%   gap {gap:+.1f}pp")

    report = {
        "note": ("post-stratification of RQ3's rollup arm on whether the session "
                 "delegated to subagents. No regrading: grades are read from the "
                 "committed majority-vote files and rates come from "
                 "rq3_raw_transcript_arm.rates, the same function the paper's "
                 "other RQ3 tables use."),
        "smoke_test_limit": args.limit or None,
        "corpus_snapshot_utc": manifest["snapshot_utc"],
        "inputs": {
            "rollup_arm": args.rollup_arm.name,
            "rollup_arm_sha256": sha256(args.rollup_arm),
            "raw_arm": args.raw_arm.name,
            "raw_arm_sha256": sha256(args.raw_arm),
            "rq1_per_session_csv_sha256": sha256(RQ1_CSV) if RQ1_CSV.exists() else None,
        },
        "baseline_gate": {
            "published_recoverable_pct": PUBLISHED_RECOVERABLE_PCT,
            "published_wrong_of_attempted_pct": PUBLISHED_WRONG_OF_ATTEMPTED_PCT,
            "recomputed_recoverable_pct": baseline["recoverable_pct"],
            "recomputed_wrong_of_attempted_pct": baseline["wrong_of_attempted_pct"],
            "reproduced_exactly": bool(gate_ok),
        },
        "baseline": baseline,
        "n_unresolved_majority_votes_dropped": len(unresolved),
        "sample_composition": {
            "n_sessions": len(session_ids),
            "n_delegating": len(deleg),
            "n_solo": len(solo),
            "unclassifiable": len(unclassifiable),
            "unclassifiable_session_ids": sorted(unclassifiable),
            "pct_delegating": round(100.0 * len(deleg) / max(1, len(deleg) + len(solo)), 1),
            "classified_from_manifest": cls["from_manifest"],
            "classified_from_live_disk_fallback": cls["from_disk_fallback"],
            "delegating_but_agent_files_pruned_since_snapshot":
                cls["pruned_since_snapshot"],
            "unclassifiable_bias_note":
                "unclassifiable sessions are excluded from both strata rather "
                "than defaulted to solo. The detection error is one-directional: "
                "a pruned subagent directory can only turn a delegating session "
                "into an apparently solo one, never the reverse, so any residual "
                "misclassification understates the delegating stratum.",
            "per_session": cls["detail"],
        },
        "strata": strata,
        "corpus_delegation_rates": corpus,
        "reweighted": reweighted,
        "per_stratum_raw_vs_rollup": per_stratum_gap,
        "interpretation": (
            "the sign of the reweighting depends entirely on the target "
            "population. Against all parent sessions the rollup looks better "
            "than RQ3 reports; against the repo's substantive filter it looks "
            "worse; against the frame RQ3 actually sampled from there is "
            "essentially nothing to correct, because the sample's delegating "
            "share is within sampling error of the frame's. The finding that "
            "survives all three is the per-stratum gap: the raw transcript beats "
            "the rollup within each stratum, so the contrast is not an artifact "
            "of composition."),
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
