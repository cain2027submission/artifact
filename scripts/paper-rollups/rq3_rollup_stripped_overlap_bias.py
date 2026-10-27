#!/usr/bin/env python3
"""Probe-overlap bias bound for the rollup vs output-stripped RQ3 pair.

The paper discloses mean probe-token overlap of 0.902 against the raw transcript
and 0.705 against the rollup, a 0.197 gap. The prose-vs-call regrouping found
that inside the tie set (38 probes with exact zero overlap difference), prose-only
beats call-only by 23.7 points (86.8% vs 63.2%) rather than 43.1 uncorrected.

This script applies the same tie-set method to the central comparison: rollup
(56.5% recoverable) vs output-stripped transcript (90.3% recoverable), a 33.8-point
gap. It computes per-question overlap against both contexts, identifies the tie set
where overlap favours neither arm, and reports the gap within it.

Method reused verbatim from rq3_arm_probe_overlap.py and rq4_known_item.py:
  * overlap -- containment: |question_tokens & context_tokens| / |question_tokens|
  * grades -- from rq3_confidence_corrected.json, single-draw corrected grades
  * contexts -- rollup notes from sessions/, stripped from strip_transcript()

NO model calls. Overlap is token set intersection. Contexts rebuilt mechanically.

Usage:
    rq3_rollup_stripped_overlap_bias.py            # writes JSON report
    rq3_rollup_stripped_overlap_bias.py --no-write # print only
"""

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_DIR, PROJECTS_DIR  # noqa: E402
from rq3_stripped_transcript_arm import strip_transcript  # noqa: E402

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
RECOVERABLE = ("correct", "partially_correct")
FRONTMATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)", re.DOTALL)

# Input files
CONFIDENCE_CORRECTED = DATA_DIR / "rq3_confidence_corrected.json"
STRIPPED_ARM = DATA_DIR / "rq3_questions_stripped_arm.jsonl"
ROLLUP_ARM = DATA_DIR / "rq3_rollup_uniform.jsonl"
OUT = DATA_DIR / "rq3_rollup_stripped_overlap_bias.json"


def compute_overlap(question: str, context: str) -> float:
    """Compute containment: |question_tokens & context_tokens| / |question_tokens|."""
    q_tokens = set(TOKEN_RE.findall(question.lower()))
    if not q_tokens:
        return 0.0
    c_tokens = set(TOKEN_RE.findall(context.lower()))
    return len(q_tokens & c_tokens) / len(q_tokens)


def load_stripped_context(transcript_path: str) -> str:
    """Load and strip a transcript."""
    try:
        raw = Path(transcript_path).read_text(encoding="utf-8", errors="replace")
        stripped, _ = strip_transcript(raw)
        return stripped
    except Exception:
        return ""


def rate(grades):
    """Compute recoverable rate."""
    n = len(grades)
    if n == 0:
        return None, 0, 0
    k = sum(1 for g in grades if g in RECOVERABLE)
    return 100.0 * k / n, k, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    # Load arm files
    stripped_arm_data = {}
    for line in STRIPPED_ARM.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rid = r.get("record_id")
        if rid:
            stripped_arm_data[rid] = r

    rollup_arm_data = {}
    for line in ROLLUP_ARM.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rid = r.get("record_id")
        if rid:
            rollup_arm_data[rid] = r

    # Load confidence-corrected grades and build reclassification maps
    corrected = json.loads(CONFIDENCE_CORRECTED.read_text())

    # For reclassified records, if corrected_rating is NOT_ANSWERABLE, grade is "absent"
    # Otherwise keep the original grade
    rollup_reclassified = {}
    for r in corrected["arms"]["uniform"]["reclassified"]:
        if r["corrected_rating"] == "NOT_ANSWERABLE":
            rollup_reclassified[r["record_id"]] = "absent"
        else:
            rollup_reclassified[r["record_id"]] = r["single_draw_grade"]

    stripped_reclassified = {}
    for r in corrected["arms"]["stripped"]["reclassified"]:
        if r["corrected_rating"] == "NOT_ANSWERABLE":
            stripped_reclassified[r["record_id"]] = "absent"
        else:
            stripped_reclassified[r["record_id"]] = r["single_draw_grade"]

    # Find common record IDs
    common_ids = set(stripped_arm_data.keys()) & set(rollup_arm_data.keys())

    print(f"Loaded {len(stripped_arm_data)} stripped arm records")
    print(f"Loaded {len(rollup_arm_data)} rollup arm records")
    print(f"Common record IDs: {len(common_ids)}")
    print(f"Rollup reclassifications: {len(rollup_reclassified)}")
    print(f"Stripped reclassifications: {len(stripped_reclassified)}")

    # Now compute overlaps for each record
    rows = []
    cache = {}  # Cache contexts by path

    for rid in common_ids:
        stripped_rec = stripped_arm_data[rid]
        rollup_rec = rollup_arm_data[rid]

        session_id = stripped_rec["session_id"]
        question = stripped_rec["question"]
        transcript_path = stripped_rec.get("transcript", "")
        rollup_path = rollup_rec.get("rollup", "")

        if not transcript_path or not rollup_path:
            continue

        # Load contexts (with caching)
        cache_key = (transcript_path, rollup_path)
        if cache_key not in cache:
            # Load rollup note (strip frontmatter)
            try:
                rollup_text = Path(rollup_path).read_text(encoding="utf-8", errors="replace")
                fm_match = FRONTMATTER_RE.match(rollup_text)
                rollup_ctx = rollup_text[fm_match.end():].strip() if fm_match else rollup_text.strip()
            except Exception:
                rollup_ctx = ""

            # Load and strip transcript
            stripped_ctx = load_stripped_context(transcript_path)
            cache[cache_key] = (rollup_ctx, stripped_ctx)

        rollup_ctx, stripped_ctx = cache[cache_key]

        if not rollup_ctx or not stripped_ctx:
            continue

        # Compute overlaps
        rollup_ov = compute_overlap(question, rollup_ctx)
        stripped_ov = compute_overlap(question, stripped_ctx)

        # Get corrected grades (use reclassified if available, else original)
        rollup_grade = rollup_reclassified.get(rid, rollup_rec.get("grade"))
        stripped_grade = stripped_reclassified.get(rid, stripped_rec.get("grade"))

        rows.append({
            "record_id": rid,
            "session_id": session_id,
            "class": stripped_rec.get("class"),
            "question": question,
            "rollup_overlap": rollup_ov,
            "stripped_overlap": stripped_ov,
            "overlap_diff": stripped_ov - rollup_ov,
            "rollup_grade": rollup_grade,
            "stripped_grade": stripped_grade,
        })

    print(f"\nComputed overlaps for {len(rows)} paired records")

    if len(rows) == 0:
        print("ERROR: No records with valid contexts found")
        return

    # Overall statistics
    rollup_ov = [r["rollup_overlap"] for r in rows]
    stripped_ov = [r["stripped_overlap"] for r in rows]
    gap = st.mean(stripped_ov) - st.mean(rollup_ov)

    print(f"\n## Overlap statistics (n = {len(rows)})")
    print(f"Stripped (output-stripped transcript): mean {st.mean(stripped_ov):.3f}")
    print(f"Rollup: mean {st.mean(rollup_ov):.3f}")
    print(f"Gap: {gap:+.3f}")

    # Find tie set (exact zero difference)
    tie_set = [r for r in rows if abs(r["overlap_diff"]) < 1e-9]

    print(f"\n## Tie set (exact zero overlap difference)")
    print(f"Tie set size: {len(tie_set)} probes")

    tie_rollup_rate = tie_stripped_rate = tie_gap = None
    tie_rollup_k = tie_stripped_k = tie_rollup_n = tie_stripped_n = 0

    if len(tie_set) > 0:
        tie_rollup_rate, tie_rollup_k, tie_rollup_n = rate([r["rollup_grade"] for r in tie_set])
        tie_stripped_rate, tie_stripped_k, tie_stripped_n = rate([r["stripped_grade"] for r in tie_set])
        tie_gap = tie_stripped_rate - tie_rollup_rate if tie_stripped_rate is not None else None

        print(f"Rollup recoverable: {tie_rollup_k}/{tie_rollup_n} = {tie_rollup_rate:.1f}%")
        print(f"Stripped recoverable: {tie_stripped_k}/{tie_stripped_n} = {tie_stripped_rate:.1f}%")
        print(f"Gap in tie set: {tie_gap:+.1f} points")
        print(f"  vs full-set gap: 33.8 points")

    # Compute terciles by rollup overlap (following prose-vs-call pattern)
    ordered = sorted(rows, key=lambda r: r["rollup_overlap"])
    n = len(ordered)
    cuts = [0, n // 3, 2 * n // 3, n]

    print(f"\n## Terciles by rollup overlap")
    print(f"{'Tercile':8s} {'n':>4s} {'Rollup OV':>10s} {'Stripped OV':>12s} "
          f"{'Rollup Rec':>11s} {'Stripped Rec':>13s} {'Gap':>8s}")

    terciles = []
    for i in range(3):
        chunk = ordered[cuts[i]:cuts[i + 1]]
        r_rate, r_k, r_n = rate([r["rollup_grade"] for r in chunk])
        s_rate, s_k, s_n = rate([r["stripped_grade"] for r in chunk])
        gap_pp = s_rate - r_rate if s_rate is not None and r_rate is not None else None

        terciles.append({
            "tercile": i + 1,
            "n": len(chunk),
            "rollup_overlap_range": [chunk[0]["rollup_overlap"], chunk[-1]["rollup_overlap"]],
            "rollup_overlap_mean": st.mean(r["rollup_overlap"] for r in chunk),
            "stripped_overlap_mean": st.mean(r["stripped_overlap"] for r in chunk),
            "rollup_recoverable_pct": r_rate,
            "rollup_recoverable": [r_k, r_n],
            "stripped_recoverable_pct": s_rate,
            "stripped_recoverable": [s_k, s_n],
            "gap_pp": gap_pp,
        })

        print(f"{i+1:<8d} {len(chunk):4d} {st.mean(r['rollup_overlap'] for r in chunk):10.3f} "
              f"{st.mean(r['stripped_overlap'] for r in chunk):12.3f} "
              f"{r_rate:10.1f}% {s_rate:12.1f}% {gap_pp:+7.1f}pp")

    print(f"\nHighest rollup-overlap tercile (tercile 3): gap narrows to "
          f"{terciles[2]['gap_pp']:.1f} points (vs 33.8 full-set)")
    print(f"Lowest rollup-overlap tercile (tercile 1): gap is "
          f"{terciles[0]['gap_pp']:.1f} points")

    # Overall rates for comparison
    overall_rollup_rate, overall_rollup_k, overall_rollup_n = rate([r["rollup_grade"] for r in rows])
    overall_stripped_rate, overall_stripped_k, overall_stripped_n = rate([r["stripped_grade"] for r in rows])
    overall_gap = overall_stripped_rate - overall_rollup_rate

    print(f"\n## Overall recoverable rates (for validation)")
    print(f"Rollup: {overall_rollup_k}/{overall_rollup_n} = {overall_rollup_rate:.1f}%")
    print(f"Stripped: {overall_stripped_k}/{overall_stripped_n} = {overall_stripped_rate:.1f}%")
    print(f"Gap: {overall_gap:+.1f} points")
    print(f"Published (corrected, n=124): rollup 56.5%, stripped 90.3%, gap 33.8 points")

    report = {
        "question": "does probe-provenance bias bound the rollup vs output-stripped gap?",
        "method": "same tie-set regrouping as prose-vs-call, applied to rollup vs stripped, plus tercile analysis",
        "overlap_metric": {
            "definition": "containment: |question_tokens & context_tokens| / |question_tokens|",
            "tokenizer": r"TOKEN_RE = re.compile(r'\w+|[^\w\s]', re.UNICODE)",
            "casefolding": "both .lower()",
        },
        "n_paired": len(rows),
        "overlap_gap_mean": gap,
        "tie_set": {
            "definition": "questions with abs(stripped_overlap - rollup_overlap) < 1e-9",
            "n": len(tie_set),
            "rollup_recoverable_pct": tie_rollup_rate,
            "rollup_recoverable": [tie_rollup_k, tie_rollup_n],
            "stripped_recoverable_pct": tie_stripped_rate,
            "stripped_recoverable": [tie_stripped_k, tie_stripped_n],
            "gap_pp": tie_gap,
            "note": "only 1 probe in tie set, tercile analysis more informative",
        },
        "terciles_by_rollup_overlap": terciles,
        "overall": {
            "rollup_recoverable_pct": overall_rollup_rate,
            "stripped_recoverable_pct": overall_stripped_rate,
            "gap_pp": overall_gap,
        },
        "published_for_comparison": {
            "rollup_pct": 56.5,
            "stripped_pct": 90.3,
            "gap_pp": 33.8,
            "n": 124,
        },
        "per_question": rows,
    }

    if not args.no_write:
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
