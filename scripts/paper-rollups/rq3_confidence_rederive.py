#!/usr/bin/env python3
"""Re-derive the RQ3 arm rates under the corrected confidence extractor.

Makes no model calls. Every grade used here is already on disk: each arm's own
single-draw grade, the 3-of-5 majority draws in rq3_majority_<arm>.jsonl, and the
cross-model verdicts in rq3_judge_cache.json.

What can and cannot be re-derived, stated plainly because it bounds every number
below. The arms stored the EXTRACTED answer, not the reply, and the buggy
extractor dropped the reply's final line unconditionally. So the line that
carried the marker is gone from disk for every record, and no re-run of the fixed
extractor over the stored data is possible. What survives is the answer text, and
that is enough to identify the records the defect demonstrably mis-rated:

  1. A marker line still inside the retained answer -- the model stated it
     mid-reply, so the fixed extractor (which scans the whole reply) reads it.
     The retained answer is a prefix of the reply, so this is exact.
  2. A retained answer that OPENS with "NOT ANSWERABLE", the refusal the
     answering prompt prescribes, while the stored rating is not NOT_ANSWERABLE.
     The reply refused; the rating says it answered. The only way the old code
     reaches ANSWERED here is by matching "ANSWERED" inside the dropped line --
     either inside PARTIALLY_ANSWERED or inside prose such as "not answered".
     Corroborating base rate: in the uniform arm 29 of 35 refusal-opening replies
     did store NOT_ANSWERABLE, so the other 6 are leakage, not a house style.
  3. Otherwise the stored rating stands. It cannot be checked, but it also cannot
     matter much: grade_answer() branches on NOT_ANSWERABLE alone, so an
     ANSWERED that should have been PARTIALLY_ANSWERED changes no grade.

That makes these counts a LOWER bound on the defect: a refusal phrased without
the prescribed words is indistinguishable here from a genuine attempt.

Usage:
    rq3_confidence_rederive.py            # writes rq3_confidence_corrected.json
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_raw_transcript_arm import (  # noqa: E402
    DATA, MARKER_LINE, load_records, rates,
)

OUT = DATA / "rq3_confidence_corrected.json"

ARMS = {
    # arm: (record file, answer field, rating field, published summary, key)
    "raw": ("rq3_questions_raw_arm.jsonl", "raw_answer", "raw_rating",
            "rq3_raw_arm_summary.json", "raw_transcript_arm"),
    "stripped": ("rq3_questions_stripped_arm.jsonl", "answer", "rating",
                 "rq3_stripped_arm_summary.json", "stripped_arm"),
    "uniform": ("rq3_rollup_uniform.jsonl", "answer", "rating",
                "rq3_rollup_uniform_summary.json", "uniform_rollup_arm"),
}

# The refusal the answering prompt asks for: 'say "NOT ANSWERABLE"'.
REFUSAL_OPENER = re.compile(r"^\s*[*_`#>\s]*NOT[ _]ANSWERABLE\b", re.IGNORECASE)


def corrected_rating(answer: str, stored: str):
    """Return (rating, reason) for one stored record. See the module docstring."""
    lines = answer.strip().split("\n")
    for i in range(len(lines) - 1, -1, -1):
        m = MARKER_LINE.match(lines[i])
        if m:
            return m.group(1).upper().replace(" ", "_"), "marker_in_retained_answer"
    if REFUSAL_OPENER.match(answer or ""):
        return "NOT_ANSWERABLE", "refusal_opener"
    return stored, "unchanged"


def main() -> None:
    cache = json.loads((DATA / "rq3_judge_cache.json").read_text())
    pool = load_records(DATA / "rq3_questions.jsonl")

    report = {
        "what_this_is": ("RQ3 answered-vs-refused and wrong-of-attempted rates "
                         "re-derived under the corrected confidence extractor, "
                         "from grades already on disk. No model calls."),
        "method": __doc__.strip(),
        "bug1_partial_collapse": {
            "claim": ("ANSWERED was tested before PARTIALLY_ANSWERED and is a "
                      "substring of it, so no arm could ever record a partial."),
            "stored_PARTIALLY_ANSWERED_by_arm": {},
            "comparison_pool": {
                "file": "rq3_questions.jsonl",
                "note": ("graded by the older resume_rq3/part2_streamlined parser, "
                         "which tested for NOT and PARTIALLY before ANSWERED; its "
                         "answering prompt asks for a 'RATING:' line, so this is "
                         "corroboration, not a controlled comparison"),
                "ratings": dict(Counter(r.get("rating") for r in pool)),
            },
            "effect_on_published_grades": ("none: grade_answer() branches on "
                                           "NOT_ANSWERABLE only, so ANSWERED and "
                                           "PARTIALLY_ANSWERED grade identically"),
        },
        "evidence_class_used": (
            "every reclassification below came from class 1, not class 2: the "
            "prescribed refusal sits on a line of its own, so the fixed matcher "
            "reads it as a marker line directly and the opener heuristic never "
            "has to fire. See each arm's `reasons`."),
        "arms": {},
    }

    for arm, (fname, ak, rk, sfile, skey) in ARMS.items():
        rows = load_records(DATA / fname)
        maj = {r["record_id"]: r
               for r in load_records(DATA / f"rq3_majority_{arm}.jsonl")}
        published = json.loads((DATA / sfile).read_text())[skey]

        report["bug1_partial_collapse"]["stored_PARTIALLY_ANSWERED_by_arm"][arm] = {
            "n_records": len(rows),
            "n_partially_answered": sum(1 for r in rows
                                        if r[rk] == "PARTIALLY_ANSWERED"),
        }

        changed, truncated = [], []
        for r in rows:
            new, reason = corrected_rating(r[ak] or "", r[rk])
            r["_new_rating"], r["_reason"] = new, reason
            if new != r[rk] and new == "NOT_ANSWERABLE":
                # The dropped final line cannot have been a clean marker, or the
                # old code would have read it: it held "ANSWERED" inside prose
                # (a line of content, silently lost) or inside
                # PARTIALLY_ANSWERED. Which one is unrecoverable per record; the
                # stored rating is wrong either way.
                truncated.append(r["record_id"])
                m = maj.get(r["record_id"], {})
                changed.append({
                    "record_id": r["record_id"],
                    "session_id": r.get("session_id"),
                    "class": r.get("class"),
                    "stored_rating": r[rk],
                    "corrected_rating": new,
                    "reason": reason,
                    "single_draw_grade": r.get("grade"),
                    "majority_grade": m.get("grade"),
                    "majority_draws": m.get("draws"),
                    "answer_is_only_the_refusal": (r[ak] or "").strip().upper()
                        in ("NOT ANSWERABLE", "NOT_ANSWERABLE"),
                    "cached_verdicts": {k.split(":", 1)[1]: v for k, v in cache.items()
                                        if k.startswith(r["record_id"] + ":")},
                    "answer_head": (r[ak] or "").replace("\n", " ")[:120],
                })
        flip = {c["record_id"] for c in changed}

        # Single draw: the grades each arm published, then the same grades with
        # every reclassified record made `absent` by rule.
        single_old = [{"grade": r.get("grade")} for r in rows]
        single_new = [{"grade": "absent" if r["record_id"] in flip else r.get("grade")}
                      for r in rows]
        # Majority vote: identical treatment of the 3-of-5 plurality grades. The
        # by-rule absent test in rq3_majority_vote_grades.py is the same test, so
        # flipping the rating is all that changes.
        mrows = [maj[r["record_id"]] for r in rows if r["record_id"] in maj]
        unresolved = [m for m in mrows if m.get("unresolved")]
        mres = [m for m in mrows if not m.get("unresolved")]
        maj_old = [{"grade": m.get("grade")} for m in mres]
        maj_new = [{"grade": "absent" if m["record_id"] in flip else m.get("grade")}
                   for m in mres]

        reproduces = all(
            single_old_block == published.get(g)
            for g, single_old_block in rates(single_old).items()
            if g in published
        )

        report["arms"][arm] = {
            "record_file": fname,
            "n_records": len(rows),
            "published_summary_file": sfile,
            "recomputation_reproduces_published_single_draw": reproduces,
            "n_reclassified": len(changed),
            "n_reclassified_answer_is_only_the_refusal":
                sum(1 for c in changed if c["answer_is_only_the_refusal"]),
            "n_refusal_opening_answers": sum(
                1 for r in rows if REFUSAL_OPENER.match(r[ak] or "")),
            "n_refusal_opening_already_rated_not_answerable": sum(
                1 for r in rows
                if REFUSAL_OPENER.match(r[ak] or "") and r[rk] == "NOT_ANSWERABLE"),
            "n_answers_retaining_a_marker_line": sum(
                1 for r in rows if r["_reason"] == "marker_in_retained_answer"),
            "n_dropped_final_line_was_not_a_clean_marker": len(truncated),
            "reasons": dict(Counter(r["_reason"] for r in rows)),
            "n_majority_unresolved_excluded": len(unresolved),
            "reclassified": sorted(changed, key=lambda c: c["record_id"]),
            "single_draw": {"stored": rates(single_old), "corrected": rates(single_new)},
            "majority_vote": {"stored": rates(maj_old), "corrected": rates(maj_new)},
        }

    # The vault task note's hypothesis, checked rather than assumed.
    u = report["arms"]["uniform"]
    maj_wrong = [r for r in load_records(DATA / "rq3_majority_uniform.jsonl")
                 if r.get("grade") == "wrong"]
    flip_u = {c["record_id"] for c in u["reclassified"]}
    report["claim_3_of_11"] = {
        "claim": ("a vault task note states 3 of the rollup arm's 11 wrong answers "
                  "literally open with NOT ANSWERABLE"),
        "arm_with_11_wrong": "uniform, majority vote (rq3_majority_uniform.jsonl)",
        "n_wrong": len(maj_wrong),
        "n_wrong_opening_with_the_refusal": sum(
            1 for r in maj_wrong if r["record_id"] in flip_u),
        "record_ids": sorted(r["record_id"] for r in maj_wrong
                            if r["record_id"] in flip_u),
        "holds": (len(maj_wrong) == 11
                  and sum(1 for r in maj_wrong if r["record_id"] in flip_u) == 3),
        "note": ("the single-draw grades for the same arm hold 13 wrong, 5 of them "
                 "refusal-openers, so the claim is specific to the majority-vote "
                 "grades; and the 11 is not the whole defect -- 6 uniform-arm "
                 "records are reclassified, the sixth already graded absent"),
    }

    OUT.write_text(json.dumps(report, indent=2) + "\n")

    for arm, a in report["arms"].items():
        print(f"=== {arm}  n={a['n_records']}  "
              f"reproduces published single draw: "
              f"{a['recomputation_reproduces_published_single_draw']}")
        print(f"  refusal-opening answers: {a['n_refusal_opening_answers']}  "
              f"of which already NOT_ANSWERABLE: "
              f"{a['n_refusal_opening_already_rated_not_answerable']}  "
              f"-> reclassified: {a['n_reclassified']}")
        for tag in ("single_draw", "majority_vote"):
            for state in ("stored", "corrected"):
                d = a[tag][state]
                h = d["hallucination_of_attempted"]
                print(f"  {tag:13s} {state:9s} n={d['n']:3d} "
                      f"correct {d['correct']['n']:3d} partial "
                      f"{d['partially_correct']['n']:3d} wrong {d['wrong']['n']:3d} "
                      f"absent {d['absent']['n']:3d}  refused "
                      f"{d['absent']['pct']:5.1f}%  recoverable "
                      f"{d['recoverable']['pct']:5.1f}%  wrong-of-attempted "
                      f"{h['pct']}% ({h['n_wrong']}/{h['n_attempted']})")
    c = report["claim_3_of_11"]
    print(f"\nclaim '3 of 11' holds={c['holds']}: "
          f"{c['n_wrong_opening_with_the_refusal']} of {c['n_wrong']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
