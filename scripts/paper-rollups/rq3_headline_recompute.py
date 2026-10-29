#!/usr/bin/env python3
"""Re-derive RQ3's wrong-of-attempted headline after the confidence-parser fix.

Zero model calls. Pure arithmetic over data already on disk.

What this establishes, in order:

1. PROVENANCE. rq3_questions.jsonl holds 349 valid records. It used to hold a
   33,278-byte NUL-filled hole at line 125 as well -- a partial write from two
   scripts appending to the same path concurrently -- and because
   rq3_retention_summary.load_records stops at the first byte it cannot decode,
   the published analysis saw exactly the 124 records before that hole. Commit
   8a9d84b7 then removed the hole as data hygiene, which silently widened this
   script's pool from 124 records to all 349 and made both published figures stop
   reproducing. The pool is therefore selected explicitly below (see
   published_pool) rather than inherited from the file's own corruption.

2. WHICH PARSER WROTE THE PUBLISHED 124. They carry the key `rating`, the schema
   of resume_rq3.py / resume_rq3_v2.py / part2_streamlined.py, and 27 of them are
   PARTIALLY_ANSWERED -- a label part2_question_retention.py's old substring loop
   could not emit at all. part2_question_retention.py contributed exactly 9
   records (lines 126-134, keyed `confidence`), all AFTER the hole and therefore
   none of them in the published set. The headline is not contaminated by the
   parser this change fixes.

3. THE PUBLISHED FIGURES, REPRODUCED. 19.5% [12.6, 29.1] is 17/87 over those 124
   records. 13.6% [7.8, 22.7] is 11/81 over rq3_majority_uniform.jsonl, the
   single-protocol re-run with majority-vote grading -- a different file. Both are
   asserted here; the script exits non-zero if either fails to reproduce.

4. HOW MUCH THE FIX MOVES IT. Nothing, and this is provable rather than assumed:
   the published records were not produced by the fixed parser's predecessor, and
   the partial-credit decision below cannot move `wrong` or `absent`. What cannot
   be established is the counterfactual size of the bug on a re-run, because no
   raw model reply is stored in any of these files -- only the post-parse
   `judge_answer`. Re-parsing that is circular. Every reclassification count here
   is a LOWER BOUND, and the lower bound is zero.

Usage:
    python3 rq3_headline_recompute.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rq3_raw_transcript_arm import MARKER_LINE, extract_confidence  # noqa: E402
from rq3_retention_summary import (  # noqa: E402
    DATA, QUESTIONS, load_records, rates, wilson)

MAJORITY_UNIFORM = DATA / "rq3_majority_uniform.jsonl"
ROLLUP_UNIFORM = DATA / "rq3_rollup_uniform.jsonl"
OUT = DATA / "rq3_headline_corrected.json"

PUBLISHED_POOLED = {"pct": 19.5, "ci95": [12.6, 29.1], "n": 124,
                    "n_wrong": 17, "n_attempted": 87}
PUBLISHED_PROTOCOL = {"pct": 13.6, "ci95": [7.8, 22.7], "n": 124,
                      "n_wrong": 11, "n_attempted": 81}

# The size of the pool the published RQ3 analysis actually scored. Stated as a
# constant because it used to be an accident: load_records halted at the NUL hole
# and 124 fell out of where the corruption happened to sit.
PUBLISHED_N = 124

failures: list[str] = []


def label(r: dict) -> str | None:
    """The confidence marker, under either of the two key names in use."""
    return r.get("rating") or r.get("confidence")


def published_pool(rows: list[dict]) -> list[dict]:
    """The PUBLISHED_N records the published RQ3 analysis scored, in file order.

    File order is not an arbitrary choice here, it is the actual provenance: the
    published consumer was load_records, whose stream decoder stopped at the NUL
    hole that used to sit at line 125, so what it scored was the file's first 124
    records and nothing after them. Verified against the pre-8a9d84b7 file: all
    124 decode byte-identically, in the same order, as the first 124 of the
    current file.

    Asserted rather than assumed, because a slice is the kind of selection that
    goes quiet when it stops matching: if the file ever holds PUBLISHED_N records
    or fewer this returns the whole thing and the reproduction below becomes
    vacuous, so that case fails loudly instead.
    """
    if len(rows) <= PUBLISHED_N:
        failures.append(
            f"{QUESTIONS.name} decodes {len(rows)} records, not more than "
            f"PUBLISHED_N={PUBLISHED_N}: the published pool can no longer be "
            "distinguished from the whole file, so nothing below is a check")
    return rows[:PUBLISHED_N]


def assert_matches(name: str, got: dict, want: dict) -> bool:
    ok = all(got.get(k) == v for k, v in want.items())
    print(f"  {'REPRODUCED' if ok else 'MISMATCH'}: {name}")
    if not ok:
        for k, v in want.items():
            if got.get(k) != v:
                print(f"      {k}: published {v!r}, recomputed {got.get(k)!r}")
        failures.append(name)
    return ok


def hoa(rows: list[dict]) -> dict:
    """wrong-of-attempted, via the published rates() helper so the arithmetic
    cannot drift from the paper."""
    r = rates(rows)["hallucination_of_attempted"]
    return {"n": len(rows), "n_wrong": r["n_wrong"],
            "n_attempted": r["n_attempted"], "pct": r["pct"],
            "ci95": r["ci95"]}


def cap_partials(rows: list[dict]) -> list[dict]:
    """PARTIALLY_ANSWERED graded `correct` becomes `partially_correct`.

    The judge declared its own answer incomplete; scoring it fully right is the
    grading inconsistency the old code allowed. `wrong` and `absent` are never
    touched, so wrong-of-attempted is arithmetically identical. Only `correct`
    moves, and `recoverable` (correct + partially_correct) is also unchanged.
    """
    out = []
    for r in rows:
        r = dict(r)
        if label(r) == "PARTIALLY_ANSWERED" and r.get("grade") == "correct":
            r["grade"] = "partially_correct"
        out.append(r)
    return out


def partial_as_own_category(rows: list[dict]) -> dict:
    """The strictest honest reading: a self-declared partial is not a full
    attempt, so wrong-of-attempted is computed over fully-ANSWERED records only
    and the partials are reported as their own bucket rather than folded in."""
    full = [r for r in rows if label(r) == "ANSWERED"]
    part = [r for r in rows if label(r) == "PARTIALLY_ANSWERED"]
    refused = [r for r in rows if label(r) == "NOT_ANSWERABLE"]
    return {
        "answered_only": hoa(full),
        "n_partially_answered": len(part),
        "partially_answered_grades": dict(Counter(r.get("grade") for r in part)),
        "n_not_answerable": len(refused),
    }


def half_credit(rows: list[dict]) -> dict:
    """recoverable with a partial worth half a correct, rather than a whole one.

    Not a Wilson-able proportion (the numerator is fractional), so it is reported
    as a point estimate only and deliberately without an interval."""
    c = Counter(r.get("grade") for r in rows)
    n = len(rows)
    score = c["correct"] + 0.5 * c["partially_correct"]
    return {"n": n, "weighted_numerator": score,
            "pct": round(100.0 * score / n, 1) if n else None,
            "interval": None,
            "note": "fractional numerator; a Wilson interval does not apply"}


def main() -> int:
    # ---------------- 1. provenance of the pool ---------------------------- #
    print("1. PROVENANCE")
    decoded, dropped = load_records(QUESTIONS)
    pub = published_pool(decoded)
    raw = QUESTIONS.read_text(errors="replace")
    all_rows = []
    for line in raw.splitlines():
        if not line.strip() or line.startswith("\x00"):
            continue
        try:
            all_rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    nul_bytes = raw.count("\x00")
    print(f"   {QUESTIONS.name}: {len(all_rows)} valid records total, "
          f"{nul_bytes} NUL bytes in a partial-write hole")
    print(f"   load_records decodes {len(decoded)}, dropping {dropped} truncated")
    # Two counts are printed here and only one of them is the pool the published
    # figures are over, so say which, rather than leaving a later reader to pick.
    print(f"   PUBLISHED POOL = first {len(pub)} records in file order; the "
          f"remaining {len(decoded) - len(pub)} were never scored in the paper")

    keys = Counter("rating" if "rating" in r else
                   "confidence" if "confidence" in r else "?" for r in all_rows)
    pub_keys = Counter("rating" if "rating" in r else
                       "confidence" if "confidence" in r else "?" for r in pub)
    print(f"   label key over the whole file: {dict(keys)}")
    print(f"   label key over the published {len(pub)}: {dict(pub_keys)}")
    print(f"   published label mix: {dict(Counter(label(r) for r in pub))}")
    part2_records = keys["confidence"]
    print(f"   -> part2_question_retention.py (key `confidence`) wrote "
          f"{part2_records} records, {pub_keys['confidence']} of them published")

    # ---------------- 2. reproduce the published figures ------------------- #
    print("\n2. PUBLISHED FIGURES, OLD PARSER, LABELS AS STORED")
    pooled_old = hoa(pub)
    print(f"   pooled  {pooled_old['n_wrong']}/{pooled_old['n_attempted']} = "
          f"{pooled_old['pct']}% {pooled_old['ci95']}")
    assert_matches("pooled 19.5% [12.6, 29.1]", pooled_old, PUBLISHED_POOLED)

    maj = [json.loads(l) for l in MAJORITY_UNIFORM.read_text().splitlines()
           if l.strip()]
    protocol_old = hoa(maj)
    print(f"   protocol {protocol_old['n_wrong']}/{protocol_old['n_attempted']} = "
          f"{protocol_old['pct']}% {protocol_old['ci95']}  "
          f"(from {MAJORITY_UNIFORM.name}, not {QUESTIONS.name})")
    assert_matches("one-protocol 13.6% [7.8, 22.7]", protocol_old,
                   PUBLISHED_PROTOCOL)

    # No-op control for the slice. If the whole file reproduced the published
    # figures too, published_pool would be doing nothing and the REPRODUCED line
    # above would be worth nothing. Measured: the full 349 give 17.2% [12.8, 22.7]
    # over 38/221, which is why 8a9d84b7 turned this gate red without touching a
    # single number in the paper.
    pooled_all = hoa(decoded)
    if len(decoded) > len(pub) and pooled_all["pct"] == PUBLISHED_POOLED["pct"]:
        failures.append(
            f"the whole {len(decoded)}-record file also gives "
            f"{pooled_all['pct']}%, so slicing to the published pool is not what "
            "makes the figures above reproduce")
    print(f"   control: whole file ({len(decoded)} recs) gives "
          f"{pooled_all['n_wrong']}/{pooled_all['n_attempted']} = "
          f"{pooled_all['pct']}% {pooled_all['ci95']} -- not the published figure, "
          "so the pool selection is load-bearing")

    # ---------------- 3. reclassification the stored data can show --------- #
    print("\n3. RECLASSIFICATION UNDER THE FIXED PARSER")
    print("   No file in data/paper-rollups stores the raw model reply; the")
    print("   schema stops at the post-parse `judge_answer`. The only evidence")
    print("   left is a marker line that survived INSIDE that answer.")
    directions: Counter = Counter()
    surviving = 0
    for r in all_rows:
        ans = (r.get("judge_answer") or "").strip()
        if not ans:
            continue
        lines = ans.split("\n")
        if not any(MARKER_LINE.match(l) for l in lines):
            continue  # no marker survived: nothing to compare against
        surviving += 1
        _, refound = extract_confidence(ans)
        if refound != label(r):
            directions[f"{label(r)}->{refound}"] += 1
    print(f"   records with a surviving marker line: {surviving} of {len(all_rows)}")
    print(f"   of those, disagreeing with the stored label: {sum(directions.values())}")
    print(f"   by direction: {dict(directions) or '{} (none)'}")

    # A separate, non-circular demonstration: how often real stored model prose
    # would itself trip the old substring test if it landed on the final line.
    prose_trips = 0
    for r in all_rows:
        ans = (r.get("judge_answer") or "").strip()
        if not ans:
            continue
        last = ans.split("\n")[-1]
        if "ANSWERED" in last.upper() and not MARKER_LINE.match(last):
            prose_trips += 1
    print(f"   stored answers whose FINAL line contains 'ANSWERED' as prose, not")
    print(f"   as a marker -- i.e. would be misread as a confident answer: "
          f"{prose_trips} of {len(all_rows)}")

    # The label mix is the one statistical trace of defect 1 on this pool.
    part2_labels = Counter(label(r) for r in all_rows if "confidence" in r)
    loose_labels = Counter(label(r) for r in all_rows if "rating" in r)
    print(f"   label mix, part2 parser ({sum(part2_labels.values())} recs): "
          f"{dict(part2_labels)}")
    print(f"   label mix, loose parsers ({sum(loose_labels.values())} recs): "
          f"{dict(loose_labels)}")
    print("   0 of 9 partials against 53 of 340 is the expected signature of")
    print("   defect 1, but at n=9 it is not evidence: P(0 partials | p=0.156)")
    print(f"   = {0.844 ** 9:.3f}.")

    # ---------------- 4. the headline, both ways on partial credit --------- #
    print("\n4. HEADLINE, BOTH WAYS ON PARTIAL CREDIT")
    pooled_capped = hoa(cap_partials(pub))
    protocol_capped = hoa(cap_partials(maj))
    pub_rates = rates(pub)
    capped_rates = rates(cap_partials(pub))
    print(f"   (a) partial-as-answered  [comparable to published]")
    print(f"       pooled   {pooled_old['n_wrong']}/{pooled_old['n_attempted']}"
          f" = {pooled_old['pct']}% {pooled_old['ci95']}")
    print(f"       protocol {protocol_old['n_wrong']}/"
          f"{protocol_old['n_attempted']} = {protocol_old['pct']}% "
          f"{protocol_old['ci95']}")
    print(f"   (b) partial capped at partially_correct  [the honest grading]")
    print(f"       pooled   {pooled_capped['n_wrong']}/"
          f"{pooled_capped['n_attempted']} = {pooled_capped['pct']}% "
          f"{pooled_capped['ci95']}   <- identical by construction")
    print(f"       correct  {pub_rates['correct']['n']} -> "
          f"{capped_rates['correct']['n']}, partially_correct "
          f"{pub_rates['partially_correct']['n']} -> "
          f"{capped_rates['partially_correct']['n']}, recoverable "
          f"{pub_rates['recoverable']['pct']}% -> "
          f"{capped_rates['recoverable']['pct']}%")
    own = partial_as_own_category(pub)
    print(f"   (c) partial as its own category, excluded from attempted")
    print(f"       ANSWERED-only pooled {own['answered_only']['n_wrong']}/"
          f"{own['answered_only']['n_attempted']} = "
          f"{own['answered_only']['pct']}% {own['answered_only']['ci95']}")
    print(f"       plus {own['n_partially_answered']} PARTIALLY_ANSWERED reported "
          f"separately: {own['partially_answered_grades']}")
    hc = half_credit(pub)
    print(f"   (d) recoverable at half credit for a partial: {hc['pct']}% "
          f"(against {pub_rates['recoverable']['pct']}% at full credit)")
    print("       Note (c) cannot be computed for the one-protocol 13.6% figure:")
    print("       rq3_rollup_uniform.jsonl records 84 ANSWERED, 40 NOT_ANSWERABLE")
    print("       and ZERO partials, because that arm ran the buggy parser. The")
    print("       partial-credit question is unanswerable there until it is re-run.")

    # ---------------- 5. write the artefact -------------------------------- #
    payload = {
        "generated_by": "scripts/paper-rollups/rq3_headline_recompute.py",
        "model_calls": 0,
        "metric": "wrong-of-attempted = wrong / (n - absent), Wilson 95% CI",
        "old_figure": {
            "pooled": {**PUBLISHED_POOLED, "source": QUESTIONS.name,
                       "as_printed": "19.5\\% [12.6, 29.1]"},
            "one_protocol": {**PUBLISHED_PROTOCOL,
                             "source": MAJORITY_UNIFORM.name,
                             "as_printed": "13.6\\% [7.8, 22.7]"},
        },
        "old_figure_reproducible": not failures,
        "old_figure_recomputed": {"pooled": pooled_old,
                                  "one_protocol": protocol_old},
        "new_figure": {
            "partial_as_answered": {"pooled": pooled_old,
                                    "one_protocol": protocol_old},
            "partial_capped_at_partially_correct": {
                "pooled": pooled_capped, "one_protocol": protocol_capped,
                "why_identical": "the cap moves correct -> partially_correct "
                                 "only; wrong and absent, and therefore "
                                 "wrong-of-attempted, cannot change",
                "pooled_grade_shift": {
                    "correct": [pub_rates["correct"]["n"],
                                capped_rates["correct"]["n"]],
                    "partially_correct": [
                        pub_rates["partially_correct"]["n"],
                        capped_rates["partially_correct"]["n"]],
                    "recoverable_pct": [pub_rates["recoverable"]["pct"],
                                        capped_rates["recoverable"]["pct"]],
                },
            },
            "partial_as_own_category": {"pooled": own},
            "recoverable_half_credit": {"pooled": hc},
        },
        "n": {"pool_file_valid_records": len(all_rows),
              "published_slice": len(pub),
              "truncated_dropped": dropped,
              "nul_bytes_in_partial_write": nul_bytes,
              "one_protocol": len(maj)},
        "reclassified_by_direction": dict(directions),
        "reclassification_is_a_lower_bound": True,
        "reclassification_limitation":
            "No raw model reply is stored in rq3_questions.jsonl, "
            "rq3_rollup_uniform.jsonl or rq3_majority_uniform.jsonl -- only the "
            "post-parse judge_answer. Re-parsing that is circular, so the "
            "counterfactual effect of the old parser on these records cannot be "
            "measured, only bounded below at zero. part2_question_retention.py "
            "now stores raw_judge_reply so the next parser change is measurable.",
        "provenance": {
            "published_slice_label_key": dict(pub_keys),
            "whole_file_label_key": dict(keys),
            "published_label_mix": dict(Counter(label(r) for r in pub)),
            "part2_parser_label_mix": dict(part2_labels),
            "loose_parser_label_mix": dict(loose_labels),
            "part2_records_in_published_slice": pub_keys["confidence"],
            "headline_touched_by_this_fix": pub_keys["confidence"] > 0,
        },
        "supporting_counts": {
            "records_with_surviving_marker_line": surviving,
            "stored_answers_whose_final_line_is_ANSWERED_prose": prose_trips,
        },
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {OUT}")
    if failures:
        print(f"FAILED to reproduce: {failures}")
        return 1
    print("both published figures reproduced exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
