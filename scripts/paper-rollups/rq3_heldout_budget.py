#!/usr/bin/env python3
"""Cost the remaining held-out run against the $10 ceiling, from MEASURED rates.

Calibrated on the 8 smoke-test pairs rather than on an assumed price: those pairs
projected $0.543 of answering input and actually moved the gateway's openai_client
bucket by $0.7029, so the realised multiplier over projected input is 1.295 and it
covers the answer output tokens and the grading call too. No API calls here.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_raw_transcript_arm import DATA, load_records, record_id  # noqa: E402

PLAN = DATA / "rq3_heldout_answer_plan.json"
QUESTIONS = DATA / "rq3_heldout_questions.jsonl"
FILES = {"action_record": DATA / "rq3_heldout_action_record_arm.jsonl",
         "output_stripped": DATA / "rq3_heldout_stripped_arm.jsonl"}
USD_IN = 2.20e-6
CHARS_PER_TOK = 2.459
MULT = 1.295        # measured on the smoke pairs: actual bucket delta / projected in
SPENT = 2.2476      # gateway openai_client cumulative after qgen + smoke
CEILING = 10.00


def main() -> None:
    plan = json.loads(PLAN.read_text())
    rows = load_records(QUESTIONS)
    for r in rows:
        r["_rid"] = record_id(r)
    done = set()
    for p in FILES.values():
        if p.exists():
            done |= {r["record_id"] for r in load_records(p)}

    def pair_usd(chars: int) -> float:
        return chars / CHARS_PER_TOK * USD_IN * MULT

    print(f"spent so far ${SPENT:.2f}   ceiling ${CEILING:.2f}   "
          f"headroom ${CEILING - SPENT:.2f}\n")

    for mid_q in (3, 2, 1):
        total = 0.0
        n_q = 0
        for prefix, s in plan["sessions"].items():
            chars = s["combined_artifact_chars"]
            cap = (5 if chars < 60_000 else mid_q if chars < 250_000 else 1)
            # Records already answered are sunk, not re-charged, and are kept
            # whatever the revised cap says.
            # Plan order, not alphabetical: the class rotation in the plan is what
            # decides which classes survive a smaller cap, and it is pre-registered.
            keep = list(s["record_ids"][:cap])
            keep += [rid for rid in s["record_ids"] if rid in done and rid not in keep]
            for rid in keep:
                n_q += 1
                if rid not in done:
                    total += pair_usd(chars)
        fits = "FITS" if SPENT + total <= CEILING else "OVER"
        print(f"mid tier (60k-250k) at {mid_q} questions: n={n_q:3d}  "
              f"remaining ${total:5.2f}  total ${SPENT + total:5.2f}  {fits}")

    print("\nper-session remaining pair cost, most expensive first:")
    for prefix, s in sorted(plan["sessions"].items(),
                            key=lambda kv: -kv[1]["combined_artifact_chars"]):
        chars = s["combined_artifact_chars"]
        undone = [r for r in rows
                  if r["session_prefix"] == prefix
                  and r["_rid"] in set(s["record_ids"])
                  and r["_rid"] not in done]
        if not undone:
            continue
        print(f"  {prefix} {chars:9,} chars  {len(undone)} undone  "
              f"${pair_usd(chars):.3f}/pair  ${len(undone)*pair_usd(chars):5.2f}")


if __name__ == "__main__":
    main()
