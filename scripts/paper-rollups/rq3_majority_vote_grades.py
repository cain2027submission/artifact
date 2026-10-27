#!/usr/bin/env python3
"""Grade both RQ3 arms by majority vote over independent draws.

Every number in this experiment previously rested on one grading call per record.
That is not safe here, because this same experiment measured the grader flipping
18 of 124 grades when simply asked again. Publishing a finding about grader
instability while resting the headline on a single draw would be incoherent.

So each record is graded 3 times independently and the plurality grade is taken.
A 1-1-1 split has no plurality, so those records get 2 further draws (5 total)
and the plurality of those; anything still tied is reported as unresolved rather
than silently assigned.

Answers are never regenerated -- they are read from each arm's output file, so
only the grading step varies. That also makes this cheap: grading prompts carry
one answer and one ground truth, not a 250k-token transcript.

The per-record agreement rate is itself an output worth reporting: it says how
much of the spread between the published 19.5% and the re-runs is just the
grader, and it belongs in the paper's methods rather than in a footnote.

Usage:
    rq3_majority_vote_grades.py --arm uniform --draws 3
    rq3_majority_vote_grades.py --arm raw --draws 3
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_raw_transcript_arm import DATA, call_llm, load_records, rates  # noqa: E402
from rq3_regrade_rollup_arm import FIXED_ORDER, grading_prompt, parse  # noqa: E402
from grading_common import GRADE_ERROR, GradeError  # noqa: E402


def parse_draw(resp, order):
    """One draw, or GradeError: a failed draw must not enter the vote."""
    g = parse(resp, order)
    if g == GRADE_ERROR:
        raise GradeError("empty or unparseable grader reply", resp)
    return g

JUDGE_MODEL = "claude-sonnet-5"

ARMS = {
    # arm: (input file, answer field, rating field)
    "uniform": ("rq3_rollup_uniform.jsonl", "answer", "rating"),
    "raw": ("rq3_questions_raw_arm.jsonl", "raw_answer", "raw_rating"),
    "stripped": ("rq3_questions_stripped_arm.jsonl", "answer", "rating"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    fname, ans_key, rate_key = ARMS[args.arm]
    rows = load_records(DATA / fname)
    out_path = DATA / f"rq3_majority_{args.arm}.jsonl"
    report_path = DATA / f"rq3_majority_{args.arm}_summary.json"

    done = {}
    if out_path.exists():
        for r in load_records(out_path):
            done[r["record_id"]] = r
        print(f"resuming: {len(done)} already voted")

    todo = [r for r in rows if r["record_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"arm={args.arm}  records={len(rows)}  to vote={len(todo)}  "
          f"draws={args.draws}\n")

    t0 = time.time()
    for i, r in enumerate(todo, 1):
        rid = r["record_id"]
        answer = r.get(ans_key) or ""
        rating = r.get(rate_key, "")

        # Deterministic by-rule absent, exactly as both arms' graders do. Calling
        # the grader here would change the protocol rather than re-measure it.
        if rating == "NOT_ANSWERABLE" or not answer:
            rec = {"record_id": rid, "class": r.get("class"),
                   "session_id": r.get("session_id"),
                   "by_rule_absent": True, "draws": [], "grade": "absent",
                   "unanimous": True, "unresolved": False,
                   "single_draw_grade": r.get("grade")}
        else:
            prompt = grading_prompt(answer, r.get("ground_truth", ""),
                                    r.get("question", ""))
            try:
                draws = []
                for _ in range(args.draws):
                    resp, _u = call_llm(prompt, model=JUDGE_MODEL, timeout=120)
                    draws.append(parse_draw(resp, FIXED_ORDER))
                counts = Counter(draws)
                top, n_top = counts.most_common(1)[0]
                # No plurality (e.g. 1-1-1): take two more draws before deciding.
                if sum(1 for g, n in counts.items() if n == n_top) > 1:
                    for _ in range(2):
                        resp, _u = call_llm(prompt, model=JUDGE_MODEL,
                                            timeout=120)
                        draws.append(parse_draw(resp, FIXED_ORDER))
                    counts = Counter(draws)
                    top, n_top = counts.most_common(1)[0]
            except GradeError:
                # Not written, so a rerun retries it; a failed draw never votes.
                print(f"[{i}/{len(todo)}] {rid} grade_error (not recorded, "
                      f"will retry)")
                continue
            unresolved = sum(1 for g, n in counts.items() if n == n_top) > 1
            rec = {"record_id": rid, "class": r.get("class"),
                   "session_id": r.get("session_id"),
                   "by_rule_absent": False, "draws": draws,
                   "grade": None if unresolved else top,
                   "unanimous": len(counts) == 1, "unresolved": unresolved,
                   "single_draw_grade": r.get("grade")}
            flag = ""
            if unresolved:
                flag = "  UNRESOLVED"
            elif not rec["unanimous"]:
                flag = f"  split {dict(counts)}"
            print(f"[{i}/{len(todo)}] {rid} -> {rec['grade']}"
                  f" (was {r.get('grade')}){flag}")

        with out_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        done[rid] = rec

    print(f"\nvoted {len(todo)} in {time.time()-t0:.0f}s")

    voted = list(done.values())
    called = [r for r in voted if not r["by_rule_absent"]]
    unresolved = [r for r in voted if r["unresolved"]]
    resolved = [r for r in voted if not r["unresolved"]]

    unanimous = sum(1 for r in called if r["unanimous"])
    changed = sum(1 for r in resolved
                  if r["grade"] != r["single_draw_grade"])

    report = {
        "arm": args.arm,
        "input_file": fname,
        "n_records": len(voted),
        "n_graded_by_model": len(called),
        "n_by_rule_absent": len(voted) - len(called),
        "draws_per_record": args.draws,
        "judge": f"{JUDGE_MODEL} via local gateway, no temperature set",
        "design": ("answers held fixed; each record graded independently "
                   f"{args.draws}x and the plurality taken; a split with no "
                   "plurality gets 2 further draws, then is reported unresolved"),
        "grader_unanimous_n": unanimous,
        "grader_unanimous_pct": round(100.0 * unanimous / len(called), 1) if called else None,
        "n_unresolved": len(unresolved),
        "n_majority_differs_from_single_draw": changed,
        "rates_majority_vote": rates(resolved),
        "rates_single_draw": rates(
            [{"grade": r["single_draw_grade"]} for r in voted]),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    def line(label, d):
        h = d["hallucination_of_attempted"]
        return (f"{label:22s} n={d['n']:3d} correct {d['correct']['n']:3d} "
                f"partial {d['partially_correct']['n']:3d} "
                f"wrong {d['wrong']['n']:3d} absent {d['absent']['n']:3d}  "
                f"recoverable {d['recoverable']['pct']:5.1f}% "
                f"{d['recoverable']['ci95']}  wrong-of-att {h['pct']}% {h['ci95']}")

    print()
    print(line("single draw", report["rates_single_draw"]))
    print(line("majority vote", report["rates_majority_vote"]))
    print(f"\ngrader unanimous on {unanimous}/{len(called)} model-graded records "
          f"({report['grader_unanimous_pct']}%)")
    print(f"majority differs from the single draw on {changed} records")
    print(f"unresolved after 5 draws: {len(unresolved)}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
