#!/usr/bin/env python3
"""Run the action-record and output-stripped arms over the HELD-OUT question set.

Both arms answer from the PREBUILT artifacts in the stage-1 artifact directory, one
.action.md and one .stripped.md per session, rather than rebuilding from the
transcript. The builders are the unmodified
rq3_action_record_arm.build_action_record and
rq3_stripped_transcript_arm.strip_transcript; stage 1 already ran them and
rq3_heldout_manifest.json records the char and token counts, so rebuilding here
could only introduce a difference between what was measured and what was asked.

The answering prompts and the grader are imported, never copied:
  * action record  -> rq3_action_record_arm.answer_from_action_record
  * output-stripped -> rq3_stripped_transcript_arm.answer_from_stripped
  * grading        -> rq3_raw_transcript_arm.grade_answer
so JUDGE_MODEL stays "claude-sonnet-5" in both and the grading prompt is the one
every published RQ3 figure was produced with. Both are called with
return_raw=True, which is the additive change that lets the confidence correction
be computed exactly here instead of bounded from below as in
rq3_confidence_rederive.py.

POPULATION DISCIPLINE. Each arm writes its own file and the analysis scores the
INTERSECTION of record_ids present in both. A question answered in one arm and not
the other is excluded from both and named. This is not decoration: a previous
version of this experiment reported a cost figure over 124 records beside a recall
figure over 89 and overstated by 50.6%, in the direction that flattered its own
conclusion.

BUDGET ALLOCATION, and why it is not 5 questions on all 25 sessions. The full
125-question two-arm run costs about $23 at this gateway's measured Sonnet 5 price
($2.20/M input, $11.00/M output, derived from /v1/usage), against a $10 ceiling.
Two numbers make it more expensive than a char count suggests: the Claude
tokenizer gives 2.468 chars/token on these artifacts against cl100k_base's 3.632,
so real tokens are 1.47x the manifest's counts, and the gateway reports
cached_tokens 0 on every call because the question precedes the artifact in the
prompt, so the five questions of one session share no cacheable prefix. Reordering
the prompt to buy that cache would change the published protocol, and is
therefore not done.

So the question count per session is set by artifact size, written to
rq3_heldout_answer_plan.json BEFORE any answer is requested, and never revised:

    combined (action + stripped) chars  questions
    < 60,000                           5  (all classes)
    60,000 - 250,000                   3
    > 250,000                          1

Every session and every class stays represented, and the two 1 MB-artifact
sessions cost $0.88 a question against $0.09 for the median session, which is why
they are sampled at 1. The allocation depends only on artifact size, fixed in
stage 1, and never on any answer or grade, so it cannot bias the paired contrast:
both arms answer exactly the same questions either way. What it does cost is
power, and the per-session weighting of the unweighted arm-level rates. Both are
reported.

Usage:
    rq3_heldout_arms.py --plan-only            # write/show the plan, no API calls
    rq3_heldout_arms.py --only 37dc4cf9,c0be1f89   # smoke test
    rq3_heldout_arms.py                        # full planned run (resumable)
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import JOBS_DIR  # noqa: E402
from rq3_raw_transcript_arm import (  # noqa: E402
    DATA, GATEWAY, grade_answer, load_records, record_id,
)
import rq3_action_record_arm as action_arm  # noqa: E402
import rq3_stripped_transcript_arm as stripped_arm  # noqa: E402

MANIFEST = DATA / "rq3_heldout_manifest.json"
QUESTIONS = DATA / "rq3_heldout_questions.jsonl"
PLAN = DATA / "rq3_heldout_answer_plan.json"
OUT = {
    "action_record": DATA / "rq3_heldout_action_record_arm.jsonl",
    "output_stripped": DATA / "rq3_heldout_stripped_arm.jsonl",
}
ARTIFACTS = Path(os.environ.get(
    "HELDOUT_ARTIFACTS", str(JOBS_DIR / "3f652bec" / "tmp" / "heldout" / "artifacts")))
SUFFIX = {"action_record": ".action.md", "output_stripped": ".stripped.md"}

CLASSES = ["decision", "dead_end", "outcome", "artifact_detail", "sequence"]
SMALL_CAP, MID_CAP = 60_000, 250_000

# Questions sampled from a mid-tier session. Revised 3 -> 2 after the smoke test
# measured the realised cost multiplier (actual gateway spend over projected
# answering input) at 1.295 rather than the 1.0 the first projection assumed, which
# put the 81-question plan at $11.08 against a $10 ceiling. At 2 the plan is 68
# questions and $9.56. The revision was made with answers from NONE of the 13
# affected sessions seen -- only 37dc4cf9 and c0be1f89 had been answered, and
# neither loses a question -- and it depends on artifact size and the measured
# price, never on a grade, so it cannot bias the paired contrast: both arms still
# answer exactly the same questions. It costs power and nothing else.
MID_Q = int(os.environ.get("HELDOUT_MID_Q", "2"))

# Record ids already answered AND graded in both arms, read before the plan is
# built so a revised cap cannot evict a pair that is already paid for.
def _already_paired() -> set:
    sets = []
    for p in OUT.values():
        if not p.exists():
            return set()
        sets.append({json.loads(l)["record_id"]
                     for l in p.read_text().splitlines() if l.strip()})
    return set.intersection(*sets) if sets else set()


ALREADY_PAIRED = _already_paired()

# Measured from this gateway's own /v1/usage, not a list price: Claude Sonnet 5
# input 35,152 tok -> $0.077319 and output 10,565 tok -> $0.116215, and the
# openai_client bucket reproduces to the cent. Used only to print a running
# projection; the reported figure comes from the /v1/usage delta.
USD_IN, USD_OUT = 2.20e-6, 11.0e-6


def artifact_path(prefix: str, arm: str) -> Path:
    return ARTIFACTS / f"{prefix}{SUFFIX[arm]}"


def build_plan(rows: List[dict]) -> dict:
    """Deterministic, outcome-independent question allocation. See module docstring."""
    by_session: Dict[str, List[dict]] = {}
    for r in rows:
        by_session.setdefault(r["session_prefix"], []).append(r)

    sessions = {}
    for prefix, qs in sorted(by_session.items()):
        combined = sum(artifact_path(prefix, a).stat().st_size for a in OUT)
        n = 5 if combined < SMALL_CAP else (MID_Q if combined <= MID_CAP else 1)
        have = {q["class"]: q for q in qs}
        # Rotation seeded by the session prefix so the sampled classes depend on
        # the session's identity and nothing else. Falls through to the next class
        # in rotation order when the generator did not emit one.
        start = int(prefix, 16) % len(CLASSES)
        order = [CLASSES[(start + k) % len(CLASSES)] for k in range(len(CLASSES))]
        chosen = [c for c in order if c in have][:n]
        extra = [q["class"] for q in qs if q["class"] not in CLASSES]
        if len(chosen) < n:
            chosen += extra[:n - len(chosen)]
        # A question both arms have already answered stays in the plan even if a
        # revised cap would now drop it. Dropping it would delete a paired record
        # that exists in both files from the denominator, which is throwing away
        # measurement rather than saving money: the spend is already sunk.
        for c in (c for c in order if c in have and c not in chosen):
            if record_id(have[c]) in ALREADY_PAIRED:
                chosen.append(c)
        sessions[prefix] = {
            "combined_artifact_chars": combined,
            "n_questions_available": len(qs),
            "n_questions_planned": len(chosen),
            "classes": chosen,
            "record_ids": [record_id(have[c]) for c in chosen if c in have],
        }

    planned = sorted(rid for s in sessions.values() for rid in s["record_ids"])
    return {
        "rule": (f"questions per session by combined action+stripped artifact "
                 f"chars: <{SMALL_CAP:,} -> 5, {SMALL_CAP:,}-{MID_CAP:,} -> 3, "
                 f">{MID_CAP:,} -> 1"),
        "class_rotation": ("CLASSES[(int(prefix,16) % 5 + k) % 5], so the sampled "
                           "classes depend only on the session prefix"),
        "written_before_any_answer_was_requested": True,
        "n_sessions": len(sessions),
        "n_planned_questions": len(planned),
        "planned_record_ids": planned,
        "sessions": sessions,
    }


def usage_snapshot() -> dict:
    """The gateway's own accounting for the openai_client bucket, which is the one
    these urllib calls land in. claude_code traffic is a separate bucket, so this
    isolates the experiment from whatever else is running on the machine."""
    try:
        with urllib.request.urlopen(f"{GATEWAY}/v1/usage", timeout=30) as r:
            d = json.loads(r.read())
        b = d["totals"]["by_tool"].get("openai_client", {})
        return {"request_count": b.get("request_count", 0),
                "input_tokens": b.get("input_tokens", 0),
                "output_tokens": b.get("output_tokens", 0),
                "cache_read_tokens": b.get("cache_read_tokens", 0),
                "cache_creation_5m_tokens": b.get("cache_creation_5m_tokens", 0),
                "estimated_cost_usd": float(b.get("estimated_cost_usd", 0) or 0),
                "as_of": d["meta"]["as_of"]}
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: /v1/usage snapshot failed: {e}")
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated 8-hex prefixes")
    ap.add_argument("--arm", default="", choices=["", "action_record", "output_stripped"])
    ap.add_argument("--budget-usd", type=float, default=9.6,
                    help="stop before a PAIR that would take projected spend past this")
    ap.add_argument("--already-spent-usd", type=float, default=0.0,
                    help=("spend outside this invocation to count against the "
                          "ceiling, which is how question generation is charged to "
                          "the same $10 budget the answering is"))
    args = ap.parse_args()

    rows = load_records(QUESTIONS)
    if not rows:
        print(f"ERROR: no questions in {QUESTIONS}. Run rq3_heldout_questions.py first.")
        sys.exit(1)
    for r in rows:
        r["_rid"] = record_id(r)

    if PLAN.exists():
        plan = json.loads(PLAN.read_text())
        print(f"plan: reusing {PLAN.name} ({plan['n_planned_questions']} questions "
              f"over {plan['n_sessions']} sessions)")
    else:
        plan = build_plan(rows)
        PLAN.write_text(json.dumps(plan, indent=2) + "\n")
        print(f"plan: wrote {PLAN.name} ({plan['n_planned_questions']} questions "
              f"over {plan['n_sessions']} sessions)")

    planned = set(plan["planned_record_ids"])
    todo_rows = [r for r in rows if r["_rid"] in planned]

    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        todo_rows = [r for r in todo_rows if r["session_prefix"] in want]

    if args.plan_only:
        print(f"\n{'sess':10s} {'comb chars':>11s} {'q':>2s}  classes")
        for prefix, s in sorted(plan["sessions"].items(),
                                key=lambda kv: kv[1]["combined_artifact_chars"]):
            print(f"{prefix:10s} {s['combined_artifact_chars']:11,} "
                  f"{s['n_questions_planned']:2d}  {','.join(s['classes'])}")
        est = 0.0
        for prefix, s in plan["sessions"].items():
            chars = s["combined_artifact_chars"]
            est += s["n_questions_planned"] * (chars / 2.459) * USD_IN
        print(f"\nplanned questions {plan['n_planned_questions']}  "
              f"projected answering input ${est:.2f}")
        print("--plan-only: no API calls made")
        return

    arms = [args.arm] if args.arm else ["action_record", "output_stripped"]
    done = {a: {r["record_id"]: r for r in load_records(OUT[a])} if OUT[a].exists()
            else {} for a in arms}
    for a in arms:
        if done[a]:
            print(f"resuming {a}: {len(done[a])} already answered")

    before = usage_snapshot()
    print(f"gateway openai_client before: {before.get('request_count')} requests, "
          f"${before.get('estimated_cost_usd', 0):.4f}")

    spent_est = args.already_spent_usd
    stopped = []
    t0 = time.time()
    # RANK-MAJOR, THEN CHEAPEST FIRST, which is what decides what a budget stop
    # costs. Every session's first planned question runs before any session's second,
    # so all 25 sessions are represented before a single marginal question is spent;
    # within a rank the cheapest goes first. Pure cheapest-first would have put the
    # two 1 MB-artifact sessions last and a stop would have deleted 2 of the 25
    # sessions from the population, which is the expensive kind of truncation. Here a
    # stop only ever trims the highest-rank question off an already-represented
    # session. Ordering depends on artifact size and the pre-registered rotation
    # alone, never on an answer.
    plan_rank = {}
    for prefix, s in plan["sessions"].items():
        for i, rid in enumerate(s["record_ids"]):
            plan_rank[rid] = i
    todo_rows.sort(key=lambda r: (plan_rank.get(r["_rid"], 99),
                                  artifact_path(r["session_prefix"], "action_record")
                                  .stat().st_size, r["session_prefix"], r["class"]))
    n_calls = 0

    # QUESTION-MAJOR AND ATOMIC PER PAIR, which is population discipline enforced in
    # the writer rather than repaired in the analysis. Arm-major would have meant a
    # budget stop or an API error partway through the second arm leaving the first
    # arm's file longer than the second's, and the arm that got the extra records
    # would be the one whose rate was then quoted over a different denominator. Here
    # both arms are answered and graded into a buffer and the pair is written only if
    # BOTH graded, so the two files are identical in record_id by construction and the
    # intersection is the whole of each.
    cache: Dict[Path, str] = {}

    def context_for(prefix: str, arm: str):
        path = artifact_path(prefix, arm)
        if not path.exists():
            return path, None
        if path not in cache:
            cache.clear()  # one session's two artifacts at a time, not all 50
            cache[path] = path.read_text(errors="replace")
        return path, cache[path]

    for r in todo_rows:
        rid = r["_rid"]
        prefix, cls = r["session_prefix"], r["class"]
        if all(rid in done[a] for a in arms):
            continue

        paths, contexts, missing = {}, {}, []
        for arm in arms:
            p, ctx = context_for(prefix, arm)
            paths[arm] = p
            contexts[arm] = ctx
            if ctx is None:
                missing.append(arm)
        if missing:
            print(f"  {prefix} {cls} MISSING ARTIFACT for {missing}, pair skipped")
            continue

        pair_cost = sum(len(contexts[a]) / 2.459 * USD_IN for a in arms) + 0.014
        projected = spent_est + pair_cost
        if projected > args.budget_usd:
            stopped.append({"record_id": rid, "prefix": prefix, "class": cls,
                            "projected_usd": round(projected, 3),
                            "pair_cost_usd": round(pair_cost, 3)})
            print(f"  BUDGET STOP before the {prefix} {cls} PAIR "
                  f"(projected ${projected:.2f} > ${args.budget_usd:.2f}); "
                  f"questions are ordered cheapest first, so stopping here")
            break

        pending, failed = {}, []
        for arm in arms:
            if rid in done[arm]:
                pending[arm] = done[arm][rid]
                continue
            context = contexts[arm]
            fn = (action_arm.answer_from_action_record if arm == "action_record"
                  else stripped_arm.answer_from_stripped)
            answer, confidence, usage, raw_reply = fn(r["question"], context,
                                                      return_raw=True)
            n_calls += 1
            spent_est += (usage.get("prompt_tokens", 0) * USD_IN
                          + usage.get("completion_tokens", 0) * USD_OUT)
            if confidence == "API_ERROR":
                # Never graded, and never graded as absent: grading an API failure
                # as a memory failure would inflate whichever arm did not fail.
                failed.append(arm)
                print(f"  {arm} {prefix} {cls} API_ERROR, pair dropped from both arms")
                continue
            grade = grade_answer(answer, r.get("ground_truth", ""), confidence)
            spent_est += 0.006  # grading call, measured flat at this prompt size
            pending[arm] = {
                "record_id": rid, "session_id": r["session_id"],
                "session_prefix": prefix, "class": cls,
                "question": r["question"], "ground_truth": r.get("ground_truth"),
                "arm": arm, "artifact": str(paths[arm]),
                "artifact_chars": len(context),
                "answer": answer, "rating": confidence, "grade": grade,
                # Additive and the point of task 4: the bytes the rating was parsed
                # out of, so the confidence correction is exact rather than the
                # lower bound rq3_confidence_rederive.py is limited to.
                "raw_reply": raw_reply,
                "usage": usage, "model": action_arm.JUDGE_MODEL,
            }

        if failed or len(pending) != len(arms):
            stopped.append({"record_id": rid, "prefix": prefix, "class": cls,
                            "reason": f"API_ERROR in {failed}, pair not written"})
            continue

        for arm in arms:
            if rid in done[arm]:
                continue
            with OUT[arm].open("a") as fh:
                fh.write(json.dumps(pending[arm]) + "\n")
            done[arm][rid] = pending[arm]
        g = "  ".join(f"{a.split('_')[0]}:{pending[a]['rating']}/{pending[a]['grade']}"
                      for a in arms)
        print(f"  {prefix} {cls:15s} "
              f"{min(len(contexts[a]) for a in arms)/1e3:7.1f}KB  {g}  "
              f"est ${spent_est:5.2f}")

    after = usage_snapshot()
    delta = {k: (after.get(k, 0) - before.get(k, 0))
             for k in ("request_count", "input_tokens", "output_tokens",
                       "cache_read_tokens", "cache_creation_5m_tokens",
                       "estimated_cost_usd") if k in after and k in before}
    print(f"\n{n_calls} answer calls in {time.time()-t0:.0f}s")
    print(f"gateway openai_client delta: {delta}")
    if stopped:
        print(f"BUDGET-STOPPED {len(stopped)} records; see the analysis for the "
              f"intersection actually scored")
    for arm in arms:
        print(f"  {arm:16s} {len(done[arm])} records in {OUT[arm].name}")

    (DATA / "rq3_heldout_run_usage.json").write_text(json.dumps({
        "note": ("gateway-reported usage for the openai_client bucket, which is "
                 "where these urllib /v1/chat/completions calls land; claude_code "
                 "traffic is a separate bucket. Appended once per invocation."),
        "price_derivation": ("from /v1/usage itself: Claude Sonnet 5 input "
                             "35,152 tok -> $0.077319 ($2.20/M), output 10,565 tok "
                             "-> $0.116215 ($11.00/M)"),
        "invocations": (json.loads((DATA / "rq3_heldout_run_usage.json").read_text())
                        ["invocations"] if (DATA / "rq3_heldout_run_usage.json").exists()
                        else []) + [{
            "argv": sys.argv[1:], "arms": arms, "answer_calls": n_calls,
            "wall_seconds": round(time.time() - t0, 1),
            "before": before, "after": after, "delta": delta,
            "budget_stopped": stopped,
        }],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
