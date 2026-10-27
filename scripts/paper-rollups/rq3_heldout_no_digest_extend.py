#!/usr/bin/env python3
"""Extend the held-out validation to a third arm: action record WITHOUT result
digests. Also extends output-stripped's held-out answers from 68 to the full
pre-registered 125-question pool (rq3_heldout_questions.jsonl), since the
original run only answered 68 under a tighter budget.

This question set is unchanged from the original held-out draw (25 sessions
dated 2026-09-27 to 2026-09-30, excluded by construction from the 124-question
design set the action record was built against). No new sessions, no new
questions, no revision of what's "held out" -- only completing answers against
an already-fixed pool, decided before any no-digest held-out answer exists.

Writes:
    data/paper-rollups/rq3_heldout_stripped_arm_extended.jsonl   (68 original + new)
    data/paper-rollups/rq3_heldout_no_digest_arm.jsonl           (new, up to 125)
    data/paper-rollups/rq3_heldout_no_digest_paired.json         (paired analysis)

Usage:
    rq3_heldout_no_digest_extend.py --dry-run
    rq3_heldout_no_digest_extend.py --limit 10
    rq3_heldout_no_digest_extend.py
"""
import argparse
import hashlib
import json
import sys
import time
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_raw_transcript_arm import (  # noqa: E402
    DATA, call_llm, extract_confidence, grade_answer, index_transcripts,
    load_records, rates, resolve,
)
from spinoff_actionrecord_ablation import build_action_record_no_result_digests  # noqa: E402
from rq3_stripped_transcript_arm import strip_transcript, answer_from_stripped  # noqa: E402
from rq3_action_record_arm import apply_confidence_correction  # noqa: E402

JUDGE_MODEL = "claude-sonnet-5"
QUESTIONS = DATA / "rq3_heldout_questions.jsonl"
STRIPPED_ORIG = DATA / "rq3_heldout_stripped_arm.jsonl"
STRIPPED_OUT = DATA / "rq3_heldout_stripped_arm_extended.jsonl"
NODIGEST_OUT = DATA / "rq3_heldout_no_digest_arm.jsonl"
PAIRED_OUT = DATA / "rq3_heldout_no_digest_paired.json"
CONTEXT_LIMIT_TOKENS = 1_180_000
BYTES_PER_TOKEN = 3.6


def rid(r: dict) -> str:
    seed = f"{r.get('session_id','')}|{r.get('class','')}|{r.get('question','')}"
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def answer_from_no_digest(question: str, record_text: str):
    prompt = f"""Someone asks: "{question}"

Below is a structured action record of an agent session, with tool calls shown as name + salient arguments and turn ordering. Tool results are NOT included. Answer the question based ONLY on this record. If it does not contain enough information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

ACTION RECORD (no result digests):
{record_text}

Provide your answer and confidence now:"""
    response, usage = call_llm(prompt, model=JUDGE_MODEL, timeout=600)
    if not response:
        return "", "API_ERROR", usage, response
    answer, confidence = extract_confidence(response)
    return answer, confidence, usage, response


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    questions = load_records(QUESTIONS)
    for q in questions:
        q["_rid"] = rid(q)
    print(f"question pool: {len(questions)}")

    by_prefix = index_transcripts()
    for q in questions:
        q["_path"] = resolve(q["session_prefix"], by_prefix)
    missing = [q for q in questions if q["_path"] is None]
    if missing:
        print(f"WARNING: {len(missing)} questions have no resolvable transcript")
    questions = [q for q in questions if q["_path"] is not None]

    if args.dry_run:
        seen = {}
        for q in questions:
            p = str(q["_path"])
            if p not in seen:
                nd_text, _ = build_action_record_no_result_digests(p)
                st_text, _ = strip_transcript(Path(p).read_text(errors="replace"))
                seen[p] = (len(nd_text), len(st_text))
        for p, (nd, st) in sorted(seen.items(), key=lambda kv: -kv[1][0]):
            print(f"{Path(p).stem[:8]:10s} no_digest={nd:8,} stripped={st:8,}")
        print(f"\n{len(seen)} unique transcripts, {len(questions)} questions. --dry-run: no API calls.")
        return

    # --- stripped arm: seed with original 68, extend to full pool ---
    stripped_done = {}
    if STRIPPED_ORIG.exists():
        for r in load_records(STRIPPED_ORIG):
            stripped_done[r["record_id"]] = r
    print(f"stripped: {len(stripped_done)} already answered (original run)")

    stripped_todo = [q for q in questions if q["_rid"] not in stripped_done]
    if args.limit:
        stripped_todo = stripped_todo[:args.limit]
    print(f"stripped: {len(stripped_todo)} to answer this run")

    # Fresh output file: original rows + new rows, never mutate the original.
    STRIPPED_OUT.write_text("")
    for r in stripped_done.values():
        with STRIPPED_OUT.open("a") as fh:
            fh.write(json.dumps(r) + "\n")

    strip_cache = {}
    t0 = time.time()
    for i, q in enumerate(stripped_todo, 1):
        p = str(q["_path"])
        if p not in strip_cache:
            strip_cache[p] = strip_transcript(Path(p).read_text(errors="replace"))
        stripped_text, st = strip_cache[p]
        if len(stripped_text) / BYTES_PER_TOKEN > CONTEXT_LIMIT_TOKENS:
            print(f"[stripped {i}/{len(stripped_todo)}] SKIP too big")
            continue
        answer, confidence, usage, raw_reply = answer_from_stripped(
            q["question"], stripped_text, return_raw=True)
        if confidence == "API_ERROR":
            print(f"[stripped {i}/{len(stripped_todo)}] API_ERROR, skip")
            continue
        grade = grade_answer(answer, q.get("ground_truth", ""), confidence)
        print(f"[stripped {i}/{len(stripped_todo)}] {q['session_prefix']} {q['class']:15s} {confidence:18s} {grade}")
        out = {"record_id": q["_rid"], "session_id": q["session_id"],
               "session_prefix": q["session_prefix"], "class": q["class"],
               "question": q["question"], "ground_truth": q["ground_truth"],
               "arm": "output_stripped", "artifact_chars": len(stripped_text),
               "answer": answer, "rating": confidence, "grade": grade,
               "raw_reply": raw_reply, "usage": usage, "model": JUDGE_MODEL}
        with STRIPPED_OUT.open("a") as fh:
            fh.write(json.dumps(out) + "\n")
        stripped_done[q["_rid"]] = out
    print(f"stripped extension done in {time.time()-t0:.0f}s, total now {len(stripped_done)}")

    # --- no-digest action record arm: answer full pool ---
    nodigest_done = {}
    if NODIGEST_OUT.exists():
        for r in load_records(NODIGEST_OUT):
            nodigest_done[r["record_id"]] = r
    nd_todo = [q for q in questions if q["_rid"] not in nodigest_done]
    if args.limit:
        nd_todo = nd_todo[:args.limit]
    print(f"\nno-digest: {len(nodigest_done)} already answered, {len(nd_todo)} to answer this run")

    nd_cache = {}
    t0 = time.time()
    for i, q in enumerate(nd_todo, 1):
        p = str(q["_path"])
        if p not in nd_cache:
            nd_cache[p] = build_action_record_no_result_digests(p)
        record_text, st = nd_cache[p]
        if len(record_text) / BYTES_PER_TOKEN > CONTEXT_LIMIT_TOKENS:
            print(f"[no_digest {i}/{len(nd_todo)}] SKIP too big")
            continue
        answer, confidence, usage, raw_reply = answer_from_no_digest(q["question"], record_text)
        if confidence == "API_ERROR":
            print(f"[no_digest {i}/{len(nd_todo)}] API_ERROR, skip")
            continue
        grade = grade_answer(answer, q.get("ground_truth", ""), confidence)
        print(f"[no_digest {i}/{len(nd_todo)}] {q['session_prefix']} {q['class']:15s} {confidence:18s} {grade}")
        out = {"record_id": q["_rid"], "session_id": q["session_id"],
               "session_prefix": q["session_prefix"], "class": q["class"],
               "question": q["question"], "ground_truth": q["ground_truth"],
               "arm": "action_record_no_digest", "artifact_chars": len(record_text),
               "answer": answer, "rating": confidence, "grade": grade,
               "raw_reply": raw_reply, "usage": usage, "model": JUDGE_MODEL}
        with NODIGEST_OUT.open("a") as fh:
            fh.write(json.dumps(out) + "\n")
        nodigest_done[q["_rid"]] = out
    print(f"no-digest done in {time.time()-t0:.0f}s, total now {len(nodigest_done)}")

    # --- paired analysis on the intersection, same method as rq3_paired_comparison.py ---
    shared = sorted(set(stripped_done) & set(nodigest_done))
    stripped_shared = [stripped_done[i] for i in shared]
    nodigest_shared = [nodigest_done[i] for i in shared]

    r_stripped = rates(stripped_shared)
    r_nodigest = rates(nodigest_shared)

    stripped_ok = {i for i in shared if stripped_done[i]["grade"] in ("correct", "partially_correct")}
    nodigest_ok = {i for i in shared if nodigest_done[i]["grade"] in ("correct", "partially_correct")}
    b = len(nodigest_ok - stripped_ok)   # no_digest recovered, stripped did not
    c = len(stripped_ok - nodigest_ok)   # stripped recovered, no_digest did not
    both = len(stripped_ok & nodigest_ok)
    neither = len(shared) - len(stripped_ok | nodigest_ok)
    n = b + c
    if n:
        k = min(b, c)
        p_value = min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0)
    else:
        p_value = 1.0

    print(f"\npaired on 'recoverable' over {len(shared)} shared records:")
    print(f"  both {both}   no_digest only {b}   stripped only {c}   neither {neither}")
    print(f"  discordant {n}, exact binomial two-sided p = {p_value:.3f}")
    print(f"\noutput-stripped (n={r_stripped['n']}): recoverable {r_stripped['recoverable']['pct']}%")
    print(f"no-digest action record (n={r_nodigest['n']}): recoverable {r_nodigest['recoverable']['pct']}%")

    report = {
        "n_shared": len(shared),
        "pool_size": len(questions),
        "output_stripped": r_stripped,
        "action_record_no_digest": r_nodigest,
        "mcnemar": {"both": both, "no_digest_only": b, "stripped_only": c,
                    "neither": neither, "discordant": n, "p_two_sided": p_value},
        "note": "powered sample size NOT computed/met; this extends from 68 to the full "
                "125-question pre-registered pool, same as the original held-out validation "
                "which was also explicitly unpowered. Not a substitute for a dedicated "
                "powered non-inferiority test.",
    }
    PAIRED_OUT.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {PAIRED_OUT}")


if __name__ == "__main__":
    main()
