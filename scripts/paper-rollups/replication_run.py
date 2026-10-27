#!/usr/bin/env python3
"""Paid runner for the replication. Modes: zero | pilot | full [--lo A --hi B].
Arms: nodigest (build_action_record_no_result_digests + rq3_heldout_no_digest_extend prompt),
stripped (strip_transcript + answer_from_stripped prompt), zero (rq3_zero_context_arm prompt).
Grade: gc.grade_with (prompt v2 + first-word parser), 3 independent draws, plurality,
2 more draws on a tie; protocol string below. Grader sees question, ground truth, one answer;
no arm label exists in the grading prompt. Per-question the order in which the two arms'
answers are answered+graded is randomised (seed ORDER_SEED, recorded in each row)."""
import argparse, hashlib, json, random, sys, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import grading_common as gc
from replication_common import RDIR, SONNET, call_paid, usd, gate, log_batch, cumulative_usd, CAP_USD
from rq3_raw_transcript_arm import extract_confidence
from spinoff_actionrecord_ablation import build_action_record_no_result_digests
from rq3_stripped_transcript_arm import strip_transcript

ORDER_SEED = 20261014
PROTOCOL = f"{gc.PROMPT_VERSION}+{gc.PARSER_VERSION}+majority3+{SONNET}"
ANS = RDIR / "answers.jsonl"
FAIL = RDIR / "failures.jsonl"
BPT = 3.6
lk = threading.Lock()
_art = {}


def verify_frozen():
    h = hashlib.sha256((RDIR / "questions.jsonl").read_bytes()).hexdigest()
    assert h == (RDIR / "questions.sha256").read_text().split()[0], "questions.jsonl changed after freeze"
    return [json.loads(l) for l in (RDIR / "questions.jsonl").open()]


def artifacts(path):
    with lk:
        if path in _art: return _art[path]
    nd, _ = build_action_record_no_result_digests(path)
    st, _ = strip_transcript(Path(path).read_text(errors="replace"))
    with lk:
        _art[path] = (nd, st)
    return _art[path]


def prompt_for(arm, q):
    question = q["question"]
    if arm == "zero":
        return f"""Someone asks: "{question}"

This question refers to a specific past agent coding session. You have been
given NO information about that session: no transcript, no filename, no
summary, no hint of any kind about what happened in it. Answer only if you
somehow already know the specific answer (you almost certainly do not, since
this refers to a private session you were never shown any part of). Do not
guess or invent plausible-sounding specifics. If you do not have enough
information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

Provide your answer and confidence now:""", 0
    nd, st = artifacts(q["transcript"])
    if arm == "nodigest":
        return f"""Someone asks: "{question}"

Below is a structured action record of an agent session, with tool calls shown as name + salient arguments and turn ordering. Tool results are NOT included. Answer the question based ONLY on this record. If it does not contain enough information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

ACTION RECORD (no result digests):
{nd}

Provide your answer and confidence now:""", len(nd)
    return f"""Someone asks: "{question}"

Below is the transcript of an agent session, with tool outputs removed. Answer the question based ONLY on this transcript. If the transcript does not contain enough information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

TRANSCRIPT:
{st}

Provide your answer and confidence now:""", len(st)


def grade(q, answer, conf, raw, acc):
    stage = "grade_zero" if acc["arm"] == "zero" else "grade"
    def call(p):
        t, u = call_paid(stage, p, SONNET, 1024, 180)
        acc["usd"] += usd(u.get("prompt_tokens", 0), u.get("completion_tokens", 0)); return t
    draws = []
    def one():
        for _ in range(2):   # at most one re-draw of a failed/unparseable draw
            try:
                return gc.grade_with(call, q["question"], q["ground_truth"], answer, confidence=conf, raw_reply=raw)
            except gc.GradeError:
                continue
        return None
    first = one()
    if first is None: return "grade_error", draws
    if first["grader_response"] is None:       # rule absent, no model call
        return first["grade"], []
    draws.append(first["grade"])
    for _ in range(2):
        r = one()
        if r is None: return "grade_error", draws
        draws.append(r["grade"])
    c = Counter(draws).most_common()
    if len(c) > 1 and c[0][1] == c[1][1] and c[0][1] == 1:
        for _ in range(2):
            r = one()
            if r is None: return "grade_error", draws
            draws.append(r["grade"])
        c = Counter(draws).most_common()
    if len(c) == 1 or c[0][1] > c[1][1]:
        return c[0][0], draws
    return "unresolved", draws


def do(job):
    q, arm, first = job
    prompt, chars = prompt_for(arm, q)
    raw, u = call_paid("answer_" + arm, prompt, SONNET, 2048, 600)
    cost = usd(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
    if not raw:
        with lk, FAIL.open("a") as f:
            f.write(json.dumps({"question_id": q["question_id"], "arm": arm, "usage": u}) + "\n")
        return ("api_error", arm)
    ans, conf = extract_confidence(raw)
    acc = {"usd": 0.0, "arm": arm}
    g, draws = grade(q, ans, conf, raw, acc)
    row = {"question_id": q["question_id"], "session_id": q["session_id"], "session_prefix": q["session_prefix"],
           "class": q["class"], "arm": arm, "artifact_chars": chars, "answer": ans, "rating": conf, "raw_reply": raw,
           "grade": g, "grade_draws": draws, "grade_protocol": PROTOCOL, "order_seed": ORDER_SEED,
           "arm_first": first, "cost": round(cost + acc["usd"], 5), "usage": u, "answer_model": SONNET, "judge_model": SONNET}
    if g in ("grade_error",):
        with lk, FAIL.open("a") as f: f.write(json.dumps(row) + "\n")
        return ("grade_error", arm)
    with lk, ANS.open("a") as f: f.write(json.dumps(row) + "\n")
    return (g, arm)


def estimate(qs, arms):
    tot = 0.0
    for q in qs:
        for a in arms:
            if a == "zero": tot += (300 * 3 + 200 * 15) / 1e6 + 0.0015; continue
            nd, st = artifacts(q["transcript"])
            ch = len(nd) if a == "nodigest" else len(st)
            tot += ((ch / BPT + 250) * 3 + 500 * 15) / 1e6 + 3 * (900 * 3 + 5 * 15) / 1e6
    return tot * 1.42   # calibrated: pilot 1.25x raw, first full batch 1.42x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode"); ap.add_argument("--lo", type=int, default=0); ap.add_argument("--hi", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    qs = verify_frozen()
    done = {(json.loads(l)["question_id"], json.loads(l)["arm"]) for l in ANS.open()} if ANS.exists() else set()
    if a.mode == "zero":
        arms, sel, stage = ["zero"], qs, "zero"
    elif a.mode == "pilot":
        arms, sel, stage = ["nodigest", "stripped"], qs[:10], "pilot"
    else:
        arms, sel, stage = ["nodigest", "stripped"], qs[a.lo:a.hi or len(qs)], f"full_{a.lo}_{a.hi}"
    rng = random.Random(ORDER_SEED)
    jobs = []
    for q in qs:                       # draw order for every question, so it is independent of batching
        arm_order = list(arms) if len(arms) == 1 else (["nodigest", "stripped"] if rng.random() < .5 else ["stripped", "nodigest"])
        if q in sel:
            for arm in arm_order:
                if (q["question_id"], arm) not in done:
                    jobs.append((q, arm, arm_order[0]))
    # biggest first is irrelevant; keep question order so answer pairs finish together
    est = estimate([j[0] for j in jobs], sorted({j[1] for j in jobs})) if False else sum(estimate([j[0]], [j[1]]) for j in jobs)
    gate(stage, est)
    before = cumulative_usd(); import time; t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(do, jobs))
    print(Counter(r[0] for r in res))
    log_batch(stage, est, note=f"jobs={len(jobs)} outcomes={dict(Counter(r[0] for r in res))} batch_usd={cumulative_usd()-before:.3f}", since=t0)


if __name__ == "__main__":
    main()
