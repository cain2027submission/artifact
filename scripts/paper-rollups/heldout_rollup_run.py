#!/usr/bin/env python3
"""Held-out rollup arm: answer the 125 held-out questions from the production
session note (the design-set rollup arm's own rollup), then grade with the
3-draw majority protocol of data/paper-rollups/regrade-2026-10-09/regrade_s4.py.
Paid; resumable; every call is costed in heldout-rollup/cost_log.jsonl and a
batch whose estimate would push the running total past CAP_USD is not run.

Usage: heldout_rollup_run.py --dry-run | --limit N [--out PATH] | (full run)
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grading_common as gc  # noqa: E402
from rq3_raw_transcript_arm import DATA, call_llm, load_records  # noqa: E402
from rq3_uniform_rollup_arm import answer_from_rollup, JUDGE_MODEL  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HD = DATA / "heldout-rollup"
QUESTIONS = DATA / "rq3_heldout_questions.jsonl"
OUT = HD / "rollup_arm.jsonl"
COST = HD / "cost_log.jsonl"
CAP_USD = 12.0
PRICE_IN, PRICE_OUT = 3.0, 15.0  # USD per 1M tokens, assumed, not billed
PROTOCOL = f"{gc.PROMPT_VERSION}+{gc.PARSER_VERSION}+majority3+{JUDGE_MODEL}"
BATCH = 25


def rid(r):
    seed = f"{r.get('session_id','')}|{r.get('class','')}|{r.get('question','')}"
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def usd(u):
    return (u.get("prompt_tokens", 0) * PRICE_IN + u.get("completion_tokens", 0) * PRICE_OUT) / 1e6


def total():
    if not COST.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in COST.read_text().splitlines() if l.strip())


def log(stage, calls, ptok, ctok, est):
    with COST.open("a") as fh:
        fh.write(json.dumps({"stage": stage, "calls": calls, "prompt_tokens": ptok,
                             "completion_tokens": ctok, "usd": round(est, 5)}) + "\n")


def majority(g):
    c = Counter(g).most_common()
    return c[0][0] if len(c) == 1 or c[0][1] > c[1][1] else None


def one(q):
    """answer once, grade 3 draws (+2 on a tie). returns row, usage totals, calls."""
    calls = ptok = ctok = 0
    text = Path(q["_rollup"]).read_text(errors="replace")
    answer = conf = None
    for attempt in range(2):  # one retry at most
        answer, conf, u = answer_from_rollup(q["question"], text)
        calls += 1; ptok += u.get("prompt_tokens", 0); ctok += u.get("completion_tokens", 0)
        if conf != "API_ERROR":
            break
    if conf == "API_ERROR":
        return None, (calls, ptok, ctok), q["_rid"]

    def grade_draw():
        nonlocal calls, ptok, ctok
        def call(p):
            nonlocal calls, ptok, ctok
            r, u = call_llm(p, timeout=120)
            calls += 1; ptok += u.get("prompt_tokens", 0); ctok += u.get("completion_tokens", 0)
            return r
        for attempt in range(2):
            try:
                return gc.grade_with(call, q["question"], q["ground_truth"], answer, conf)["grade"]
            except gc.GradeError:
                continue
        return gc.GRADE_ERROR

    draws = [grade_draw() for _ in range(3)]
    valid = [g for g in draws if g != gc.GRADE_ERROR]
    maj = majority(valid) if len(valid) >= 2 else None
    if maj is None and len(valid) >= 2:
        draws += [grade_draw() for _ in range(2)]
        valid = [g for g in draws if g != gc.GRADE_ERROR]
        maj = majority(valid)
    if maj is None:
        return None, (calls, ptok, ctok), q["_rid"]
    row = {"record_id": q["_rid"], "question_id": q["_rid"], "session_id": q["session_id"],
           "session_prefix": q["session_prefix"], "class": q["class"], "question": q["question"],
           "ground_truth": q["ground_truth"], "arm": "rollup_production_note",
           "rollup": f"sessions/{q['session_id']}.md", "rollup_chars": len(text),
           "answer": answer, "rating": conf, "grade": maj, "grade_draws": draws[:3],
           "grade_protocol": PROTOCOL, "model": JUDGE_MODEL}
    if len(draws) > 3:
        row["grade_tiebreak_draws"] = draws[3:]
    return row, (calls, ptok, ctok), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--stage", default="full")
    args = ap.parse_args()
    out = Path(args.out)
    qs = load_records(QUESTIONS)
    for q in qs:
        q["_rid"] = rid(q)
        q["_rollup"] = str(ROOT / "sessions" / f"{q['session_id']}.md")
        assert Path(q["_rollup"]).exists(), q["_rollup"]
    done = {json.loads(l)["record_id"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    todo = [q for q in qs if q["_rid"] not in done]
    if args.limit:
        # spread across sessions, smallest and largest included
        todo = sorted(todo, key=lambda q: Path(q["_rollup"]).stat().st_size)
        todo = [todo[0], todo[-1], todo[len(todo)//2], todo[len(todo)//4], todo[3*len(todo)//4]][:args.limit]
    # estimate: answer prompt + 3 grade prompts (+ ~15% for ties), output ~150 answer / 5 grade tokens
    chars = lambda q: Path(q["_rollup"]).stat().st_size
    def est(qs_):
        pin = sum(chars(q) / 3.2 + 150 + 3 * (len(q["question"]) + len(q["ground_truth"]) + 700) / 3.2 * 1.15 for q in qs_)
        return (pin * PRICE_IN + len(qs_) * (200 + 3 * 5) * PRICE_OUT) / 1e6
    print(f"records {len(qs)}  done {len(done)}  todo {len(todo)}  full-todo estimate ${est(todo):.3f}  running total ${total():.3f}")
    if args.dry_run:
        return
    errs = []
    for b in range(0, len(todo), BATCH):
        batch = todo[b:b + BATCH]
        e = est(batch)
        print(f"batch {b//BATCH+1}: estimate ${e:.3f}, running total ${total():.3f}, cap ${CAP_USD}")
        if total() + e > CAP_USD:
            print("STOP: batch would exceed cap"); sys.exit(2)
        t0 = time.time()
        with ThreadPoolExecutor(5) as ex:
            res = list(ex.map(one, batch))
        calls = ptok = ctok = 0
        for row, (c, p, k), err in res:
            calls += c; ptok += p; ctok += k
            if row:
                with out.open("a") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                errs.append(err)
        spent = (ptok * PRICE_IN + ctok * PRICE_OUT) / 1e6
        log(args.stage, calls, ptok, ctok, spent)
        print(f"  batch done {time.time()-t0:.0f}s calls {calls} tokens {ptok}+{ctok} est ${spent:.3f}; total ${total():.3f}; failed {len(errs)}")
    if errs:
        print("FAILED (excluded, rerun to retry once more is NOT allowed beyond the in-run retry):", errs)


if __name__ == "__main__":
    main()
