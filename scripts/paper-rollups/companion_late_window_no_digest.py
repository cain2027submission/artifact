#!/usr/bin/env python3
"""Companion test (b): no-digest action record on the late-window mid/late questions.
DESCRIPTIVE only, no pass/fail. Cap USD 10 (estimates run ~1.5x real spend).

Pre-registration: tasks/2026-10-13 a fresh-session replication decides whether the no-digest
action record can be the primary recommendation, pre-registered before any run.md

Same questions as data/paper-rollups/late-window (those with an existing output-stripped answer:
LATE 17 sessions / 85 questions, MID 5 sessions / 25 questions), the output-stripped answers and
grades are REUSED. Only the `nodigest` arm is bought: answered with
rq3_action_record_no_digest_arm.answer_from_record over build_action_record_no_result_digests
(built from the same pinned transcript bytes as the other arms), graded with the late-window
protocol (rq3_late_window.grade_record: v2 prompt, firstword parser, 3-draw majority, early stop).

Stages: estimate | answer | grade | analyse   (cost log: companion/cost_log.jsonl)
"""
from __future__ import annotations
import argparse, json, sys, tempfile, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq3_late_window as lw  # noqa: E402
import rq3_late_common as lc  # noqa: E402
import rq3_late_analyse as la  # noqa: E402
import cluster_bootstrap as cb  # noqa: E402
from rq3_action_record_no_digest_arm import answer_from_record  # noqa: E402
from spinoff_actionrecord_ablation import build_action_record_no_result_digests  # noqa: E402

DATA = lc.DATA
CO = DATA / "companion"
CO.mkdir(exist_ok=True)
COST = CO / "cost_log.jsonl"
CAP = 10.0
ARM = "nodigest"
lw.OUT = CO / "late_window_nd"          # keep this test's ledger, draws and errors separate
lw.OUT.mkdir(exist_ok=True)
lw.LEDGER = lw.OUT / "ledger.jsonl"
lw.DRAWS = lw.OUT / "grade_draws.jsonl"
AFILE = lw.OUT / "answers.jsonl"
GFILE = lw.OUT / "grades.jsonl"
RESULT = CO / "late_window_no_digest.json"
PRICE = (3.0, 15.0)


def usd(i, o):
    return i / 1e6 * PRICE[0] + o / 1e6 * PRICE[1]


def test_b_total():
    return sum(r["usd"] for r in lc.read_jsonl(COST) if r.get("test") == "b")


def log_cost(stage, rows):
    i = sum(r["in"] for r in rows); o = sum(r["out"] for r in rows)
    row = {"t": time.time(), "test": "b", "stage": stage, "calls": len(rows), "in_tokens": i,
           "out_tokens": o, "usd": round(usd(i, o), 4)}
    with COST.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print("cost log:", row, "running total test b $%.2f of $%.0f" % (test_b_total(), CAP))


def scope():
    """Questions with an existing output-stripped answer, per window."""
    stripped = {r["qid"] for r in lc.read_jsonl(lc.OUTDIR / "answers.jsonl") if r["arm"] == "stripped"}
    qs = [q for q in lc.read_jsonl(lc.OUTDIR / "questions.jsonl") if q["qid"] in stripped]
    man = {s["uuid"]: s for s in json.loads((lc.OUTDIR / "manifest.json").read_text())["sessions"]}
    return qs, man


_rec = {}


def record(s):
    if s["uuid"] not in _rec:
        raw = lc.pinned_raw(s["transcript"], s["pinned_bytes"])
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write(raw)
        _rec[s["uuid"]] = build_action_record_no_result_digests(f.name)[0]
        Path(f.name).unlink()
    return _rec[s["uuid"]]


def measured_cpt():
    """chars per token measured on this test's own answers so far (x0.9 safety), else 2.5."""
    r = lc.read_jsonl(AFILE)
    if not r:
        return 2.5
    return 0.9 * sum(x["context_chars"] for x in r) / sum(x["usage"]["prompt_tokens"] for x in r)


def estimate(max_q=0):
    qs, man = scope()
    done = {r["qid"] for r in lc.read_jsonl(AFILE)}
    qs = sorted([q for q in qs if q["qid"] not in done], key=lambda q: (len(record(man[q["uuid"]])), q["uuid"], q["window"], q["qid"]))
    if max_q:
        qs = qs[:max_q]
    cpt = measured_cpt()
    ans_in = ans_out = 0
    for q in qs:
        ans_in += len(record(man[q["uuid"]])) / cpt + 150
        ans_out += 550  # measured 534/answer in batch 1
    # conservative 2.5 chars/token (output-stripped measured 2.35 in this set, 3.74 is the record's own ratio).
    # grading: measured 735 in / 28 out per draw in the late-window ledger; use 800/100, 2.4 draws/answer
    g_in = len(qs) * 2.4 * 800; g_out = len(qs) * 2.4 * 100
    a, g = usd(ans_in, ans_out), usd(g_in, g_out)
    cum, acc = [], 0.0
    for q in qs:
        acc += usd(len(record(man[q["uuid"]])) / cpt + 150, 550)
        cum.append(round(acc, 2))
    print("cumulative answer usd by question count (ascending session size):", {i + 1: c for i, c in enumerate(cum) if (i + 1) % 5 == 0})
    print(f"chars/token used {cpt:.2f}")
    print(f"questions {len(qs)}  sessions {len({q['uuid'] for q in qs})}")
    print(f"answer est: in {ans_in:,.0f} tok out {ans_out:,.0f} = ${a:.2f}; grade est ${g:.2f}; total ${a+g:.2f}; "
          f"already logged ${test_b_total():.2f}; cap ${CAP}")
    return a, g, qs


def answer(workers, max_q):
    a, g, todo = estimate(max_q)
    if test_b_total() + a + g > CAP:
        print("WOULD EXCEED CAP, not running"); return
    _, man = scope()
    for u in {q["uuid"] for q in todo}:
        record(man[u])
    made = []

    def work(q):
        text = record(man[q["uuid"]])
        for attempt in range(2):     # one retry only
            ans, conf, usage = answer_from_record(q["question"], text)
            made.append({"in": usage.get("prompt_tokens", 0), "out": usage.get("completion_tokens", 0)})
            if conf != "API_ERROR":
                with lw.lock:
                    with AFILE.open("a") as f:
                        f.write(json.dumps({"qid": q["qid"], "arm": ARM, "uuid": q["uuid"], "window": q["window"],
                                            "class": q["class"], "answer": ans, "confidence": conf,
                                            "usage": usage, "context_chars": len(text)}) + "\n")
                return True
            time.sleep(5)
        return False
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(work, todo))
    print(f"answered {sum(res)} of {len(todo)}, failed after one retry {len(res)-sum(res)}")
    log_cost("answer", made)


def grade(workers):
    n_ans = len(lc.read_jsonl(AFILE))
    g = usd(n_ans * 2.4 * 800, n_ans * 2.4 * 100)
    print(f"grade est for {n_ans} answers (no cached draws assumed): ${g:.2f}; logged ${test_b_total():.2f}")
    if test_b_total() + g > CAP:
        print("WOULD EXCEED CAP, not running"); return
    qs, _ = scope()
    qd = {q["qid"]: q for q in qs}
    answers = lc.read_jsonl(AFILE)
    cache = {r["key"]: r for r in lc.read_jsonl(lw.DRAWS)}
    before = len(lc.read_jsonl(lw.LEDGER))

    def work(ans):
        q = qd[ans["qid"]]
        r = lw.grade_record(q, ans, cache)
        return {"qid": q["qid"], "uuid": q["uuid"], "prefix": q["prefix"], "window": q["window"],
                "class": q["class"], "arm": ARM, "answer": ans["answer"], "confidence": ans["confidence"],
                "grade": r["grade"], "grade_draws": r["draws"], "rule_absent": r["rule"]}
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(work, answers))
    GFILE.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("graded", len(rows), "unresolved", sum(r["grade"] is None for r in rows))
    new = lc.read_jsonl(lw.LEDGER)[before:]
    log_cost("grade", [{"in": r["in"], "out": r["out"]} for r in new])


def analyse():
    nd = lc.read_jsonl(GFILE)
    out = {"pre_registration": "tasks/2026-10-13 a fresh-session replication ... pre-registered before any run",
           "design": "descriptive only, no pass/fail; session-clustered paired bootstrap 95% CI "
                     "(cluster_bootstrap.Boot, B=10000, seed 20261008); output-stripped answers and grades reused",
           "windows": {}}
    for w in ("mid", "late"):
        arms = {"stripped": la.late_rows(w)["stripped"],
                ARM: [{"rid": r["qid"], "session": r["prefix"], "class": r["class"], "grade": r["grade"]}
                      for r in nd if r["window"] == w and r["grade"] is not None]}
        qids = {r["rid"] for r in arms["stripped"]} & {r["rid"] for r in arms[ARM]}
        arms = {k: [r for r in v if r["rid"] in qids] for k, v in arms.items()}
        p = la.paired(arms, ARM, "stripped")
        d = {k: la.describe(v) for k, v in arms.items()}
        out["windows"][w] = {"n_paired": len(qids), "paired_nodigest_minus_stripped": p, "descriptive": d,
                             "absent_rate_pct": {k: round(100 * d[k]["absent"] / d[k]["n"], 1) for k in d},
                             "wrong_given_attempted_pct": {k: d[k]["wrong_of_attempted_pct"] for k in d}}
    out["coverage"] = {"questions_in_scope": 110, "answered_and_graded": len(nd),
                       "reason_partial": "cap: remaining 30 questions (largest sessions) estimated above the USD 10 cap; run in ascending record size, whole sessions"}
    out["cost_usd_estimate"] = round(test_b_total(), 2)
    out["cap_usd"] = CAP
    RESULT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage"); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--max-q", type=int, default=0)
    a = ap.parse_args()
    {"estimate": lambda: estimate(a.max_q), "answer": lambda: answer(a.workers, a.max_q),
     "grade": lambda: grade(a.workers), "analyse": analyse}[a.stage]()
