#!/usr/bin/env python3
"""Later-window retention experiment (EXPERIMENT 1): does the rule cut's recall
lead over the rollup hold for questions drawn from the MIDDLE (30-60%) and END
(60-100%) of a session, not only its first 30,000 characters?

Stages (each resumable; every model reply is cached on disk before use):
  select --n N          rank eligible sessions, write manifest.json
  gen --windows mid,late [--first K]   generate 5 questions per session-window
  answer --arms ... [--windows ...] [--first K]
  grade [--windows ...] [--first K]
  cost                  spend from the ledger

Reuses, unchanged: the published generator prompt and model
(part2_question_retention.generate_questions, now with an `offset`), the arm
builders and answer prompts (rq3_uniform_rollup_arm, rq3_stripped_transcript_arm,
rq3_prose_only_arm, rq3_tool_inputs_only_arm, rq3_zero_context_arm), and the
revised grader (grading_common.grade_with, prompt v2-question, first-word parser).
Model ids are the existing scripts' own constants. No temperature is set.

Data: data/paper-rollups/late-window/{manifest.json,questions.jsonl,
answers.jsonl,grades.jsonl,ledger.jsonl,gen_failures.jsonl,answer_errors.jsonl}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
import urllib.error
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq3_late_common as lc  # noqa: E402
import part2_question_retention as p2  # noqa: E402
import grading_common as gc  # noqa: E402
from rq3_raw_transcript_arm import GATEWAY, post  # noqa: E402
from rq3_uniform_rollup_arm import answer_from_rollup  # noqa: E402
from rq3_stripped_transcript_arm import (  # noqa: E402
    strip_transcript, answer_from_stripped, CONTEXT_LIMIT_TOKENS, BYTES_PER_TOKEN)
from rq3_prose_only_arm import prose_only_context, answer_from_prose_only  # noqa: E402
from rq3_tool_inputs_only_arm import (  # noqa: E402
    tool_inputs_only_context, answer_from_tool_inputs_only)
from rq3_zero_context_arm import answer_zero_context  # noqa: E402

OUT = lc.OUTDIR
MANIFEST = OUT / "manifest.json"
QFILE = OUT / "questions.jsonl"
AFILE = OUT / "answers.jsonl"
GFILE = OUT / "grades.jsonl"
LEDGER = OUT / "ledger.jsonl"
GEN_FAIL = OUT / "gen_failures.jsonl"
ANS_ERR = OUT / "answer_errors.jsonl"
DRAWS = OUT / "grade_draws.jsonl"  # gate (b) points this elsewhere

GRADER_MODEL = p2.JUDGE_MODEL      # "claude-sonnet-5", as regrade-2026-10-09
GEN_MODEL = p2.QUESTION_MODEL      # "claude-sonnet-5"
ARMS = ["rollup", "stripped", "prose", "callonly", "noop"]
PRICE = {"claude-sonnet-5": (3.0, 15.0)}  # USD per 1M tokens, as the regrade assumed
# Disclosed deviation, the same one rq3_heldout_questions.py makes: 2048 output
# tokens truncates 5 questions with verbatim ground truths and silently drops the
# session. 4096 changes only whether the reply fits, not the prompt.
p2.QUESTION_MAX_TOKENS = 4096

lock = threading.Lock()


def append(path: Path, row: dict) -> None:
    with lock:
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")


def ledger(stage: str, model: str, usage: dict, ok: bool, tag: str) -> None:
    append(LEDGER, {"t": time.time(), "stage": stage, "model": model, "tag": tag, "ok": ok,
                    "in": (usage or {}).get("prompt_tokens", 0),
                    "out": (usage or {}).get("completion_tokens", 0)})


# ------------------------------------------------------------------ sessions
def preflight_rows():
    return json.loads((OUT / "preflight.json").read_text())["rows"]


def ctx_cost_chars(r):
    return r["stripped_chars"] + r["prose_chars"] + r["callonly_chars"]


def cmd_select(a):
    """Eligible = surviving transcript, pinned rollup, MID and LATE each >= 30k
    chars (harness entries excluded), no window slice >50% one entry, stripped
    context under the answerer's limit. Ranked by output-stripped context size
    ascending (the cost driver), ties by uuid; the first N are taken. This is a
    cost rule, stated in DESIGN.md."""
    rows = [r for r in preflight_rows() if r["eligible"]
            and r["stripped_chars"] / BYTES_PER_TOKEN <= CONTEXT_LIMIT_TOKENS]
    rows.sort(key=lambda r: (r["stripped_chars"], r["uuid"]))
    sel = rows[:a.n]
    MANIFEST.write_text(json.dumps({
        "rule": "eligible sessions ranked by output-stripped chars ascending, ties by uuid; first N",
        "n": len(sel), "n_eligible": len(rows),
        "sessions": [{k: r[k] for k in ("uuid", "prefix", "transcript", "pinned_bytes",
                                         "rollup_note", "gen_text_chars", "mid_offset",
                                         "late_offset", "delegating", "in_existing_68",
                                         "in_graded_25", "slug", "stripped_chars",
                                         "prose_chars", "callonly_chars", "rollup_chars")}
                     for r in sel]}, indent=1))
    print(f"selected {len(sel)} of {len(rows)} eligible; delegating "
          f"{sum(r['delegating'] for r in sel)}, from existing 68: "
          f"{sum(r['in_existing_68'] for r in sel)}, from graded 25: "
          f"{sum(r['in_graded_25'] for r in sel)}")


def manifest(first=0, only=None):
    s = json.loads(MANIFEST.read_text())["sessions"]
    if only:
        s = [x for x in s if x["prefix"] in only]
    return s[:first] if first else s


_raw_cache = {}


def raw_of(s):
    if s["uuid"] not in _raw_cache:
        _raw_cache[s["uuid"]] = lc.pinned_raw(s["transcript"], s["pinned_bytes"])
    return _raw_cache[s["uuid"]]


# ------------------------------------------------------------------ generation
def gen_call(prompt, model, max_tokens=2048):
    """p2.call_llm with the usage kept for the ledger. Same payload."""
    try:
        data = post(f"{GATEWAY}/v1/chat/completions",
                    {"model": model, "max_tokens": max_tokens,
                     "messages": [{"role": "user", "content": prompt}]}, timeout=300)
        ch = data["choices"][0]
        u = data.get("usage", {})
        text = (ch["message"]["content"] or "").strip()
        ok = bool(text) and ch.get("finish_reason") != "length"
        ledger("gen", model, u, ok, "gen")
        if ch.get("finish_reason") == "length":
            print(f"  WARNING: generator hit max_tokens={max_tokens}")
        return text
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
        ledger("gen", model, {}, False, f"error {str(e)[:80]}")
        return ""


# Spike 1 (SPIKE.md) found OUTCOME questions phrased as "by the end of the
# transcript" whose ground truth was the state at the END OF THE EXCERPT, so a
# whole-session arm answering with the true final state was graded wrong. The
# note below is the one fix; it is passed only for MID and LATE.
EXCERPT_NOTE = (
    "NOTE: the transcript below is an EXCERPT of {n:,} characters taken from the "
    "{where} of a longer session (characters {a:,} to {b:,} of {total:,}). The session "
    "continued after this excerpt, so do not ask about the final state of the session "
    "or about what happened 'by the end of the transcript'. Every question, including "
    "the OUTCOME question, must be about something that happened and was concluded "
    "within this excerpt (for OUTCOME: the result of a piece of work completed inside "
    "the excerpt), phrased so that it has the same answer whether or not one knows "
    "what happened later.\n\n")
WHERE = {"mid": "middle (30-60%)", "late": "later part (60-100%)"}


def valid_set(qs):
    if not isinstance(qs, list) or len(qs) != 5:
        return False
    cls = sorted((q.get("class") or "").lower() for q in qs if isinstance(q, dict))
    return cls == sorted(lc.CLASSES) and all(q.get("question") and q.get("ground_truth") for q in qs)


def cmd_gen(a):
    p2.call_llm = gen_call  # usage capture only; prompt and model unchanged
    done = {(q["uuid"], q["window"]) for q in lc.read_jsonl(QFILE)}
    for s in manifest(a.first, a.only):
        text = lc.generator_text(raw_of(s))
        assert len(text) == s["gen_text_chars"], (s["prefix"], len(text), s["gen_text_chars"])
        for w in a.windows:
            if (s["uuid"], w) in done:
                continue
            off = s[f"{w}_offset"]
            qs = None
            for attempt in range(2):
                note = EXCERPT_NOTE.format(n=lc.WINDOW_CHARS, where=WHERE[w], a=off,
                                           b=off + lc.WINDOW_CHARS, total=len(text))
                qs = p2.generate_questions(text, s["prefix"], offset=off, excerpt_note=note)
                if valid_set(qs):
                    break
                append(GEN_FAIL, {"uuid": s["uuid"], "window": w, "attempt": attempt,
                                  "got": qs if isinstance(qs, list) else None})
            if not valid_set(qs):
                print(f"  {s['prefix']} {w}: no valid 5-class set after 2 attempts, skipped")
                continue
            for q in qs:
                cls = q["class"].lower()
                append(QFILE, {"qid": lc.qid(s["uuid"], w, cls, q["question"]),
                               "uuid": s["uuid"], "prefix": s["prefix"], "window": w,
                               "offset": off, "window_chars": lc.WINDOW_CHARS,
                               "class": cls, "question": q["question"],
                               "ground_truth": q["ground_truth"], "model": GEN_MODEL,
                               "gen_max_tokens": p2.QUESTION_MAX_TOKENS,
                               "excerpt_note": True})
            print(f"  {s['prefix']} {w}: 5 questions (offset {off:,} of {len(text):,})")


# ------------------------------------------------------------------ answering
_ctx = {}


def contexts(s):
    if s["uuid"] not in _ctx:
        raw = raw_of(s)
        _ctx[s["uuid"]] = {
            "rollup": lc.pinned_rollup(s["rollup_note"]),
            "stripped": strip_transcript(raw)[0],
            "prose": prose_only_context(raw)[0],
            "callonly": tool_inputs_only_context(raw)[0],
        }
    return _ctx[s["uuid"]]


def answer_one(arm, q, c):
    if arm == "rollup":
        return answer_from_rollup(q, c["rollup"])
    if arm == "stripped":
        return answer_from_stripped(q, c["stripped"])
    if arm == "prose":
        return answer_from_prose_only(q, c["prose"])
    if arm == "callonly":
        return answer_from_tool_inputs_only(q, c["callonly"])
    if arm == "noop":
        return answer_zero_context(q)
    raise ValueError(arm)


def cmd_answer(a):
    sess = {s["uuid"]: s for s in manifest(a.first, a.only)}
    done = {(r["qid"], r["arm"]) for r in lc.read_jsonl(AFILE)}
    qs = [q for q in lc.read_jsonl(QFILE) if q["uuid"] in sess and q["window"] in a.windows]
    todo = [(q, arm) for q in qs for arm in a.arms if (q["qid"], arm) not in done]
    print(f"answers to make: {len(todo)}")
    for u in {q["uuid"] for q, _ in todo}:
        contexts(sess[u])  # build serially; the pool only makes calls

    def work(item):
        q, arm = item
        if total_usd() > HARD_STOP_USD:
            append(ANS_ERR, {"qid": q["qid"], "arm": arm, "skipped": "hard stop"})
            return False
        c = contexts(sess[q["uuid"]])
        for attempt in range(2):
            ans, conf, usage = answer_one(arm, q["question"], c)
            ledger("answer", p2.JUDGE_MODEL, usage, conf != "API_ERROR", f"{arm}|{q['qid']}")
            if conf != "API_ERROR":
                append(AFILE, {"qid": q["qid"], "arm": arm, "uuid": q["uuid"],
                               "window": q["window"], "class": q["class"],
                               "answer": ans, "confidence": conf, "usage": usage,
                               "context_chars": len(c[arm]) if arm in c and c[arm] else 0,
                               "model": p2.JUDGE_MODEL})
                return True
            time.sleep(5 * (attempt + 1))
        append(ANS_ERR, {"qid": q["qid"], "arm": arm, "usage": usage})
        return False

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(work, todo))
    print(f"answered {sum(res)}, API errors after retry {len(res) - sum(res)}")


# ------------------------------------------------------------------ grading
def grade_call(prompt):
    """Same payload as regrade-2026-10-09 call_model: 1024 cap, widened to 4096
    once on truncation, up to 4 attempts on a gateway error."""
    cap = 1024
    for attempt in range(4):
        try:
            d = post(f"{GATEWAY}/v1/chat/completions",
                     {"model": GRADER_MODEL, "max_tokens": cap,
                      "messages": [{"role": "user", "content": prompt}]}, timeout=180)
            ch = d["choices"][0]
            text = (ch["message"]["content"] or "").strip()
            fin = ch.get("finish_reason")
            ok = bool(text) and fin != "length"
            ledger("grade", GRADER_MODEL, d.get("usage", {}), ok, "grade")
            if fin == "length" and cap == 1024:
                cap = 4096
                continue
            return text if ok else ""
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
            ledger("grade", GRADER_MODEL, {}, False, f"error {str(e)[:80]}")
            time.sleep(2 * (attempt + 1))
    return ""


def majority(gs):
    c = Counter(gs).most_common()
    if len(c) == 1 or c[0][1] > c[1][1]:
        return c[0][0]
    return None


def grade_draw(q, ans, draw, cache):
    k = f"{q['qid']}|{ans['arm']}|{draw}"
    if k in cache:
        return cache[k]["grade"]
    try:
        frag = gc.grade_with(grade_call, q["question"], q["ground_truth"], ans["answer"],
                             confidence=ans["confidence"])
        g = frag["grade"]
    except gc.GradeError as e:
        # A failed call is never scored and never cached, so a rerun retries it.
        append(OUT / "grade_errors.jsonl", {"key": k, "response": (e.response or "")[:200]})
        return gc.GRADE_ERROR
    row = {"key": k, "qid": q["qid"], "arm": ans["arm"], "draw": draw, "grade": g,
           "rule": frag["grader_response"] is None,
           "grader_response": frag["grader_response"],
           "answer_sha": hashlib.sha256(ans["answer"].encode()).hexdigest()[:12],
           "prompt_version": gc.PROMPT_VERSION, "parser_version": gc.PARSER_VERSION,
           "model": GRADER_MODEL}
    append(DRAWS, row)
    cache[k] = row
    return g


def grade_record(q, ans, cache):
    """Majority of three draws. A rule-absent answer (first-line refusal,
    NOT_ANSWERABLE, empty) is `absent` with no call. Draws 0 and 1 are made
    first; when they agree the majority of three is already decided, so draw 2
    is made only when they differ, and draws 3-4 only on a three-way split
    (as regrade-2026-10-09). A draw that errors is retried as a fresh draw
    index, never scored."""
    if gc.is_rule_absent(ans["answer"], ans["confidence"]):
        return {"grade": "absent", "draws": ["absent(rule)"], "rule": True}
    valid, idx = [], 0
    while idx < 8:
        if len(valid) == 2 and valid[0] == valid[1]:
            break  # two of three already agree
        if len(valid) >= 3 and majority(valid) is not None:
            break
        if len(valid) >= 5:
            break
        g = grade_draw(q, ans, idx, cache)
        idx += 1
        if g != gc.GRADE_ERROR:
            valid.append(g)
    m = majority(valid) if len(valid) >= 2 else None
    return {"grade": m, "draws": valid, "rule": False}


def cmd_grade(a):
    sess = {s["uuid"] for s in manifest(a.first, a.only)}
    qs = {q["qid"]: q for q in lc.read_jsonl(QFILE) if q["uuid"] in sess and q["window"] in a.windows}
    answers = [r for r in lc.read_jsonl(AFILE) if r["qid"] in qs]
    cache = {r["key"]: r for r in lc.read_jsonl(DRAWS)}

    def work(ans):
        q = qs[ans["qid"]]
        r = grade_record(q, ans, cache)
        return {"qid": q["qid"], "uuid": q["uuid"], "prefix": q["prefix"],
                "window": q["window"], "class": q["class"], "arm": ans["arm"],
                "question": q["question"], "ground_truth": q["ground_truth"],
                "answer": ans["answer"], "confidence": ans["confidence"],
                "grade": r["grade"], "grade_draws": r["draws"], "rule_absent": r["rule"],
                "grade_protocol": f"{gc.PROMPT_VERSION}+{gc.PARSER_VERSION}+majority3-earlystop+{GRADER_MODEL}"}

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(work, answers))
    rows.sort(key=lambda r: (r["window"], r["prefix"], r["class"], r["arm"]))
    # grades.jsonl is a pure function of the caches; rewritten whole each run.
    GFILE.write_text("".join(json.dumps(r) + "\n" for r in rows))
    unres = [r for r in rows if r["grade"] is None]
    print(f"graded {len(rows)}; unresolved (no majority or failed draws) {len(unres)}")
    print(Counter((r["window"], r["arm"], r["grade"]) for r in rows))


# ------------------------------------------------------------------ cost
def spend(rows=None):
    rows = lc.read_jsonl(LEDGER) if rows is None else rows
    out = {}
    for r in rows:
        pi, po = PRICE.get(r["model"], (3.0, 15.0))
        st = out.setdefault(r["stage"], {"calls": 0, "failed": 0, "in": 0, "out": 0, "usd": 0.0})
        st["calls"] += 1
        st["failed"] += (not r["ok"])
        st["in"] += r["in"]
        st["out"] += r["out"]
        st["usd"] += r["in"] / 1e6 * pi + r["out"] / 1e6 * po
    return out


HARD_STOP_USD = 65.0


def total_usd():
    """Spend across this run's ledger and the superseded spike 1."""
    rows = lc.read_jsonl(LEDGER) + lc.read_jsonl(OUT / "spike1" / "ledger.jsonl")
    return sum(v["usd"] for v in spend(rows).values())


def cmd_cost(a):
    s = spend()
    tot = {"calls": sum(v["calls"] for v in s.values()), "usd": sum(v["usd"] for v in s.values())}
    for k, v in s.items():
        print(f"{k:7s} calls {v['calls']:5d} failed {v['failed']:4d} in {v['in']:>11,} out {v['out']:>9,} ${v['usd']:.2f}")
    print(f"total calls {tot['calls']} ${tot['usd']:.2f} (this ledger); "
          f"with spike 1: ${total_usd():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--only", type=lambda s: s.split(","), default=None)
    ap.add_argument("--windows", type=lambda s: s.split(","), default=["mid", "late"])
    ap.add_argument("--arms", type=lambda s: s.split(","), default=ARMS)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    {"select": cmd_select, "gen": cmd_gen, "answer": cmd_answer, "grade": cmd_grade,
     "cost": cmd_cost}[a.stage](a)


if __name__ == "__main__":
    main()
