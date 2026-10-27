#!/usr/bin/env python3
"""EXPERIMENT 2 (CAIN RQ3): what a Sonnet-written summary recovers under the ~400-token
(1,500-character) read cap, against same-length rule cuts and the existing capped arms.

Design: data/paper-rollups/cap400/DESIGN.md (committed before the full run).

Arms (all answered by claude-sonnet-5, graded by the revised grader: grading_common's
prompt + first-word parser, claude-sonnet-5 judge, majority of three with two more draws
on a 1-1-1 split, a failed grader call retried and never scored, refusal = absent):
  sonnet400   claude-sonnet-5 summary of the output-stripped transcript, the matched
              arm's prompt (rq3_sonnet_rollup_124.build_prompt_matched) with only the
              length target changed to 400 tokens / ~1,500 characters
  haiku400    same prompt, claude-haiku-4-5@20251001 (the production rollup model)
  head1500    prose-only context, first 1,500 characters (truncate_to_budget, word boundary)
  tail1500    prose-only context, last 1,500 characters (mirror of truncate_to_budget)
  zero        question only (rq3_zero_context_arm prompt)
  rollup      the production Haiku rollup note, whole (rq3_uniform_rollup_arm prompt)
  prose2k     prose-only first 2,000 characters (rq3_prose_only_budget_ladder)

Units: (arm, gen, draw). gen is 1/2 for the summary arms (independent generations), 0 for
the rest. Every generation, answer and grader draw is cached on disk; reruns resume.

Re-use on another question set:  --questions Q.jsonl --sessions S.jsonl --out-dir DIR
  Q.jsonl rows: record_id, session_id, class, question, ground_truth
  S.jsonl rows: session_id, transcript  (path to the session .jsonl)

Modes: --estimate | --generate | --answer | --analyse ; --arms ... --units arm:gen:draw ...
       --only SID-substring ...
Run: python3 scripts/paper-rollups/rq3_cap400_run.py ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROLLUP_SRC_DIR))

import grading_common as gc                                    # noqa: E402
from rq3_raw_transcript_arm import GATEWAY, extract_confidence, wilson  # noqa: E402
from rq3_stripped_transcript_arm import strip_transcript       # noqa: E402
from rq3_prose_only_arm import prose_only_context              # noqa: E402
from rq3_prose_only_budget_ladder import truncate_to_budget    # noqa: E402
import rq3_sonnet_rollup_124 as SR                             # noqa: E402
import rq3_long_rollup_spike as L                              # noqa: E402

REPO = HERE.parents[2]
DATA = REPO / "data" / "paper-rollups"

SONNET = "claude-sonnet-5"                   # generator (L.GEN_MODEL), answerer, judge
HAIKU = SR.HAIKU_MODEL                       # "claude-haiku-4-5@20251001"
assert L.GEN_MODEL == SONNET and L.ANSWER_MODEL == SONNET and L.JUDGE_MODEL == SONNET
PRICE = {SONNET: (3.0 / 1e6, 15.0 / 1e6), HAIKU: (1.0 / 1e6, 5.0 / 1e6)}

TARGET = 400
CHARS = 1500
# The matched arm's LENGTH_INSTR with only the length target changed: the token figure and
# a stated character figure. Everything else, including "a substantial paragraph or two",
# is byte-identical to rq3_sonnet_rollup_124.LENGTH_INSTR.
LENGTH_INSTR_400 = ("\n\nLength: write about {n} tokens (roughly {w} words, about 1,500 characters) in total across "
                    "the intent, thinking and outcome fields combined: never fewer than {lo} tokens and no more than {hi}. "
                    "Each field should be a substantial paragraph or two. Be selective: keep the decisive facts, numbers, "
                    "identifiers, files and dead ends, and drop the rest.")
assert SR.LENGTH_INSTR.replace("(roughly {w} words)", "(roughly {w} words, about 1,500 characters)") == LENGTH_INSTR_400
GEN_MAX_TOKENS = SR.MATCHED_MAX_TOKENS
# The one permitted retune (DESIGN.md length rule). Spike 1 medians with LENGTH_INSTR_400:
# sonnet400 593, haiku400 638 cl100k on 3 sessions (+48%, +60%), both outside 300-500. One
# shared retune for both arms, so the two generators still see the same prompt: state the
# character figure as a hard ceiling and replace "a substantial paragraph or two" with
# "one short paragraph". Nothing else differs.
LENGTH_INSTR_400_V2 = ("\n\nLength: write about {n} tokens (roughly {w} words, about 1,500 characters) in total across "
                       "the intent, thinking and outcome fields combined: never fewer than {lo} tokens and no more than {hi}. "
                       "This is a hard limit: a note longer than 1,500 characters is too long. Each field should be one "
                       "short paragraph. Be selective: keep the decisive facts, numbers, identifiers, files and dead ends, "
                       "and drop the rest.")
RETUNE = {"sonnet400": LENGTH_INSTR_400_V2, "haiku400": LENGTH_INSTR_400_V2}

SUMMARY_ARMS = {"sonnet400": SONNET, "haiku400": HAIKU}
ALL_ARMS = ("sonnet400", "haiku400", "head1500", "tail1500", "zero", "rollup", "prose2k")

lock = threading.Lock()


# ------------------------------------------------------------------ io
def jl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def append(p, row):
    with lock:
        with Path(p).open("a") as f:
            f.write(json.dumps(row) + "\n")


class Store:
    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.gen_f = self.out / "generations.jsonl"
        self.ans_f = self.out / "answers.jsonl"
        self.jud_f = self.out / "judge_draws.jsonl"
        self.led_f = self.out / "ledger.jsonl"
        self.graded_f = self.out / "graded.jsonl"
        self.gens = {(r["arm"], r["session_id"], r["gen"]): r for r in jl(self.gen_f)}
        self.ans = {(r["arm"], r["record_id"], r["gen"], r["draw"]): r for r in jl(self.ans_f)}
        self.jud = {}
        for r in jl(self.jud_f):
            self.jud.setdefault(r["key"], {})[r["jdraw"]] = r["grade"]


# ------------------------------------------------------------------ model calls
def call(store, prompt, model, kind, tag, max_tokens, timeout, need_stop=False):
    """One logical call -> (text, usage). Every HTTP attempt goes in the ledger.
    finish=length: one retry at double cap (call_llm / regrade convention). Network errors:
    up to 3 retries with backoff. Returns "" on failure; callers never score a failure."""
    cap = max_tokens
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                f"{GATEWAY}/v1/chat/completions",
                data=json.dumps({"model": model, "max_tokens": cap,
                                 "messages": [{"role": "user", "content": prompt}]}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                d = json.loads(resp.read())
            ch = d["choices"][0]
            u = d.get("usage", {})
            fin = ch.get("finish_reason")
            u["finish_reason"] = fin
            text = (ch["message"]["content"] or "").strip()
            ok = bool(text) and fin != "length" and (fin == "stop" or not need_stop)
            append(store.led_f, {"t": time.time(), "kind": kind, "model": model, "tag": tag,
                                 "in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0),
                                 "ok": ok, "finish": fin})
            if fin == "length" and cap == max_tokens:
                cap = max_tokens * 2
                continue
            if ok:
                return text, u
            if need_stop and attempt < 2:
                continue
            return "", u
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, OSError) as e:
            append(store.led_f, {"t": time.time(), "kind": kind, "model": model, "tag": tag,
                                 "in": 0, "out": 0, "ok": False, "error": str(e)[:160]})
            time.sleep(3 * (attempt + 1))
    return "", {}


# ------------------------------------------------------------------ inputs
def default_questions():
    return [{k: r[k] for k in ("record_id", "session_id", "class", "question", "ground_truth")}
            | {"rollup": r.get("rollup")} for r in jl(DATA / "rq3_rollup_uniform.jsonl")]


def default_sessions():
    out = {}
    for r in jl(DATA / "rq3_questions_stripped_arm.jsonl"):
        out.setdefault(r["session_id"], r["transcript"])
    return out


def load_inputs(a):
    qs = jl(a.questions) if a.questions else default_questions()
    if a.sessions:
        sp = Path(a.sessions)
        raw = json.loads(sp.read_text()) if sp.suffix == ".json" else jl(sp)
        sess = raw if isinstance(raw, dict) else {r["session_id"]: r["transcript"] for r in raw}
        sess = {k: (v["transcript"] if isinstance(v, dict) else v) for k, v in sess.items()}
    else:
        sess = default_sessions()
    if a.only:
        qs = [q for q in qs if any(o in q["session_id"] for o in a.only)]
    missing = sorted({q["session_id"] for q in qs} - set(sess))
    if missing:
        sys.exit(f"sessions with no transcript: {missing}")
    return qs, sess


_TX = {}


def texts(sid, path):
    if sid not in _TX:
        raw = Path(path).read_text(errors="replace")
        stripped, _ = strip_transcript(raw)
        prose, _ = prose_only_context(raw)
        _TX[sid] = {"stripped": stripped, "prose": prose}
    return _TX[sid]


def tail_cut(prose, budget):
    """Mirror of truncate_to_budget: LAST n chars, cut forward to a word boundary."""
    if len(prose) <= budget:
        return prose
    t = prose[-budget:]
    return "…" + (t.split(" ", 1)[1] if " " in t else t)


def rollup_text(q):
    p = q.get("rollup")
    if p and Path(p).exists():
        return Path(p).read_text()
    import rq3_uniform_rollup_arm as U
    f = U.find_rollup(q["session_id"])
    return f.read_text() if f else None


# ------------------------------------------------------------------ generation
def gen_prompt(arm, stripped):
    instr = RETUNE.get(arm, LENGTH_INSTR_400)
    return SR.build_prompt_matched(stripped, TARGET, instr)


def generate(store, qs, sess, arms, gens, workers):
    jobs = []
    for sid in sorted({q["session_id"] for q in qs}):
        for arm in arms:
            if arm not in SUMMARY_ARMS:
                continue
            for g in gens:
                if (arm, sid, g) not in store.gens:
                    jobs.append((arm, sid, g))
    print(f"generations to run: {len(jobs)}")

    def work(j):
        arm, sid, g = j
        tx = texts(sid, sess[sid])
        prompt = gen_prompt(arm, tx["stripped"])
        text, u = call(store, prompt, SUMMARY_ARMS[arm], "gen", f"{arm}|{sid}|{g}",
                       GEN_MAX_TOKENS, 900, need_stop=True)
        row = {"arm": arm, "session_id": sid, "gen": g, "model": SUMMARY_ARMS[arm],
               "target_tokens": TARGET, "retuned": arm in RETUNE, "ok": bool(text),
               "stripped_tokens": L.ntok(tx["stripped"]), "usage": u, "summary": text,
               "summary_cl100k": L.ntok(text) if text else None, "summary_chars": len(text)}
        if text:
            append(store.gen_f, row)
            store.gens[(arm, sid, g)] = row
        else:
            append(store.out / "generation_failures.jsonl", row)
        print(f"  gen {arm} {sid[:40]} g{g} ok={bool(text)} tok={row['summary_cl100k']} "
              f"in={u.get('prompt_tokens')} fin={u.get('finish_reason')}")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, jobs))


# ------------------------------------------------------------------ answering
def note_prompt(question, note):   # rq3_uniform_rollup_arm / rq3_long_rollup_spike, verbatim
    return f"""Someone asks: "{question}"

Below is a compressed note summarizing an agent session. Answer the question based ONLY on this note. If the note does not contain enough information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

NOTE:
{note}

Provide your answer and confidence now:"""


def prose_prompt(question, prose):  # rq3_prose_only_budget_ladder, verbatim
    return f"""Someone asks: "{question}"

Below is a transcript with only the prose (user and assistant text). Tool calls and their results have been removed. Answer the question based ONLY on this prose. If it does not contain enough information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

PROSE:
{prose}

Provide your answer and confidence now:"""


def zero_prompt(question):          # rq3_zero_context_arm, verbatim
    return f"""Someone asks: "{question}"

This question refers to a specific past agent coding session. You have been
given NO information about that session: no transcript, no filename, no
summary, no hint of any kind about what happened in it. Answer only if you
somehow already know the specific answer (you almost certainly do not, since
this refers to a private session you were never shown any part of). Do not
guess or invent plausible-sounding specifics. If you do not have enough
information to answer, say "NOT ANSWERABLE".

After your answer, on a new line, state your confidence as one of: ANSWERED, PARTIALLY_ANSWERED, or NOT_ANSWERABLE.

Provide your answer and confidence now:"""


def context_for(store, arm, gen, q, sess):
    """-> (prompt, context_text) or (None, reason)."""
    sid = q["session_id"]
    if arm in SUMMARY_ARMS:
        g = store.gens.get((arm, sid, gen))
        if not g:
            return None, "no_generation"
        return note_prompt(q["question"], g["summary"]), g["summary"]
    if arm == "zero":
        return zero_prompt(q["question"]), ""
    if arm == "rollup":
        t = rollup_text(q)
        if t is None:
            return None, "no_rollup"
        return note_prompt(q["question"], t), t
    prose = texts(sid, sess[sid])["prose"]
    ctx = {"head1500": lambda: truncate_to_budget(prose, CHARS),
           "tail1500": lambda: tail_cut(prose, CHARS),
           "prose2k": lambda: truncate_to_budget(prose, 2000)}[arm]()
    return prose_prompt(q["question"], ctx), ctx


def jkey(arm, gen, draw, rid, answer):
    return f"{arm}|{gen}|{draw}|{rid}|{hashlib.sha256(answer.encode()).hexdigest()[:12]}"


def grade(store, q, arm, gen, draw, answer, conf, raw):
    """Revised grader, majority of three. Draws stop once two agree (the third cannot
    change a 2-of-3 majority, so the verdict is identical to always drawing three);
    on a 1-1-1 split, two more draws. Returns (grade|None, draws, n_failed_calls)."""
    key = jkey(arm, gen, draw, q["record_id"], answer)
    cached = store.jud.setdefault(key, {})
    rule = gc.is_rule_absent(answer, conf) or gc.first_line_refusal(raw)
    draws, fails = [], 0
    while True:
        n = len(draws)
        if n:
            top = Counter(draws).most_common(1)[0][1]
            if rule or (n == 2 and top == 2) or (n == 3 and top >= 2) or n == 5:
                break
        jd = n
        if jd in cached:
            g = cached[jd]
        else:
            g, isrule = None, False
            for _ in range(3):
                try:
                    frag = gc.grade_with(lambda p: call(store, p, SONNET, "judge", key, 1024, 180)[0],
                                         q["question"], q["ground_truth"], answer, confidence=conf,
                                         raw_reply=raw)
                    g, isrule = frag["grade"], frag["grader_response"] is None
                    break
                except gc.GradeError:
                    fails += 1
            if g is None:
                return "grade_error", draws, fails
            append(store.jud_f, {"key": key, "jdraw": jd, "grade": g, "rule": isrule})
            cached[jd] = g
        draws.append(g)
    cnt = Counter(draws).most_common()
    if len(cnt) == 1 or cnt[0][1] > cnt[1][1]:
        return cnt[0][0], draws, fails
    return None, draws, fails


def answer_all(store, qs, sess, units, workers):
    todo = [(arm, gen, draw, q) for (arm, gen, draw) in units for q in qs
            if (arm, q["record_id"], gen, draw) not in store.ans]
    print(f"answers to run: {len(todo)}")

    def work(t):
        arm, gen, draw, q = t
        prompt, ctx = context_for(store, arm, gen, q, sess)
        if prompt is None:
            return
        raw, u = call(store, prompt, SONNET, "answer", f"{arm}|{gen}|{draw}|{q['record_id']}", 2048, 600)
        if not raw:
            append(store.out / "answer_failures.jsonl", {"arm": arm, "gen": gen, "draw": draw,
                                                        "record_id": q["record_id"], "usage": u})
            return
        ans, conf = extract_confidence(raw)
        g, draws, fails = grade(store, q, arm, gen, draw, ans, conf, raw)
        row = {"arm": arm, "gen": gen, "draw": draw, "record_id": q["record_id"], "session_id": q["session_id"],
               "class": q["class"], "question": q["question"], "ground_truth": q["ground_truth"],
               "context_chars": len(ctx), "context_cl100k": L.ntok(ctx) if ctx else 0,
               "raw_reply": raw, "answer": ans, "confidence": conf, "grade": g, "judge_draws": draws,
               "unresolved": g is None, "grade_error": g == "grade_error", "judge_failed_calls": fails,
               "usage": u, "answer_model": SONNET, "judge_model": SONNET,
               "grade_protocol": f"{gc.PROMPT_VERSION}+{gc.PARSER_VERSION}+majority3(early-stop)+{SONNET}"}
        if g == "grade_error":
            append(store.out / "grade_failures.jsonl", row)
            return
        append(store.ans_f, row)
        store.ans[(arm, q["record_id"], gen, draw)] = row

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, todo))


# ------------------------------------------------------------------ analysis
REC = ("correct", "partially_correct")
B, SEED = 10_000, 20261010


def series(store, arm, gen, draw):
    return {r["record_id"]: (r["session_id"], int((r["grade"] or "absent") in REC))
            for (a, _, g, d), r in store.ans.items() if a == arm and g == gen and d == draw}


def boot_diff(sa, sb):
    shared = sorted(set(sa) & set(sb))
    by = {}
    for rid in shared:
        by.setdefault(sa[rid][0], []).append((sa[rid][1], sb[rid][1]))
    ks = sorted(by)
    rng = random.Random(SEED)
    ds = []
    for _ in range(B):
        x = y = n = 0
        for k in (rng.choice(ks) for _ in ks):
            for a_, b_ in by[k]:
                x += a_; y += b_; n += 1
        ds.append(100 * (x - y) / n)
    ds.sort()
    point = 100 * sum(sa[r][1] - sb[r][1] for r in shared) / len(shared)
    p = 2 * min((sum(d <= 0 for d in ds) + 1) / (B + 1), (sum(d >= 0 for d in ds) + 1) / (B + 1))
    b_only = sum(sa[r][1] and not sb[r][1] for r in shared)
    c_only = sum(sb[r][1] and not sa[r][1] for r in shared)
    k = min(b_only, c_only)
    mc = min(1.0, 2 * sum(math.comb(b_only + c_only, i) for i in range(k + 1)) / 2 ** (b_only + c_only)) if b_only + c_only else 1.0
    return {"n": len(shared), "n_sessions": len(ks), "diff_pts": round(point, 1),
            "cluster95_pts": [round(ds[int(0.025 * B)], 1), round(ds[int(0.975 * B) - 1], 1)],
            "boot_p": round(min(1.0, p), 4), "mcnemar_exact_p": round(mc, 4),
            "discordant_a_only_b_only": [b_only, c_only]}


def arm_stats(store, arm, gen, draw):
    rows = [r for (a, _, g, d), r in store.ans.items() if a == arm and g == gen and d == draw]
    if not rows:
        return None
    n = len(rows)
    gr = [r["grade"] or "absent" for r in rows]
    rec, wrong, absent = sum(g in REC for g in gr), gr.count("wrong"), gr.count("absent")
    att = n - absent
    toks = [r["context_cl100k"] for r in rows]
    lo, hi = wilson(rec, n)
    return {"n": n, "n_sessions": len({r["session_id"] for r in rows}), "recoverable": rec,
            "pct": round(100 * rec / n, 1), "wilson95": [lo, hi],
            "wrong": wrong, "attempted": att,
            "wrong_of_attempted_pct": round(100 * wrong / att, 1) if att else None,
            "unresolved": sum(r["unresolved"] for r in rows),
            "median_context_cl100k_per_question": statistics.median(toks)}


def ledger_summary(store):
    led = jl(store.led_f)
    out = {}
    for r in led:
        k = (r["kind"], r["model"])
        o = out.setdefault(f"{k[0]}:{k[1]}", {"calls": 0, "ok": 0, "failed": 0, "in": 0, "out": 0, "usd": 0.0})
        o["calls"] += 1
        o["ok"] += bool(r["ok"])
        o["failed"] += not r["ok"]
        o["in"] += r["in"]; o["out"] += r["out"]
        pi, po = PRICE[r["model"]]
        o["usd"] += r["in"] * pi + r["out"] * po
    for o in out.values():
        o["usd"] = round(o["usd"], 3)
    tot = {"calls": sum(o["calls"] for o in out.values()), "failed": sum(o["failed"] for o in out.values()),
           "usd": round(sum(o["usd"] for o in out.values()), 2)}
    return out, tot


def analyse(store, out_json):
    units = sorted({(a, g, d) for (a, _, g, d) in store.ans})
    res = {"arms": {}, "generation_tokens": {}, "paired": {}, "spread": {}}
    for u in units:
        res["arms"][f"{u[0]}:g{u[1]}:d{u[2]}"] = arm_stats(store, *u)
    for (arm, g) in sorted({(a, g) for (a, _, g) in store.gens}):
        rows = [r for (a, _, gg), r in store.gens.items() if a == arm and gg == g]
        t = [r["summary_cl100k"] for r in rows]
        res["generation_tokens"][f"{arm}:g{g}"] = {
            "n_sessions": len(rows), "median": statistics.median(t), "mean": round(statistics.mean(t)),
            "min_max": [min(t), max(t)], "median_chars": statistics.median(r["summary_chars"] for r in rows),
            "pct_miss_of_400": round(100 * (statistics.median(t) - TARGET) / TARGET, 1)}
    S = {f"{a}:g{g}:d{d}": series(store, a, g, d) for (a, g, d) in units}
    prim = "sonnet400:g1:d1"
    for b in ("head1500:g0:d1", "haiku400:g1:d1", "rollup:g0:d1", "tail1500:g0:d1", "prose2k:g0:d1"):
        if prim in S and b in S:
            res["paired"][f"{prim} - {b}"] = boot_diff(S[prim], S[b])
    for a_, b_ in (("haiku400:g1:d1", "head1500:g0:d1"), ("haiku400:g1:d1", "rollup:g0:d1"),
                   ("head1500:g0:d1", "tail1500:g0:d1")):
        if a_ in S and b_ in S:
            res["paired"][f"{a_} - {b_}"] = boot_diff(S[a_], S[b_])
    # spread: answer-draw (same generation) and generator (different generation)
    for arm in SUMMARY_ARMS:
        for lab, x, y in (("answer_draw", f"{arm}:g1:d1", f"{arm}:g1:d2"),
                          ("generation", f"{arm}:g1:d1", f"{arm}:g2:d1")):
            if x in S and y in S:
                sh = set(S[x]) & set(S[y])
                flips = sum(S[x][r][1] != S[y][r][1] for r in sh)
                res["spread"][f"{arm} {lab}"] = {"n": len(sh), "pct_x": round(100 * sum(S[x][r][1] for r in sh) / len(sh), 1),
                                                 "pct_y": round(100 * sum(S[y][r][1] for r in sh) / len(sh), 1),
                                                 "recoverable_flips": flips, **{"paired": boot_diff(S[x], S[y])}}
    for b_ in ("head1500:g0:d2", "rollup:g0:d2"):
        a_ = b_.replace("d2", "d1")
        if a_ in S and b_ in S:
            sh = set(S[a_]) & set(S[b_])
            res["spread"][f"{b_.split(':')[0]} answer_draw"] = {
                "n": len(sh), "pct_x": round(100 * sum(S[a_][r][1] for r in sh) / len(sh), 1),
                "pct_y": round(100 * sum(S[b_][r][1] for r in sh) / len(sh), 1),
                "recoverable_flips": sum(S[a_][r][1] != S[b_][r][1] for r in sh)}
    # gate (b) and the run-to-run spread on fixed contexts: this pass vs the stored
    # regrade-2026-10-09 pass of the same arm (same context, fresh answer and grader draws)
    stored = {"rollup:g0:d1": ("rq3_rollup_uniform.jsonl", None, 69),
              "prose2k:g0:d1": ("rq3_prose_only_budget_ladder.jsonl", 2000, 47)}
    for u, (f, bud, published) in stored.items():
        if u not in S or not (DATA / f).exists():
            continue
        st = {r["record_id"]: (r["session_id"], int(r["grade"] in REC)) for r in jl(DATA / f)
              if bud is None or r.get("budget") == bud}
        sh = set(S[u]) & set(st)
        res["spread"][f"{u.split(':')[0]} this pass vs stored pass"] = {
            "n": len(sh), "this_pass": sum(S[u][r][1] for r in sh), "stored": sum(st[r][1] for r in sh),
            "published_recoverable": published,
            "gate_b_within_one_question": abs(sum(S[u][r][1] for r in sh) - published) <= 1,
            "recoverable_flips": sum(S[u][r][1] != st[r][1] for r in sh), "paired": boot_diff(S[u], st)}
    res["failures"] = {f: len(jl(store.out / f)) for f in
                       ("generation_failures.jsonl", "answer_failures.jsonl", "grade_failures.jsonl")}
    res["ledger"], res["ledger_total"] = ledger_summary(store)
    Path(out_json).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    for f in ("generate", "answer", "analyse", "inputs"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--questions"); ap.add_argument("--sessions")
    ap.add_argument("--out-dir", default=str(DATA / "cap400"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--arms", nargs="*", default=list(SUMMARY_ARMS))
    ap.add_argument("--gens", nargs="*", type=int, default=[1])
    ap.add_argument("--units", nargs="*", default=[], help="arm:gen:draw")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--summary", default=None)
    a = ap.parse_args()
    store = Store(a.out_dir)
    qs, sess = load_inputs(a)
    if a.inputs:
        print(f"questions {len(qs)} sessions {len({q['session_id'] for q in qs})}")
        for sid in sorted({q['session_id'] for q in qs}):
            tx = texts(sid, sess[sid])
            print(f"  {sid[:50]:50s} stripped {L.ntok(tx['stripped']):>8,} prose_chars {len(tx['prose']):>8,} "
                  f"head {L.ntok(truncate_to_budget(tx['prose'], CHARS))} tail {L.ntok(tail_cut(tx['prose'], CHARS))}")
    if a.generate:
        generate(store, qs, sess, a.arms, a.gens, a.workers)
    if a.answer:
        units = [(u.split(":")[0], int(u.split(":")[1]), int(u.split(":")[2])) for u in a.units]
        answer_all(store, qs, sess, units, a.workers)
    if a.analyse:
        analyse(store, a.summary or (Path(a.out_dir) / "summary.json"))


if __name__ == "__main__":
    main()
